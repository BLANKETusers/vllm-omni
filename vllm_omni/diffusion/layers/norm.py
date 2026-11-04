# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.errors import NpuRmsNormError, is_aclnn_rms_norm_tiling_failure

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None
_NPU_RMS_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _tensor_brief(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}(shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
        f"dtype={tensor.dtype}, device={tensor.device}, contiguous={tensor.is_contiguous()})"
    )


def _describe_rms_norm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
) -> str:
    parts = [_tensor_brief("x", x), _tensor_brief("gamma", weight)]
    if residual is not None:
        parts.append(_tensor_brief("residual", residual))
    return "; ".join(parts)


def _rms_norm_launch_problem(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
) -> str | None:
    """Return why aclnnRmsNorm must not be launched, or None when the contract holds.

    These are the constraints CANN reports as 561002 (``ACLNN_ERR_INNER_TILING_ERROR``)
    at tiling time: rank outside 1..8, empty gamma, and a normalized axis that
    does not match gamma. Refusing the launch keeps that failure off the NPU
    stream. A dtype or device mismatch is included because the op requires the
    input and gamma to be the same fp16/bf16/fp32 tensor on one device; that
    check does not by itself explain an intermittent 561002 on legal inputs.
    """
    problems: list[str] = []
    if x.ndim < 1 or x.ndim > 8:
        problems.append(f"input rank {x.ndim} is outside aclnnRmsNorm's 1..8 range")
    if weight.ndim != 1 or weight.numel() == 0:
        problems.append(
            f"gamma must be a non-empty 1D tensor (empty gamma fails tiling), got shape {tuple(weight.shape)}"
        )
    elif x.ndim >= 1 and int(x.shape[-1]) != int(weight.numel()):
        problems.append(
            f"normalized size x.shape[-1]={int(x.shape[-1])} does not equal gamma.numel()={int(weight.numel())}"
        )
    if x.dtype not in _NPU_RMS_NORM_DTYPES:
        problems.append(f"input dtype {x.dtype} is outside fp16/bf16/fp32")
    if weight.dtype not in _NPU_RMS_NORM_DTYPES:
        problems.append(f"gamma dtype {weight.dtype} is outside fp16/bf16/fp32")
    elif x.dtype != weight.dtype:
        problems.append(f"input dtype {x.dtype} does not match gamma dtype {weight.dtype}")
    if weight.device != x.device:
        problems.append(f"gamma device {weight.device} does not match input device {x.device}")
    if residual is not None:
        if residual.shape != x.shape:
            problems.append(f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}")
        if residual.dtype != x.dtype:
            problems.append(f"residual dtype {residual.dtype} does not match input dtype {x.dtype}")
        if residual.device != x.device:
            problems.append(f"residual device {residual.device} does not match input device {x.device}")
    if not problems:
        return None
    return "; ".join(problems)


def _format_npu_rms_norm_error(
    reason: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
    original: BaseException | None = None,
) -> str:
    original_text = f" Original error: {original}." if original is not None else ""
    return (
        "npu_rms_norm/aclnnRmsNorm failed (561002 ACLNN_ERR_INNER_TILING_ERROR: "
        f"CANN tiling rejected the launch). {reason}. {_describe_rms_norm_inputs(x, weight, residual)}."
        f"{original_text}"
    )


def _as_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    # aclnn tiling rejects some non-standard strides with 561002 even when the
    # logical shape is legal. Already-contiguous tensors keep their identity.
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _npu_context_usable() -> bool:
    """Clear a sticky NPU error so a later fp32 fallback can launch.

    ``synchronize()`` re-raises a pending device error. When that happens the
    context is poisoned and the native fallback would fail the same way. An
    uninitialized NPU runtime has nothing to clear.
    """
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None:
        return True
    is_initialized = getattr(npu_mod, "is_initialized", None)
    if callable(is_initialized) and not is_initialized():
        return True
    synchronize = getattr(npu_mod, "synchronize", None)
    if not callable(synchronize):
        return True
    try:
        synchronize()
    except Exception as exc:
        logger.warning("NPU synchronize after aclnnRmsNorm 561002 failed: %s", exc)
        return False
    return True


class LayerNorm(nn.LayerNorm, CustomOp):
    """
    LayerNorm implementation that inherits from both ``nn.LayerNorm`` and ``CustomOp``.
    NPU:
        Uses ``mindiesd.fast_layernorm(self, x)`` when MindIE-SD is installed.
    CUDA / HIP / XPU / native:
        Falls back to FP32 nn.LayerNorm implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__(  # type: ignore[call-arg]
            normalized_shape=dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        # CustomOp.__init__ cannot be called here because it would re-run
        # nn.Module initialization and clear LayerNorm parameters.
        self._forward_method = CustomOp.dispatch_forward(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_method(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import fast_layernorm

                return fast_layernorm(self, x)
            except ImportError as e:
                # warning_once caches its arguments; pass a string so the cache
                # does not retain the traceback frame holding ``x``.
                logger.warning_once(
                    "mindiesd.fast_layernorm import failed, falling back to FP32 layer_norm: %s",
                    str(e),
                )

        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNorm(CustomOp):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        weight_dtype = dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.hidden_size = hidden_size
        self.variance_epsilon = eps

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        from vllm._custom_ops import rms_norm as fused_rms_norm

        orig_shape = x.shape
        hidden_size = orig_shape[-1]
        x_2d = x.reshape(-1, hidden_size)
        out = torch.empty_like(x_2d)
        fused_rms_norm(out, x_2d, self.weight.data, self.variance_epsilon)
        return out.reshape(orig_shape)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return self.forward_native(x, residual)
        # During torch.compile tracing, fused_rms_norm writes to `out` in-place
        # (returns None) and accesses self.weight.data, which is a DTensor under
        # HSDP. Both patterns confuse inductor's compute_ancestors scheduler.
        # Fall back to forward_native so inductor can fuse the pure-PyTorch ops
        # itself. The fused kernel also cannot launch on an empty tensor.
        if torch.compiler.is_compiling() or x.numel() == 0:
            return self.forward_native(x)
        try:
            return self._forward_fused(x)
        except Exception:
            return self.forward_native(x)

    def forward_musa(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return self.forward_native(x, residual)
        # Preserve the aten::rms_norm graph so dynamic Inductor can fuse the
        # H3 Q/K norm with its inline RoPE path on MUSA.
        return F.rms_norm(
            x,
            (self.hidden_size,),
            self.weight,
            self.variance_epsilon,
        )

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not _is_npu_tensor(x):
            return self.forward_native(x, residual)
        problem = _rms_norm_launch_problem(x, self.weight, residual)
        if problem is not None:
            raise NpuRmsNormError(_format_npu_rms_norm_error(problem, x, self.weight, residual))
        # Empty gamma is rejected above. An empty activation (numel 0) with a
        # matching last dim is valid RMSNorm math, but aclnnRmsNorm tiling
        # rejects it on several CANN versions.
        if x.numel() == 0 or (residual is not None and residual.numel() == 0):
            return self.forward_native(x, residual)
        import torch_npu

        x_op = _as_contiguous(x)
        weight = _as_contiguous(self.weight)
        try:
            if residual is not None:
                output, _, updated_residual = torch_npu.npu_add_rms_norm(
                    x_op,
                    _as_contiguous(residual),
                    weight,
                    self.variance_epsilon,
                )
                return output, updated_residual
            return torch_npu.npu_rms_norm(x_op, gamma=weight, epsilon=self.variance_epsilon)[0]
        except RuntimeError as exc:
            if not is_aclnn_rms_norm_tiling_failure(exc):
                raise
            # Identical shapes succeed outside the reported windows, so this is
            # treated as an intermittent CANN tiling fault. The fp32 reference
            # is the same math CUDA uses when its fused kernel cannot launch.
            # A poisoned context is not recovered that way: surface it instead.
            detail = _format_npu_rms_norm_error(
                "aclnnRmsNorm returned 561002 for inputs that satisfy the launch contract",
                x,
                self.weight,
                residual,
                exc,
            )
            if not _npu_context_usable():
                raise NpuRmsNormError(
                    detail + " NPU context remained unusable after synchronize; the worker needs a restart."
                ) from exc
            logger.warning_once(
                "aclnnRmsNorm error 561002 (ACLNN_ERR_INNER_TILING_ERROR); using the fp32 RMSNorm fallback. %s",
                detail,
            )
            return self.forward_native(x, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            residual = residual + x
            x = residual
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight.to(torch.float32) * out
        out = out.to(input_dtype)
        return (out, residual) if residual is not None else out

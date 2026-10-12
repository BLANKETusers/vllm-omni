# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.platforms import current_platform

from vllm_omni.diffusion.lora.layers import (
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
    DiffusionRowParallelLinearWithLoRA,
)


def _select_moe_lora_wrapper_cls() -> type:
    """Pick the MoE LoRA wrapper class for the current platform.

    GPU reuses upstream vLLM's native ``FusedMoEWithLoRA``. NPU uses
    ``AscendFusedMoEWithLoRA`` from vllm-ascend, imported directly here.
    """
    if current_platform.device_type == "npu":
        from vllm_ascend.lora.fused_moe import AscendFusedMoEWithLoRA

        return AscendFusedMoEWithLoRA
    return FusedMoEWithLoRA


def _move_moe_lora_buffers(instance: FusedMoEWithLoRA, device: torch.device) -> None:
    """Keep MoE LoRA sidecars resident while DLO moves base weights.

    The upstream wrapper keeps the per-expert matrices in separate W13 and W2
    stacks. ``lora_a_stacked`` / ``lora_b_stacked`` are views into those
    tensors, so regenerate them after moving the source stacks.
    """
    for attr in (
        "w13_lora_a_stacked",
        "w13_lora_b_stacked",
        "w2_lora_a_stacked",
        "w2_lora_b_stacked",
    ):
        buffers = getattr(instance, attr)
        setattr(instance, attr, tuple(buffer.to(device=device) for buffer in buffers))
    instance.adapter_enabled = instance.adapter_enabled.to(device=device)

    lora_a_stacked: list[torch.Tensor] = []
    lora_b_stacked: list[torch.Tensor] = []
    for lora_id in range(instance.max_loras):
        for expert_id in range(instance.local_num_experts):
            w13_a_expert_id = 0 if instance.enable_moe_shared_loras else expert_id
            w2_b_expert_id = 0 if instance.enable_moe_shared_loras else expert_id
            lora_a_stacked.extend(
                (
                    instance.w13_lora_a_stacked[0][lora_id][w13_a_expert_id],
                    instance.w2_lora_a_stacked[0][lora_id][expert_id],
                )
            )
            lora_b_stacked.extend(
                (
                    instance.w13_lora_b_stacked[0][lora_id][expert_id],
                    instance.w2_lora_b_stacked[0][lora_id][w2_b_expert_id],
                )
            )
            if instance._w13_slices == 2:
                lora_a_stacked.append(instance.w13_lora_a_stacked[1][lora_id][w13_a_expert_id])
                lora_b_stacked.append(instance.w13_lora_b_stacked[1][lora_id][expert_id])
    instance.lora_a_stacked = lora_a_stacked
    instance.lora_b_stacked = lora_b_stacked


def _match_target_modules(module_name: str, target_modules: list[str]) -> bool:
    """from vllm/lora/model_manager.py _match_target_modules, helper function"""
    import regex as re

    return any(
        re.match(rf".*\.{target_module}$", module_name) or target_module == module_name
        for target_module in target_modules
    )


def _expand_expected_modules_for_packed_layers(
    supported_modules: set[str],
    packed_modules_mapping: dict[str, list[str]] | None,
) -> set[str]:
    """Expand expected LoRA module suffixes for packed (fused) projections.

    Some diffusion models use packed projections like `to_qkv` or `w13`, while
    LoRA checkpoints are typically saved against the logical sub-projections
    (e.g. `to_q`/`to_k`/`to_v`, `w1`/`w3`). The packed layer name is present in
    `supported_modules`, but the sublayer names are not. Expanding the set
    ensures these sublayer keys are not dropped when loading a LoRA checkpoint.

    The packed→sublayer mapping is model-specific and is derived from each
    diffusion model's `stacked_params_mapping` (used by `load_weights()`), so
    new packed layers are added alongside the model implementation rather than
    hard-coded in the LoRA framework.
    """
    expanded = set(supported_modules)
    if not packed_modules_mapping:
        return expanded

    for packed_name, sub_names in packed_modules_mapping.items():
        if packed_name in supported_modules:
            expanded.update(sub_names)

    return expanded


def from_layer_diffusion(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list[str],
    model_config: PretrainedConfig | None = None,
) -> nn.Module:
    """
    Diffusion-specific layer replacement. similar to vLLM's `from_layer`
    """
    # MoE runner bridge: upstream vLLM's FusedMoEWithLoRA (GPU) /
    # vllm-ascend's AscendFusedMoEWithLoRA (NPU) already wrap MoERunner. The
    # diffusion manager only needs to select the platform wrapper and call it
    # directly — omni does not maintain a second MoE LoRA compute. The MoE
    # wrapper carries its own target semantics (gate_up_proj/down_proj), so
    # packed_modules_list is irrelevant here; the branch must come before the
    # dense classes so a MoERunner is not mistaken for a dense linear.
    if isinstance(layer, MoERunner):
        wrapper_cls = _select_moe_lora_wrapper_cls()
        instance = wrapper_cls(layer)
        buffer_device: list[torch.device | None] = [None]
        original_create_lora_weights = instance.create_lora_weights

        def _set_diffusion_lora_buffer_device(device: torch.device) -> None:
            buffer_device[0] = device
            _move_moe_lora_buffers(instance, device)

        def _create_lora_weights(max_loras, lora_config, model_config=None) -> None:
            original_create_lora_weights(max_loras, lora_config, model_config)
            if buffer_device[0] is not None:
                _move_moe_lora_buffers(instance, buffer_device[0])

        instance._set_diffusion_lora_buffer_device = _set_diffusion_lora_buffer_device  # type: ignore[method-assign]
        instance.create_lora_weights = _create_lora_weights  # type: ignore[method-assign]
        instance.create_lora_weights(max_loras, lora_config, model_config)
        # Runtime context forwarding: upstream FusedMoEWithLoRA.forward()
        # delegates to ``base_layer.forward(*args, **kwargs)`` directly. A
        # direct ``.forward()`` call bypasses ``nn.Module.__call__``, so the
        # runner's forward pre-hooks never fire. Those hooks are load-bearing:
        # omni's ``_num_tokens_pre_hook`` sets ForwardContext.num_tokens, and
        # on NPU ``fused_moe_forward_context_pre_hook`` installs
        # ``moe_comm_method`` per-forward — without it the routed-experts
        # forward crashes with "'NoneType' object has no attribute 'prepare'".
        # Override the wrapper's forward to route through ``base_layer(...)``
        # (i.e. ``__call__``) so the pre-hooks fire as intended. Assert the
        # runner still carries its pre-hooks so a future upstream change that
        # strips them is caught.
        assert len(layer._forward_pre_hooks) > 0, (
            "MoERunner lost its forward pre-hooks after LoRA wrapping; "
            "ForwardContext.num_tokens / NPU moe_comm_method would be "
            "uninitialized."
        )

        # Bind as a closure, not ``instance.forward = fn`` with a ``self``
        # param: a function assigned as an *instance* attribute is not
        # descriptor-bound, so ``nn.Module._call_impl`` would hand the first
        # positional arg to ``self``. The MoE wrapper is called kwargs-only
        # (``self.experts(hidden_states=..., router_logits=...)``), so that form
        # raises ``TypeError: missing 1 required positional argument: 'self'``.
        # The closure captures ``instance`` directly; ``instance.base_layer(...)``
        # goes through ``nn.Module.__call__`` so the runner's forward pre-hooks
        # fire as intended.
        def _forward_via_base_call(*args, **kwargs):
            return instance.base_layer(*args, **kwargs)

        instance.forward = _forward_via_base_call  # type: ignore[method-assign]

        # Suspend/resume fast path: omni's DiffusionLoRAManager toggles an
        # adapter off/on without re-uploading weights via suspend_lora/
        # resume_lora (the dense layers gate a slice mask read by apply()).
        # Upstream FusedMoEWithLoRA defines neither — its on/off switch is the
        # ``adapter_enabled`` tensor that the routed-experts LoRA delta
        # injection reads (moe_lora_apply_w13/w2 pass it as adapter_enabled).
        # Mirror the dense semantics: suspend flips the bound slot's flag to 0
        # (delta suppressed), resume restores it. The toggle is in-place on the
        # same tensor object the per-layer MoELoRAContext references, so once
        # set_mapping publishes the context the forward path observes the
        # change without rebuilding it. Bound as closures taking no ``self``
        # param for the same descriptor reason as ``forward`` above: an
        # instance-attribute function is not descriptor-bound, and the manager
        # calls these with no positional args.
        _suspended: list[int | None] = [None]

        def _suspend_lora() -> None:
            if _suspended[0] is not None:
                return
            _suspended[0] = int(instance.adapter_enabled[0].item())
            instance.adapter_enabled[0] = 0

        def _resume_lora() -> None:
            prev = _suspended[0]
            if prev is None:
                return
            instance.adapter_enabled[0] = prev
            _suspended[0] = None

        instance.suspend_lora = _suspend_lora  # type: ignore[method-assign]
        instance.resume_lora = _resume_lora  # type: ignore[method-assign]
        return instance

    diffusion_lora_classes = [
        DiffusionMergedQKVParallelLinearWithLoRA,
        DiffusionQKVParallelLinearWithLoRA,
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionColumnParallelLinearWithLoRA,
        DiffusionRowParallelLinearWithLoRA,
        DiffusionReplicatedLinearWithLoRA,
    ]

    for lora_cls in diffusion_lora_classes:
        if lora_cls.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance = lora_cls(layer)  # type: ignore[arg-type]
            instance.create_lora_weights(max_loras, lora_config, model_config)
            return instance

    return layer

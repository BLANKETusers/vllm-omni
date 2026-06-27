# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Used by Hop1 (GPU worker <-> scheduler) to avoid pickling large video tensors
through the MessageQueue. Tensors above ``_SHM_TENSOR_THRESHOLD`` are copied
into a named shared-memory segment; only a lightweight metadata dict is
serialised through the queue.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput

logger = init_logger(__name__)

_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB
DIFFUSION_RPC_RESULT_ENVELOPE = "diffusion_rpc_result"


def _tensor_to_shm(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy a tensor into POSIX shared memory and return a metadata handle.

    The shared memory segment remains alive after this call (the local fd is
    closed, but the segment persists until ``_tensor_from_shm`` unlinks it).
    """
    from multiprocessing import shared_memory

    import numpy as np

    tensor = tensor.detach().cpu().contiguous()
    original_dtype = tensor.dtype
    # NumPy does not support bfloat16; promote to float32 for the SHM
    # transfer and record the original dtype so _tensor_from_shm can
    # convert back.  The round-trip is lossless for bfloat16 values.
    if original_dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    arr = tensor.numpy()
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:nbytes])
    np.copyto(shm_arr, arr)
    handle = {
        "__tensor_shm__": True,
        "name": shm.name,
        "shape": list(tensor.shape),
        "torch_dtype": str(original_dtype),
        "numpy_dtype": str(arr.dtype),
        "nbytes": nbytes,
    }
    shm.close()
    return handle


def _tensor_from_shm(handle: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from a shared-memory handle and free the segment."""
    from multiprocessing import shared_memory

    import numpy as np

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        np_dtype = np.dtype(handle["numpy_dtype"])
        arr = np.ndarray(handle["shape"], dtype=np_dtype, buffer=shm.buf[: handle["nbytes"]])
        tensor = torch.from_numpy(arr.copy())
        # Restore the original dtype if it differs from the numpy-compatible
        # dtype used for the SHM transfer (e.g. bfloat16 → float32 → bfloat16).
        torch_dtype_str = handle.get("torch_dtype", "")
        if torch_dtype_str:
            original_dtype = getattr(torch, torch_dtype_str.replace("torch.", ""), None)
            if original_dtype is not None and tensor.dtype != original_dtype:
                tensor = tensor.to(original_dtype)
    finally:
        shm.close()
        shm.unlink()
    return tensor


# Deferred SHM: enqueue empty shell, fill in background thread so _busy_loop
# can start next request's GPU work without waiting for D2H.


def _create_deferred_shm(tensor: torch.Tensor) -> dict[str, Any]:
    """Create an empty SHM shell + ready flag for deferred D2H."""
    from multiprocessing import shared_memory

    original_dtype = tensor.dtype
    if original_dtype == torch.bfloat16:
        nbytes = tensor.nelement() * 4  # promoted to float32
        numpy_dtype = "float32"
    else:
        nbytes = tensor.nelement() * tensor.element_size()
        numpy_dtype = str(torch.zeros(1, dtype=original_dtype).numpy().dtype)

    shm_name: str | None = None
    try:
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        shm_name = shm.name
        shm.close()

        # 1-byte ready flag SHM (0=pending, 1=filled, 2=error)
        ready_shm = shared_memory.SharedMemory(create=True, size=1)
        ready_shm.buf[0] = 0
        ready_shm.close()
    except Exception:
        # Unlink the data segment if ready_shm creation fails, so it
        # does not leak in /dev/shm forever.
        if shm_name is not None:
            try:
                lingering = shared_memory.SharedMemory(name=shm_name)
                lingering.close()
                lingering.unlink()
            except Exception:
                pass
        raise

    return {
        "__tensor_shm_deferred__": True,
        "name": shm.name,
        "ready_name": ready_shm.name,
        "shape": list(tensor.shape),
        "torch_dtype": str(original_dtype),
        "numpy_dtype": numpy_dtype,
        "nbytes": nbytes,
    }


def _fill_deferred_shm(
    handle: dict[str, Any],
    tensor: torch.Tensor,
    gpu_event: torch.cuda.Event | None = None,
) -> None:
    """Fill deferred SHM via side CUDA stream + pinned memory.

    *gpu_event* is an optional event recorded on the default stream at the
    point where the tensor was produced.  The side stream waits on it so that
    the copy does not start until the default stream has finished writing the
    tensor — without this cross-stream ordering the read may see partially
    written data and produce corrupted output.
    """
    from multiprocessing import shared_memory

    import numpy as np

    original_dtype = tensor.dtype
    if tensor.is_cuda:
        # Pinned staging buffer + side stream: D2H never blocks the default stream.
        cpu_buf = torch.empty(tensor.shape, dtype=original_dtype, pin_memory=True)
        d2h_stream = torch.cuda.Stream()
        with torch.cuda.stream(d2h_stream):
            # Cross-stream ordering: the producer (default stream) must finish
            # writing the tensor before this side-stream consumer reads it.
            if gpu_event is not None:
                d2h_stream.wait_event(gpu_event)
            cpu_buf.copy_(tensor.detach(), non_blocking=True)
            event = d2h_stream.record_event()
        # Synchronize only the side stream, not the default stream.
        event.synchronize()
        del d2h_stream
    else:
        cpu_buf = tensor.detach().cpu()

    # Promote bfloat16 to float32 for NumPy.
    if original_dtype == torch.bfloat16:
        cpu_buf = cpu_buf.to(torch.float32)

    arr = cpu_buf.contiguous().numpy()

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[: handle["nbytes"]])
        np.copyto(shm_arr, arr)
    finally:
        shm.close()

    # Signal: data is ready.
    ready_shm = shared_memory.SharedMemory(name=handle["ready_name"])
    ready_shm.buf[0] = 1
    ready_shm.close()


def _is_deferred_handle(val: object) -> bool:
    """Return True if *val* is a deferred SHM handle dict."""
    return isinstance(val, dict) and bool(val.get("__tensor_shm_deferred__"))


def _resolve_deferred_shm(handle: dict[str, Any]) -> torch.Tensor:
    """Block until deferred SHM is filled, then reconstruct the tensor.

    Polls with time.sleep(0.001) for the D2H duration (typically 14-60ms),
    blocking the async event loop. Acceptable for the current single-request
    serial model (max_num_seqs=1); for concurrent requests or streaming,
    callers should offload this via loop.run_in_executor so the event loop
    stays responsive.
    """
    import time
    from multiprocessing import shared_memory

    import numpy as np

    deadline = time.monotonic() + 30.0
    while True:
        try:
            ready_shm = shared_memory.SharedMemory(name=handle["ready_name"])
        except FileNotFoundError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Deferred SHM '{handle['name']}' not ready after 30 s")
            time.sleep(0.001)
            continue
        try:
            if ready_shm.buf[0] == 1:
                break
            if ready_shm.buf[0] == 2:
                raise RuntimeError(f"Deferred SHM '{handle['name']}' fill failed on worker")
        finally:
            ready_shm.close()
        if time.monotonic() > deadline:
            raise TimeoutError(f"Deferred SHM '{handle['name']}' not ready after 30 s")
        time.sleep(0.001)

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        np_dtype = np.dtype(handle["numpy_dtype"])
        arr = np.ndarray(handle["shape"], dtype=np_dtype, buffer=shm.buf[: handle["nbytes"]])
        tensor = torch.from_numpy(arr.copy())
        torch_dtype_str = handle.get("torch_dtype", "")
        if torch_dtype_str:
            original_dtype = getattr(torch, torch_dtype_str.replace("torch.", ""), None)
            if original_dtype is not None and tensor.dtype != original_dtype:
                tensor = tensor.to(original_dtype)
    finally:
        shm.close()
        shm.unlink()

    # Clean up the ready-flag SHM.
    try:
        ready_shm = shared_memory.SharedMemory(name=handle["ready_name"])
        ready_shm.close()
        ready_shm.unlink()
    except FileNotFoundError:
        pass

    return tensor


def _resolve_deferred_tree(val: object) -> object:
    """Recursively resolve deferred SHM handles, mirroring ``_unpack_if_shm_handle``."""
    if _is_deferred_handle(val):
        return _resolve_deferred_shm(val)
    if isinstance(val, dict):
        return {key: _resolve_deferred_tree(value) for key, value in val.items()}
    if isinstance(val, list):
        return [_resolve_deferred_tree(item) for item in val]
    if isinstance(val, tuple):
        return tuple(_resolve_deferred_tree(item) for item in val)
    return val


def resolve_deferred_outputs(output_data: object) -> object:
    """Resolve any deferred SHM handles in *output_data* tree before postprocess."""
    return _resolve_deferred_tree(output_data)


def _pack_tensor_deferred(val: torch.Tensor) -> dict:
    """Always pack tensor as a deferred SHM handle (no D2H here)."""
    return _create_deferred_shm(val)


def _pack_value_deferred(
    val: object,
    deferred_items: list[tuple[dict[str, Any], torch.Tensor]],
) -> object:
    """Recursively replace large tensors with deferred SHM handles."""
    if isinstance(val, torch.Tensor):
        if val.nelement() * val.element_size() > _SHM_TENSOR_THRESHOLD:
            handle = _pack_tensor_deferred(val)
            deferred_items.append((handle, val))
            return handle
        return val
    if isinstance(val, dict):
        return {key: _pack_value_deferred(value, deferred_items) for key, value in val.items()}
    if isinstance(val, list):
        return [_pack_value_deferred(item, deferred_items) for item in val]
    if isinstance(val, tuple):
        return tuple(_pack_value_deferred(item, deferred_items) for item in val)
    return val


def _pack_diffusion_fields_deferred(
    output: DiffusionOutput,
) -> list[tuple[dict[str, Any], torch.Tensor]]:
    """Replace large tensor fields in DiffusionOutput with deferred SHM handles."""
    deferred_items: list[tuple[dict[str, Any], torch.Tensor]] = []
    if output.output is not None:
        output.output = _pack_value_deferred(output.output, deferred_items)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
        output.trajectory_latents = _pack_value_deferred(output.trajectory_latents, deferred_items)
    if output.trajectory_timesteps is not None and isinstance(output.trajectory_timesteps, torch.Tensor):
        output.trajectory_timesteps = _pack_value_deferred(output.trajectory_timesteps, deferred_items)
    if output.trajectory_log_probs is not None and isinstance(output.trajectory_log_probs, torch.Tensor):
        output.trajectory_log_probs = _pack_value_deferred(output.trajectory_log_probs, deferred_items)
    return deferred_items


def pack_diffusion_output_shm_deferred(
    output: object,
) -> list[tuple[dict[str, Any], torch.Tensor]]:
    """Deferred SHM variant of pack_diffusion_output_shm; returns (handle, tensor) pairs."""
    if isinstance(output, DiffusionOutput):
        return _pack_diffusion_fields_deferred(output)

    if _is_rpc_result_envelope(output):
        result = output.get("result")
        if isinstance(result, DiffusionOutput):
            return _pack_diffusion_fields_deferred(result)

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        return _pack_diffusion_fields_deferred(result)

    return []


def _signal_deferred_error(handle: dict[str, Any]) -> None:
    """Signal consumer that a deferred SHM fill failed.

    Sets the ready flag to 2 (error) so the consumer can fail fast instead
    of waiting for the 30 s timeout.  Best-effort — must not raise.
    """
    from multiprocessing import shared_memory

    try:
        ready_shm = shared_memory.SharedMemory(name=handle["ready_name"])
        ready_shm.buf[0] = 2
        ready_shm.close()
    except Exception:
        pass


def _fill_deferred_handles(
    items: list[tuple[dict[str, Any], torch.Tensor]],
    gpu_event: torch.cuda.Event | None = None,
) -> None:
    """Fill all deferred SHM handles (called from worker background thread)."""
    for handle, tensor in items:
        try:
            _fill_deferred_shm(handle, tensor, gpu_event=gpu_event)
        except Exception:
            logger.exception(
                "Deferred SHM fill failed for handle '%s'; signalling error to consumer",
                handle.get("name", "unknown"),
            )
            # Signal error (2) so the consumer does not hang for 30 s.
            _signal_deferred_error(handle)


def _pack_tensor_if_large(val: torch.Tensor) -> torch.Tensor | dict:
    """Replace a tensor with an SHM handle if it exceeds the threshold."""
    if val.nelement() * val.element_size() > _SHM_TENSOR_THRESHOLD:
        return _tensor_to_shm(val)
    return val


def _pack_value_if_large(val: object) -> object:
    """Recursively replace large tensors with SHM handles.

    Walks the container shapes pipelines return as ``DiffusionOutput.output``:
    bare tensors, dicts (e.g. Cosmos3 ``{"image"/"video": ...}``), and
    tuples/lists (e.g. LTX2 and DreamID ``(video, audio)``). Other values pass
    through unchanged. ``_unpack_if_shm_handle`` must mirror these shapes — keep
    the two in sync.
    """
    if isinstance(val, torch.Tensor):
        return _pack_tensor_if_large(val)
    if isinstance(val, dict):
        return {key: _pack_value_if_large(value) for key, value in val.items()}
    if isinstance(val, list):
        return [_pack_value_if_large(item) for item in val]
    if isinstance(val, tuple):
        return tuple(_pack_value_if_large(item) for item in val)
    return val


def _unpack_if_shm_handle(val: object) -> object:
    """Reconstruct tensors from SHM handles; deferred handles pass through for later resolve."""
    if isinstance(val, dict) and val.get("__tensor_shm_deferred__"):
        return val  # resolved later in step() → postprocess_output
    if isinstance(val, dict) and val.get("__tensor_shm__"):
        return _tensor_from_shm(val)
    if isinstance(val, dict):
        return {key: _unpack_if_shm_handle(value) for key, value in val.items()}
    if isinstance(val, list):
        return [_unpack_if_shm_handle(item) for item in val]
    if isinstance(val, tuple):
        return tuple(_unpack_if_shm_handle(item) for item in val)
    return val


def _pack_diffusion_fields(output: DiffusionOutput) -> DiffusionOutput:
    if output.output is not None:
        output.output = _pack_value_if_large(output.output)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
        output.trajectory_latents = _pack_tensor_if_large(output.trajectory_latents)
    if output.trajectory_timesteps is not None and isinstance(output.trajectory_timesteps, torch.Tensor):
        output.trajectory_timesteps = _pack_tensor_if_large(output.trajectory_timesteps)
    if output.trajectory_log_probs is not None and isinstance(output.trajectory_log_probs, torch.Tensor):
        output.trajectory_log_probs = _pack_tensor_if_large(output.trajectory_log_probs)
    return output


def _is_rpc_result_envelope(output: object) -> bool:
    return isinstance(output, dict) and output.get("type") == DIFFUSION_RPC_RESULT_ENVELOPE


def pack_diffusion_output_shm(output: object) -> object:
    """Replace large tensors in diffusion worker outputs with SHM handles.

    Supports either a bare ``DiffusionOutput`` or a wrapper object carrying one
    in ``.result`` (for example ``RunnerOutput``), or an RPC result envelope
    carrying the diffusion output in ``["result"]``.
    """
    if isinstance(output, DiffusionOutput):
        return _pack_diffusion_fields(output)

    if _is_rpc_result_envelope(output):
        result = output.get("result")
        if isinstance(result, DiffusionOutput):
            output["result"] = _pack_diffusion_fields(result)
        return output

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _pack_diffusion_fields(result)
    return output


def _unpack_diffusion_fields(output: DiffusionOutput) -> DiffusionOutput:
    output.output = _unpack_if_shm_handle(output.output)
    output.trajectory_latents = _unpack_if_shm_handle(output.trajectory_latents)
    output.trajectory_timesteps = _unpack_if_shm_handle(output.trajectory_timesteps)
    output.trajectory_log_probs = _unpack_if_shm_handle(output.trajectory_log_probs)
    return output


def unpack_diffusion_output_shm(output: object) -> object:
    """Reconstruct tensors from SHM handles in diffusion worker outputs."""
    if isinstance(output, DiffusionOutput):
        return _unpack_diffusion_fields(output)

    if _is_rpc_result_envelope(output):
        result = output.get("result")
        if isinstance(result, DiffusionOutput):
            output["result"] = _unpack_diffusion_fields(result)
        return output

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _unpack_diffusion_fields(result)
    return output

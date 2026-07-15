# Plan: Extend Async Diffusion Output to `execute_model_batch` and `execute_stepwise`

## Problem

When `--enable-async-diffusion-output` is enabled, only `execute_model` (single-request path) uses the async D2H pipeline. Both `execute_model_batch` (used by Qwen-Image) and `execute_stepwise` bypass the async path entirely, resulting in synchronous `Memcpy DtoH (Device -> Pageable)` blocking the GPU default stream alongside `pipeline_forward_batch`.

This means:
- GPU compute finishes → D2H memcpy starts on default stream → blocks next request's GPU forward
- The async optimization (COMPUTE_DONE signal → background thread D2H on side stream) is completely unused

## Root Cause (3 locations)

### 1. Executor gate condition (multiproc_executor.py:458)
```python
if self.od_config.enable_async_diffusion_output and method == "execute_model":
```
Only `execute_model` triggers the async Path 1. `execute_model_batch` and `execute_stepwise` fall through to Path 2/3 (sync).

### 2. Worker return_result gate (diffusion_worker.py:847)
```python
if self.od_config.enable_async_diffusion_output and isinstance(output, DiffusionOutput):
```
`execute_model_batch` returns `BatchRunnerOutput`, not `DiffusionOutput`, so this check fails.

### 3. DiffusionEngine._busy_loop output handling (diffusion_engine.py:424)
The busy loop calls `self.execute_fn(sched_output)` which returns `BaseRunnerOutput`. For batch mode, `execute_fn = executor.execute_batch` which returns a `BatchRunnerOutput`. The async path in `step()` (lines 260-262) checks `output.output_token`, but `_busy_loop` → `_handle_finished_requests` → `_finalize_finished_request` only produces `output_token` when the **RunnerOutput** (per-request) has it, not when the `BatchRunnerOutput` wrapper has it.

## Design

The key insight: **async output should be per-DiffusionOutput, not per-RPC-method**. Whether the worker returns one `DiffusionOutput` or a `BatchRunnerOutput` containing N `DiffusionOutput`s, each `DiffusionOutput` that has GPU tensors should go through the background D2H thread.

### Approach: "Wrap at DiffusionOutput level, not at RPC level"

Instead of gating on `method == "execute_model"`, we gate on whether the **response contains DiffusionOutput(s) with GPU tensors**. The Worker's `return_result` already has the right intuition — check `isinstance(output, DiffusionOutput)`. We extend it to also handle `BatchRunnerOutput`.

## Changes (4 files)

### File 1: `vllm_omni/diffusion/worker/diffusion_worker.py` — WorkerProc.return_result

**Current logic (line 847-868):**
- Only `DiffusionOutput` triggers async path
- `BatchRunnerOutput` falls through to sync path

**New logic:**
- Helper `_has_diffusion_output(output)` checks if output carries any `DiffusionOutput` with GPU tensors
- For `DiffusionOutput`, `BatchRunnerOutput`, or `RunnerOutput` with a `DiffusionOutput` result: submit to async queue
- The background thread packs ALL `DiffusionOutput`s inside the output using `pack_diffusion_output_shm(output, d2h_stream=d2h_stream)`
- Send `COMPUTE_DONE` with `output_token` and `rpc_id`

```python
def _has_diffusion_output(output: Any) -> bool:
    """Check if output contains a DiffusionOutput that needs async D2H."""
    if isinstance(output, DiffusionOutput):
        return True
    if isinstance(output, RunnerOutput):
        return isinstance(output.result, DiffusionOutput)
    if isinstance(output, BatchRunnerOutput):
        return any(isinstance(ro.result, DiffusionOutput) for ro in output.runner_outputs)
    return False

def return_result(self, output: Any, rpc_id: str | None = None):
    if self.result_mq is None:
        return
    if isinstance(output, OmniACK):
        self.result_mq.enqueue(output)
        return

    # Async path: handle DiffusionOutput, BatchRunnerOutput, and RunnerOutput
    if self.od_config.enable_async_diffusion_output and _has_diffusion_output(output):
        try:
            output_token = WorkerProc._generate_output_token()
            gpu_event = WorkerProc._record_gpu_event()
            self._async_output_queue.put((output, output_token, gpu_event))
            msg = AsyncDiffusionOutput(
                kind=AsyncDiffusionOutput.COMPUTE_DONE,
                rpc_id=rpc_id,
                output_token=output_token,
            )
            self.result_mq.enqueue(msg)
            return
        except Exception as e:
            logger.warning("Async output submission failed, falling back to sync: %s", e)

    # Sync path (original, or async fallback)
    ...
```

### File 2: `vllm_omni/diffusion/ipc.py` — pack_diffusion_output_shm

**Current logic:** Only handles `DiffusionOutput` and `dict` envelopes.

**New logic:** Also handle `BatchRunnerOutput` — walk `runner_outputs` and pack each `RunnerOutput.result` (which is a `DiffusionOutput`) via `_pack_diffusion_fields`.

```python
def pack_diffusion_output_shm(output, d2h_stream=None):
    if isinstance(output, DiffusionOutput):
        return _pack_diffusion_fields(output, d2h_stream=d2h_stream)
    if isinstance(output, BatchRunnerOutput):
        for runner_output in output.runner_outputs:
            if runner_output.result is not None:
                _pack_diffusion_fields(runner_output.result, d2h_stream=d2h_stream)
        return output
    ...
```

Same for `unpack_diffusion_output_shm`:

```python
def unpack_diffusion_output_shm(output):
    if isinstance(output, BatchRunnerOutput):
        for runner_output in output.runner_outputs:
            if runner_output.result is not None:
                _unpack_diffusion_fields(runner_output.result)
        return output
    ...
```

### File 3: `vllm_omni/diffusion/executor/multiproc_executor.py` — collective_rpc async gate

**Current logic (line 458):**
```python
if self.od_config.enable_async_diffusion_output and method == "execute_model":
```

**New logic:** Extend to also cover `execute_model_batch` and `execute_stepwise`:

```python
_ASYNC_METHODS = {"execute_model", "execute_model_batch", "execute_stepwise"}

if self.od_config.enable_async_diffusion_output and method in _ASYNC_METHODS:
```

Also need to update `execute_request` and `execute_batch` to handle the new async response shape. Currently `execute_request` checks for `AsyncDiffusionOutput(COMPUTE_DONE)` and produces a `RunnerOutput(output_token=...)`. We need `execute_batch` to do the same for `BatchRunnerOutput` responses:

In `execute_batch` (line 394-411):
```python
def execute_batch(self, scheduler_output):
    result = self.collective_rpc(...)
    if isinstance(result, AsyncDiffusionOutput) and result.kind == AsyncDiffusionOutput.COMPUTE_DONE:
        # Batch async: all requests in this batch share one output_token
        # The DiffusionEngine._busy_loop will handle output_token via _finalize_finished_request
        runner_outputs = []
        for new_req in scheduler_output.scheduled_new_reqs:
            runner_outputs.append(
                RunnerOutput(
                    request_id=new_req.request_id,
                    step_index=None,
                    finished=True,
                    result=None,
                    output_token=result.output_token,
                )
            )
        return BatchRunnerOutput.from_list(runner_outputs)
    if not isinstance(result, BatchRunnerOutput):
        raise RuntimeError(...)
    return result
```

### File 4: `vllm_omni/diffusion/diffusion_engine.py` — step/step_streaming output_token handling

**Current logic (lines 260-262):**
```python
if output.output_token:
    fut = self.executor.wait_output_ready(output.output_token)
    output = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=_ASYNC_OUTPUT_TIMEOUT)
```

This works for single-request mode. For batch mode, `_finalize_finished_request` already produces `DiffusionOutput(output_token=runner_output.output_token)` for each request (line 1102-1103). The `step()` method then checks `output.output_token` and waits. This should already work correctly once the executor changes are in place.

**However, there's a subtlety:** In batch mode with `_busy_loop`, multiple requests finish simultaneously. Each request's `DiffusionOutput` in the `_out_queue` future will carry `output_token`. When `step()` is called per-request in the orchestrator, each one will call `wait_output_ready(output_token)`. But all requests in the same batch share the same `output_token` from the Worker.

**This is fine** because `wait_output_ready` handles this:
- First `wait_output_ready` call creates a Future in `_output_futures`
- Worker sends one `OUTPUT_READY` with the batch's `BatchRunnerOutput`
- The `_result_pump` receives it and sets the Future result with the full `BatchRunnerOutput`

Wait — there's a problem. `_result_pump` currently assumes `OUTPUT_READY.output` is a single `DiffusionOutput`. We need it to also handle `BatchRunnerOutput`.

In `_result_pump` (lines 553-574), when it receives `OUTPUT_READY`:
```python
elif msg.kind == AsyncDiffusionOutput.OUTPUT_READY:
    with self._futures_lock:
        fut = self._output_futures.pop(msg.output_token, None) if msg.output_token else None
    if fut is not None and not fut.done():
        if msg.error:
            fut.set_exception(RuntimeError(msg.error))
        else:
            try:
                unpack_diffusion_output_shm(msg.output)
            except Exception as e:
                ...
                fut.set_exception(e)
                continue
            fut.set_result(msg.output)
```

Here `msg.output` will be a `BatchRunnerOutput` (or `DiffusionOutput`). The `fut.set_result(msg.output)` passes the whole object. Then `DiffusionEngine.step()` does `output = await ... fut` and checks `output.output_token`.

**The issue:** For batch mode, all N requests share the same `output_token`. When `step()` is called per-request in the orchestrator, each `wait_output_ready(token)` would overwrite `_output_futures[token]` — only the last Future gets set, the others hang.

**Solution: One output_token per batch, list of Futures per token**

Change `_output_futures` from `dict[str, Future]` to `dict[str, list[Future]]`. Each `wait_output_ready` call appends a new Future to the list. When `_result_pump` receives `OUTPUT_READY`, it resolves all Futures in the list.

### multiproc_executor.py — `_output_futures` and `_result_pump`

**Change `_output_futures` type:**
```python
# Before: dict[str, concurrent.futures.Future[DiffusionOutput]]
# After:  dict[str, list[concurrent.futures.Future]]
self._output_futures: dict[str, list[concurrent.futures.Future]] = {}
```

**Change `wait_output_ready`:**
```python
def wait_output_ready(self, output_token: str) -> concurrent.futures.Future:
    with self._futures_lock:
        cached = self._completed_outputs.pop(output_token, None)
        if cached is not None:
            # Result already arrived; return a pre-resolved Future.
            fut = concurrent.futures.Future()
            if isinstance(cached, BatchRunnerOutput):
                # For batch results, store the whole BatchRunnerOutput;
                # DiffusionEngine.step() will extract per-request result.
                fut.set_result(cached)
            else:
                fut.set_result(cached)
            return fut
        # Result not yet arrived; append Future to the waiting list.
        fut = concurrent.futures.Future()
        self._output_futures.setdefault(output_token, []).append(fut)
    return fut
```

**Change `_result_pump` OUTPUT_READY handling:**
```python
elif msg.kind == AsyncDiffusionOutput.OUTPUT_READY:
    with self._futures_lock:
        waiting_futs = self._output_futures.pop(msg.output_token, []) if msg.output_token else []
    # Resolve all waiting Futures with the same result object.
    for fut in waiting_futs:
        if fut.done():
            continue
        if msg.error:
            fut.set_exception(RuntimeError(msg.error))
        else:
            try:
                unpack_diffusion_output_shm(msg.output)
            except Exception as e:
                logger.exception("SHM unpack failed in result pump")
                fut.set_exception(e)
                continue
            fut.set_result(msg.output)
    # Also cache for late arrivals (requests whose step() calls
    # wait_output_ready after OUTPUT_READY already arrived).
    if msg.output_token and not msg.error:
        self._completed_outputs[msg.output_token] = msg.output
```

### diffusion_engine.py — `step()` and `step_streaming()`

When `wait_output_ready` returns a `BatchRunnerOutput`, extract the per-request `DiffusionOutput`:

```python
# In step():
if output.output_token:
    fut = self.executor.wait_output_ready(output.output_token)
    raw_output = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=_ASYNC_OUTPUT_TIMEOUT)
    if isinstance(raw_output, BatchRunnerOutput):
        per_req = raw_output.get_request_output(request.request_id)
        if per_req is not None and per_req.result is not None:
            output = per_req.result
        else:
            output = DiffusionOutput(error="Async batch output missing for request")
    else:
        output = raw_output
```

Similarly for `step_streaming()`.

For `add_req_and_wait_for_response()` (sync path, used only in `_dummy_run`):
```python
if output.output_token:
    fut = self.executor.wait_output_ready(output.output_token)
    raw_output = fut.result(timeout=_ASYNC_OUTPUT_TIMEOUT)
    if isinstance(raw_output, BatchRunnerOutput):
        per_req = raw_output.get_request_output(target_request_id)
        if per_req is not None and per_req.result is not None:
            output = per_req.result
        else:
            output = DiffusionOutput(error="Async batch output missing for request")
    else:
        output = raw_output
```

## Summary of Changes

| File | Change | Scope |
|------|--------|-------|
| `diffusion_worker.py` | `return_result`: extend async gate to handle `BatchRunnerOutput` (covers both `execute_model_batch` and `execute_stepwise`) | WorkerProc |
| `diffusion_worker.py` | Import `BatchRunnerOutput` from utils | imports |
| `ipc.py` | `pack_diffusion_output_shm`: handle `BatchRunnerOutput` and `RunnerOutput` — pack nested `DiffusionOutput` tensors on d2h_stream | IPC |
| `ipc.py` | `unpack_diffusion_output_shm`: handle `BatchRunnerOutput` and `RunnerOutput` — unpack nested SHM handles | IPC |
| `multiproc_executor.py` | `collective_rpc`: extend async gate from `method == "execute_model"` to `_ASYNC_METHODS = {"execute_model", "execute_model_batch", "execute_stepwise"}` | Executor |
| `multiproc_executor.py` | `execute_batch`: handle `AsyncDiffusionOutput(COMPUTE_DONE)` → create `BatchRunnerOutput` with `output_token` per request | Executor |
| `multiproc_executor.py` | `execute_step`: handle `AsyncDiffusionOutput(COMPUTE_DONE)` → propagate `output_token` through | Executor |
| `multiproc_executor.py` | `_output_futures`: change from `dict[str, Future]` to `dict[str, list[Future]]` so multiple waiters for same output_token don't overwrite each other | Executor |
| `multiproc_executor.py` | `wait_output_ready`: append Future to list, handle cached `BatchRunnerOutput` | Executor |
| `multiproc_executor.py` | `_result_pump`: resolve all Futures in list for `OUTPUT_READY`, cache result for late arrivals | Executor |
| `diffusion_engine.py` | `step()`: when `wait_output_ready` returns `BatchRunnerOutput`, extract per-request `DiffusionOutput` via `get_request_output` | Engine |
| `diffusion_engine.py` | `step_streaming()`: same extraction logic as `step()` | Engine |
| `diffusion_engine.py` | `add_req_and_wait_for_response()`: same extraction logic (sync path) | Engine |

## Testing

- Verify Qwen-Image with `--enable-async-diffusion-output` shows D2H on a side stream in profiler traces
- Verify `Memcpy DtoH (Device -> Pinned)` appears on the d2h_stream, not the default stream
- Verify `pipeline_forward_batch` no longer overlaps with D2H memcpy on the same stream
- Verify batch concurrency: 2 concurrent Qwen-Image requests should see GPU compute for request B start while request A's D2H is still running on the side stream

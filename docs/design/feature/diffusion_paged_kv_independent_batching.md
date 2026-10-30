# Paged Independent-Request Batching and Step Execution

Status: **Proposed**. This page is the design contract for the
[RFC #9192](https://github.com/vllm-project/vllm-omni/issues/9192) workstream
**Independent-request batching and step execution** on HunyuanImage-3.0 under
`diffusion_kv_mode=paged_scheduler`. It does not change runtime behavior.
The claim that scoped this workstream is
[issue comment 6512521276](https://github.com/vllm-project/vllm-omni/issues/9192#issuecomment-6512521276).

Scheduler-managed paged KV itself stays described in
[Scheduler-Managed Paged KV Cache](diffusion_paged_kv_cache.md). Execution-mode
names follow
[Diffusion Continuous Batching](diffusion_continuous_batching.md).

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Status Today](#status-today)
4. [Diagnosis](#diagnosis)
5. [Proposed Contract](#proposed-contract)
6. [Grouping Rule](#grouping-rule)
7. [Open Review Question](#open-review-question)
8. [Out of Scope](#out-of-scope)
9. [Related Files](#related-files)

## Overview

Hunyuan `paged_scheduler` currently runs one public request to completion.
Step execution is rejected while paged KV is active, including a single
request (`max_num_seqs=1`) and a grouped step batch (`max_num_seqs>1`).

The follow-up implementation should allow step execution, and should allow
compatible independent requests to advance together, without giving up the
paged KV layout. This page fixes the contract those code changes must follow.
The blocker recorded below is the span representation passed into attention,
not the lack of a grouped paged kernel.

This documentation PR does not edit the pipeline, the piecewise planner, or
tests. An implementation PR comes after this contract is reviewed.

## Motivation

Paged KV keeps a stable prefix in scheduler-owned pages and rewrites only the
image target on later denoising steps. That layout is already the supported
request path. Operators who want step execution or multi-request step
batching still have to leave `paged_scheduler`:

- One request that should advance one denoise step per scheduler tick cannot
  do so while paged KV is on.
- Several independent requests that share an image-token count, but not a
  prompt-prefix length, cannot share a step batch on the paged path.

The dense step path can already group those requests, but only by padding
every row onto one max-prefix grid and only with `TORCH_SDPA`. Reusing that
grid on the paged path is what the planner rejects. The contract below keeps
the dense grid for the dense path and keeps per-row coordinates for the paged
path.

## Status Today

| Execution path | Configuration | Hunyuan `paged_scheduler` today |
| --- | --- | --- |
| Request execution | `step_execution=false`, `max_num_seqs=1` | Supported. Unchanged by this proposal |
| Request-level batching | `step_execution=false`, `max_num_seqs>1` | Unsupported. `HunyuanImage3Pipeline.supports_request_batch` is `False` |
| Step execution | `step_execution=true`, `max_num_seqs=1` | Rejected |
| Step continuous batching | `step_execution=true`, `max_num_seqs>1` | Rejected |

Two pipeline checks enforce the step-execution rejection:

- `HunyuanImage3Pipeline._validate_step_request` raises when
  `diffusion_kv_mode` is `paged_scheduler`, before a step state is prepared.
- `HunyuanImage3Pipeline.denoise_step` raises the same error for the whole
  step batch, before any group runs.

Grouped dense step batches have a second guard.
`_ensure_grouped_attention_backend_supported` allows `max_num_seqs>1` only
when the resolved diffusion attention backend is `TORCH_SDPA`. Paged KV
requires `FLASH_ATTN` and does not build a dense mask, so that guard and the
paged path cannot be true together.

The dense merge that grouped steps use today lives in
`_merge_step_model_inputs`:

- `_merge_attention_masks` pads each row onto a uniform grid of
  `max(prefix_lens)` plus `max(current_lens)`, then publishes one `query_lens`
  entry and one `seq_lens` entry per row, all equal to that padded mask.
- `_shift_full_attn_spans` moves spans that start at or after the row prefix
  by `max_prefix_len - prefix_len`, so those spans sit on the same grid.
  Spans that start inside the prefix are left in place.

That pair keeps a 4D dense mask well-formed. It is the wrong coordinate
system for paged attention.

## Diagnosis

The conflict is the span form handed to attention. It is not a missing
grouped kernel on the paged backend.

On a later denoising step the image target is the query, and rows may share
that image-token count while keeping different prefixes. The claim's GPU
probe used `prefix_lens=[9, 5]` and per-row `seq_lens=[17, 13]` (image length
8 on both rows). Paged output matched a per-row dense reference at
`max_abs_err=0.00000` / `0.00024`. This documentation PR does not re-run that
probe. The numbers are the ones recorded in the
[claim comment](https://github.com/vllm-project/vllm-omni/issues/9192#issuecomment-6512521276).

`_shift_full_attn_spans` turns the true per-row spans `[[(9, 17)], [(5, 13)]]`
into `[[(9, 17)], [(9, 17)]]`, because the shorter row is shifted by
`9 - 5 = 4`. Those two forms disagree with the two checkers:

| Spans given to attention | Dense `piecewise_attn._check_homogeneous` | Paged `build_paged_piecewise_plan` |
| --- | --- | --- |
| `[[(9, 17)], [(9, 17)]]`, the `_shift_full_attn_spans` result | Pass. Every row matches the first row | **Rejected.** Span `(9, 17)` is outside per-row `seq_len=13` |
| `[[(9, 17)], [(5, 13)]]`, true per-row coordinates | **Rejected.** Row spans are not identical | Pass. `homogeneous_batch_shape=(2, 8)` |

The range check is `0 <= start < end <= seq_len` inside
`build_paged_piecewise_plan`. The homogeneous check is equality of the full
span list across rows inside `_check_homogeneous`. The two representations
are mutually exclusive: the form the dense merge emits is the form the paged
planner rejects, and the form the paged planner accepts is the form the dense
helper rejects.

When the true spans and the true per-row lengths are kept, both rows have
query length 8 and the same local segment layout, so the planner reports
`homogeneous_batch_shape=(2, 8)`. Different global offsets and different KV
lengths stay row-specific in the native metadata. The planner comment already
states that a shared local range is enough for that fast path.

The `TORCH_SDPA`-only guard is therefore not the blocker for paged grouped
steps. The paged path builds no dense mask and does no prefix padding. Dropping
the guard without also skipping the span shift would still hand the planner
out-of-range spans.

## Proposed Contract

The follow-up implementation PR should apply all three rules. This page only
records them.

1. **Keep per-row coordinates when paged is active.** Skip
   `_shift_full_attn_spans`. Do not pad rows through `_merge_attention_masks`,
   and do not replace per-row `query_lens` / `seq_lens` with the padded mask
   shape. Pass each row's true `seq_len` and true `full_attn_spans` to
   `build_paged_piecewise_plan`. The dense path keeps the current shift and
   the uniform mask; this rule is only for `paged_scheduler`.
2. **Allow the paged backend for grouped steps.** Do not require `TORCH_SDPA`
   for a grouped step batch that runs under `paged_scheduler`. Grouped paged
   steps use the existing paged / `FLASH_ATTN` path. The `TORCH_SDPA` guard
   remains the dense grouped-step requirement.
3. **Do not batch mismatched i2i prefix offsets until review chooses how.**
   Reference-image spans (`joint_image_slices`) can sit at different offsets
   inside the prefix. On the first denoising step the query covers the whole
   sequence, and `build_paged_piecewise_plan` rejects rows whose piecewise
   segments are not aligned (different segment counts, or a different mode in
   the same segment slot). That case stays out of a shared plan until the
   [open question](#open-review-question) is decided.

Lifting the blanket `paged_scheduler` raise in `_validate_step_request` and
`denoise_step` is part of the same follow-up. A one-request step
(`max_num_seqs=1`) has no cross-request prefix grid to build; it still cannot
run today because those raises fire before the merge.

Request-level fused forward (`step_execution=false` with `max_num_seqs>1`)
stays unsupported. This workstream is step execution and step continuous
batching, not `supports_request_batch`.

## Grouping Rule

A paged step group may share one piecewise plan only when every row has the
same:

- **CFG shape.** Same CFG factor (no-CFG, or the cond/uncond pair) and the
  same latent spatial shape. `_step_group_key` already splits on CFG factor
  and latent shape.
- **Sequence length.** The generated-image token count (`num_image_tokens` /
  target length). Prefix length may differ, so full `seq_len` may differ, as
  in the `[9, 5]` / `[17, 13]` case. The planner's homogeneous fast path needs
  equal query lengths, not equal prefixes.
- **Prefix-hit boundary.** The ordered full-attention spans that sit in the
  prefix (reference-image spans). Rows that do not produce the same piecewise
  segment layout must not share a plan.

One mismatched boundary fails that group, not the whole scheduler batch.
Other groups in the same `denoise_step` call still run. Today's
`_step_group_key` does not include this prefix-hit boundary. Adding it is how
rule 3 is met if review chooses to split groups rather than extend the
planner.

## Open Review Question

For i2i rows whose reference-image spans sit at different prefix offsets, pick
one of these before coding. This documentation PR does not choose:

| Option | Change | Effect |
| --- | --- | --- |
| Split groups | Extend the step group key with the prefix-hit boundary | Mismatched rows never enter one `build_paged_piecewise_plan` call. The planner stays unchanged |
| Extend the planner | Teach `build_paged_piecewise_plan` to accept those unaligned reference spans | One group can hold the mismatched rows. The planner's aligned-segment rule changes |

The text-to-image geometry in [Diagnosis](#diagnosis) is not this question.
Those rows have a single generated-image span, equal image length, and no
conflicting reference span. They should share a plan once rules 1 and 2 land,
whether or not review extends the planner.

## Out of Scope

This contract does not cover, and the first implementation slice should not
bundle:

- Ring attention, AllGather-KV, and Ulysses combined with those layouts.
  AllGather-KV and strict Ulysses + AllGather-KV are
  [#9396](https://github.com/vllm-project/vllm-omni/pull/9396). Ring stays on
  the sequence-parallel workstream. The current paged Hunyuan path remains
  request-mode strict Ulysses only.
- The default-paged flip and dense-path removal. That workstream is
  [#9542](https://github.com/vllm-project/vllm-omni/pull/9542). `dense_legacy`
  stays available, and this proposal does not change the default of
  `diffusion_kv_mode`.
- AR KV import combined with the local prefix, and KV-page offload. Imported
  AR-to-DiT transfer remains the separate follow-up described in the paged KV
  design. `ar_diffusion_kv` is not `paged_scheduler`.
- Runtime edits in this documentation PR. Pipeline, planner, scheduler, and
  test changes belong to the implementation PR that follows this contract.

## Related Files

- Step merge and guards:
  [`pipeline_hunyuan_image3.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py)
  (`_validate_step_request`, `denoise_step`, `_merge_step_model_inputs`,
  `_merge_attention_masks`, `_shift_full_attn_spans`,
  `_ensure_grouped_attention_backend_supported`, `_step_group_key`)
- Piecewise planning:
  [`piecewise_attn.py`](gh-file:vllm_omni/diffusion/attention/backends/utils/piecewise_attn.py)
  (`_check_homogeneous`, `build_paged_piecewise_plan`)
- Paged KV contract and mode matrix:
  [diffusion_paged_kv_cache.md](diffusion_paged_kv_cache.md)
- Execution-mode names:
  [diffusion_continuous_batching.md](diffusion_continuous_batching.md)

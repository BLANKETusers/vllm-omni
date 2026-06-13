# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import vllm.forward_context as _vllm_fc
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.platforms import current_omni_platform


def _set_forward_context_num_tokens(num_tokens: int) -> None:
    """Set num_tokens on the vLLM ForwardContext for MoE routing.

    After the rebase to vLLM 0.18.0, FusedMoE expects
    ForwardContext.num_tokens to be set. Without it, MoE expert
    routing may produce incorrect results (silent correctness bug).
    """
    if not _vllm_fc.is_forward_context_available():
        return
    forward_context = _vllm_fc.get_forward_context()
    forward_context.num_tokens = num_tokens
    if not hasattr(forward_context, "in_profile_run"):
        forward_context.in_profile_run = False


class HunyuanFusedMoEDefault(FusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        # Current vLLM FusedMoE handles output reduction internally.
        kwargs.pop("reduce_results", None)
        super().__init__(prefix=prefix, **kwargs)
        self._prefix = prefix

        # vLLM assumes sp=tp, so flatten_tp only merges dp+pcp+tp into ep_size, missing sp.
        # In omni, EP group includes sp (ep = tp * sp * cfg * dp), so fix ep_size/ep_rank here.
        from vllm.distributed import get_ep_group
        ep_group = get_ep_group()
        if ep_group.world_size > 1:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"[EP fix] Before: ep_size={self.moe_parallel_config.ep_size}, ep_rank={self.moe_parallel_config.ep_rank}; "
                      f"After: ep_size={ep_group.world_size}, ep_rank={ep_group.rank_in_group}, "
                      f"local_num_experts={self.expert_map_manager.local_num_experts}")
            self.moe_parallel_config.ep_size = ep_group.world_size
            self.moe_parallel_config.ep_rank = ep_group.rank_in_group
            # When EP is active with SP (ep_size includes sp dimension), vLLM's
            # _maybe_reduce_final_output would trigger tensor_model_parallel_all_reduce
            # over the TP group (ws=2) — but this only covers 2 TP partners, not the
            # full EP group (ws=4).  Setting sp_size = ep_group.world_size makes
            # is_sequence_parallel = True (property: sp_size > 1), which ensures the
            # TP all-reduce is skipped because the guard is
            # ``not self.moe_config.is_sequence_parallel``.
            # EP dispatch/combine is handled manually in forward() since vLLM's
            # do_naive_dispatch_combine requires dp_size>1, but in tp2+sp2
            # ep_size=4 while dp_size=1.
            self.moe_parallel_config.sp_size = ep_group.world_size
            self.expert_map_manager.update(
                self.moe_parallel_config, self.global_num_experts
            )
            self.update_expert_map_info()
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"[EP fix] Updated local_num_experts={self.local_num_experts}, "
                      f"is_sequence_parallel=True, sp_size={self.moe_parallel_config.sp_size}")

        self._init_hook_handle = self.register_forward_pre_hook(self._initialize_kernel_hook, with_kwargs=True)
        self._debug_call_count = 0

    def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
        if self.quant_method and getattr(self.quant_method, "moe_kernel", None) is None:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        from vllm.distributed import get_ep_group, get_sp_group
        ep_group = get_ep_group()
        ep_size = ep_group.world_size

        if ep_size > 1:
            sp_group = get_sp_group()
            sp_size = sp_group.world_size if sp_group is not None else 1
            tp_in_ep = ep_size // sp_size  # TP replication factor within EP

            ep_pg = ep_group.device_group

            # ---- EP dispatch: all-gather across the full EP group ----
            # vLLM's built-in do_naive_dispatch_combine only fires when dp_size>1,
            # but in omni's tp2+sp2 layout ep_size=4 while dp_size=1, so the
            # built-in path never runs.
            gathered_hs = [torch.empty_like(hidden_states) for _ in range(ep_size)]
            torch.distributed.all_gather(gathered_hs, hidden_states, group=ep_pg)

            gathered_rl = [torch.empty_like(router_logits) for _ in range(ep_size)]
            torch.distributed.all_gather(gathered_rl, router_logits, group=ep_pg)

            # Only use first sp_size unique token chunks (skip TP duplicates).
            # Within EP, ranks are ordered [tp, sp]: gathered_hs has sp_size
            # unique token sets, each replicated tp_in_ep times.
            unique_hs = torch.cat(gathered_hs[:sp_size], dim=0)
            unique_rl = torch.cat(gathered_rl[:sp_size], dim=0)

            # Verify: print once per rank on first call (hidden inside compile guard).
            self._debug_call_count += 1
            if self._debug_call_count <= 1 and not torch.compiler.is_compiling():
                print(f"[MoE Opt] rank={torch.distributed.get_rank()}, "
                      f"ep_size={ep_size}, sp_size={sp_size}, tp_in_ep={tp_in_ep}, "
                      f"hs_in={hidden_states.shape[0]}, hs_unique={unique_hs.shape[0]}, "
                      f"ratio={unique_hs.shape[0] / hidden_states.shape[0]:.1f}x, "
                      f"local_experts={self.local_num_experts}")

            # ---- FusedMoE compute on unique tokens ----
            # Each rank computes only its local experts (16 of 64).
            # is_sequence_parallel=True skips _maybe_reduce_final_output
            # (TP all-reduce, which only covers 2 ranks instead of 4).
            _set_forward_context_num_tokens(unique_hs.shape[0])
            result = super(HunyuanFusedMoEDefault, self).forward(
                unique_hs, unique_rl
            )

            # ---- EP combine: all-reduce + SP slice + shared-expert fix ----
            # all_reduce across EP group combines all 64 experts' contributions
            # (each rank contributes 16 local experts on the same unique tokens).
            torch.distributed.all_reduce(result, group=ep_pg)

            # Keep only this rank's SP portion.
            local_N = hidden_states.shape[0]
            my_sp_idx = ep_group.rank_in_group // tp_in_ep
            result = result[my_sp_idx * local_N : (my_sp_idx + 1) * local_N]

            # Shared expert was all-reduced ep_size times; keep only one copy.
            if self.shared_experts is not None:
                shared_out = self.shared_experts._layer(hidden_states)
                result = result - (ep_size - 1) * shared_out

            return result

        # No EP: standard FusedMoE forward
        _set_forward_context_num_tokens(hidden_states.shape[0])
        return super(HunyuanFusedMoEDefault, self).forward(
            hidden_states, router_logits
        )


class HunyuanFusedMoE:
    def __new__(cls, *, prefix: str = "", **kwargs: Any) -> Any:
        op_name = "hunyuan_fused_moe"
        current_omni_platform.prepare_diffusion_op_runtime(op_name)
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname(op_name),
        )
        return impl(prefix=prefix, **kwargs)

    @classmethod
    def make_expert_params_mapping(
        cls,
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname("hunyuan_fused_moe"),
        )
        return impl.make_expert_params_mapping(
            model,
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )

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
        self._debug_call_count = 0  # limit per-step debug prints

    def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
        if self.quant_method and getattr(self.quant_method, "moe_kernel", None) is None:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        from vllm.distributed import get_ep_group
        ep_group = get_ep_group()
        ep_size = ep_group.world_size

        if ep_size > 1:
            # ---- EP dispatch: all-gather across the full EP group ----
            # vLLM's built-in do_naive_dispatch_combine only fires when dp_size>1,
            # but in omni's tp2+sp2 layout ep_size=4 while dp_size=1, so the
            # built-in path never runs.  We manually do EP all-gather here so
            # that every rank sees ALL tokens and can compute its local-expert
            # contributions for them.
            # Use torch.distributed directly on ep_group.device_group (ProcessGroup)
            # because GroupCoordinator may not expose all_gather/reduce_scatter when
            # use_device_communicator=False.
            ep_pg = ep_group.device_group

            # all_gather hidden_states [N/sp, D] -> [N, D]
            gathered_hs = [torch.empty_like(hidden_states) for _ in range(ep_size)]
            torch.distributed.all_gather(gathered_hs, hidden_states, group=ep_pg)
            hidden_states_all = torch.cat(gathered_hs, dim=0)

            # all_gather router_logits [N/sp, E] -> [N, E]
            gathered_rl = [torch.empty_like(router_logits) for _ in range(ep_size)]
            torch.distributed.all_gather(gathered_rl, router_logits, group=ep_pg)
            router_logits_all = torch.cat(gathered_rl, dim=0)

            # ---- FusedMoE compute on all-gathered tokens ----
            # With is_sequence_parallel=True (sp_size=ep_size>1) set in __init__,
            # _maybe_reduce_final_output is SKIPPED (the guard is
            # ``not self.moe_config.is_sequence_parallel``, which is False).
            # _maybe_dispatch/_maybe_combine are also SKIPPED because dp_size=1.
            # This means super().forward() only does the local expert +
            # shared-expert kernel — exactly what we want.
            self._debug_call_count += 1
            if self._debug_call_count <= 3:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
                print(f"[MoE GEMM Debug] call={self._debug_call_count} rank={rank}, "
                      f"hs_in.shape={hidden_states.shape}, hs_in.stride={hidden_states.stride()}, "
                      f"hs_all.shape={hidden_states_all.shape}, hs_all.stride={hidden_states_all.stride()}, "
                      f"hs_all.is_contiguous={hidden_states_all.is_contiguous()}, "
                      f"local_num_experts={self.local_num_experts}, global_num_experts={self.global_num_experts}")
            _set_forward_context_num_tokens(hidden_states_all.shape[0])
            result = super(HunyuanFusedMoEDefault, self).forward(
                hidden_states_all, router_logits_all
            )

            # ---- EP combine: reduce-scatter across the full EP group ----
            # reduce_scatter sums each token's contributions from all EP ranks
            # (fused experts are partial — only local-expert slices — so the
            # sum reconstructs the full Σ weight·expert) and then scatters
            # each rank its original N/sp tokens.
            # [N, D] -> [N/sp, D]
            output = torch.empty_like(hidden_states)
            torch.distributed.reduce_scatter(output, result, group=ep_pg)
            result = output

            # ---- Fix shared-expert over-counting ----
            # reduce_scatter sums the shared-expert output ep_size times
            # (every rank computes the same shared-expert value because the
            # input is identical after all-gather), but the correct result
            # should include it only once.  Subtract (ep_size - 1) × shared_out
            # to compensate.
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

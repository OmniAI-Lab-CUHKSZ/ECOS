# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.trainer.ppo.selective_gigpo import selective_route_loss_terms
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _response_only_rmpad_selectors(
    attention_mask: torch.Tensor,
    response_length: int,
    unpad_indices: torch.Tensor,
):
    """Map valid response tokens to their predictors in a packed sequence.

    ``unpad_input`` concatenates every row's valid tokens and returns the
    original flattened positions in ``unpad_indices``.  Response token ``j``
    is predicted by absolute column ``prompt_length - 1 + j``.  This helper
    maps those predictor columns into the packed sequence so Qwen's tensor
    ``logits_to_keep`` can skip every prompt-only LM-head projection.

    The final return value is the actual ``logits_to_keep`` tensor.  For an
    empty response micro-batch it contains one harmless fallback position so
    every FSDP rank still executes the same model forward; the first three
    returned tensors remain empty and the fallback logits are discarded.
    """

    if attention_mask.ndim != 2:
        raise ValueError("response-only selectors require a 2-D attention mask")
    response_length = int(response_length)
    seqlen = int(attention_mask.size(1))
    prompt_length = seqlen - response_length
    if response_length <= 0 or prompt_length <= 0:
        raise ValueError(
            "response-only selectors require positive prompt and response lengths"
        )
    if unpad_indices.ndim != 1:
        raise ValueError("unpad_indices must be a 1-D tensor")

    unpad_indices = unpad_indices.long()
    active_pairs = torch.nonzero(
        attention_mask[:, -response_length:] != 0, as_tuple=False
    )
    if int(active_pairs.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=attention_mask.device)
        if int(unpad_indices.numel()) == 0:
            raise ValueError("cannot select fallback logits from an empty sequence")
        fallback = torch.tensor(
            [int(unpad_indices.numel()) - 1],
            dtype=torch.long,
            device=attention_mask.device,
        )
        return empty, empty, empty, fallback

    active_rows = active_pairs[:, 0].long()
    response_positions = active_pairs[:, 1].long()
    predictor_flat = (
        active_rows * seqlen + prompt_length - 1 + response_positions
    )
    packed_positions = torch.searchsorted(unpad_indices, predictor_flat)
    within = packed_positions < int(unpad_indices.numel())
    exact = torch.zeros_like(within)
    if bool(within.any().item()):
        exact[within] = (
            unpad_indices.index_select(0, packed_positions[within])
            == predictor_flat[within]
        )
    if not bool(exact.all().item()):
        raise ValueError(
            "a valid response token has no valid predecessor in the packed sequence"
        )
    return (
        active_rows,
        response_positions,
        packed_positions.long(),
        packed_positions.long(),
    )


def _scatter_compact_response_values(
    compact_values: torch.Tensor,
    active_rows: torch.Tensor,
    response_positions: torch.Tensor,
    *,
    batch_size: int,
    response_length: int,
) -> torch.Tensor:
    """Scatter compact active-token values back to the padded response axes."""

    if compact_values.ndim != 1:
        raise ValueError("compact response values must be a 1-D tensor")
    if (
        active_rows.ndim != 1
        or response_positions.ndim != 1
        or active_rows.shape != response_positions.shape
        or compact_values.shape != active_rows.shape
    ):
        raise ValueError("compact response selectors must be equal 1-D tensors")
    output = compact_values.new_zeros((int(batch_size), int(response_length)))
    if int(compact_values.numel()) == 0:
        return output
    return output.index_put(
        (active_rows.long(), response_positions.long()),
        compact_values,
        accumulate=False,
    )


def _resolve_command_final_norm(module: nn.Module) -> nn.Module:
    """Find Qwen's final decoder norm through the common FSDP wrappers."""

    pending = [module]
    visited = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        base_model = getattr(candidate, "model", None)
        final_norm = getattr(base_model, "norm", None)
        if isinstance(final_norm, nn.Module):
            return final_norm
        pending.extend(
            [
                getattr(candidate, "module", None),
                getattr(candidate, "_fsdp_wrapped_module", None),
            ]
        )
    raise RuntimeError(
        "command scorer could not resolve the Qwen final decoder norm"
    )


def _command_model_forward(
    actor_module: nn.Module,
    *,
    need_hidden: bool,
    final_hidden_only: bool,
    **kwargs,
):
    """Run one command forward and optionally retain only the final hidden.

    ``output_hidden_states=True`` keeps every decoder layer output alive even
    though observation grouping needs only the final decoder hidden state.  A temporary
    hook on Qwen's final norm captures that identical tensor without retaining
    all intermediate layers.  The legacy path remains available for isolated
    A/B comparison.
    """

    if "output_hidden_states" in kwargs:
        raise ValueError("_command_model_forward owns output_hidden_states")
    if not need_hidden or not final_hidden_only:
        output = actor_module(output_hidden_states=need_hidden, **kwargs)
        final_hidden = output.hidden_states[-1] if need_hidden else None
        return output, final_hidden

    captured = []

    def capture_final_hidden(_module, _inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(value)

    handle = _resolve_command_final_norm(actor_module).register_forward_hook(
        capture_final_hidden
    )
    try:
        output = actor_module(output_hidden_states=False, **kwargs)
    finally:
        handle.remove()
    if len(captured) != 1 or not isinstance(captured[0], torch.Tensor):
        raise RuntimeError(
            "command final-hidden hook expected exactly one tensor output, "
            f"captured={len(captured)}"
        )
    return output, captured[0]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        reflect_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.reflect_optimizer = reflect_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        # Fixed CFI CountSketch hashes are reused across every feature
        # micro-batch.  The cache is Actor-local so it follows the worker and
        # never crosses devices or serialization boundaries.
        self._score_sketch_hash_cache = {}

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        response_only_logits=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)
        response_only_logits = bool(response_only_logits)
        if response_only_logits and self.use_fused_kernels:
            raise ValueError(
                "R/S response-only logits do not support fused PPO kernels"
            )
        if response_only_logits and self.use_remove_padding and self.use_ulysses_sp:
            raise ValueError(
                "R/S response-only packed logits require sequence parallel size 1"
            )
        if response_only_logits and multi_modal_inputs:
            raise ValueError("R/S response-only logits support text-only batches")

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                active_rows = response_positions = None
                if response_only_logits:
                    (
                        active_rows,
                        response_positions,
                        _,
                        logits_to_keep,
                    ) = _response_only_rmpad_selectors(
                        attention_mask,
                        response_length,
                        indices,
                    )

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # The legacy path scores every valid input position.  R/S can
                # instead provide exact response predictor positions directly.
                input_ids_rmpad_rolled = None
                if not response_only_logits:
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                if not response_only_logits:
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if response_only_logits:
                    extra_args["logits_to_keep"] = logits_to_keep

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if response_only_logits:
                    compact_logits = output.logits.squeeze(0)
                    if int(active_rows.numel()) == 0:
                        # Keep a zero-gradient connection to the model so an
                        # all-empty micro-batch remains backward-safe.
                        anchor = compact_logits.sum().float() * 0.0
                        log_probs = anchor.expand(batch_size, response_length)
                        if calculate_entropy:
                            entropy = anchor.expand(batch_size, response_length)
                    else:
                        compact_logits.div_(temperature)
                        compact_labels = micro_batch["responses"][
                            active_rows, response_positions
                        ]
                        compact_log_probs = logprobs_from_logits(
                            logits=compact_logits,
                            labels=compact_labels,
                            inplace_backward=not calculate_entropy,
                        )
                        log_probs = _scatter_compact_response_values(
                            compact_log_probs,
                            active_rows,
                            response_positions,
                            batch_size=batch_size,
                            response_length=response_length,
                        )
                        if calculate_entropy:
                            compact_entropy = self.compute_entropy_from_logits(
                                compact_logits
                            )
                            entropy = _scatter_compact_response_values(
                                compact_entropy,
                                active_rows,
                                response_positions,
                                batch_size=batch_size,
                                response_length=response_length,
                            )
                elif self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if not response_only_logits and self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                if not response_only_logits:
                    # pad back to (bsz, seqlen)
                    if calculate_entropy:
                        full_entropy = pad_input(
                            hidden_states=entropy_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    full_log_probs = pad_input(
                        hidden_states=log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                    # only return response part:
                    if calculate_entropy:
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                if response_only_logits:
                    # Keep the final prompt predictor plus the fixed response
                    # suffix, then discard the final unused response logit.
                    extra_args["logits_to_keep"] = response_length + 1
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    if response_only_logits:
                        logits = logits[:, :-1, :]
                    else:
                        logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self, optimizer=None):
        assert self.config.grad_clip is not None

        optimizer = optimizer or self.actor_optimizer
        isolated_optimizer = optimizer is not self.actor_optimizer

        if isolated_optimizer:
            # With use_orig_params=True each LoRA parameter exposes its local
            # FSDP shard.  Sum those shard norms across ranks so clipping is
            # based only on the adapter, never on the discarded base gradients.
            params = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            device = (
                params[0].grad.device
                if params
                else next(self.actor_module.parameters()).device
            )
            local_sq = torch.zeros((), dtype=torch.float32, device=device)
            for parameter in params:
                local_sq.add_(parameter.grad.detach().float().square().sum())
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_sq)
            grad_norm = local_sq.sqrt()
            if torch.isfinite(grad_norm):
                coefficient = min(
                    1.0,
                    float(self.config.grad_clip) / (float(grad_norm.item()) + 1e-6),
                )
                if coefficient < 1.0:
                    for parameter in params:
                        parameter.grad.mul_(coefficient)
        elif isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
        else:
            if self.config.get("stream_optimizer_states", False):
                from streamed_adamw import streamed_adamw_step
                streamed_adamw_step(optimizer)
            else:
                optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        rs_response_only_logits = bool(
            data.meta_info.get("rs_response_only_logits", False)
            or data.meta_info.get("response_only_logits", False)
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    response_only_logits=rs_response_only_logits,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto, rs_data: DataProto = None):
        """Update the policy with PPO + optional signed route loss."""
        import time

        self.actor_module.train()
        elastic_steps=data.meta_info.get('elastic_loss_steps')
        joint_diagnostics = bool(self.config.get('rs_joint_diagnostics', False)) and rs_data is not None
        first_last_diagnostics = bool(self.config.get('rs_joint_diagnostics_first_last', False))
        joint_progress = bool(self.config.get('rs_joint_progress', False)) and rs_data is not None
        progress_rank = (torch.distributed.get_rank() if torch.distributed.is_available()
                         and torch.distributed.is_initialized() else 0)

        def report_progress(message):
            if joint_progress and progress_rank == 0:
                print(f"[JointUpdate] {message}", flush=True)
        if rs_data is not None:
            if not self.config.get('rs_joint_update', False) or self.reflect_optimizer is not None:
                raise ValueError('Joint R/S requires the shared full Actor optimizer')
            if elastic_steps is None or len(rs_data.meta_info['elastic_loss_steps']) != len(elastic_steps) * self.config.ppo_epochs:
                raise ValueError('R/S must use exactly the existing Actor optimizer slots')
        if elastic_steps is not None:
            from verl.trainer.ppo.elastic_dispatch import planned_microbatches
            if not self.config.use_dynamic_bsz or self.ulysses_sequence_parallel_size!=1:
                raise ValueError('Elastic loss plan requires dynamic batches and sequence parallel size1')

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        route_enabled = "route_gate" in data.batch
        has_route_token_mask = "route_token_mask" in data.batch
        has_selective_route = "selective_route_mask" in data.batch
        if route_enabled:
            select_keys += ["teacher_log_probs", "route_gate"]
            if has_selective_route:
                select_keys.append('selective_route_mask')
            if has_route_token_mask:
                select_keys.append("route_token_mask")
        route_coef = data.meta_info.get("route_loss_coef", 0.5)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        if elastic_steps is not None:
            if has_multi_modal_inputs:raise ValueError('Elastic multimodal updates are not implemented')
            dataloader=[planned_microbatches(batch,step) for step in elastic_steps]
        metrics = {}
        optimizer_steps = len(dataloader) * int(self.config.ppo_epochs)
        optimizer_steps_applied = 0
        diagnostic_sample_count = 0
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                optimizer_index = epoch * len(dataloader) + batch_idx
                detailed_step = joint_diagnostics and (
                    not first_last_diagnostics or optimizer_index in (0, optimizer_steps - 1)
                )
                step_label = f"optimizer={optimizer_index + 1}/{optimizer_steps}"
                update_start = time.perf_counter()
                report_progress(f"{step_label} stage=main_backward")
                # split batch into micro_batches
                mini_batch = data
                if elastic_steps is not None:
                    micro_batches=mini_batch
                elif has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                from verl.utils.training_execution import accumulation_microbatches
                with accumulation_microbatches(
                    self.actor_module, micro_batches,
                    enabled=bool(self.config.get("defer_gradient_sync", False)),
                ) as synced_batches:
                    for data in synced_batches:
                        if elastic_steps is not None:
                            data,elastic_scale,elastic_padding=data
                            if elastic_padding:
                                saved_metrics=metrics;metrics={}
                        # Support all hardwares
                        if isinstance(data, DataProto):
                            data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                        else:
                            data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                        responses = data["responses"]
                        response_length = responses.size(1)
                        attention_mask = data["attention_mask"]
                        if multi_turn:
                            response_mask = data["loss_mask"][:, -response_length:]
                        else:
                            response_mask = attention_mask[:, -response_length:]

                        old_log_prob = data["old_log_probs"]
                        advantages = data["advantages"]

                        clip_ratio = self.config.clip_ratio
                        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        # all return: (bsz, response_length)
                        calculate_entropy = False
                        if entropy_coeff != 0:
                            calculate_entropy = True
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            response_only_logits=bool(self.config.get("response_only_logits", False)),
                        )
                        
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        if loss_mode != "vanilla":
                            raise ValueError(f"The released Actor uses vanilla PPO clipping, got {loss_mode}")
                        policy_loss_fn = compute_policy_loss

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                        pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss * pg_loss_coef

                        if self.config.use_kl_loss:
                            ref_log_prob = data["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if route_enabled:
                            route_weight = data["route_gate"].detach()
                            route_loss_mask = (
                                data["route_token_mask"]
                                if has_route_token_mask
                                else response_mask
                            ).to(response_mask.dtype)
                            # Detached weighted likelihood. SDAR/D3 keeps its
                            # existing positive token gate.
                            route_term = route_weight * (data["teacher_log_probs"].detach() - log_prob)
                            if has_selective_route:
                                route_term = selective_route_loss_terms(
                                    log_prob, old_log_prob, data['teacher_log_probs'], route_weight,
                                    data['selective_route_mask'], clip_ratio_low, clip_ratio_high,
                                )
                            route_l = agg_loss(
                                loss_mat=route_term,
                                loss_mask=route_loss_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss + route_coef * route_l
                            valid_tokens = route_loss_mask.sum().clamp(min=1)
                            gap_student_log_prob = old_log_prob
                            route_gap = (
                                data["teacher_log_probs"] - gap_student_log_prob
                            ).detach()
                            route_weight_max = (
                                route_weight.masked_fill(
                                    ~route_loss_mask.bool(), float("-inf")
                                ).max()
                                if bool(route_loss_mask.bool().any().item())
                                else route_weight.new_zeros(())
                            )
                            append_to_dict(
                                metrics,
                                {
                                    "actor/route_loss": route_l.detach().item(),
                                    "actor/route_coef": route_coef,
                                    "route/token_weight_rms_scale": 1.0,
                                    "route/token_weight_mean": (
                                        (route_weight * route_loss_mask).sum() / valid_tokens
                                    ).detach().item(),
                                    "route/token_weight_abs_mean": (
                                        (route_weight.abs() * route_loss_mask).sum() / valid_tokens
                                    ).detach().item(),
                                    "route/token_weight_max": route_weight_max.detach().item(),
                                    "route/token_weight_negative_frac": (
                                        ((route_weight < 0).float() * route_loss_mask).sum()
                                        / valid_tokens
                                    ).detach().item(),
                                    "route/token_weight_active_frac": (
                                        ((route_weight.abs() > 1e-8).float() * route_loss_mask).sum()
                                        / valid_tokens
                                    ).detach().item(),
                                    "route/teacher_student_gap_mean": (
                                        (route_gap * route_loss_mask).sum() / valid_tokens
                                    ).detach().item(),
                                },
                            )

                        if elastic_steps is not None:
                            loss=policy_loss*elastic_scale
                        elif self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss.backward()

                        if elastic_steps is not None and elastic_padding:
                            metrics=saved_metrics
                            continue

                        data = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                        append_to_dict(metrics, data)

                main_backward_time = time.perf_counter() - update_start
                diagnostic_start = time.perf_counter()
                auxiliary_time = 0.
                component_metrics = {}
                if rs_data is not None:
                    from verl.utils.joint_rs_diagnostics import capture_gradients, gradient_components, capture_parameters, parameter_change
                    append_to_dict(metrics, {'rs_joint/main_backward_wall_s': main_backward_time})
                    report_progress(f"{step_label} stage=rs_backward main_s={main_backward_time:.3f}")
                    snapshot = capture_gradients(self.actor_module) if detailed_step else None
                    auxiliary_start = time.perf_counter()
                    rs_metrics = self.update_reflect_policy(
                        rs_data, accumulate_only=True,
                        step_index=optimizer_index,
                    )
                    for key, values in rs_metrics.items():
                        metrics.setdefault(key, []).extend(values if isinstance(values, list) else [values])
                    auxiliary_time = time.perf_counter() - auxiliary_start
                    append_to_dict(metrics, {'rs_joint/auxiliary_backward_wall_s': auxiliary_time})
                    if detailed_step:
                        component_metrics = gradient_components(self.actor_module, snapshot)
                        append_to_dict(metrics, component_metrics)
                        del snapshot
                    report_progress(f"{step_label} stage=optimizer_step rs_s={auxiliary_time:.3f} detailed={int(detailed_step)}")
                before_parameters = capture_parameters(self.actor_module) if detailed_step else None
                grad_norm = self._optimizer_step()
                applied = bool(torch.isfinite(grad_norm).all().item())
                optimizer_steps_applied += int(applied)
                if detailed_step:
                    append_to_dict(metrics, parameter_change(self.actor_module, before_parameters))
                    del before_parameters
                    diagnostic_sample_count += 1
                if rs_data is not None:
                    diagnostics_and_step_time = time.perf_counter() - diagnostic_start - auxiliary_time
                    append_to_dict(metrics, {
                        'rs_joint/diagnostics_and_step_wall_s': diagnostics_and_step_time,
                        'rs/joint_update': 1., 'rs/extra_optimizer_steps': 0.,
                    })
                    detail = ''.join(f" {key.rsplit('/', 1)[-1]}={component_metrics[key]:.8g}"
                                     for key in ('rs_joint/rs_to_main_grad_ratio', 'rs_joint/grad_cosine')
                                     if key in component_metrics)
                    report_progress(
                        f"{step_label} stage=complete applied={int(applied)} "
                        f"grad_norm={grad_norm.detach().item():.8g} "
                        f"diagnostics_step_s={diagnostics_and_step_time:.3f} "
                        f"total_s={time.perf_counter() - update_start:.3f}{detail}"
                    )
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        metrics['actor/optimizer_steps_attempted'] = optimizer_steps
        metrics['actor/optimizer_steps_applied'] = optimizer_steps_applied
        if rs_data is not None:
            metrics['rs_joint/diagnostic_sample_count'] = diagnostic_sample_count
        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_reflect_policy(self, data: DataProto, accumulate_only=False, step_index=0):
        """Alpha-weighted GRPO update over Stage-A GRSD reflection tokens.

        Mirrors update_policy but is intentionally minimal:
          * single-turn response_mask (reflection is a plain prompt->response gen);
          * no KL / teacher-distillation / entropy terms (pure GRPO on s_i);
          * the policy-gradient loss is scaled by reflect_loss_coef (= alpha in the
            paper's L = L_task + alpha * L_ref);
          * gradient accumulation is derived from the actual micro-batch count, so
            the reflection batch size need not equal ppo_mini_batch_size.
        Legacy mode retains its separate optimizer pass. In joint mode this
        method ONLY accumulates the assigned R/S gradients; it never clears
        existing Actor gradients, clips, or advances an optimizer/scheduler.
        """
        self.actor_module.train()
        elastic_steps=data.meta_info.get('elastic_loss_steps')

        temperature = data.meta_info["temperature"]  # must be in meta_info to avoid silent error
        reflect_loss_coef = data.meta_info.get("reflect_loss_coef", 0.1)
        reflect_loss_agg_mode = data.meta_info.get(
            "reflect_loss_agg_mode", self.config.loss_agg_mode
        )
        metric_prefix = str(data.meta_info.get("reflect_metric_prefix", "reflect"))
        rs_response_only_logits = bool(
            data.meta_info.get("rs_response_only_logits", False)
        )
        if rs_response_only_logits and metric_prefix != "rs":
            raise ValueError(
                "response-only logits may only be enabled for the R/S optimizer pass"
            )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        batch = data.select(batch_keys=select_keys).batch

        if elastic_steps is not None:
            from verl.trainer.ppo.elastic_dispatch import planned_microbatches
            if (not accumulate_only and len(elastic_steps)!=1) or not self.config.use_dynamic_bsz or self.ulysses_sequence_parallel_size!=1:
                raise ValueError('R/S elastic update must contain exactly one dynamic optimizer step')
            micro_batches=planned_microbatches(batch,elastic_steps[step_index])
        elif self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(self.config.ppo_micro_batch_size_per_gpu)
        grad_accum = max(1, len(micro_batches))

        metrics = {}
        optimizer = self.reflect_optimizer or self.actor_optimizer
        if accumulate_only and (elastic_steps is None or optimizer is not self.actor_optimizer):
            raise ValueError('Joint R/S needs a planned shared Actor update')
        # Reflection backward traverses the shared base graph, but only the
        # isolated adapter optimizer is allowed to step.  Clear both sets so
        # no discarded base gradient can leak into the following Actor pass.
        if not accumulate_only:
            self.actor_optimizer.zero_grad()
            if optimizer is not self.actor_optimizer:
                optimizer.zero_grad()
        from verl.utils.training_execution import accumulation_microbatches
        with accumulation_microbatches(
            self.actor_module, micro_batches,
            enabled=bool(self.config.get("defer_gradient_sync", False)),
        ) as synced_batches:
            for micro in synced_batches:
                if elastic_steps is not None:
                    micro,elastic_scale,elastic_padding=micro
                micro = micro.to(get_torch_device().current_device())
                responses = micro["responses"]
                response_length = responses.size(1)
                attention_mask = micro["attention_mask"]
                response_mask = attention_mask[:, -response_length:]

                old_log_prob = micro["old_log_probs"].detach()
                advantages = micro["advantages"].detach()

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                loss_agg_mode = reflect_loss_agg_mode

                _, log_prob = self._forward_micro_batch(
                    micro_batch=micro,
                    temperature=temperature,
                    calculate_entropy=False,
                    response_only_logits=rs_response_only_logits,
                )

                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )

                if elastic_steps is not None:
                    loss=pg_loss*reflect_loss_coef*elastic_scale
                elif self.config.use_dynamic_bsz:
                    loss = pg_loss * reflect_loss_coef * (len(micro) / len(batch))
                else:
                    loss = (pg_loss * reflect_loss_coef) / grad_accum
                loss.backward()

                if elastic_steps is not None and elastic_padding:
                    continue

                append_to_dict(metrics, {
                    f"{metric_prefix}/pg_loss": pg_loss.detach().item(),
                    f"{metric_prefix}/pg_clipfrac": pg_clipfrac.detach().item(),
                    f"{metric_prefix}/ppo_kl": ppo_kl.detach().item(),
                })

        if accumulate_only:
            append_to_dict(metrics, {f'{metric_prefix}/coef': reflect_loss_coef,
                                     f'{metric_prefix}/isolated_optimizer': 0.})
            return metrics
        grad_norm = self._optimizer_step(optimizer=optimizer)
        append_to_dict(
            metrics,
            {
                f"{metric_prefix}/grad_norm": grad_norm.detach().item(),
                f"{metric_prefix}/coef": reflect_loss_coef,
                f"{metric_prefix}/isolated_optimizer": float(
                    optimizer is not self.actor_optimizer
                ),
            },
        )
        self.actor_optimizer.zero_grad()
        if optimizer is not self.actor_optimizer:
            optimizer.zero_grad()
        return metrics

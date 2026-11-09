from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Tuple
from pathlib import Path

import pytest
import torch


ACTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl"
    / "workers"
    / "actor"
    / "dp_actor.py"
)


def _load_function(name: str):
    tree = ast.parse(ACTOR_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(ACTOR_PATH), "exec"), namespace)
    return namespace[name]


def _load_actor_method(name: str, namespace: dict):
    tree = ast.parse(ACTOR_PATH.read_text(encoding="utf-8"))
    actor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor"
    )
    method = next(
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ACTOR_PATH), "exec"), namespace)
    return namespace[name]


def _forward_test_namespace():
    select = _load_function("_response_only_rmpad_selectors")
    scatter = _load_function("_scatter_compact_response_values")

    def unpad_input(values, attention_mask):
        indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
        flattened = values.reshape(-1, *values.shape[2:])
        return flattened.index_select(0, indices), indices, None, None

    def index_first_axis(values, indices):
        return values.index_select(0, indices)

    def rearrange(values, _pattern):
        return values.reshape(-1, *values.shape[2:])

    def logprobs_from_logits(logits, labels, inplace_backward=True):
        del inplace_backward
        return _response_log_probs(logits, labels)

    return {
        "Tuple": Tuple,
        "torch": torch,
        "unpad_input": unpad_input,
        "index_first_axis": index_first_axis,
        "rearrange": rearrange,
        "_response_only_rmpad_selectors": select,
        "_scatter_compact_response_values": scatter,
        "logprobs_from_logits": logprobs_from_logits,
    }


class _SelectiveLogitsActor(torch.nn.Module):
    def __init__(self, vocab_size=13):
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids, logits_to_keep, **_kwargs):
        positions = torch.arange(
            input_ids.size(1), dtype=torch.float32, device=input_ids.device
        ).view(1, -1, 1)
        vocab = torch.arange(
            self.vocab_size, dtype=torch.float32, device=input_ids.device
        ).view(1, 1, -1)
        full_logits = torch.remainder((positions + 1.0) * (vocab + 2.0), 11.0) / 3.0
        selected = full_logits.index_select(1, logits_to_keep.long())
        return SimpleNamespace(logits=selected)


def _forward_test_actor():
    return SimpleNamespace(
        use_remove_padding=True,
        use_ulysses_sp=False,
        use_fused_kernels=False,
        device_name="cpu",
        actor_module=_SelectiveLogitsActor(),
    )


def _response_log_probs(logits: torch.Tensor, labels: torch.Tensor):
    return torch.log_softmax(logits.float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)


def _clipped_ppo_loss(log_probs, old_log_probs, advantages, mask):
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 0.8, 1.2)
    loss = torch.maximum(unclipped, clipped)
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def test_response_only_selectors_map_left_pad_and_right_pad_exactly():
    select = _load_function("_response_only_rmpad_selectors")
    # prompt width 5, response width 3; prompts are left padded and responses
    # are right padded with different valid lengths.
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    unpad_indices = torch.nonzero(
        attention_mask.reshape(-1), as_tuple=False
    ).flatten()

    rows, response_positions, packed, logits_to_keep = select(
        attention_mask, 3, unpad_indices
    )

    expected_rows = torch.tensor([0, 1, 1, 1, 2, 2])
    expected_response_positions = torch.tensor([0, 0, 1, 2, 0, 1])
    predictor_flat = expected_rows * attention_mask.size(1) + 4 + expected_response_positions
    assert torch.equal(rows, expected_rows)
    assert torch.equal(response_positions, expected_response_positions)
    assert torch.equal(unpad_indices.index_select(0, packed), predictor_flat)
    assert torch.equal(logits_to_keep, packed)


def test_response_only_selectors_reject_a_response_without_valid_predecessor():
    select = _load_function("_response_only_rmpad_selectors")
    # Response position 1 is marked valid while its position-0 predecessor is
    # padding, which violates the normal right-padded response contract.
    attention_mask = torch.tensor([[1, 1, 1, 0, 1]], dtype=torch.long)
    unpad_indices = torch.nonzero(
        attention_mask.reshape(-1), as_tuple=False
    ).flatten()
    with pytest.raises(ValueError, match="no valid predecessor"):
        select(attention_mask, 2, unpad_indices)


def test_forward_micro_batch_remove_padding_response_only_scores_exact_tokens():
    """Exercise the exact online R/S old-logprob branch end to end.

    This combination deliberately does not construct the legacy rolled labels.
    The regression protects against dereferencing that absent tensor before the
    compact response labels are selected.
    """

    forward = _load_actor_method("_forward_micro_batch", _forward_test_namespace())
    actor = _forward_test_actor()
    input_ids = torch.tensor(
        [
            [0, 4, 5, 6, 7, 8, 0, 0],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor([[8, 0, 0], [6, 7, 8]], dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    micro_batch = {
        "input_ids": input_ids,
        "responses": responses,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(8).expand(2, -1),
    }

    entropy, log_probs = forward(
        actor,
        micro_batch,
        temperature=1.0,
        calculate_entropy=False,
        response_only_logits=True,
    )

    active_rows = torch.tensor([0, 1, 1, 1])
    response_positions = torch.tensor([0, 0, 1, 2])
    packed_positions = torch.tensor([3, 9, 10, 11])
    compact_logits = actor.actor_module(
        input_ids=torch.tensor([[4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]]),
        logits_to_keep=packed_positions,
    ).logits.squeeze(0)
    expected_compact = _response_log_probs(
        compact_logits, responses[active_rows, response_positions]
    )
    expected = torch.zeros(2, 3)
    expected[active_rows, response_positions] = expected_compact

    assert entropy is None
    assert torch.allclose(log_probs, expected, atol=1e-6, rtol=1e-6)


def test_forward_micro_batch_remove_padding_response_only_all_empty_is_safe():
    forward = _load_actor_method("_forward_micro_batch", _forward_test_namespace())
    actor = _forward_test_actor()
    micro_batch = {
        "input_ids": torch.tensor([[0, 3, 4, 5, 0, 0]], dtype=torch.long),
        "responses": torch.tensor([[0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.long),
        "position_ids": torch.arange(6).unsqueeze(0),
    }

    entropy, log_probs = forward(
        actor,
        micro_batch,
        temperature=1.0,
        calculate_entropy=False,
        response_only_logits=True,
    )

    assert entropy is None
    assert torch.equal(log_probs, torch.zeros(1, 2))


def test_selective_lm_head_matches_full_lp_ppo_loss_and_gradients():
    select = _load_function("_response_only_rmpad_selectors")
    scatter = _load_function("_scatter_compact_response_values")
    generator = torch.Generator().manual_seed(19)

    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    response_length = 3
    unpad_indices = torch.nonzero(
        attention_mask.reshape(-1), as_tuple=False
    ).flatten()
    rows, response_positions, packed, _ = select(
        attention_mask, response_length, unpad_indices
    )

    packed_tokens = torch.randint(
        0, 11, (len(unpad_indices),), generator=generator
    )
    response_tokens = torch.randint(0, 11, (2, 3), generator=generator)
    packed_tokens[packed] = response_tokens[rows, response_positions]
    base_hidden = torch.randn(
        len(unpad_indices), 5, generator=generator, dtype=torch.float64
    )
    base_weight = torch.randn(11, 5, generator=generator, dtype=torch.float64)

    hidden_full = base_hidden.clone().requires_grad_(True)
    weight_full = base_weight.clone().requires_grad_(True)
    all_logits = hidden_full @ weight_full.t()
    all_log_probs = _response_log_probs(all_logits, packed_tokens)
    full_response_lp = scatter(
        all_log_probs.index_select(0, packed),
        rows,
        response_positions,
        batch_size=2,
        response_length=response_length,
    )

    hidden_selective = base_hidden.clone().requires_grad_(True)
    weight_selective = base_weight.clone().requires_grad_(True)
    compact_logits = hidden_selective.index_select(0, packed) @ weight_selective.t()
    compact_lp = _response_log_probs(
        compact_logits, response_tokens[rows, response_positions]
    )
    selective_response_lp = scatter(
        compact_lp,
        rows,
        response_positions,
        batch_size=2,
        response_length=response_length,
    )

    response_mask = attention_mask[:, -response_length:].bool()
    old_log_probs = torch.randn(2, 3, generator=generator, dtype=torch.float64)
    advantages = torch.randn(2, 3, generator=generator, dtype=torch.float64)
    full_loss = _clipped_ppo_loss(
        full_response_lp, old_log_probs, advantages, response_mask
    )
    selective_loss = _clipped_ppo_loss(
        selective_response_lp, old_log_probs, advantages, response_mask
    )

    assert torch.equal(full_response_lp, selective_response_lp)
    assert torch.equal(full_loss, selective_loss)

    full_loss.backward()
    selective_loss.backward()
    assert torch.allclose(weight_full.grad, weight_selective.grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(hidden_full.grad, hidden_selective.grad, atol=1e-12, rtol=1e-12)
    gradient_cosine = torch.nn.functional.cosine_similarity(
        weight_full.grad.flatten(), weight_selective.grad.flatten(), dim=0
    )
    assert gradient_cosine.item() == pytest.approx(1.0, abs=1e-12)

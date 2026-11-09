"""Reschedule existing loss units without changing their statistical weights.

Not wired into the live trainer yet. A unit is an EXISTING microbatch and its
global objective coefficient, not a new token/global normalization convention.
DDP/FSDP average gradients across physical ranks, hence multiply by world size.
Each optimizer step has its own plan; never mix work from adjacent steps.
"""
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class LossUnit:
    rows: tuple[int, ...]
    coefficient: float
    token_cost: int

    def __post_init__(self):
        if not self.rows or any(type(i) is not int or i < 0 for i in self.rows):
            raise ValueError("A loss unit needs nonnegative row indices")
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("Invalid objective coefficient")
        if type(self.token_cost) is not int or self.token_cost < 1:
            raise ValueError("token_cost must be a positive integer")


@dataclass(frozen=True)
class WorkSlot:
    unit_index: int
    backward_scale: float
    padding: bool = False


def plan_optimizer_step(units: Sequence[LossUnit], world_size: int):
    """Return equally many collective rounds on every rank.

    Longer units first reduces the maximum per-round token cost. Sorting never
    changes unit membership, normalization, coefficient, or optimizer boundary.
    Empty slots forward the smallest unit with zero backward weight, so FSDP
    still executes the same collective sequence. They are excluded from metrics.
    The model must have no mutable batch statistics; non-finite losses remain
    errors even in padding slots (zero times NaN is not zero).
    """
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be positive")
    if not units:
        return tuple(() for _ in range(world_size))
    order = sorted(range(len(units)), key=lambda i: (-units[i].token_cost, i))
    dummy = min(range(len(units)), key=lambda i: (units[i].token_cost, i))
    ranks = [[] for _ in range(world_size)]
    rounds = (len(order) + world_size - 1) // world_size
    for index in range(rounds * world_size):
        rank = index % world_size
        if index < len(order):
            unit_index = order[index]
            slot = WorkSlot(unit_index, world_size * units[unit_index].coefficient)
        else:
            slot = WorkSlot(dummy, 0.0, padding=True)
        ranks[rank].append(slot)
    return tuple(tuple(slots) for slots in ranks)


def original_dynamic_coefficient(micro_rows: int, local_denominator: int,
                                 original_dp_size: int) -> float:
    """Exactly the current actor/R-S dynamic accumulation coefficient.

    Actor denominator is its configured local minibatch size, including for a
    partial last minibatch. R/S denominator is its actual local batch size.
    Caller must pass the original values; do not renormalize the final minibatch.
    """
    if min(micro_rows, local_denominator, original_dp_size) <= 0:
        raise ValueError("Counts must be positive")
    return micro_rows / (local_denominator * original_dp_size)


def build_dynamic_loss_steps(shards, sequence_lengths, *, local_minibatch_size,
                             max_token_len, partition_fn, reflection=False):
    """Reproduce existing dynamic loss units before physical-rank redistribution.

    `shards` are the already ordered original logical-rank row assignments.
    This function does not rebalance, repad, or change sequence loss masks.
    In particular existing zero-advantage R/S padding is still part of the
    original denominator. `partition_fn` is the repository's existing
    get_seqlen_balanced_partitions function.
    """
    if not shards or not all(shards):
        raise ValueError('Original logical shards must be nonempty')
    if len({len(s) for s in shards})!=1:
        raise ValueError('Original dispatch requires equal-size logical shards')
    if local_minibatch_size<=0 or max_token_len<=0:
        raise ValueError('Minibatch and token limits must be positive')
    flat=[i for shard in shards for i in shard]
    if len(set(flat))!=len(flat):
        raise ValueError('Use distinct physical row indices, including padding rows')
    if any(type(sequence_lengths[i]) is not int or sequence_lengths[i]<=0 for i in flat):
        raise ValueError('Sequence lengths must be positive integer token counts')
    if max(sequence_lengths[i] for i in flat)>max_token_len:
        raise ValueError('A sequence exceeds the existing token budget')
    stride=len(shards[0]) if reflection else local_minibatch_size
    steps=[]
    for offset in range(0,len(shards[0]),stride):
        minibatches=[tuple(shard[offset:offset+stride]) for shard in shards]
        # Equivalent to the current MAX all_reduce before repartitioning.
        count=max(min(len(rows),math.ceil(sum(sequence_lengths[i] for i in rows)/max_token_len))
                  for rows in minibatches)
        if any(count>len(rows) for rows in minibatches):
            raise ValueError('Existing dynamic partition cannot form this many microbatches')
        units=[]
        for rows in minibatches:
            lengths=[sequence_lengths[i] for i in rows]
            partitions=partition_fn(lengths,count,equal_size=False)
            denominator=len(rows) if reflection else local_minibatch_size
            for partition in partitions:
                members=tuple(rows[j] for j in partition)
                units.append(LossUnit(members,original_dynamic_coefficient(len(members),denominator,len(shards)),
                                      sum(sequence_lengths[i] for i in members)))
        steps.append(tuple(units))
    return tuple(steps)

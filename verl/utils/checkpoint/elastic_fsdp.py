"""On-demand, topology-independent FSDP1 snapshots for planned GPU changes.

Regular checkpoints stay unchanged. A SCALE_UP_REQUEST in the run directory
asks the next save to also export named full model/optimizer states.
"""
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType


def combined_optimizer(manager):
    """Serialization-only view: FSDP requires all parameters in a flat handle.

    Actor and R/S remain separate live optimizers; this view never steps them.
    """
    optimizers = {"actor": manager.optimizer, **manager.auxiliary_optimizers}
    groups = [dict(group) for opt in optimizers.values() if opt is not None for group in opt.param_groups]
    combined = torch.optim.Optimizer(groups, defaults={})
    for opt in optimizers.values():
        if opt is not None:
            combined.state.update(opt.state)
    return combined


def restore_separate_optimizers(manager, combined):
    saved = combined.state_dict()
    offset = 0
    for opt in [manager.optimizer, *manager.auxiliary_optimizers.values()]:
        if opt is None:
            continue
        groups = saved["param_groups"][offset:offset + len(opt.param_groups)]
        ids = {pid for group in groups for pid in group["params"]}
        opt.load_state_dict({"param_groups": groups,
                             "state": {pid: state for pid, state in saved["state"].items() if pid in ids}})
        offset += len(opt.param_groups)


def requested(actor_path):
    flag = [os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(actor_path)), "SCALE_UP_REQUEST"))
            if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(flag, src=0)
    return flag[0]


def save_portable(manager, path):
    model = manager.model
    if not isinstance(model, FSDP):
        raise TypeError("Planned elastic export currently requires FSDP1")
    rng = manager.get_rng_state()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    if manager.rank == 0:
        torch.save(state, os.path.join(path, "elastic_model.pt"))
    del state
    optimizers = {"actor": manager.optimizer, **manager.auxiliary_optimizers}
    combined = combined_optimizer(manager)
    state = FSDP.full_optim_state_dict(model, combined, rank0_only=True)
    if manager.rank == 0:
        torch.save(state, os.path.join(path, "elastic_optim_all.pt"))
    del state, combined
    dist.barrier()
    if manager.rank == 0:
        with open(os.path.join(path, "elastic.json"), "w", encoding="utf-8") as stream:
            json.dump({"world_size": manager.world_size,
                       "optimizers": [name for name, opt in optimizers.items() if opt is not None]}, stream)
    dist.barrier()
    manager.load_rng_state(rng)
    print(f"[Elastic] portable checkpoint saved: {path}", flush=True)


def load_portable(manager, path):
    if os.environ.get("WEBSHOP_STREAM_PORTABLE_LOAD") == "1":
        from streamed_native_load import load_portable_streamed
        return load_portable_streamed(manager, path)
    with open(os.path.join(path, "elastic.json"), encoding="utf-8") as stream:
        metadata = json.load(stream)
    source_world = int(metadata["world_size"])
    optimizers = {"actor": manager.optimizer, **manager.auxiliary_optimizers}
    expected = {name for name, opt in optimizers.items() if opt is not None}
    if set(metadata["optimizers"]) != expected:
        raise ValueError("Elastic checkpoint optimizer set differs from current training")
    state = torch.load(os.path.join(path, "elastic_model.pt"), map_location="cpu", weights_only=False, mmap=True)
    with FSDP.state_dict_type(manager.model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
        manager.model.load_state_dict(state, strict=True)
    del state
    combined = combined_optimizer(manager)
    state = (torch.load(os.path.join(path, "elastic_optim_all.pt"),
                        map_location="cpu", weights_only=False, mmap=True) if manager.rank == 0 else None)
    sharded = FSDP.scatter_full_optim_state_dict(state, manager.model, optim=combined)
    combined.load_state_dict(sharded)
    restore_separate_optimizers(manager, combined)
    del state, sharded, combined
    source_rank = manager.rank % source_world
    extra = torch.load(os.path.join(path, f"extra_state_world_size_{source_world}_rank_{source_rank}.pt"),
                       map_location="cpu", weights_only=False, mmap=True)
    if manager.lr_scheduler is not None:
        manager.lr_scheduler.load_state_dict(extra["lr_scheduler"])
    for name, scheduler in manager.auxiliary_lr_schedulers.items():
        saved = torch.load(os.path.join(path, f"{name}_extra_world_size_{source_world}_rank_{source_rank}.pt"),
                           map_location="cpu", weights_only=False, mmap=True)
        if scheduler is not None:
            scheduler.load_state_dict(saved["lr_scheduler"])
    manager.load_rng_state(extra["rng"])
    # Existing ranks retain their exact RNG state. New ranks deterministically
    # branch from the saved parent stream, not identical copied random streams.
    # A topology change is stateful continuation, not bitwise replay.
    if manager.rank >= source_world:
        generator = torch.Generator(device="cpu")
        generator.set_state(extra["rng"]["cpu"])
        seed = int(torch.randint(0, 2**31 - 1, (manager.rank + 1,), generator=generator)[-1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    dist.barrier()
    print(f"[Elastic] rank {manager.rank}: restored model + {sorted(expected)} optimizers, "
          f"schedulers and RNG from {source_world} -> {manager.world_size} ranks", flush=True)

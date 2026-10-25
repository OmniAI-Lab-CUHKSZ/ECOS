"""Full FP32/Adam/per-rank RNG recovery on the checkpoint's original topology."""
import json
from pathlib import Path
import time


def resume_position(state, config, world):
    if state['config'] != config:
        changed = {k: (state['config'].get(k), config.get(k))
                   for k in set(state['config']) | set(config)
                   if state['config'].get(k) != config.get(k)}
        raise ValueError(f'Resume configuration differs: {changed}')
    if state['scheduler'] is not None or config['lr_schedule'] != 'constant':
        raise ValueError('This trainer has constant LR and no scheduler object')
    if len(state['rng_by_rank']) != world:
        raise ValueError('Restore on the same rank topology to preserve RNG')
    step = state['step']
    if not isinstance(step, int) or not 0 <= step < config['expected_steps']:
        raise ValueError(f'Checkpoint is not a resumable intermediate step: {step}')
    epoch, index = divmod(step, config['train_tasks'])
    if (state['next_epoch'], state['next_task_index']) != (epoch, index):
        raise ValueError('Checkpoint task position does not match its step')
    return step, epoch, index


def assert_same(actual, expected, path='state'):
    import torch
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
            raise AssertionError(f'Non-identical restored tensor: {path}')
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f'Restored keys differ: {path}')
        for key, value in expected.items():
            assert_same(actual[key], value, f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f'Restored sequence differs: {path}')
        for i, value in enumerate(expected):
            assert_same(actual[i], value, f'{path}[{i}]')
    elif actual != expected:
        raise AssertionError(f'Restored value differs: {path}: {actual!r} != {expected!r}')


def restore_full_checkpoint(model, optimizer, checkpoint, config, data, rank, world, full_states):
    import gc
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig

    started = time.time()
    checkpoint = Path(checkpoint)
    marker = json.loads((checkpoint/'COMPLETE.json').read_text())
    expected_tasks = json.loads((checkpoint.parent/'training_tasks.json').read_text())['train']
    current_tasks = [dict(group=g['group'], task_key=g['task_key']) for g in data]
    if current_tasks != expected_tasks:
        raise ValueError('Training task order changed since this checkpoint')
    saved = torch.load(checkpoint/'optimizer_rng.pt', map_location='cpu', weights_only=False, mmap=True)
    step, epoch, index = resume_position(saved, config, world)
    assert marker['step'] == step and marker['optimizer_rng_saved'] and marker['master_weights_saved']
    rng = saved['rng_by_rank'][rank]
    assert set(rng) == {'torch', 'cuda', 'python', 'numpy'}
    print('RESTORE_LOADING', json.dumps(dict(rank=rank, step=step, checkpoint=str(checkpoint))), flush=True)
    # Each rank reads the same full CPU state, and FSDP places only its own
    # FP32 parameter/Adam shards on its GPU. HF BF16 files are not resume input.
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)):
        model.load_state_dict(saved['model'], strict=True)
        local_optim = FSDP.optim_state_dict_to_load(model, optimizer, saved['optimizer'])
        optimizer.load_state_dict(local_optim)
    del local_optim
    # Validate the real FSDP load path, not only the on-disk format: export
    # the loaded state and compare every FP32 weight and Adam value exactly.
    actual_model, actual_optim = full_states()
    if rank == 0:
        assert_same(actual_model, saved['model'], 'model')
        assert_same(actual_optim, saved['optimizer'], 'optimizer')
        assert {float(v['step']) for v in actual_optim['state'].values()} == ({float(step)} if step else set())
    record = dict(checkpoint=str(checkpoint), step=step, next_epoch=epoch,
                  next_task_index=index, first_new_step=step+1,
                  model_tensors=len(saved['model']), optimizer_parameters=len(saved['optimizer']['state']),
                  optimizer_tensors=sum(torch.is_tensor(t) for v in saved['optimizer']['state'].values() for t in v.values()),
                  model_and_adam_exact=True, rng_ranks=world,
                  scheduler='constant/no scheduler object', training_task_order_exact=True)
    del actual_model, actual_optim, saved
    gc.collect()
    dist.barrier()
    # Restore all random streams only after initialization/loading/validation.
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(rng['cuda'])
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    assert torch.equal(torch.get_rng_state(), rng['torch'])
    assert torch.equal(torch.cuda.get_rng_state(), rng['cuda'])
    assert random.getstate() == rng['python']
    restored_np = np.random.get_state()
    assert restored_np[0] == rng['numpy'][0] and np.array_equal(restored_np[1], rng['numpy'][1])
    assert restored_np[2:] == rng['numpy'][2:]
    record.update(rng_restored=True, seconds=time.time()-started, time=time.time())
    print('RANK_RESTORE_VERIFIED', json.dumps(dict(record, rank=rank)), flush=True)
    return record

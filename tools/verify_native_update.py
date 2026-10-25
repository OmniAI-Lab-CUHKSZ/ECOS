"""CPU-only inspection of saved GRPO state and actual parameter differences."""
import argparse
import contextlib
import gc
import json
import math
import os
import pickle
from pathlib import Path
import time

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


class SavedShardedTensor:
    """Deserialize saved local shards without creating a distributed group."""
    def __setstate__(self, state):
        self.saved_state = state


class SavedShardUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.distributed._shard.sharded_tensor.api' and name == 'ShardedTensor':
            return SavedShardedTensor
        return super().find_class(module, name)


class SavedShardPickle:
    Unpickler = SavedShardUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def write(path, value):
    partial = path.with_name(path.name + '.partial')
    with partial.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def local_entries(value):
    if isinstance(value, DTensor):
        shape, offsets = compute_local_shape_and_global_offset(value.shape, value.device_mesh, value.placements)
        assert tuple(shape) == tuple(value._local_tensor.shape)
        return [(value._local_tensor, offsets)]
    if isinstance(value, SavedShardedTensor):
        return [(shard.tensor, shard.metadata.shard_offsets) for shard in value.saved_state[0]]
    if isinstance(value, torch.Tensor):
        return [(value, [0]*value.ndim)]
    raise TypeError(type(value))


def sample_slice(tensor):
    sizes = list(tensor.shape)
    sizes[0] = min(sizes[0], max(1, 8192//max(1, math.prod(sizes[1:]))))
    return sizes, tuple(slice(0, size) for size in sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--inspect-serialization-only', action='store_true')
    parser.add_argument('--capture-samples', type=Path)
    parser.add_argument('--reference-samples', type=Path)
    parser.add_argument('--receipt-name', default='first_real_update_verified.json')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(2)
    checkpoint = Path(plan['output']) / f'global_step_{args.step}'
    assert (checkpoint / 'CHECKPOINT_COMPLETE').read_text().strip() == str(args.step)
    loaded = []
    # Map all shards before the next complete checkpoint can prune this one.
    for rank in range(plan['world_size']):
        folder = checkpoint / 'actor'
        model = torch.load(folder / f'model_world_size_{plan["world_size"]}_rank_{rank}.pt', map_location='cpu', weights_only=False, mmap=True, pickle_module=SavedShardPickle)
        if args.inspect_serialization_only:
            first = next(iter(model.values()))
            detail = dict(key=next(iter(model)),type=type(first).__name__,state_types=[type(x).__name__ for x in first.saved_state] if isinstance(first,SavedShardedTensor) else None)
            if isinstance(first, DTensor):
                detail.update(local_shape=list(first._local_tensor.shape),shape=list(first.shape),placements=[str(x) for x in first.placements],coordinate=first.device_mesh.get_coordinate(),computed=compute_local_shape_and_global_offset(first.shape,first.device_mesh,first.placements))
            print(json.dumps(detail))
            return
        optim = torch.load(folder / f'optim_world_size_{plan["world_size"]}_rank_{rank}.pt', map_location='cpu', weights_only=False, mmap=True)
        extra = torch.load(folder / f'extra_state_world_size_{plan["world_size"]}_rank_{rank}.pt', map_location='cpu', weights_only=False)
        loaded.append((rank, model, optim, extra))
    data = torch.load(checkpoint / 'data.pt', map_location='cpu', weights_only=False)
    env = torch.load(checkpoint / plan['environment_state_file'], map_location='cpu', weights_only=False)
    assert len(env['train']['workers']) == 128 and len(env['validation']['workers']) == 128
    if args.capture_samples:
        assert not args.capture_samples.exists()
        captured = []
        for rank, model, optim, extra in loaded:
            samples = {}
            for name, value in model.items():
                for entry, (tensor, offsets) in enumerate(local_entries(value)):
                    if tensor.numel():
                        _, selection = sample_slice(tensor)
                        samples[f'{name}::{entry}'] = tensor[selection].clone()
            captured.append(dict(rank=rank, samples=samples,
                                 adam_steps=sorted(set(float(x['step']) for x in optim['state'].values()))))
        with args.capture_samples.open('xb') as handle:
            torch.save(dict(step=args.step, checkpoint=str(checkpoint), ranks=captured), handle)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(dict(captured_from_step=args.step, ranks=len(captured), path=str(args.capture_samples), bytes=args.capture_samples.stat().st_size)),flush=True)
        return
    reference = torch.load(args.reference_samples, map_location='cpu', weights_only=False) if args.reference_samples else None
    if reference:
        assert args.step == reference['step'] + 1
    index = json.loads((Path(plan['model']) / 'model.safetensors.index.json').read_text())['weight_map']
    results = []
    with contextlib.ExitStack() as stack:
        safes = {name:stack.enter_context(safe_open(str(Path(plan['model'])/name), framework='pt', device='cpu')) for name in set(index.values())}
        for position, (rank, model, optim, extra) in enumerate(loaded):
            assert 'lr_scheduler' in extra and 'rng' in extra
            adam_steps = sorted(set(float(x['step']) for x in optim['state'].values()))
            tensors = [v for state in optim['state'].values() for key,v in state.items() if key != 'step' and isinstance(v, torch.Tensor)]
            adam_finite = all(bool(torch.isfinite(t).all()) for t in tensors)
            adam_dtypes = sorted(set(str(t.dtype) for t in tensors))
            assert adam_steps and min(adam_steps)>0 and adam_finite and adam_dtypes==['torch.float32']
            assert all(g['lr']==1e-6 for g in optim['param_groups'])
            finite = True
            dtypes = set()
            compared = 0
            changed = 0
            delta_squared = 0.0
            max_delta = 0.0
            sample_keys = []
            for name, value in model.items():
                entries = local_entries(value)
                for entry, (tensor, offsets) in enumerate(entries):
                    finite = finite and bool(torch.isfinite(tensor).all())
                    dtypes.add(str(tensor.dtype))
                    if tensor.numel()==0:
                        continue
                    key = name.replace('_fsdp_wrapped_module.', '')
                    if key == 'lm_head.weight' and key not in index:
                        assert json.loads((Path(plan['model'])/'config.json').read_text())['tie_word_embeddings']
                        key = 'model.embed_tokens.weight'
                    assert key in index, key
                    sizes, local_slice = sample_slice(tensor)
                    source_slice = tuple(slice(o,o+n) for o,n in zip(offsets,sizes))
                    initial = (reference['ranks'][rank]['samples'][f'{name}::{entry}'].float() if reference
                               else safes[index[key]].get_slice(key)[source_slice].float())
                    current = tensor[local_slice].float()
                    assert current.shape == initial.shape
                    delta = current-initial
                    compared += delta.numel()
                    changed += int(torch.count_nonzero(delta))
                    delta_squared += float(delta.double().square().sum())
                    max_delta = max(max_delta,float(delta.abs().max()))
                    if bool(torch.count_nonzero(delta)) and len(sample_keys)<4:sample_keys.append(key)
            assert finite and dtypes=={'torch.float32'} and changed>0
            result=dict(rank=rank,adam_steps=adam_steps,adam_all_finite=adam_finite,adam_dtypes=adam_dtypes,lr=[g['lr'] for g in optim['param_groups']],scheduler=extra['lr_scheduler'],rng_keys=list(extra['rng']),model_all_finite=finite,model_dtypes=sorted(dtypes),compared_parameter_elements=compared,changed_parameter_elements=changed,sampled_delta_l2=math.sqrt(delta_squared),max_abs_delta=max_delta,example_changed_keys=sample_keys)
            if reference:
                result['source_adam_steps'] = reference['ranks'][rank]['adam_steps']
                assert min(adam_steps) > max(result['source_adam_steps'])
            results.append(result)
            write(Path(plan['root'])/'native_update_progress.json',dict(step=args.step,checked_ranks=rank+1,last=result,time=time.time()))
            loaded[position]=None
            del model,optim,extra,tensors,tensor,value,entries
            gc.collect()
    receipt=dict(time=time.time(),step=args.step,checkpoint=str(checkpoint),initialization_model=plan['model'],ranks=results,data_keys=list(data),environment={k:dict(env_num=v['env_num'],group_n=v['group_n'],workers=len(v['workers']),has_sampler='sampler' in v) for k,v in env.items()},nonzero_parameter_change_verified=True,comparison='Actual saved FP32 shard slices versus corresponding initial safetensors slices; no model generation',all_model_and_adam_finite=True)
    if reference:
        receipt.update(resume_source_step=reference['step'],reference_samples=str(args.reference_samples),
                       comparison='Actual saved FP32 shard slices versus retained samples from the complete resume checkpoint')
    assert Path(args.receipt_name).name == args.receipt_name
    method=json.loads((checkpoint/'method_observed.json').read_text())
    assert method['rs_joint_update'] and method['route_mode']=='selective_gigpo'
    assert method['actor_signal_scope']=='action_command' and method['modulation_gap_unit']=='action_command_sum'
    assert method['actor_gap_unit']=='action_command_sum'
    assert method['modulation_mode']=='symmetric_alignment' and method['modulation_lambda']==1. and method['modulation_tau']==.5
    actual=method['actor_metrics'];applied=actual['actor/optimizer_steps_applied']
    selective=method['selective_metrics']
    assert selective['selective/actor_response_token_scope']==0. and selective['selective/action_advantage_shared_across_response']==0.
    assert selective['selective/distilled_nonaction_tokens']==0.
    assert selective['selective/skip_teacher_on_all_success']==1.
    # Natural groups may have zero Teacher or R/S credit. Record that outcome honestly.
    assert applied>0 and applied==actual['actor/optimizer_steps_attempted']
    for row in results:
        source_steps=row.get('source_adam_steps',[0.])
        assert row['adam_steps']==sorted(set(x+applied for x in source_steps))
        assert row['scheduler']['last_epoch']==args.step
    assert math.isclose(actual['actor/route_coef'], .02, rel_tol=0.0, abs_tol=4*math.ulp(.02))
    assert math.isclose(actual['rs/coef'], .1, rel_tol=0.0, abs_tol=4*math.ulp(.1))
    assert actual['rs/joint_update']==1 and actual['rs/extra_optimizer_steps']==0
    assert math.isfinite(actual['rs_joint/scaled_rs_grad_norm']) and actual['rs_joint/scaled_rs_grad_norm']>=0
    receipt.update(method_observed=method,rs_training_verified=actual['rs_joint/scaled_rs_grad_norm']>0,rs_extra_optimizer_steps=0,
                   adam_matches_applied_updates=True,shared_actor_joint_optimizer=True)
    import io
    import zipfile
    import numpy as np
    signal_source=checkpoint/'teacher_action_signals.npz'
    if signal_source.is_file():
        signal_bytes=signal_source.read_bytes()
    else:
        # Only step 1 has a dedicated top-level copy. Resumed first-step
        # diagnostics are retained in the same native checkpoint raw archive.
        with zipfile.ZipFile(checkpoint/'attempt_raw.zip') as archive:
            signal_bytes=archive.read(f'training/unified_diagnostics/step{args.step}_natural_signals.npz')
    with np.load(io.BytesIO(signal_bytes),allow_pickle=False) as signal:
        from response_rs.teacher_signals import verify_action_teacher_signals
        receipt['action_teacher_verified']=verify_action_teacher_signals(signal)
        receipt['teacher_group_counts']=receipt['action_teacher_verified']['teacher_group_counts']
    signal_copy=Path(plan['root'])/'teacher_action_signals.npz'
    signal_copy.write_bytes(signal_bytes)
    assert signal_copy.read_bytes()==signal_bytes
    with signal_copy.open('r+b') as handle:os.fsync(handle.fileno())
    receipt['durable_receipt']=json.loads((checkpoint/'STORAGE_ARCHIVE_VERIFIED.json').read_text())
    assert receipt['durable_receipt']['all_bytes_equal'] and receipt['durable_receipt']['fsync']
    write(Path(plan['root'])/args.receipt_name,receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()

"""Full-parameter 3B/7B ALFWorld/WebShop SFT using the retained original task objective and existing FSDP kernel.

All ranks cooperate on ONE task per optimizer update, not multiple tasks. Each
rank runs ceil(9/world) forwards/backwards; dummy slots have zero loss so FSDP has the
same collective sequence. Gradient averaging is compensated by world_size.
The task objective remains mean(R1..R8)/2 + S/2. No trajectory truncation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


from .sft_data import slots


def order_for_epoch(data, epoch):
    order = list(range(len(data)))
    random.Random(epoch).shuffle(order)
    return [data[i] for i in order]


def load_data(material, annotator_model=None):
    from .sft_data import load_groups
    groups, info = load_groups(material)
    if annotator_model and annotator_model != info['annotator']:
        raise ValueError('Requested annotator differs from the supplied material')
    return dict(train=groups, val=[], metadata=info)


def run(a):
    import yaml
    import datetime
    from functools import partial
    import shutil
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert 2 <= world <= 8, 'One node, 2..8 cooperating ranks per task'
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    torch.set_num_threads(2); torch.manual_seed(0); random.seed(0); np.random.seed(0)
    annotator_model = getattr(a, 'annotator_model', None)
    data = load_data(a.material, annotator_model)
    recipe = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/sft.yaml').read_text(encoding='utf8'))
    annotator_model = data['metadata']['annotator']
    a.output.mkdir(parents=True, exist_ok=True)
    if a.smoke and a.resume_from:
        raise ValueError('Use --resume-test with --resume-from, not --smoke')
    if a.resume_test and not a.resume_from:
        raise ValueError('--resume-test requires a full checkpoint')
    if rank == 0:
        if (a.output / 'progress.json').exists():
            raise FileExistsError('Use a fresh output directory; preserve previous run')
        required_gib = 120 if (a.smoke or a.resume_test) else 380
        if shutil.disk_usage(a.output).free < required_gib * 2**30:
            raise RuntimeError('Insufficient shared space for full optimizer checkpoints')
        write(a.output / 'progress.json', dict(state='LOADING_MODEL', time=time.time()))
    dist.barrier()
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', local_files_only=True).float()
    base.requires_grad_(True)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    from .response_loss import ResponseLoss
    model = FSDP(ResponseLoss(base), device_id=local_rank, sync_module_states=True,
                 auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                          transformer_layer_cls={Qwen2DecoderLayer}),
                 use_orig_params=True, limit_all_gathers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], betas=tuple(recipe['betas']),
                                  eps=recipe['eps'], weight_decay=recipe['weight_decay'], foreach=False)
    config = dict(base_model=str(a.model), annotator=annotator_model, reasoning_effort='max',
                  train_tasks=len(data['train']), val_tasks=0, epochs=recipe['epochs'], expected_steps=len(data['train'])*recipe['epochs'],
                  environment=data['metadata']['environment'], source_model=data['metadata']['model_family'],
                  group_batch_size=1, text_batch_size=9, world_size=world, train_seed=0,
                  lr=recipe['lr'], betas=recipe['betas'], eps=recipe['eps'], weight_decay=recipe['weight_decay'], grad_clip=recipe['grad_clip'],
                  lr_schedule='constant', warmup=0, master_dtype='float32', autocast='bfloat16',
                  response_only_logits=True, padding=False, max_length=32768,
                  truncation='error', lora=False, objective='0.5*mean(R8)+0.5*S')
    if rank == 0:
        write(a.output / 'config.json', config)
        write(a.output / 'training_tasks.json', dict(train=[dict(group=g['group'],
              task_key=g['task_key']) for g in data['train']], validation=[],
              train_texts=1800, original_split='All 200 supplied training tasks'))
    import psutil, socket
    actual_uuid = str(torch.cuda.get_device_properties(local_rank).uuid).removeprefix('GPU-')
    allowed = json.loads(os.environ['TRAINING_GPU_UUIDS'])
    assert actual_uuid == allowed[local_rank].removeprefix('GPU-')
    write(a.output/f'identity_rank{rank}.json',dict(rank=rank,pid=os.getpid(),
          create_time=psutil.Process().create_time(),host=socket.gethostname(),gpu_uuid='GPU-'+actual_uuid,
          master_dtype='float32',world_size=world,model=str(a.model),time=time.time()))

    def loss_of(row):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(row['prompt_ids'], row['response_ids'])

    @torch.no_grad()
    def evaluate(groups, epoch, step):
        model.eval()
        sums = torch.zeros(2, dtype=torch.float64, device='cuda')
        for group in groups:
            for row, weight in slots(group['reflections'] + group['skills'], rank, world):
                value = loss_of(row)
                sums[0 if row['kind'] == 'reflection' else 1] += value.double() * weight
        dist.all_reduce(sums)
        sums /= len(groups)
        if not torch.isfinite(sums).all():
            raise ValueError('Nonfinite fixed-target validation')
        return dict(epoch=epoch, step=step, fixed_train_nll=float(sums.sum()), split='train_fixed',
                    reflection_nll=float(2*sums[0]), skill_nll=float(2*sums[1]), time=time.time())

    def full_states():
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
        return state, optim_state

    def save_checkpoint(epoch, step, next_task_index=0):
        ckpt = a.output / f'checkpoint{step:03d}'
        if rank == 0:
            ckpt.mkdir(exist_ok=False)
            write(a.output/'progress.json', dict(state='SAVING_CHECKPOINT', step=step, epoch=epoch, time=time.time()))
        dist.barrier()
        state, optim_state = full_states()
        rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                   python=random.getstate(), numpy=np.random.get_state())
        rngs = [None] * world if rank == 0 else None
        dist.gather_object(rng, rngs, dst=0)
        if rank == 0:
            assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
            assert all(torch.isfinite(t).all() for values in optim_state['state'].values()
                       for t in values.values() if torch.is_tensor(t))
            # Save full FP32 model and FSDP full optimizer names for genuine resume.
            torch.save(dict(model=state, optimizer=optim_state, rng_by_rank=rngs,
                            step=step, epoch=epoch,
                            next_epoch=epoch if next_task_index == 0 else epoch-1,
                            next_task_index=next_task_index,
                            scheduler=None, config=config), ckpt/'optimizer_rng.pt')
            hf_state = {n.removeprefix('model.'): p.to(torch.bfloat16) for n,p in state.items()}
            base.save_pretrained(ckpt, state_dict=hf_state, safe_serialization=True, max_shard_size='4GB')
            tok.save_pretrained(ckpt)
            for path in ckpt.iterdir():
                if path.is_file():
                    with path.open('r+b') as handle: os.fsync(handle.fileno())
            write(ckpt/'COMPLETE.json', dict(step=step, epoch=epoch, optimizer_rng_saved=True,
                                          master_weights_saved=True,all_finite=True,fsync=True,time=time.time()))
        del state, optim_state
        dist.barrier()

    history = []; metrics = []; step = 0
    start_epoch = start_index = 0
    if a.resume_from:
        from .sft_resume_state import restore_full_checkpoint
        restored = restore_full_checkpoint(model, optimizer, a.resume_from, config,
                     data['train'], rank, world, full_states)
        step, start_epoch, start_index = restored['step'], restored['next_epoch'], restored['next_task_index']
        if rank == 0:
            write(a.output/'resume.json', restored)
            print('FULL_STATE_RESUMED', json.dumps(restored), flush=True)
        dist.barrier()
        # No random work after restoration; constructing epoch orders uses
        # dedicated Python Random(epoch), not the saved global random stream.
    if a.smoke:
        # One complete real task containing the longest training Reflection,
        # repeated twice so the second pass includes initialized Adam states.
        longest = max(data['train'], key=lambda g:max(len(r['prompt_ids'])+len(r['response_ids'])
                      for r in g['reflections']+g['skills']))
        epochs = [[longest, longest]]
    else:
        # No held-out material NLL: all 200 labeled tasks are now training data.
        epochs = [order_for_epoch(data['train'], epoch) for epoch in range(recipe['epochs'])]
    performed = 0
    if not a.smoke and not a.resume_test and not a.resume_from:
        write(a.output/f'rank{rank}_initialization.json',dict(rank=rank,master_dtype='float32',optimizer_state_empty=not optimizer.state,time=time.time()))
        save_checkpoint(0,0)
    if not a.smoke and not a.resume_test and not a.resume_from:
        if rank == 0:
            write(a.output/'progress.json',dict(state='FIXED_TARGET_EVALUATION',step=0,epoch=0,time=time.time()))
        initial = evaluate(data['train'],0,0)
        if rank == 0:
            metrics.append(initial);write(a.output/'epoch_metrics.json',metrics)
            print('FIXED_TRAIN_NLL',json.dumps(initial),flush=True)
    for epoch, groups in enumerate(epochs, 1):
        if not a.smoke and epoch - 1 < start_epoch:
            continue
        for group_index, group in enumerate(groups):
            if not a.smoke and epoch - 1 == start_epoch and group_index < start_index:
                continue
            start = time.time(); model.train(); optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), dtype=torch.float64, device='cuda')
            for row, weight in slots(group['reflections']+group['skills'], rank, world):
                loss = loss_of(row)
                finite = torch.tensor(int(torch.isfinite(loss)), device='cuda')
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise ValueError('Nonfinite R/S loss')
                (loss * weight * world).backward()
                local_loss += loss.detach().double() * weight
            norm = model.clip_grad_norm_(recipe['grad_clip'])
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm; no optimizer step')
            # A small local parameter slice confirms an actual update without
            # cloning the full model; aggregate squared change over the ranks.
            local_params = [(n,p) for n,p in model.named_parameters()
                            if p.requires_grad and p.numel() > 0 and 'layers.' in n]
            probe_name, probe_param = next(((n,p) for n,p in local_params
                              if n.endswith('self_attn.q_proj.weight')), local_params[0])
            probe = probe_param.reshape(-1)[:4096]
            before = probe.detach().clone()
            optimizer.step(); step += 1; performed += 1
            change = (probe.detach()-before).double().square().sum()
            dist.all_reduce(change); dist.all_reduce(local_loss)
            if not torch.isfinite(change) or change.item() == 0:
                raise ValueError('Parameter update probe is zero or nonfinite')
            peak = torch.tensor(torch.cuda.max_memory_allocated()/2**30, device='cuda')
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            entry = dict(state='TRAINING', step=step, epoch=epoch, group=group['group'],
                         loss=local_loss.item(), grad_norm=float(norm), lr=recipe['lr'],
                         parameter_probe_update_l2=math.sqrt(change.item()),
                         seconds=time.time()-start, peak_allocated_gib=float(peak), time=time.time())
            assert all(p.dtype == torch.float32 for p in model.parameters())
            local_adam_steps = {float(state['step']) for state in optimizer.state.values()}
            assert local_adam_steps == {float(step)}
            assert all(state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32 for state in optimizer.state.values())
            write(a.output/f'optimizer_rank{rank}_latest.json',dict(rank=rank,step=step,
                  parameters_with_adam=len(optimizer.state),adam_steps=sorted(local_adam_steps),
                  master_dtype='float32',adam_dtype='float32',time=time.time()))
            if rank == 0:
                history.append(entry); write(a.output/'progress.json', entry)
                with (a.output/'history.jsonl').open('a', encoding='utf8') as out:
                    out.write(json.dumps(entry)+'\n')
                print('FULL_SFT_STEP', json.dumps(entry), flush=True)
            optimizer.zero_grad(set_to_none=True)
            if not a.smoke and not a.resume_test:
                # A task-private file request is checked only after a real update.
                # Rank0 broadcasts once, so every rank saves the same boundary.
                pause = torch.tensor(int(rank == 0 and (a.output/'PAUSE_REQUEST').exists()),
                                     device='cuda')
                dist.broadcast(pause, src=0)
                if pause.item():
                    next_index = (group_index + 1) % len(groups)
                    save_checkpoint(epoch, step, next_task_index=next_index)
                    if rank == 0:
                        result = dict(state='PAUSED_BY_REQUEST', step=step, epoch=epoch,
                                      checkpoint=str(a.output/f'checkpoint{step:03d}'), time=time.time())
                        write(a.output/'PAUSED.json', result)
                        write(a.output/'progress.json', result)
                        print('PAUSED_BY_REQUEST', json.dumps(result), flush=True)
                    dist.barrier(); dist.destroy_process_group()
                    return
            if a.resume_test and performed == 2:
                break
        if a.resume_test and performed == 2:
            break
        if not a.smoke:
            if not a.resume_test:
                metric = evaluate(data['train'],epoch,step)
                if rank == 0:
                    metrics.append(metric);write(a.output/'epoch_metrics.json',metrics)
                    print('FIXED_TRAIN_NLL',json.dumps(metric),flush=True)
            save_checkpoint(epoch, step)
    if a.smoke or a.resume_test:
        # Exercise the same cross-rank state export used by real checkpoints,
        # without retaining a large checkpoint from a disposable two-step test.
        state, optim_state = full_states()
        if rank == 0:
            assert state and optim_state['state']
            assert {float(v['step']) for v in optim_state['state'].values()} == {float(step)}
            assert all(t.dtype == torch.float32 for t in state.values() if t.is_floating_point())
            assert all(torch.isfinite(t).all() for t in state.values())
            assert all(torch.isfinite(t).all() for values in optim_state['state'].values()
                       for t in values.values() if torch.is_tensor(t))
            write(a.output/'state_export_test.json', dict(model_tensors=len(state),
                  optimizer_parameters=len(optim_state['state']), steps=sorted({float(v['step'])
                  for v in optim_state['state'].values()}), all_finite=True,
                  master_dtype='float32', time=time.time()))
        del state, optim_state
        dist.barrier()
    if not a.smoke and not a.resume_test:
        assert step == config['expected_steps'], step
    if rank == 0:
        result = dict(state='RESUME_TEST_COMPLETE' if a.resume_test else ('SMOKE_COMPLETE' if a.smoke else 'TRAIN_COMPLETE'), step=step,
                      smoke=a.smoke, epoch_metrics=metrics, config=config, time=time.time(),
                      final_checkpoint=None if (a.smoke or a.resume_test) else str(a.output/f'checkpoint{step:03d}'))
        write(a.output/'COMPLETE.json', result); write(a.output/'progress.json', result)
        print(result['state'], json.dumps(result), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--resume-test', action='store_true')
    parser.add_argument('--annotator-model', default=None)
    run(parser.parse_args())

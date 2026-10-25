"""Load exact rank-local flat slices without materializing full Adam flat copies."""
import gc
import json
import math
import os
import socket
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def flat_layout(model):
    result={}
    for name,module in model.named_modules():
        if not isinstance(module,FSDP) or module._handle is None:continue
        flat=module._handle.flat_param
        prefix=name.replace('_fsdp_wrapped_module.','').removesuffix('._fsdp_wrapped_module')
        if prefix=='_fsdp_wrapped_module':prefix=''
        entries=[dict(name=(prefix+'.' if prefix else '')+fqn,shape=tuple(shape),numel=int(n))
                 for fqn,shape,n in zip(flat._fqns,flat._shapes,flat._numels)]
        assert not getattr(module,'_use_orig_params',False)
        assert not flat._shared_param_infos, 'Tied parameters require explicit alias handling'
        result[flat]=entries
    assert {id(p) for p in result}=={id(p) for p in model.parameters()}
    return result


def fill_slice(destination,entries,values,rank,world):
    total=sum(e['numel'] for e in entries);width=math.ceil(total/world)
    assert destination.numel()==width
    start=rank*width;end=min(start+width,total);cursor=0;copied=0
    for entry in entries:
        value=values(entry['name'])
        assert value.dtype==destination.dtype and tuple(value.shape)==entry['shape']
        lo=max(start,cursor);hi=min(end,cursor+entry['numel'])
        if hi>lo:
            source=value.reshape(-1)
            for offset in range(lo,hi,2**20):
                finish=min(offset+2**20,hi)
                destination[offset-start:finish-start].copy_(source[offset-cursor:finish-cursor])
            copied+=hi-lo
        cursor+=entry['numel']
    assert copied==max(0,end-start)
    if copied<width:destination[copied:].zero_()


def release_copy(manager,path,name,phase):
    """The durable source was byte-verified before the attempt was launched."""
    dist.barrier()
    env=os.environ.get('WEBSHOP_DURABLE_PLAN')
    if manager.rank==0 and env:
        plan=json.loads(Path(env).read_text());source=Path(path)
        assert source.resolve()==(Path(plan['resume_from_path'])/'actor').resolve()
        rollback=plan['durable_rollback_source'];receipt=rollback['receipt']
        assert receipt['all_values_equal'] and receipt['fsync'] and receipt['step']==plan['resume_source_step']
        file=source/name;assert not file.is_symlink() and file.stat().st_size==rollback['files'][name]
        file.unlink()
        from durable_native_store import write
        write(source.parent/phase,dict(time=time.time(),source_host=socket.gethostname(),source=rollback['source'],file=name,step=receipt['step'],loaded_all_ranks=True))
    dist.barrier()


@torch.no_grad()
def load_portable_streamed(manager,path):
    path=Path(path);metadata=json.loads((path/'elastic.json').read_text())
    assert metadata['optimizers']==['actor'] and not manager.auxiliary_optimizers
    layout=flat_layout(manager.model)
    state=torch.load(path/'elastic_model.pt',map_location='cpu',weights_only=False,mmap=True)
    assert set(state)=={entry['name'] for entries in layout.values() for entry in entries}
    for parameter,entries in layout.items():fill_slice(parameter.data,entries,lambda name:state[name],manager.rank,manager.world_size)
    del state;gc.collect()
    release_copy(manager,path,'elastic_model.pt','ELASTIC_MODEL_LOADED.json')
    state=torch.load(path/'elastic_optim_all.pt',map_location='cpu',weights_only=False,mmap=True)
    assert len(state['param_groups'])==len(manager.optimizer.param_groups)
    manager.optimizer.state.clear()
    for group,saved_group in zip(manager.optimizer.param_groups,state['param_groups']):
        names=[e['name'] for p in group['params'] for e in layout[p]]
        assert names==saved_group['params']
        group.update({k:v for k,v in saved_group.items() if k!='params'})
        for parameter in group['params']:
            entries=layout[parameter];first=state['state'][entries[0]['name']];restored={}
            for key,value in first.items():
                assert all(key in state['state'][e['name']] for e in entries)
                if isinstance(value,torch.Tensor) and value.ndim>0:
                    restored[key]=torch.empty(parameter.numel(),dtype=value.dtype,device='cpu')
                    fill_slice(restored[key],entries,lambda name:state['state'][name][key],manager.rank,manager.world_size)
                else:
                    assert all(torch.equal(state['state'][e['name']][key],value) if isinstance(value,torch.Tensor) else state['state'][e['name']][key]==value for e in entries)
                    restored[key]=value.clone() if isinstance(value,torch.Tensor) else value
            manager.optimizer.state[parameter]=restored
    del state,first,value;gc.collect()
    release_copy(manager,path,'elastic_optim_all.pt','ELASTIC_OPTIMIZER_LOADED.json')
    source_world=int(metadata['world_size']);source_rank=manager.rank%source_world
    extra=torch.load(path/f'extra_state_world_size_{source_world}_rank_{source_rank}.pt',map_location='cpu',weights_only=False)
    if manager.lr_scheduler is not None:manager.lr_scheduler.load_state_dict(extra['lr_scheduler'])
    manager.load_rng_state(extra['rng'])
    assert manager.world_size<=source_world, 'Expansion RNG branching is provided by the existing loader'
    dist.barrier()
    print(f'[Elastic] streamed rank {manager.rank}: exact model/Adam/scheduler/RNG restored {source_world} -> {manager.world_size}',flush=True)

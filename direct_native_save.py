"""Serialize native tensor shards to durable storage without a tmpfs payload copy."""
import http.client
import json
import os
from pathlib import Path
import re
import time
import urllib.parse

import torch

from durable_native_store import write


class Stream:
    def __init__(self, consume=None):
        self.consume=consume
        self.count=0

    def write(self, data):
        view=memoryview(data)
        for offset in range(0,len(view),8*1024**2):
            if self.consume:self.consume(view[offset:offset+8*1024**2])
        self.count+=len(view)
        return len(view)

    def flush(self):
        pass


def save_tensor_state(state, filename, plan=None):
    if plan is None:
        config=os.environ.get('WEBSHOP_DURABLE_PLAN')
        plan=json.loads(Path(config).read_text()) if config else {}
    if not plan.get('direct_native_save'):
        return torch.save(state,filename)
    path=Path(filename)
    assert path.parent.name=='actor' and path.parent.parent.parent.resolve()==Path(plan['output']).resolve()
    assert re.fullmatch(r'(model|optim)_world_size_[0-9]+_rank_[0-9]+\.pt',path.name)
    assert not path.exists() and not path.with_name(path.name+'.DIRECT_SAVED.json').exists()
    checkpoint=path.parent.parent
    step=int(checkpoint.name.removeprefix('global_step_'))
    name=path.relative_to(checkpoint).as_posix()
    counter=Stream();torch.save(state,counter)
    size=counter.count
    token=Path(plan['archive_token_file']).read_text().strip()
    headers={'Authorization':'Bearer '+token}
    url='/'+checkpoint.name+'/'+urllib.parse.quote(name,safe='/')
    def connect():return http.client.HTTPConnection(plan['archive_ip'],plan['archive_http_port'],timeout=300)
    connection=connect()
    try:
        connection.putrequest('PUT',url)
        for key,value in dict(headers,**{'Content-Length':str(size)}).items():connection.putheader(key,value)
        connection.endheaders()
        sender=Stream(connection.send);torch.save(state,sender)
        assert sender.count==size
        response=connection.getresponse()
        assert response.status==201,(response.status,response.read())
        response.read()
    finally:connection.close()
    connection=connect()
    try:
        connection.request('GET',url,headers=headers)
        response=connection.getresponse();assert response.status==200
        assert int(response.getheader('Content-Length'))==size
        def compare(block):
            assert response.read(len(block))==block, 'Native serialized readback differs'
        verifier=Stream(compare);torch.save(state,verifier)
        assert verifier.count==size and response.read(1)==b''
    finally:connection.close()
    receipt=dict(step=step,name=name,bytes=size,all_bytes_equal=True,fsync=True,time=time.time(),
                 target=plan['archive_storage']+'/'+checkpoint.name+'/'+name,target_host=plan['archive_host'])
    write(path.with_name(path.name+'.DIRECT_SAVED.json'),receipt)
    print('[Direct durable shard]',step,name,size,flush=True)
    return receipt


def gather_rank_metadata(actor_path, rank, world_size):
    """Gather only small per-rank metadata when nodes have separate local disks.

    Tensor bytes are already persisted and verified by their owning rank.
    Rank zero needs every receipt and extra state before the durable commit.
    """
    config = os.environ.get('WEBSHOP_DURABLE_PLAN')
    plan = json.loads(Path(config).read_text()) if config else {}
    if not (plan.get('direct_native_save') and len(plan.get('nodes', [])) > 1):
        return
    import torch.distributed as dist
    actor = Path(actor_path)
    names = [f'{prefix}_world_size_{world_size}_rank_{rank}.pt.DIRECT_SAVED.json'
             for prefix in ['model', 'optim']]
    names += [f'extra_state_world_size_{world_size}_rank_{rank}.pt']
    payload = {name: (actor/name).read_bytes() for name in names}
    received = [None]*world_size if rank == 0 else None
    dist.gather_object(payload, received, dst=0)
    if rank == 0:
        for item in received:
            for name, data in item.items():
                path = actor/name
                if path.exists():
                    assert path.read_bytes() == data
                else:
                    with path.open('xb') as stream:
                        stream.write(data);stream.flush();os.fsync(stream.fileno())
        fd = os.open(actor, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    dist.barrier()

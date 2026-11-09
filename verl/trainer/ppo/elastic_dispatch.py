"""Redistribute existing logical-rank loss units to physical training workers."""
import copy
import torch
from verl.protocol import DataProto
from verl.trainer.ppo.elastic_workplan import build_dynamic_loss_steps, plan_optimizer_step
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


def pack_loss_packets(data, *, logical_world_size, physical_world_size,
                      global_minibatch_size, max_token_len, reflection=False):
    if len(data)%logical_world_size or global_minibatch_size%logical_world_size:
        raise ValueError('The original logical dispatch/minibatch must divide exactly')
    local=len(data)//logical_world_size
    shards=[tuple(range(i*local,(i+1)*local)) for i in range(logical_world_size)]
    lengths=data.batch['attention_mask'].sum(-1).tolist()
    steps=build_dynamic_loss_steps(shards,lengths,
        local_minibatch_size=global_minibatch_size//logical_world_size,
        max_token_len=max_token_len,partition_fn=get_seqlen_balanced_partitions,
        reflection=reflection)
    indices=[[] for _ in range(physical_world_size)]
    descriptors=[[] for _ in range(physical_world_size)]
    for units in steps:
        for rank,slots in enumerate(plan_optimizer_step(units,physical_world_size)):
            pieces=[]
            for slot in slots:
                begin=len(indices[rank]);indices[rank].extend(units[slot.unit_index].rows)
                pieces.append((begin,len(indices[rank]),slot.backward_scale,slot.padding))
            descriptors[rank].append(pieces)
    packets=[]
    for rank in range(physical_world_size):
        packet=data.select_idxs(torch.tensor(indices[rank],dtype=torch.long))
        packet.meta_info=copy.deepcopy(data.meta_info)
        packet.meta_info['elastic_loss_steps']=descriptors[rank]
        packet.meta_info['elastic_physical_world_size']=physical_world_size
        packets.append(packet)
    return packets


def collect_loss_metrics(outputs):
    metrics={}
    for output in outputs:
        for key,value in output.meta_info['metrics'].items():
            metrics.setdefault(key,[]).extend(value if isinstance(value,list) else [value])
    return DataProto(meta_info={'metrics':metrics})


def pack_joint_rs_packets(data, *, physical_world_size, optimizer_steps, max_token_len):
    """Consume R/S once, spread over the EXISTING Actor optimizer slots.

    Advantages already contain N_active * fixed group/stage weight. Every
    active sequence therefore retains coefficient 1/N_active, including when
    some groups have zero reward. No logical-rank padding or new denominator.
    """
    from verl.trainer.ppo.elastic_workplan import LossUnit
    import math
    if optimizer_steps < 1 or len(data) < 1:
        raise ValueError('Joint R/S requires nonempty data and Actor steps')
    lengths = data.batch['attention_mask'].sum(-1).tolist()
    if min(lengths) <= 0 or max(lengths) > max_token_len:
        raise ValueError('R/S sequence outside configured token budget')
    indices = [[] for _ in range(physical_world_size)]
    descriptors = [[] for _ in range(physical_world_size)]
    for step in range(optimizer_steps):
        rows = list(range(step, len(data), optimizer_steps))
        units = []
        if rows:
            count = min(len(rows), math.ceil(sum(lengths[i] for i in rows) / max_token_len))
            partitions = get_seqlen_balanced_partitions([lengths[i] for i in rows], count, equal_size=False)
            for partition in partitions:
                members = tuple(rows[j] for j in partition)
                units.append(LossUnit(members, len(members) / len(data), sum(lengths[i] for i in members)))
        for rank, slots in enumerate(plan_optimizer_step(units, physical_world_size)):
            pieces = []
            for slot in slots:
                start = len(indices[rank])
                indices[rank].extend(units[slot.unit_index].rows)
                pieces.append((start, len(indices[rank]), slot.backward_scale, slot.padding))
            descriptors[rank].append(pieces)
    packets = []
    for rank in range(physical_world_size):
        packet = data.select_idxs(torch.tensor(indices[rank], dtype=torch.long))
        packet.meta_info = copy.deepcopy(data.meta_info)
        packet.meta_info.update(elastic_loss_steps=descriptors[rank],
                                elastic_physical_world_size=physical_world_size,
                                joint_rs_original_rows=len(data))
        packets.append(packet)
    return packets


def planned_microbatches(batch, descriptors):
    result=[]
    end=0
    for start,stop,scale,padding in descriptors:
        if start<0 or stop<=start or stop>len(batch) or scale<0:
            raise ValueError('Invalid scheduled microbatch')
        if padding and scale!=0:
            raise ValueError('Padding must have zero backward weight')
        result.append((batch[start:stop],scale,padding));end=stop
    return result

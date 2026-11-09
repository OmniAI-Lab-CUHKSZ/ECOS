"""Execute actual Actor/R-S loss methods on CPU FSDP at4 vs6 physical ranks.

Only model forward and device-specific microbatch MAX collective are CPU
adapters; PPO, entropy, KL, selective Teacher route and accumulation code are
loaded verbatim from dp_actor.py. This is not a Qwen/NCCL E2E test.
"""
import ast
import datetime
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import OmegaConf
from verl.protocol import DataProto,pad_dataproto_to_divisor
from verl.trainer.ppo.core_algos import agg_loss,compute_policy_loss,kl_penalty
from verl.trainer.ppo.selective_gigpo import selective_route_loss_terms
from verl.trainer.ppo.elastic_dispatch import pack_loss_packets
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches as actual_splitter


def rearrange_micro_batches(batch,max_token_len):
    n=min(len(batch),(int(batch['attention_mask'].sum())+max_token_len-1)//max_token_len)
    n=torch.tensor(n);dist.all_reduce(n,op=dist.ReduceOp.MAX)
    return actual_splitter(batch,max_token_len,same_micro_num_in_dp=False,min_num_micro_batch=int(n))


def load_actor_methods(device):
    path=Path(__file__).resolve().parents[3]/'verl/workers/actor/dp_actor.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataParallelPPOActor')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['update_policy','update_reflect_policy','_optimizer_step']]
    for method in methods:method.decorator_list=[]
    ns=dict(globals());ns['get_torch_device']=lambda:types.SimpleNamespace(current_device=lambda:device)
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(path),'exec'),ns)
    return ns


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Parameter(torch.tensor([.2,-.1,.4],dtype=torch.float32))
        self.skill=torch.nn.Parameter(torch.tensor([.01,.02,-.03],dtype=torch.float32))
    def forward(self,ids,skill):
        x=ids.to(torch.float32)
        features=torch.stack((torch.sin(x),torch.cos(x),x.remainder(3)/3),dim=-1)
        w=self.base+self.skill if skill else self.base
        z=features@w
        lp=torch.nn.functional.logsigmoid(z)
        p=lp.exp();entropy=-p*lp-(1-p)*torch.nn.functional.logsigmoid(-z)
        return entropy,lp


def fixture(rows,reflection=False):
    ids=torch.arange(rows*17).reshape(rows,17)%97
    mask=torch.arange(17)[None,:]<(torch.arange(rows)%16+1)[:,None]
    advantage=torch.sin(ids.to(torch.float32)*.3)
    tensors=dict(input_ids=ids,responses=ids,attention_mask=mask.long(),position_ids=ids,
        old_log_probs=torch.full((rows,17),-.7,dtype=torch.float32),advantages=advantage,
        ref_log_prob=torch.full((rows,17),-.8,dtype=torch.float32),
        teacher_log_probs=torch.full((rows,17),-.6,dtype=torch.float32),
        route_gate=advantage*.2,route_token_mask=mask,selective_route_mask=mask)
    result=DataProto.from_dict(tensors=tensors)
    result.meta_info=dict(temperature=1.,reflect_loss_coef=.01,reflect_metric_prefix='rs',
        reflect_loss_agg_mode='seq-mean-token-mean',rs_response_only_logits=True,route_loss_coef=.01)
    if reflection:
        result,n=pad_dataproto_to_divisor(result,4)
        if n:result.batch['advantages'][-n:]=0
    return result


def worker(rank,world,store,result_path):
    torch.set_num_threads(1)
    gpu=os.environ.get('ELASTIC_GPU_TEST')=='1' and world!=4
    device=torch.device('cuda',rank) if gpu else torch.device('cpu')
    if gpu:torch.cuda.set_device(device)
    dist.init_process_group('nccl' if gpu else 'gloo',init_method=Path(store).as_uri(),rank=rank,world_size=world,
                            timeout=datetime.timedelta(seconds=90))
    try:
        raw=Toy();model=FSDP(raw,device_id=device,use_orig_params=True)
        actor=types.SimpleNamespace(actor_module=model,ulysses_sequence_parallel_size=1)
        actor.config=OmegaConf.create(dict(use_dynamic_bsz=True,use_kl_loss=True,kl_loss_type='low_var_kl',
            kl_loss_coef=.01,entropy_coeff=.001,grad_clip=1.,ppo_mini_batch_size=64,ppo_micro_batch_size_per_gpu=32,
            ppo_max_token_len_per_gpu=128,ppo_epochs=1,clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2,
            clip_ratio_c=3.,loss_agg_mode='token-mean',policy_loss=dict(loss_mode='vanilla')))
        actor.actor_optimizer=torch.optim.AdamW([raw.base],lr=1e-3)
        actor.reflect_optimizer=torch.optim.AdamW([raw.skill],lr=1e-3)
        steps={'actor':0,'rs':0}
        methods=load_actor_methods(device)
        def optimizer_step(optimizer=None):
            optimizer=optimizer or actor.actor_optimizer
            norm=methods['_optimizer_step'](actor,optimizer)
            steps['rs' if optimizer is actor.reflect_optimizer else 'actor']+=1
            return norm
        actor._optimizer_step=optimizer_step
        for reflection,nrows in [(True,13),(False,640),(True,1),(False,128)]:
            data=fixture(nrows,reflection)
            if world==4:packet=data.chunk(4)[rank]
            else:packet=pack_loss_packets(data,logical_world_size=4,physical_world_size=world,
                global_minibatch_size=256,max_token_len=128,reflection=reflection)[rank]
            def forward(micro_batch,temperature,calculate_entropy,response_only_logits):
                return model(micro_batch['input_ids'],reflection)
            actor._forward_micro_batch=forward
            methods['update_reflect_policy' if reflection else 'update_policy'](actor,packet)
        with FSDP.summon_full_params(model,writeback=False):
            output=dict(base=raw.base.tolist(),skill=raw.skill.tolist(),steps=steps)
        if rank==0:Path(result_path).write_text(json.dumps(output))
    finally:dist.destroy_process_group()


class ActualActorExecutionTest(unittest.TestCase):
    @unittest.skipUnless(os.name=='posix','CPU FSDP collective test requires Linux')
    def test_four_to_six_actual_loss_updates(self):
        with tempfile.TemporaryDirectory(prefix='bepo_actual_actor_') as d:
            results=[]
            target=int(os.environ.get('ELASTIC_TEST_RANKS','6'))
            for world in (4,target):
                p=Path(d)/str(world);p.mkdir()
                mp.spawn(worker,args=(world,str(p/'store'),str(p/'out.json')),nprocs=world,join=True)
                results.append(json.loads((p/'out.json').read_text()))
            for key in ('base','skill'):
                torch.testing.assert_close(torch.tensor(results[0][key]),torch.tensor(results[1][key]),atol=1e-6,rtol=1e-6)
            self.assertEqual(results[0]['steps'],results[1]['steps'])
            print('ACTUAL_ACTOR_COMPARISON',json.dumps(dict(target_ranks=target,gpu=os.environ.get('ELASTIC_GPU_TEST')=='1',results=results)))

if __name__=='__main__':unittest.main(verbosity=2)

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from response_rs.config import PROFILES, load_config
from response_rs.response_loss import ResponseLoss
from response_rs.sft_data import load_groups, slots

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('actual_selective',ROOT/'verl/trainer/ppo/selective_gigpo.py')
actual=importlib.util.module_from_spec(spec)
spec.loader.exec_module(actual)


@pytest.mark.parametrize('profile',PROFILES)
def test_all_four_profiles_share_current_method(profile):
    c=load_config(profile)
    s=c['algorithm']['selective_gigpo']
    assert s['skip_teacher_on_all_success'] is True
    assert s['actor_signal_scope']=='action_command'
    assert s['modulation_mode']=='symmetric_alignment'
    assert s['modulation_lambda']==1. and s['modulation_tau']==.5
    assert s['rs_reward_clip']==3.
    assert c['algorithm']['route']['lambda_route']==.02
    assert c['algorithm']['grsd']['reflect_loss_coef']==.1
    assert c['actor_rollout_ref']['actor']['rs_joint_update']
    assert not c['actor_rollout_ref']['model']['reflect_lora']['enabled']
    assert not c['algorithm']['grsd']['judge_enabled']
    assert c['env']['rollout']['n']==8
    assert c['data']['train_batch_size']==16
    assert c['actor_rollout_ref']['actor']['elastic_logical_world_size']==8
    assert c['actor_rollout_ref']['actor']['ppo_mini_batch_size']==256
    assert c['actor_rollout_ref']['actor']['optim']['lr']==1e-6
    assert c['actor_rollout_ref']['actor']['entropy_coeff']==.001
    assert c['actor_rollout_ref']['actor']['kl_loss_coef']==.01


@pytest.mark.parametrize('profile',PROFILES)
def test_real_hydra_composition_resolves_all_four_profiles(profile,monkeypatch):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    monkeypatch.setenv('SFT_MODEL','/models/sft600')
    monkeypatch.setenv('RL_TRAIN_DATA','/data/train.parquet')
    monkeypatch.setenv('RL_VAL_DATA','/data/test.parquet')
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs')):
        config=compose(config_name=profile);OmegaConf.resolve(config)
        assert config.actor_rollout_ref.model.path=='/models/sft600'
        assert config.algorithm.selective_gigpo.skip_teacher_on_all_success
        assert config.algorithm.selective_gigpo.actor_signal_scope=='action_command'
        assert config.algorithm.route.lambda_route==.02
        assert config.algorithm.grsd.reflect_loss_coef==.1


def rows(successes):
    return [dict(group='g',trajectory=str(i),turn=0,outcome=float(i<successes),
                 anchor='s',action='look',reward=float(i<successes),valid=True,
                 admissible=['look'],done=True,episode_length=1) for i in range(8)]


@pytest.mark.parametrize('successes',[0,7,8])
def test_success_mask_ignores_padding_and_real_revisits(successes):
    evidence=rows(successes)
    evidence += [dict(evidence[0]),dict(evidence[1],turn=1)]
    skip, counts=actual.all_success_group_rows(evidence,8)
    assert skip.tolist()==[successes==8]*10
    assert counts['all_success']==int(successes==8)
    weights=torch.tensor([[2.,-1.,.2]]*10,requires_grad=False)
    selected=torch.ones_like(weights,dtype=torch.bool)
    after, chosen=actual.mask_all_success_teacher(weights,selected,skip)
    if successes==8:
        assert not after.any() and not chosen.any()
    else:
        torch.testing.assert_close(after,weights)
        assert chosen.all()
    torch.testing.assert_close(weights,torch.tensor([[2.,-1.,.2]]*10))


def test_action_only_teacher_and_original_response_denominator():
    old=torch.full((16,5),-10.,dtype=torch.float64)
    delta=torch.tensor([[2.,-1.,-.3,.1,0.]]*16,dtype=torch.float64)
    response=torch.tensor([[1,1,1,1,0]]*16,dtype=torch.bool)
    action=torch.tensor([[0,0,1,1,0]]*16,dtype=torch.bool)
    adv=np.full(16,.5)
    weights, selected, raw_gap=actual.token_route(old+delta,old,action,response,adv,[True]*16,
        modulation_lambda=1.,modulation_tau=.5,actor_signal_scope='action_command',
        modulation_mode='symmetric_alignment')
    # The signed command gap is shared only by its command tokens.
    assert not weights[~action].any() and not selected[~action].any()
    assert (weights[:,2:4]<0).all()
    torch.testing.assert_close(weights[:,2],weights[:,3])
    before=raw_gap.clone()
    skip=np.array([True]*8+[False]*8)
    weights, selected=actual.mask_all_success_teacher(weights,selected,skip)
    current=old.clone().requires_grad_()
    loss=.02*(actual.selective_route_loss_terms(current,old,old+delta,weights,selected,.2,.2)*response).sum()/response.sum()
    loss.backward()
    torch.testing.assert_close(current.grad,-.02*weights/64)
    assert not current.grad[:8].any()
    assert not current.grad[~action].any()
    assert current.grad[8:,2:4].count_nonzero()==16
    torch.testing.assert_close(raw_gap,before)


@pytest.mark.parametrize('world',range(2,9))
def test_sft_task_weights_and_padding_are_world_invariant(world):
    samples=[dict(kind='reflection',id=i) for i in range(8)]+[dict(kind='skill',id=8)]
    mass=torch.zeros(9,dtype=torch.float64)
    lengths=[]
    for rank in range(world):
        rank_slots=list(slots(samples,rank,world));lengths.append(len(rank_slots))
        for row,weight in rank_slots:mass[row['id']]+=weight
    assert len(set(lengths))==1
    torch.testing.assert_close(mass,torch.tensor([.5/8]*8+[.5],dtype=torch.float64))


class CausalBody(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(11,7)
    def forward(self,input_ids,**kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.emb(input_ids).cumsum(1))


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=CausalBody();self.lm_head=torch.nn.Linear(7,11,bias=False)


def test_real_response_loss_matches_masked_causal_cross_entropy():
    torch.manual_seed(91)
    model=TinyCausal().double()
    prompt=[1,2,3];response=[4,5]
    loss=ResponseLoss(model)(prompt,response)
    ids=torch.tensor([prompt+response])
    logits=model.lm_head(model.model(ids).last_hidden_state).float()
    expected=F.cross_entropy(logits[:,len(prompt)-1:-1].reshape(-1,11),torch.tensor(response))
    torch.testing.assert_close(loss,expected)
    actual_grads=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    expected_grads=torch.autograd.grad(expected,tuple(model.parameters()))
    for a,b in zip(actual_grads,expected_grads):torch.testing.assert_close(a,b)


@pytest.mark.parametrize('profile',PROFILES)
def test_packaged_material_has_all_1800_targets(profile):
    groups, info=load_groups(ROOT/'data/sft'/profile)
    assert len(groups)==200
    assert sum(len(g['reflections']) for g in groups)==1600
    assert sum(len(g['skills']) for g in groups)==200
    assert max(len(r['prompt_ids'])+len(r['response_ids']) for g in groups for r in g['reflections']+g['skills'])==info['max_observed_sequence_tokens']

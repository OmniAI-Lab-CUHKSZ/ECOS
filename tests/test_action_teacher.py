import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from response_rs.teacher_signals import verify_action_teacher_signals

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('action_selective',ROOT/'verl/trainer/ppo/selective_gigpo.py')
actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)


@pytest.mark.parametrize('gap',[-.4,.4])
@pytest.mark.parametrize('advantage',[-.5,0.,.5])
def test_command_modulation_keeps_signed_gap_and_has_no_think_gradient(gap,advantage):
    delta=torch.tensor([[100.,gap*.75,gap*.25,-999.]],dtype=torch.float64)
    action=torch.tensor([[0,1,1,0]],dtype=torch.bool)
    response=torch.tensor([[1,1,1,0]],dtype=torch.bool)
    weights,selected,raw=actual.token_route(delta,torch.zeros_like(delta),action,response,[advantage],[True],
        1.,.5,actor_signal_scope='action_command',modulation_mode='symmetric_alignment')
    coefficient=gap*(1+np.sign(gap)*np.tanh(advantage/.5))
    torch.testing.assert_close(weights,torch.tensor([[0.,coefficient,coefficient,0.]],dtype=torch.float64))
    assert not selected[~action].any()
    changed=delta.clone();changed[:,0]*=-500
    after=actual.token_route(changed,torch.zeros_like(delta),action,response,[advantage],[True],
        1.,.5,actor_signal_scope='action_command',modulation_mode='symmetric_alignment')
    torch.testing.assert_close(weights,after[0]);torch.testing.assert_close(raw,after[2])


def test_real_action_parser_excludes_think_tags_whitespace_and_incomplete_commands():
    from verl.trainer.ppo.route_loss import compute_action_mask
    class Tokenizer:
        pieces=['<think>','reason','</think>','<action>','go',' to',' desk','</action>',' ','padding']
        def decode(self,ids,**kwargs):return ''.join(self.pieces[i] for i in ids)
    response=torch.tensor([[0,1,2,3,4,5,6,7,8,9]])
    mask=torch.tensor([[1]*9+[0]],dtype=torch.bool)
    action=compute_action_mask(Tokenizer(),response,mask)
    assert action.tolist()==[[False]*4+[True]*3+[False]*3]
    incomplete_mask=torch.tensor([[1]*7+[0]*3],dtype=torch.bool)
    assert not compute_action_mask(Tokenizer(),response,incomplete_mask).any()


def payload():
    rows=[dict(group=str(i//8),trajectory=str(i%8),turn=0,outcome=float(i!=15),
               anchor='state',action='look',reward=float(i!=15),valid=True) for i in range(16)]
    delta=torch.tensor([[50.,-.3,.1,200.]]*16,dtype=torch.float64)
    response=torch.tensor([[1,1,1,0]]*16,dtype=torch.bool)
    action=torch.tensor([[0,1,1,0]]*16,dtype=torch.bool)
    advantage=np.linspace(-.5,.5,16)
    weights,selected,gap=actual.token_route(delta,torch.zeros_like(delta),action,response,advantage,[True]*16,
        1.,.5,actor_signal_scope='action_command',modulation_mode='symmetric_alignment')
    skipped,_=actual.all_success_group_rows(rows,8)
    weights,selected=actual.mask_all_success_teacher(weights,selected,skipped)
    return dict(actor_signal_scope=np.array('action_command'),modulation_mode=np.array('symmetric_alignment'),
        modulation_lambda=np.array(1.),modulation_tau=np.array(.5),route_loss_coef=np.array(.02),
        skip_teacher_on_all_success=np.array(True),teacher_skipped_rows=skipped,
        rows_json=np.array(json.dumps(rows)),raw_teacher_token_gap=delta.numpy(),raw_command_gap=gap.numpy(),
        raw_actor_gap=(gap[:,None]*action).numpy(),actor_token_weights=weights.numpy(),
        actor_selected_mask=selected.numpy(),actor_eligible_mask=action.numpy(),
        actor_response_mask=response.numpy(),action_token_mask=action.numpy(),advantage=advantage)


def test_observer_accepts_actual_action_route_and_8of8_skip():
    result=verify_action_teacher_signals(payload())
    assert result['nonaction_nonzero_tokens']==0 and result['action_nonzero_tokens']==16
    assert result['teacher_group_counts']==dict(all_success=1,all_fail=0,mixed=1)


@pytest.mark.parametrize('change',['think_signal','skipped_group_signal','old_scope','old_coefficient'])
def test_observer_rejects_wrong_scope_coefficient_or_mask(change):
    signal=copy.deepcopy(payload())
    if change=='think_signal':signal['actor_token_weights'][8,0]=1.
    if change=='skipped_group_signal':signal['actor_token_weights'][0,1]=1.
    if change=='old_scope':signal['actor_signal_scope']=np.array('response_token')
    if change=='old_coefficient':signal['route_loss_coef']=np.array(.01)
    with pytest.raises(AssertionError):verify_action_teacher_signals(signal)


def test_actual_trainer_saves_action_arrays_used_by_native_observer(tmp_path):
    signal=payload();path=ROOT/'verl/trainer/ppo/grsd_reflect_ray_trainer.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='savez_compressed' and any(k.arg=='raw_command_gap' for k in n.keywords)]
    assert len(calls)==1
    tensors=lambda k:torch.as_tensor(signal[k])
    self=SimpleNamespace(global_steps=1,selective_actor_signal_scope='action_command',
        selective_modulation_mode='symmetric_alignment',selective_lambda=1.,selective_max_factor=16.,
        selective_tau=.5,route_lambda=.02,skip_teacher_on_all_success=True)
    namespace=dict(np=np,json=json,os=__import__('os'),self=self,diagnostic_dir=str(tmp_path),
        selective_data_rows=json.loads(str(signal['rows_json'])),observation_embeddings=np.zeros((16,2)),
        batch=SimpleNamespace(batch={'old_log_probs':torch.zeros(16,4,dtype=torch.float64)},
                              non_tensor_batch={'anchor_obs':['state']*16}),
        selective_adv=signal['advantage'],raw_gap=signal['raw_command_gap'],
        skipped_teacher_rows=signal['teacher_skipped_rows'],eligible_mask=tensors('actor_eligible_mask'),
        action_mask=tensors('action_token_mask'),route_token_mask=tensors('actor_response_mask'),
        teacher_log_probs=tensors('raw_teacher_token_gap'),raw_actor_gap=tensors('raw_command_gap')[:,None],
        actor_eligible_mask=tensors('actor_eligible_mask'),route_gate=tensors('actor_token_weights'),
        selected_mask=tensors('actor_selected_mask'),response_scope=False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])],type_ignores=[])),str(path),'exec'),namespace)
    with np.load(tmp_path/'step1_natural_signals.npz',allow_pickle=False) as saved:
        result=verify_action_teacher_signals(saved)
    assert result['actual_scope']=='action_command' and result['route_loss_coef']==.02

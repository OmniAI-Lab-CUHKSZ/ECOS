import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('selective_under_test',ROOT/'verl/trainer/ppo/selective_gigpo.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def fixture(outcomes=(1,1,1,1,0,0,0,0)):
    n=len(outcomes)
    meta=dict(uid=['g']*n,traj_uid=[str(i) for i in range(n)],turn_step=[0]*n,
              anchor_obs=['s']*n,executed_action=['good' if v else 'bad' for v in outcomes],
              rewards=list(outcomes),episode_rewards=list(outcomes),is_action_valid=[True]*n)
    return s.prepare_rows(meta)


def test_step_only_formula_and_stability():
    adv,valid=s.stable_advantages(fixture())
    np.testing.assert_allclose(adv[:4],1/(1+1e-6/5))
    np.testing.assert_array_equal(adv,s.rollout_advantages(fixture()))
    assert valid.all()


def test_actor_and_skill_ignore_episode_term_when_step_returns_are_unchanged():
    rows=fixture()
    # Isolate the episode addend: hold the full step-return evidence fixed.
    changed=[dict(r,outcome=10.-r['outcome']) for r in rows]
    np.testing.assert_array_equal(s.rollout_advantages(rows),s.rollout_advantages(changed))
    before,ok=s.stable_advantages(rows)
    after,ok2=s.stable_advantages(changed)
    np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(ok,ok2)
    gap=np.sign(before)
    c,n=s.skill_credit(rows,['g'],gap,np.ones(len(rows),bool))
    c2,n2=s.skill_credit(changed,['g'],gap,np.ones(len(rows),bool))
    np.testing.assert_array_equal(c,c2)
    np.testing.assert_array_equal(n,n2)
    assert c[0]>0 and n[0]==8


def test_step_zero_ignores_nonzero_episode_and_keeps_teacher_gap():
    rows=fixture()
    # Each anchor has only one visit: episode differs, local evidence is zero.
    rows=[dict(r,anchor=f's{i}') for i,r in enumerate(rows)]
    adv=s.rollout_advantages(rows)
    np.testing.assert_array_equal(adv,np.zeros(8))
    old=torch.zeros(8,1);teacher=torch.tensor([[2.],[-2.]]*4)
    mask=torch.ones_like(old,dtype=torch.bool)
    weights,selected,_=s.token_route(teacher,old,mask,mask,adv,[True]*8)
    torch.testing.assert_close(weights,teacher)
    assert selected.all()
    credit,counts=s.skill_credit(rows,['g'],np.ones(8),np.ones(8,bool))
    np.testing.assert_array_equal(credit,[0.]);np.testing.assert_array_equal(counts,[8])


def test_same_trajectory_visits_remain_step_evidence_but_padding_does_not():
    rows=fixture((0,0))
    rows[0]['rtg']=-.1;rows[1]['rtg']=-.4
    rows.append(dict(rows[0],turn=1,rtg=-.2))
    expected=s._standardize([-.1,-.4,-.2])
    np.testing.assert_allclose(s.rollout_advantages(rows),expected)
    np.testing.assert_array_equal(s.rollout_advantages(rows+[dict(rows[2])]),expected[[0,1,2,2]])


def test_teacher_veto_follows_step_when_composite_sign_disagrees():
    rows=fixture()
    for i,r in enumerate(rows):
        r['anchor']='success_anchor' if i<4 else 'failure_anchor'
        if i<4:r['rtg']=10.-i
    composite,step=s.advantages(rows)
    assert composite[2]>0 and step[2]<0
    adv=s.rollout_advantages(rows)
    old=torch.zeros(8,1);teacher=torch.ones(8,1)
    teacher[2,0]=-1.
    mask=torch.ones_like(old,dtype=torch.bool)
    weight,selected,_=s.token_route(teacher,old,mask,mask,adv,[True]*8)
    assert selected[2,0] and weight[2,0]==-1.


def test_all_failed_step_still_contains_invalid_action_return_signal():
    meta=dict(uid=['g']*4,traj_uid=[str(i) for i in range(4)],turn_step=[0]*4,
              anchor_obs=['s']*4,executed_action=['look','look','invalid','invalid'],
              rewards=[0.]*4,episode_rewards=[0.]*4,is_action_valid=[True,True,False,False])
    rows=s.prepare_rows(meta)
    expected=s._standardize([0.,0.,-.1,-.1])
    np.testing.assert_allclose(s.rollout_advantages(rows),expected)
    assert expected[0]>0 and expected[2]<0


def test_padding_copy_does_not_create_extra_evidence():
    rows=fixture();adv,ok=s.stable_advantages(rows)
    duplicate=rows+[dict(rows[0])]*5
    a2,o2=s.stable_advantages(duplicate)
    np.testing.assert_array_equal(a2[:8],adv)
    np.testing.assert_array_equal(o2[:8],ok)
    c1,n1=s.skill_credit(rows,['g'],adv,np.ones(8,bool))
    c2,n2=s.skill_credit(duplicate,['g'],a2,np.ones(13,bool))
    np.testing.assert_array_equal(c1,c2);np.testing.assert_array_equal(n1,n2)


def test_conflicting_duplicate_is_rejected():
    rows=fixture();rows.append(dict(rows[0],action='different'))
    with pytest.raises(ValueError,match='Conflicting duplicate'):s.stable_advantages(rows)


def test_target_is_retained_not_crossfit():
    rows=fixture((1,0,0,0));_,valid=s.stable_advantages(rows)
    assert valid[0] and not valid[1:].any()


def test_same_batch_skill_feedback_four_quadrants():
    rows=fixture();adv,ok=s.stable_advantages(rows)
    positive,n=s.skill_credit(rows,['g'],np.sign(adv),ok)
    negative,_=s.skill_credit(rows,['g'],-np.sign(adv),ok)
    assert positive[0]>0 and negative[0]<0 and n[0]==8


def test_zero_advantage_skill_is_zero_not_penalized():
    rows=fixture((0,0,0,0));c,n=s.skill_credit(rows,['g'],np.ones(4),np.ones(4,bool))
    assert c[0]==0 and n[0]==4


def test_same_sign_command_gap_and_zero_fallback():
    old=torch.tensor([[-3.,-2.,-1.],[-2.,-2.,-2.]])
    teacher=old+torch.tensor([[1.,-2.,.3],[.1,.2,.3]])
    action=torch.tensor([[0,1,0],[0,1,0]],dtype=torch.bool)
    rm=torch.ones_like(action)
    w,sel,gap=s.token_route(teacher,old,action,rm,[-7.,-1.],[True,True])
    assert w[0,1]==-2 and sel[1,1] and 0.<w[1,1]<.2
    assert not w[~sel].any()
    assert not w.requires_grad


def test_no_action_span_abstains():
    old=torch.zeros(1,3);teacher=old+1
    w,sel,_=s.token_route(teacher,old,old.bool(),torch.ones_like(old),[1],[True])
    assert not sel.any();assert not w.any()


def test_all_quadrants_zeros_and_invalid_rows():
    old=torch.zeros(8,1);gap=torch.tensor([[2.],[-2.],[2.],[-2.],[2.],[0.],[2.],[2.]])
    action=torch.ones_like(old,dtype=torch.bool)
    w,sel,_=s.token_route(old+gap,old,action,action,[3,3,-3,-3,0,3,3,3],
                         [True,True,True,True,True,True,False,True])
    k=.75*np.tanh(3.)
    torch.testing.assert_close(w[:,0],torch.tensor([2*(1+k),-2*(1-k),2*(1-k),-2.,2.,0.,0.,2*(1+k)],dtype=w.dtype))
    assert sel[:,0].tolist()==[True,True,True,True,True,False,False,True]


def test_command_sum_not_individual_token_agreement_and_padding():
    old=torch.zeros(1,4);tea=torch.tensor([[3.,-1.,100.,100.]])
    action=torch.ones_like(old,dtype=torch.bool);response=torch.tensor([[1,1,0,0]])
    w,sel,gap=s.token_route(tea,old,action,response,[9],[True])
    torch.testing.assert_close(gap,torch.tensor([2.]))
    torch.testing.assert_close(w,torch.tensor([[2*(1+.75*np.tanh(9.))]*2+[0.,0.]],dtype=w.dtype))
    w2,_,_=s.token_route(tea,old,action,response,[.01],[True])
    assert 2.<w2[0,0]<w[0,0]


def test_unfiltered_advantage_deduplicates_without_stability_filter():
    rows=fixture((1,0));adv=s.rollout_advantages(rows)
    _,stable=s.stable_advantages(rows)
    assert not stable.any() and np.all(adv!=0)
    np.testing.assert_array_equal(s.rollout_advantages(rows+[dict(rows[0])]*3),adv[[0,1,0,0,0]])
    mask=torch.ones(2,1,dtype=torch.bool)
    weights,selected,_=s.token_route(torch.as_tensor(np.sign(adv))[:,None],torch.zeros(2,1),mask,mask,adv,[True,True])
    assert selected.all()


def test_trainer_step_soft_modulation_and_raw_gap_credit():
    source=(ROOT/'verl/trainer/ppo/grsd_reflect_ray_trainer.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='rollout_advantages']
    assert len(calls)==1 and calls[0].args[0].id=='selective_data_rows'
    assert any(k.arg=='estimator' and isinstance(k.value,ast.Attribute)
               and k.value.attr=='selective_estimator' for k in calls[0].keywords)
    assert 'stable_advantages(' not in source
    assert "'selective/zero_advantage_uses_teacher_gap':1." in source
    assert "'selective/conflict_weakened':1." in source
    assert "'selective/step_only_advantage':float(self.selective_estimator=='gigpo_step')" in source
    assert "advantage_source=self.selective_estimator" in source
    assert "skill_advantage_source=self.selective_estimator" in source
    assert 'raw_gap=command_gap.detach().cpu().numpy()' in source
    assert 'selective_no_skill_lp' not in source
    assert 'centered_credit(selective_data_rows,selective_adv,raw_gap,' in source
    assert "state_keys,group_keys=reflect_ctx['prior_uids']" in source


def test_selected_gradient_directions_and_teacher_detachment():
    old=torch.zeros(1,2);current=old.clone().requires_grad_();teacher=torch.ones(1,2,requires_grad=True)
    weights=torch.tensor([[2.,-3.]],requires_grad=True)
    loss=s.selective_route_loss_terms(current,old,teacher,weights,torch.ones(1,2,dtype=torch.bool),.2,.2).sum()
    loss.backward()
    torch.testing.assert_close(current.grad,torch.tensor([[-2.,3.]]))
    assert teacher.grad is None and weights.grad is None


def test_negative_update_is_clipped():
    old=torch.zeros(1,1);current=torch.tensor([[-1.]],requires_grad=True)
    loss=s.selective_route_loss_terms(current,old,old,torch.tensor([[-2.]]),torch.ones(1,1,dtype=torch.bool),.2,.2).sum()
    loss.backward();assert current.grad.item()==0


def test_unselected_loss_and_gradient_are_zero_even_with_stale_weights():
    old=torch.zeros(1,2);new=torch.tensor([[.3,-.1]],requires_grad=True);teacher=torch.ones(1,2)
    w=torch.tensor([[.8,.2]])
    actual=s.selective_route_loss_terms(new,old,teacher,w,torch.zeros(1,2,dtype=torch.bool),.2,.2)
    assert not actual.any()
    actual.sum().backward();torch.testing.assert_close(new.grad,torch.zeros_like(new))


def test_no_reflection_skill_cross_validation_was_added():
    trainer=(ROOT/'verl/trainer/ppo/grsd_reflect_ray_trainer.py').read_text(encoding='utf-8')
    assert '_selective_crossfit_priors' not in trainer
    assert "reflect_ctx['prior_uids']" in trainer
    assert 'same_batch_feedback=True, crossfit=False' in trainer
    tree=ast.parse(trainer)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_reflect_and_build_priors')
    source=ast.get_source_segment(trainer,method)
    assert 'selective' not in source
    skill_method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_selective_rs_train_batch')
    assert 'reflect_out' in ast.get_source_segment(trainer,skill_method)


def test_public_profiles_use_shared_full_actor_joint_rs():
    from response_rs.config import PROFILES,load_config
    for profile in PROFILES:
        config=load_config(profile)
        assert config['algorithm']['route']['mode']=='selective_gigpo'
        assert config['actor_rollout_ref']['actor']['rs_joint_update']
        assert not config['actor_rollout_ref']['model']['reflect_lora']['enabled']


def test_nonfinite_and_unexpected_rewards():
    with pytest.raises(ValueError,match='nonfinite'):
        s.token_route(torch.tensor([[float('nan')]]),torch.zeros(1,1),torch.ones(1,1),torch.ones(1,1),[1],[1])
    rows=fixture()
    with pytest.raises(ValueError,match='nonfinite'):s.skill_credit(rows,['g'],[float('nan')]*8,[True]*8)


def test_two_optimizer_steps_agreement_direction_and_main_update_survives():
    # Log-probability coordinates isolate the auxiliary gradient from the main
    # objective. This is not a language model or an environment smoke.
    current=torch.nn.Parameter(torch.full((4,2),-5.))
    opt=torch.optim.SGD([current],lr=.01)
    action=torch.tensor([[1,0]]*4,dtype=torch.bool)
    response=torch.ones_like(action)
    advantage=s.rollout_advantages(fixture((1,0,1,0)))
    delta=torch.tensor([[2.,3.],[-2.,3.],[-2.,3.],[2.,3.]])
    for _ in range(2):
        old=current.detach().clone();teacher=old+delta
        weight,selected,_=s.token_route(teacher,old,action,response,advantage,[True]*4)
        terms=s.selective_route_loss_terms(current,old,teacher,weight,selected,.2,.2)
        gradient=torch.autograd.grad(terms.sum(),current,retain_graph=True)[0]
        k=.75*np.tanh(advantage[0])
        torch.testing.assert_close(gradient,torch.tensor([[-2*(1+k),0.],[2.,0.],[2*(1-k),0.],[-2*(1-k),0.]],dtype=gradient.dtype))
        main=-.1*current.sum()
        opt.zero_grad();(main+terms.sum()).backward();opt.step()
        assert current[0,0]>old[0,0] and current[1,0]<old[1,0]
        assert torch.all(current.detach()[~selected]>old[~selected])
        assert torch.isfinite(current).all()


def test_all_failed_zero_gigpo_retains_positive_and_negative_teacher_gap():
    adv=s.rollout_advantages(fixture((0,0,0,0)))
    np.testing.assert_array_equal(adv,np.zeros(4))
    old=torch.full((4,2),-5.)
    teacher=(old+torch.tensor([[2.,3.],[-2.,3.],[0.,3.],[1.,3.]])).requires_grad_()
    action=torch.tensor([[1,0]]*4,dtype=torch.bool)
    weights,selected,gap=s.token_route(teacher,old,action,torch.ones_like(action),adv,[True,True,True,False])
    torch.testing.assert_close(weights,torch.tensor([[2.,0.],[-2.,0.],[0.,0.],[0.,0.]]))
    assert selected[:,0].tolist()==[True,True,False,False]
    current=torch.nn.Parameter(old.clone());opt=torch.optim.SGD([current],lr=.01)
    for _ in range(2):
        opt.zero_grad()
        s.selective_route_loss_terms(current,old,teacher,weights,selected,.2,.2).sum().backward()
        opt.step()
    assert current[0,0]>old[0,0] and current[1,0]<old[1,0]
    torch.testing.assert_close(current.detach()[~selected],old[~selected])
    assert teacher.grad is None


@pytest.mark.parametrize('adv', [1e-50,-1e-50,1e-10,-1e-10])
def test_small_advantage_precision_is_preserved_before_tau_scaling(adv):
    old=torch.zeros(1,1);teacher=torch.full_like(old,-1. if adv>0 else 1.)
    mask=torch.ones_like(old,dtype=torch.bool)
    weights,selected,_=s.token_route(teacher,old,mask,mask,[adv],[True],modulation_tau=abs(adv))
    assert selected.all()
    torch.testing.assert_close(weights,teacher*(1.-.75*np.tanh(1.)))


@pytest.mark.parametrize('lam,tau', [(-.1,1.),(1.1,1.),(float('nan'),1.),(.5,0.),(.5,-1.),(.5,float('inf'))])
def test_bad_modulation_parameters_are_rejected(lam,tau):
    x=torch.zeros(1,1);mask=torch.ones_like(x,dtype=torch.bool)
    with pytest.raises(ValueError,match='modulation_'):
        s.token_route(x+1,x,mask,mask,[1.],[True],lam,tau)


@pytest.mark.parametrize('lam', [0.,.75,1.])
def test_soft_modulation_never_reverses_teacher_and_zero_lambda_is_identity(lam):
    adv=np.repeat([-1e4,-1.,0.,1.,1e4],3)
    gap=torch.tensor([-2.,0.,2.]*5)[:,None]
    mask=torch.ones_like(gap,dtype=torch.bool)
    w,_,raw=s.token_route(gap,torch.zeros_like(gap),mask,mask,adv,[True]*15,lam,1.)
    assert torch.all(w*gap>=0.)
    assert torch.all(w.abs()<=2*gap.abs())
    torch.testing.assert_close(raw,gap[:,0])
    if lam==0.:torch.testing.assert_close(w,gap)


def test_lambda_one_all_quadrants_zero_fallback_and_saturated_conflict():
    advantage=[1.,-1.,1.,-1.,0.,0.,-100.,100.]
    gap=torch.tensor([2.,2.,-2.,-2.,2.,-2.,2.,-2.])[:,None]
    mask=torch.ones_like(gap,dtype=torch.bool)
    weights,selected,raw=s.token_route(gap,torch.zeros_like(gap),mask,mask,
                                     advantage,[True]*8,1.,1.)
    k=float(np.tanh(1.))
    expected=torch.tensor([2*(1+k),2*(1-k),-2*(1-k),-2.,2.,-2.,0.,0.])[:,None]
    torch.testing.assert_close(weights,expected)
    torch.testing.assert_close(raw,gap[:,0])
    assert selected[:,0].tolist()==[True]*6+[False,False]
    current=torch.nn.Parameter(torch.zeros_like(gap))
    loss=s.selective_route_loss_terms(current,torch.zeros_like(gap),gap,
                                     weights,selected,.2,.2).sum()
    loss.backward()
    torch.testing.assert_close(current.grad,-expected)


def test_rs_unfiltered_credit_uses_raw_gap_and_equal_trajectory_means():
    rows=fixture((1,0))
    rows[0]['rtg']=3.;rows[1]['rtg']=-1.
    rows += [dict(rows[1],turn=1),dict(rows[1],turn=2)]
    # Only two trajectories: old stability support is absent, but new R/S works.
    assert not s.stable_advantages(rows)[1].any()
    adv=s.rollout_advantages(rows)
    gap=np.array([2.,1.,3.,5.])
    result=s.rs_credit(rows,[('g','1'),('g','0'),('g','missing')],['g','missing'],gap,[True]*4)
    utility=adv*gap
    np.testing.assert_allclose(result['reflection_credit'],[utility[1:].mean(),utility[0],0.])
    expected=(utility[0]+utility[1:].mean())/2
    np.testing.assert_allclose(result['skill_credit'],[expected,0.])
    assert not np.isclose(expected,utility.mean())
    assert result['reflection_credit'][0]<0 and result['reflection_credit'][1]>0
    padded=s.rs_credit(rows+[dict(rows[0])],[('g','0')],['g'],list(gap)+[gap[0]],[True]*5)
    np.testing.assert_allclose(padded['skill_credit'],[expected])
    assert padded['skill_turn_counts'].tolist()==[4]


def test_rs_keeps_zero_utilities_in_denominator_and_excludes_unscored_turns():
    rows=fixture((1,0));rows.append(dict(rows[0],turn=1,anchor='unique'))
    # The extra turn has A=0 but is still a scored turn of trajectory 0.
    result=s.rs_credit(rows,[('g','0'),('g','1')],['g'],[2.,-2.,3.],[True,False,True])
    utility=s.rollout_advantages(rows)[0]*2
    np.testing.assert_allclose(result['reflection_credit'],[utility/2,0.])
    np.testing.assert_allclose(result['skill_credit'],[utility/2])
    assert result['skill_trajectory_counts'].tolist()==[1]


def test_final_reward_hard_clip_preserves_sign_zeros_small_values_and_inputs():
    raw=np.array([-1e40,-3.,-.2,-1e-50,0.,1e-50,.2,3.,1e40])
    before=raw.copy()
    clipped=s.hard_clip_rs_reward(raw)
    np.testing.assert_array_equal(clipped,[-3.,-3.,-.2,-1e-50,0.,1e-50,.2,3.,3.])
    np.testing.assert_array_equal(np.sign(clipped),np.sign(raw))
    np.testing.assert_array_equal(raw,before)
    assert np.isfinite(clipped.astype(np.float32)).all()
    assert s.hard_clip_rs_reward([]).size==0
    np.testing.assert_array_equal(s.hard_clip_rs_reward([0.,0.]),[0.,0.])


@pytest.mark.parametrize('limit',[0.,-1.,float('nan'),float('inf'),-float('inf')])
def test_final_reward_rejects_invalid_clip_limit(limit):
    with pytest.raises(ValueError,match='rs_reward_clip'):
        s.hard_clip_rs_reward([1.],limit)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_final_reward_clip_cannot_hide_nonfinite_credit(value):
    with pytest.raises(ValueError,match='nonfinite R/S reward'):
        s.hard_clip_rs_reward([value])


def test_skill_uses_raw_reflection_means_then_both_rewards_are_clipped():
    rows=fixture((1,1,0,0));adv=s.rollout_advantages(rows)
    raw=np.array([10.,-4.,-4.,-4.])
    feedback=s.rs_credit(rows,[('g',str(i)) for i in range(4)],['g'],raw/adv,[True]*4)
    np.testing.assert_allclose(feedback['action_credit'],raw)
    np.testing.assert_allclose(feedback['reflection_credit'],raw)
    np.testing.assert_allclose(feedback['reflection_reward'],[3.,-3.,-3.,-3.])
    np.testing.assert_allclose(feedback['skill_credit'],[-.5])
    np.testing.assert_allclose(feedback['skill_reward'],[-.5])
    assert not np.isclose(feedback['skill_reward'][0],feedback['reflection_reward'].mean())
    changed=s.rs_credit(rows,[('g',str(i)) for i in range(4)],['g'],raw/adv,[True]*4,reward_clip=2.)
    np.testing.assert_array_equal(changed['reflection_credit'],feedback['reflection_credit'])
    np.testing.assert_allclose(changed['reflection_reward'],[2.,-2.,-2.,-2.])


def test_no_action_clipping_before_trajectory_mean_and_missing_reward_stays_zero():
    rows=fixture((1,0));rows.append(dict(rows[0],turn=1))
    adv=s.rollout_advantages(rows);credit=np.array([10.,-1.,-4.])
    feedback=s.rs_credit(rows,[('g','0'),('g','1'),('g','missing')],['g','missing'],credit/adv,[True]*3)
    np.testing.assert_allclose(feedback['action_credit'],credit)
    np.testing.assert_allclose(feedback['reflection_credit'],[3.,-1.,0.])
    np.testing.assert_allclose(feedback['reflection_reward'],[3.,-1.,0.])
    np.testing.assert_allclose(feedback['skill_reward'],[1.,0.])


def test_fixed_reward_cap_stops_common_scale_growth_and_handles_large_finite_values():
    rows=fixture((1,0));adv=s.rollout_advantages(rows)
    for scale in (1e4,1e40):
        feedback=s.rs_credit(rows,[('g','0'),('g','1')],['g'],scale*np.sign(adv),[True,True])
        assert feedback['skill_credit'][0]>3.
        np.testing.assert_array_equal(feedback['reflection_reward'],[3.,3.])
        np.testing.assert_array_equal(feedback['skill_reward'],[3.])


def test_rs_clip_default_config_and_actual_trainer_reward_entry():
    from omegaconf import OmegaConf
    config=OmegaConf.load(ROOT/'configs/rl_common.yaml')
    assert config.algorithm.selective_gigpo.rs_reward_clip==s.DEFAULT_RS_REWARD_CLIP==3.
    source=(ROOT/'verl/trainer/ppo/grsd_reflect_ray_trainer.py').read_text(encoding='utf-8')
    assert 'limit=self.selective_rs_reward_clip' in source
    assert "feedback['reflection_reward']" in source and "feedback['skill_reward']" in source
    from response_rs.config import PROFILES,load_config
    for profile in PROFILES:
        assert load_config(profile)['algorithm']['selective_gigpo']['rs_reward_clip']==3.

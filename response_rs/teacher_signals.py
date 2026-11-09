"""Check saved natural action-Teacher signals without running another rollout."""
import json
import math

import numpy as np


def verify_action_teacher_signals(signal):
    from verl.trainer.ppo.selective_gigpo import all_success_group_rows

    assert str(signal['actor_signal_scope'])=='action_command', 'Expected action-command Teacher scope'
    assert str(signal['modulation_mode'])=='symmetric_alignment'
    assert float(signal['modulation_lambda'])==1. and float(signal['modulation_tau'])==.5
    assert math.isclose(float(signal['route_loss_coef']),.01,rel_tol=0.,abs_tol=4*math.ulp(.01))
    skip_all_success=bool(signal['skip_teacher_on_all_success'])
    weights=np.asarray(signal['actor_token_weights'])
    selected=np.asarray(signal['actor_selected_mask'],dtype=bool)
    eligible=np.asarray(signal['actor_eligible_mask'],dtype=bool)
    response=np.asarray(signal['actor_response_mask'],dtype=bool)
    action=np.asarray(signal['action_token_mask'],dtype=bool)
    delta=np.asarray(signal['raw_teacher_token_gap'])
    gap=np.asarray(signal['raw_command_gap'])
    advantage=np.asarray(signal['advantage'],dtype=np.float64)
    assert weights.ndim==2 and weights.shape==selected.shape==eligible.shape==response.shape==action.shape==delta.shape
    assert gap.shape==advantage.shape==(weights.shape[0],)
    assert all(np.isfinite(x).all() for x in [weights,delta,gap,advantage])
    assert not np.any(eligible & ~(action & response))
    assert not np.any(weights[~action]) and not np.any(selected[~action]), 'Teacher signal outside action commands'
    assert not np.any(weights[~eligible]) and not np.any(selected[~eligible])
    assert not np.any(weights[~selected])
    np.testing.assert_allclose((delta*(action & response)).sum(-1),gap,rtol=2e-6,atol=2e-6)
    mean_gap=gap/np.maximum((action & response).sum(-1),1)
    factor=(1.+np.sign(mean_gap)*np.tanh(advantage/.5)).astype(delta.dtype)
    expected=delta*factor[:,None]
    rows=json.loads(str(signal['rows_json']))
    assert len(rows)==weights.shape[0]
    all_success,counts=all_success_group_rows(rows,8)
    skipped=all_success if skip_all_success else np.zeros(len(rows),dtype=bool)
    np.testing.assert_array_equal(skipped,signal['teacher_skipped_rows'])
    assert not np.any(weights[skipped]) and not np.any(selected[skipped])
    expected[~eligible]=0.;expected[skipped]=0.
    np.testing.assert_allclose(weights,expected,rtol=2e-6,atol=2e-6)
    expected_raw=delta*eligible
    np.testing.assert_allclose(signal['raw_actor_gap'],expected_raw,rtol=2e-6,atol=2e-6)
    return dict(actual_scope='action_command',command_gap_verified=True,
                same_turn_action_advantage_verified=True,route_loss_coef=.01,
                shared_command_mean_multiplier_verified=True,per_token_gap_verified=True,
                skip_teacher_on_all_success=skip_all_success,
                teacher_group_counts=counts,all_success_skipped_rows=int(skipped.sum()),
                nonaction_nonzero_tokens=0,action_nonzero_tokens=int(np.count_nonzero(weights*action)),
                action_gap_for_rs_unchanged=True,max_abs_weight_error=float(np.max(np.abs(weights-expected),initial=0.)))

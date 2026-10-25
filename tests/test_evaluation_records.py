import pytest

from response_rs.webshop_evaluation import OneEnvironment,validate_record


def test_partial_score_is_not_success_and_incomplete_result_is_rejected():
    row=dict(won=False,done=True,task_score=.75,prompt_ids=[1],response_ids=[2],
             observation='start',next_observation='finish',request_seed=0)
    record=dict(goal_index=0,seed=0,model='model',optimizer_steps=0,success=False,rows=[row])
    assert not validate_record(record,0,0,'model')['success']
    with pytest.raises(ValueError,match='final environment won'):
        validate_record(dict(record,success=True),0,0,'model')
    with pytest.raises(ValueError,match='Incomplete'):
        validate_record(dict(record,rows=[dict(row,done=False)]),0,0,'model')


def test_single_worker_adapter_preserves_actual_won_and_action():
    class Worker:
        def reset(self,goal):return f'goal {goal}',dict(won=False)
        def step(self,action):return 'done',10.,True,dict(won=True,executed_action=action)
        def close(self):self.closed=True
    worker=Worker();env=OneEnvironment(worker,17)
    assert env.reset()==(['goal 17'],[dict(won=False)])
    result=env.step(['click[Buy Now]'])
    assert result==(['done'],[10.],[True],[dict(won=True,executed_action='click[Buy Now]')])
    env.close();assert worker.closed

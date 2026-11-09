import pytest

from response_rs.webshop_evaluation import (OneEnvironment,validate_record,goal_panel,
                                          seed_summary,aggregate_summaries)


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


def test_alfworld_macro_is_not_task_weighted_micro():
    from response_rs.alfworld_evaluation import CATEGORIES,counts
    sizes=[35,13,27,16,25,24]
    records=[dict(category=category,success=category!=CATEGORIES[1])
             for category,size in zip(CATEGORIES,sizes) for _ in range(size)]
    result=counts(records)
    assert result['accuracy']==pytest.approx(5/6)
    assert result['micro_accuracy']==pytest.approx(127/140)
    assert result['categories'][CATEGORIES[1]]['successes']==0
    with pytest.raises(ValueError,match='Missing'):
        counts([r for r in records if r['category']!=CATEGORIES[0]])


def test_webshop_panels_are_seeded_without_replacement_and_method_independent():
    import random
    before=random.getstate()
    panels=[goal_panel(seed) for seed in [0,1,2]]
    assert all(len(panel)==len(set(panel))==128 for panel in panels)
    assert all(0<=goal<500 for panel in panels for goal in panel)
    assert panels[0]==goal_panel(0) and panels[0]!=panels[1]
    assert random.getstate()==before


def test_webshop_score_preserves_partial_credit_and_uses_sample_std():
    summaries=[]
    for seed,score in enumerate([.25,.5,.75]):
        records=[dict(success=False,rows=[dict(task_score=score)]),
                 dict(success=True,rows=[dict(task_score=1.)])]
        summaries.append(seed_summary(records,seed))
    result=aggregate_summaries(summaries)
    assert result['success']==dict(mean=50.,sample_std=0.)
    assert result['score']['mean']==pytest.approx(75.)
    assert result['score']['sample_std']==pytest.approx(12.5)
    assert summaries[0]['score']==62.5

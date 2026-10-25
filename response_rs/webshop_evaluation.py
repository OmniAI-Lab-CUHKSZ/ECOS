"""Frozen full500 WebShop evaluation using the training observation/action adapter."""
import json
from pathlib import Path
import statistics


class OneEnvironment:
    """Expose one original WebShop worker through the vector manager interface."""
    def __init__(self,worker,goal_index):
        self.worker=worker;self.goal_index=goal_index
    def reset(self):
        observation,info=self.worker.reset(self.goal_index)
        return [observation],[info]
    def step(self,actions):
        if len(actions)!=1:raise ValueError('Expected one action')
        observation,reward,done,info=self.worker.step(actions[0])
        return [observation],[reward],[done],[info]
    def close(self):self.worker.close()


def validate_record(record,goal,seed,model):
    if (record['goal_index'],record['seed'],record['model'])!=(goal,seed,model):
        raise ValueError('Saved trajectory belongs to a different evaluation')
    rows=record['rows']
    if not 1<=len(rows)<=30 or record['optimizer_steps']!=0:
        raise ValueError('Invalid evaluation length or optimizer updates')
    if type(record['success']) is not bool or record['success']!=rows[-1]['won']:
        raise ValueError('Success must equal final environment won')
    if not (rows[-1]['done'] or len(rows)==30):raise ValueError('Incomplete trajectory')
    for turn,row in enumerate(rows):
        if row['request_seed']!=seed*1_000_000+goal*30+turn:raise ValueError('Changed sampling seed')
        if len(row['prompt_ids'])>2048 or len(row['response_ids'])>512:raise ValueError('Changed token budget')
        if turn+1<len(rows) and (row['done'] or row['next_observation']!=rows[turn+1]['observation']):
            raise ValueError('Inconsistent trajectory transitions')
    return record


def evaluate(model,output,seeds):
    from omegaconf import OmegaConf
    from vllm import LLM,SamplingParams
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    from agent_system.environments.env_package.webshop.projection import webshop_projection
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from response_rs.alfworld_evaluation import read,write

    output=Path(output);model=str(Path(model).resolve())
    config=OmegaConf.create(dict(env=dict(history_length=5)))
    base=None;llm=None
    try:
        base=WebshopWorker(2026,dict(observation_mode='text',num_products=1000,human_goals=False,
                                    catalog_seed=2026,auto_reset=False))
        goals=base.get_goals()
        if len(goals)<500:raise ValueError('Expected the full500 WebShop test pool')
        write(output/'panel.json',dict(catalog_seed=2026,goals=[dict(goal_index=i,task=goals[i]['instruction_text']) for i in range(500)]))
        llm=LLM(model=model,tokenizer=model,dtype='bfloat16',tensor_parallel_size=1,
                 gpu_memory_utilization=.35,max_model_len=2560,max_num_seqs=24,max_num_batched_tokens=8192,
                 enforce_eager=True,enable_prefix_caching=False,seed=0,disable_log_stats=True)
        tok=llm.get_tokenizer();summaries=[]
        for seed in seeds:
            destination=output/f'seed{seed}';destination.mkdir(exist_ok=True)
            for start in range(0,500,24):
                episodes=[]
                try:
                    for goal in range(start,min(start+24,500)):
                        path=destination/f'goal{goal:03d}.json'
                        if path.exists():validate_record(read(path),goal,seed,model);continue
                        worker=WebshopWorker(2026,dict(server=base.env.server,session=goal,observation_mode='text',
                            num_products=1000,human_goals=False,catalog_seed=2026,auto_reset=False))
                        backend=OneEnvironment(worker,goal)
                        manager=WebshopEnvironmentManager(backend,webshop_projection,config)
                        observation,info=manager.reset({})
                        if manager.tasks[0]!=goals[goal]['instruction_text']:raise ValueError('Test goal changed')
                        episodes.append(dict(goal=goal,manager=manager,backend=backend,observation=observation,rows=[],done=False))
                    for turn in range(30):
                        active=[e for e in episodes if not e['done']]
                        if not active:break
                        prompts=[e['observation']['text'][0] for e in active]
                        encoded=[tok.apply_chat_template([dict(role='user',content=p)],tokenize=True,
                            add_generation_prompt=True,enable_thinking=False) for p in prompts]
                        ids=[p[-2048:] for p in encoded]
                        params=[SamplingParams(n=1,temperature=.4,top_p=1.,top_k=-1,max_tokens=512,
                            seed=seed*1_000_000+e['goal']*30+turn) for e in active]
                        outputs=llm.generate([dict(prompt_token_ids=p) for p in ids],params,use_tqdm=False)
                        for e,prompt,pi,po,result in zip(active,prompts,ids,encoded,outputs):
                            text=result.outputs[0].text
                            observation,reward,done,info=e['manager'].step([text])
                            final=info[0]
                            e['rows'].append(dict(turn=turn,observation=e['observation']['anchor'][0],
                                prompt=prompt,prompt_ids=pi,prompt_original_tokens=len(po),response=text,
                                response_ids=list(result.outputs[0].token_ids),action=final['executed_action'],
                                valid=bool(final['is_action_valid']),next_observation=observation['anchor'][0],
                                won=bool(final['won']),done=bool(done[0]),task_score=float(final['task_score']),
                                request_seed=seed*1_000_000+e['goal']*30+turn))
                            e.update(observation=observation,done=bool(done[0]))
                    for e in episodes:
                        record=dict(goal_index=e['goal'],seed=seed,model=model,optimizer_steps=0,
                                    success=e['rows'][-1]['won'],rows=e['rows'])
                        validate_record(record,e['goal'],seed,model)
                        write(destination/f'goal{e["goal"]:03d}.json',record)
                finally:
                    for e in episodes:e['backend'].close()
                print('BATCH_COMPLETE',seed,min(start+24,500),flush=True)
            records=[validate_record(read(destination/f'goal{g:03d}.json'),g,seed,model) for g in range(500)]
            won=sum(r['success'] for r in records)
            result=dict(seed=seed,successes=won,total=500,accuracy=won/500)
            write(destination/'summary.json',result);summaries.append(result)
            print('SEED_COMPLETE',json.dumps(result),flush=True)
        values=[r['accuracy'] for r in summaries]
        summary=dict(seeds=summaries,mean=statistics.mean(values),
                     sample_std=statistics.stdev(values) if len(values)>1 else None,optimizer_steps=0)
        write(output/'summary.json',summary)
        return summary
    finally:
        if base is not None:base.close()
        if llm is not None and hasattr(llm.llm_engine,'engine_core'):llm.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()

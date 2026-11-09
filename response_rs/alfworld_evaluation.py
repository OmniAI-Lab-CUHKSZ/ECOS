"""Frozen ALFWorld full274 evaluation and native Model export (no updates)."""
import json
import os
from pathlib import Path
import random
import sys
import time

SPLITS={'seen':('eval_in_distribution',140),'unseen':('eval_out_of_distribution',134)}
CATEGORIES=('pick_and_place_simple','look_at_obj_in_light','pick_clean_then_place_in_recep',
            'pick_heat_then_place_in_recep','pick_cool_then_place_in_recep','pick_two_obj_and_place')

def read(p):return json.loads(Path(p).read_text())

def write(p,r):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    with tmp.open('w') as f:json.dump(r,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(p)

def base(p,split):
    import yaml
    sys.path.insert(0,p['repo'])
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
    c=yaml.safe_load((Path(p['repo'])/'agent_system/environments/env_package/alfworld/configs/config_tw.yaml').read_text())
    c['dataset']['num_eval_games']=-1
    return get_environment(c['env']['type'])(c,train_eval=SPLITS[split][0])

def export(p):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    from response_rs.native_shards import SavedShardPickle,local_entries
    from safetensors import safe_open
    torch.set_num_threads(4)
    source=Path(p['checkpoint']);target=Path(p['model'])
    assert not target.exists() and (source/'CHECKPOINT_COMPLETE').read_text().strip()==str(p['source_step'])
    receipt=read(source/'STORAGE_ARCHIVE_VERIFIED.json');assert receipt['step']==p['source_step'] and receipt['all_bytes_equal'] and receipt['fsync']
    paths=[source/'actor'/f'model_world_size_{p["world_size"]}_rank_{r}.pt' for r in range(p['world_size'])]
    before=[(f.stat().st_size,f.stat().st_mtime_ns) for f in paths]
    shards=[torch.load(f,map_location='cpu',weights_only=False,mmap=True,pickle_module=SavedShardPickle) for f in paths]
    assert all(set(s)==set(shards[0]) for s in shards)
    c=AutoConfig.from_pretrained(source/'actor',local_files_only=True)
    with init_empty_weights():model=AutoModelForCausalLM.from_config(c)
    shapes={k:tuple(v.shape) for k,v in model.state_dict().items()};state={}
    for key in shards[0]:
        name=key.replace('_fsdp_wrapped_module.','');assert name in shapes
        entries={}
        for shard in shards:
            for tensor,offset in local_entries(shard[key]):
                off=tuple(offset)
                assert tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),name
                if off in entries:assert torch.equal(entries[off],tensor)
                else:entries[off]=tensor
        full=torch.empty(shapes[name],dtype=torch.bfloat16);cursor=0
        for offset,tensor in sorted(entries.items()):
            assert offset[0]==cursor and not any(offset[1:]) and tuple(tensor.shape[1:])==shapes[name][1:]
            full[cursor:cursor+tensor.shape[0]].copy_(tensor)
            assert torch.equal(full[cursor:cursor+tensor.shape[0]],tensor.to(torch.bfloat16))
            cursor+=tensor.shape[0]
        assert cursor==shapes[name][0] and bool(torch.isfinite(full).all())
        state[name]=full
    if c.tie_word_embeddings:
        assert torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight'])
        state['lm_head.weight']=state['model.embed_tokens.weight']
    model.load_state_dict(state,strict=True,assign=True);model.tie_weights();model.eval().requires_grad_(False)
    model.save_pretrained(target,safe_serialization=True,max_shard_size='2GB')
    AutoTokenizer.from_pretrained(source/'actor',local_files_only=True).save_pretrained(target)
    index=read(target/'model.safetensors.index.json')['weight_map'];compared=0
    for file in sorted(set(index.values())):
        with safe_open(str(target/file),framework='pt',device='cpu') as saved:
            for key in saved.keys():assert torch.equal(saved.get_tensor(key),state[key]);compared+=1
        with (target/file).open('r+b') as f:os.fsync(f.fileno())
    assert compared==len(state)-int(c.tie_word_embeddings)
    assert before==[(f.stat().st_size,f.stat().st_mtime_ns) for f in paths]
    write(target/'export.json',dict(time=time.time(),step=p['source_step'],source=str(source),source_world_size=p['world_size'],dtype='bfloat16',
        tensor_count=len(state),saved_tensor_count=compared,strict_load=True,all_bfloat16_source_values_equal=True,
        saved_tensor_readback_equal=True,source_native_unchanged=True,model_only=True,optimizer_steps=0))
    print('MODEL_EXPORT_COMPLETE',compared,flush=True)

def panel(p):
    result=[]
    for split,(_,count) in SPLITS.items():
        b=base(p,split);games=sorted(str(Path(x).resolve()) for x in b.game_files)
        assert len(games)==len(set(games))==count,(split,len(games))
        for game in games:
            env=b.init_env_for_game(game)
            try:
                s=env.reset();text=str(s['feedback']);assert 'Your task is to: ' in text
                result.append(dict(index=len(result),split=split,gamefile=game,category=read(Path(game).with_name('traj_data.json'))['task_type'],
                    initial_observation=text,task=text.split('Your task is to: ',1)[1].strip(),admissible=list(s['admissible_commands'])))
            finally:env.close()
        print('PANEL_SPLIT',split,count,flush=True)
    assert len(result)==274 and len({x['gamefile'] for x in result})==274
    write(Path(p['root'])/'panel.json',dict(tasks=result,split_sizes={k:v[1] for k,v in SPLITS.items()},selection='all sorted game files; no subsampling'))

def action_prompt(e,templates,history=5):
    commands='\n '.join(f"'{str(x).strip()}'" for x in e['state']['admissible_commands'] if str(x).strip() and str(x).strip()!='help')
    if not e['rows']:return templates[0].format(current_observation=e['initial_observation'],admissible_actions=commands)
    recent=e['rows'][-history:];start=len(e['rows'])-len(recent)
    text='\n'.join(f"[Observation {start+j+1}: '{r['observation']}', Action {start+j+1}: '{r['action']}']" for j,r in enumerate(recent))
    return templates[1].format(task_description=e['task'],step_count=len(e['rows']),history_length=len(recent),
        action_history=text,current_step=len(e['rows'])+1,current_observation=str(e['state']['feedback']),admissible_actions=commands)

def request_seed(seed,task,turn):return seed*1_000_000+task*30+turn

def validate(r,item,seed,p):
    assert r['seed']==seed and r['task_index']==item['index'] and r['gamefile']==item['gamefile'] and r['split']==item['split']
    assert r['checkpoint']==p['checkpoint'] and r['model']==p['model'] and r['optimizer_steps']==0
    rows=r['rows'];assert r['length']==len(rows) and 1<=len(rows)<=30
    assert type(r['success']) is bool and r['success']==rows[-1]['won']
    assert r['done']==rows[-1]['done'] and (r['done'] or len(rows)==30)
    assert rows[0]['observation']==item['initial_observation']
    for turn,row in enumerate(rows):
        assert row['request_seed']==request_seed(seed,item['index'],turn)
        assert len(row['prompt_ids'])<=2048 and len(row['response_ids'])<=512
        if turn+1<len(rows):assert not row['done'] and row['next_observation']==rows[turn+1]['observation']
    return r

def counts(rows):
    """Paper accuracy is the equal-weight mean of the six category rates."""
    n=len(rows);won=sum(r['success'] for r in rows)
    categories={}
    unknown={r['category'] for r in rows}-set(CATEGORIES)
    if unknown:raise ValueError(f'Unknown ALFWorld task categories: {sorted(unknown)}')
    for category in CATEGORIES:
        selected=[r for r in rows if r['category']==category]
        if not selected:raise ValueError(f'Missing ALFWorld evaluation category: {category}')
        successes=sum(r['success'] for r in selected)
        categories[category]=dict(successes=successes,total=len(selected),accuracy=successes/len(selected))
    macro=sum(r['accuracy'] for r in categories.values())/len(CATEGORIES)
    return dict(successes=won,total=n,accuracy=macro,macro_accuracy=macro,
                micro_accuracy=won/n,categories=categories,metric='six_category_macro')

def worker(p,worker_id):
    import fcntl,psutil
    import numpy as np
    import torch
    from vllm import LLM,SamplingParams
    own=p['workers'][worker_id];assert __import__('socket').gethostname()==own['host']
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(own['gpu'])
    root=Path(p['root']);tasks=read(root/'panel.json')['tasks'];assert len(tasks)==274
    exported=read(Path(p['model'])/'export.json');assert exported['source']==p['checkpoint'] and exported['saved_tensor_readback_equal']
    lock=(root/f'worker{worker_id}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,p['repo'])
    from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE_NO_HIS,ALFWORLD_TEMPLATE
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    templates=(ALFWORLD_TEMPLATE_NO_HIS,ALFWORLD_TEMPLATE)
    bases={s:base(p,s) for s in SPLITS}
    status_path=root/f'worker{worker_id}_status.json'
    identity=dict(worker=worker_id,pid=os.getpid(),create_time=psutil.Process().create_time(),gpu=own['gpu'],uuid=own['uuid'],host=own['host'],optimizer_steps=0)
    write(status_path,dict(identity,state='LOADING_MODEL',time=time.time()))
    llm=LLM(model=p['model'],tokenizer=p['model'],dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.35,
        max_model_len=2560,max_num_seqs=24,max_num_batched_tokens=8192,enforce_eager=True,enable_prefix_caching=False,seed=0,disable_log_stats=True)
    try:
        tok=llm.get_tokenizer()
        for seed in own['seeds']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            dest=root/f'seed{seed}';dest.mkdir(exist_ok=True)
            for start in range(0,len(tasks),24):
                episodes=[]
                try:
                    for item in tasks[start:start+24]:
                        file=dest/f'task{item["index"]:03d}.json'
                        if file.exists():validate(read(file),item,seed,p);continue
                        env=bases[item['split']].init_env_for_game(item['gamefile'])
                        if hasattr(env,'seed'):env.seed(seed)
                        state=env.reset();assert str(state['feedback'])==item['initial_observation']
                        episodes.append(dict(item,env=env,state=state,rows=[],done=False,success=False))
                    for turn in range(30):
                        active=[e for e in episodes if not e['done']]
                        if not active:break
                        prompts=[action_prompt(e,templates) for e in active]
                        encoded=[tok.apply_chat_template([dict(role='user',content=s)],tokenize=False,add_generation_prompt=True,enable_thinking=False) for s in prompts]
                        original=[tok.encode(s,add_special_tokens=False) for s in encoded]
                        ids=[s[-2048:] for s in original]
                        params=[SamplingParams(n=1,temperature=.4,top_p=1.,top_k=-1,max_tokens=512,seed=request_seed(seed,e['index'],turn)) for e in active]
                        outputs=llm.generate([dict(prompt_token_ids=t) for t in ids],params,use_tqdm=False);assert len(outputs)==len(active)
                        texts=[o.outputs[0].text for o in outputs]
                        actions,valids=alfworld_projection(texts,[list(e['state']['admissible_commands']) for e in active])
                        for e,text,action,valid,output,prompt,pi,po in zip(active,texts,actions,valids,outputs,prompts,ids,original):
                            state,_,done=e['env'].step(action);won=bool(state.get('won',False));done=bool(done or won)
                            e['rows'].append(dict(turn=turn,observation=str(e['state']['feedback']),admissible=list(e['state']['admissible_commands']),
                                prompt=prompt,prompt_ids=pi,prompt_original_tokens=len(po),response=text,response_ids=list(output.outputs[0].token_ids),
                                action=action,valid=bool(valid),next_observation=str(state['feedback']),won=won,done=done,request_seed=request_seed(seed,e['index'],turn)))
                            e.update(state=state,done=done,success=won)
                        write(status_path,dict(identity,state='EVALUATING',seed=seed,batch_start=start,turn=turn+1,time=time.time()))
                    for e in episodes:
                        item=tasks[e['index']]
                        r=dict({k:v for k,v in e.items() if k not in ('env','state')},task_index=e['index'],seed=seed,
                            checkpoint=p['checkpoint'],model=p['model'],optimizer_steps=0,length=len(e['rows']))
                        validate(r,item,seed,p);write(dest/f'task{item["index"]:03d}.json',r)
                finally:
                    for e in episodes:e['env'].close()
                print('BATCH_COMPLETE',worker_id,seed,min(start+24,274),flush=True)
            records=[validate(read(dest/f'task{item["index"]:03d}.json'),item,seed,p) for item in tasks]
            summary=dict(seed=seed,**{s:counts([r for r in records if r['split']==s]) for s in SPLITS},total=counts(records))
            write(dest/'summary.json',summary);print('SEED_COMPLETE',json.dumps(summary),flush=True)
        write(status_path,dict(identity,state='COMPLETE',time=time.time(),completed_seeds=own['seeds']))
    finally:
        if hasattr(llm.llm_engine,'engine_core'):llm.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()

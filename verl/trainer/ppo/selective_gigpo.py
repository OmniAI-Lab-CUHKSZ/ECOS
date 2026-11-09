"""Step-only soft Teacher modulation and same-batch Reflection/Skill credit.

No rollout/query/critic is added. Both feedback paths preserve real visits and
deduplicate batch padding. Only auxiliary Actor/R/S signals use this estimator;
main GRPO is unchanged. Legacy stability helpers remain diagnostic-only.
"""
from collections import defaultdict
import json
import numpy as np

EPS = 1e-12
DEFAULT_RS_REWARD_CLIP = 3.0


def unique_turns(rows):
    """Undo existing adjust_batch(copy) padding using the rollout primary key."""
    owners={};unique=[];inverse=[]
    for i,r in enumerate(rows):
        key=(r['group'],r['trajectory'],r['turn'])
        if key not in owners:
            owners[key]=len(unique);unique.append(i)
        else:
            first=rows[unique[owners[key]]]
            for field in ('anchor','action','outcome','reward','valid','admissible','done','episode_length'):
                if r.get(field)!=first.get(field):raise ValueError(f'Conflicting duplicate turn: {key}, {field}')
        inverse.append(owners[key])
    return np.asarray(unique,dtype=int),np.asarray(inverse,dtype=int)


def all_success_group_rows(rows, expected_group_size):
    """Classify complete natural rollout groups using environment outcomes only.

    Real revisits and padding copies are not extra trajectories. Format penalties,
    step advantages and Teacher scores do not enter the success definition.
    """
    unique,_=unique_turns(rows)
    groups=defaultdict(dict)
    for i in unique:
        r=rows[i];trajectories=groups[r['group']]
        outcome=r['outcome']>0.
        if r['trajectory'] in trajectories and trajectories[r['trajectory']]!=outcome:
            raise ValueError('Conflicting environment outcomes within a trajectory')
        trajectories[r['trajectory']]=outcome
    counts=dict(all_success=0,all_fail=0,mixed=0)
    all_success=set()
    for group,trajectories in groups.items():
        if len(trajectories)!=expected_group_size:
            raise ValueError(f'Teacher group needs {expected_group_size} natural trajectories: {group} has {len(trajectories)}')
        successes=sum(trajectories.values())
        kind='all_success' if successes==expected_group_size else 'all_fail' if successes==0 else 'mixed'
        counts[kind]+=1
        if kind=='all_success':all_success.add(group)
    return np.asarray([r['group'] in all_success for r in rows],dtype=bool),counts


def mask_all_success_teacher(weights, selected, all_success_rows):
    """Zero only Actor Teacher contributions; leave raw gap and loss masks alone."""
    import torch
    skip=torch.as_tensor(all_success_rows,device=weights.device,dtype=torch.bool)
    if skip.shape!=(weights.shape[0],):raise ValueError('Teacher group row alignment')
    return weights.masked_fill(skip[:,None],0.),selected & ~skip[:,None]


def _standardize(values):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std()) if len(values) else 0.
    return (values-values.mean())/(std+1e-6) if std > EPS else np.zeros_like(values)


def _sign(values):
    values = np.asarray(values)
    return np.where(np.abs(values)>EPS, np.sign(values), 0)


def prepare_rows(metadata, gamma=.95, invalid_penalty=.1, require_graph_metadata=False):
    required = ('uid','traj_uid','turn_step','anchor_obs','executed_action',
                'rewards','is_action_valid','episode_rewards')
    if require_graph_metadata:
        required += ('admissible_commands_json','step_done','episode_lengths')
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f'Selective GiGPO requires rollout metadata: {missing}')
    if not 0. <= gamma <= 1. or invalid_penalty < 0.:
        raise ValueError('invalid return configuration')
    rows=[]
    for i in range(len(metadata['uid'])):
        reward=float(metadata['rewards'][i])
        outcome=float(metadata['episode_rewards'][i])
        if not np.isfinite(reward) or not np.isfinite(outcome):
            raise ValueError('nonfinite environment reward')
        if reward not in (0.,1.,10.):
            raise ValueError(f'Unexpected ALFWorld raw reward {reward}; do not double-penalize')
        rows.append(dict(group=str(metadata['uid'][i]),trajectory=str(metadata['traj_uid'][i]),
                         turn=int(metadata['turn_step'][i]),anchor=' '.join(str(metadata['anchor_obs'][i]).split()),
                         action=' '.join(str(metadata['executed_action'][i]).split()),
                         reward=10. if reward>0 else 0.,outcome=10. if outcome>0 else 0.,
                         valid=bool(metadata['is_action_valid'][i])))
        if require_graph_metadata:
            commands=json.loads(str(metadata['admissible_commands_json'][i]))
            if not isinstance(commands,list) or not all(isinstance(v,str) for v in commands):
                raise ValueError('Graph advantage requires the observed admissible command list')
            rows[-1].update(admissible=tuple(sorted(commands)),
                            done=bool(metadata['step_done'][i]),
                            episode_length=int(metadata['episode_lengths'][i]))
    trajectories=defaultdict(list)
    unique,inverse=unique_turns(rows)
    for i in unique:
        r=rows[i];trajectories[(r['group'],r['trajectory'])].append(i)
    for ix in trajectories.values():
        ix.sort(key=lambda i:rows[i]['turn'])
        running=0.
        for pos in range(len(ix)-1,-1,-1):
            r=rows[ix[pos]]
            reward=r['reward']
            if pos==len(ix)-1 and reward==0. and r['outcome']>0.:reward=r['outcome']
            r['immediate']=reward-(0. if r['valid'] else invalid_penalty)
            running=r['immediate']+gamma*running
            r['rtg']=running
    for i,owner in enumerate(inverse):
        for field in ('rtg','immediate'):rows[i][field]=rows[unique[owner]][field]
    return rows


def step_advantages(rows):
    """Standardized return-to-go within task/anchor, with no episode addend.

    RTG still contains terminal success and the existing invalid-action penalty.
    Padding copies must be removed by the caller; real repeated visits stay.
    """
    cells=defaultdict(list)
    for i,r in enumerate(rows):
        cells[(r['group'],r['anchor'])].append(i)
    local=np.zeros(len(rows))
    for ix in cells.values():local[ix]=_standardize([rows[i]['rtg'] for i in ix])
    return local


def advantages(rows):
    """Legacy composite/step pair for diagnostics, not Actor or Skill routing."""
    groups=defaultdict(dict)
    for r in rows:groups[r['group']][r['trajectory']]=r['outcome']
    episode={g:dict(zip(v,_standardize(list(v.values())))) for g,v in groups.items()}
    ep=np.asarray([episode[r['group']][r['trajectory']] for r in rows])
    local=step_advantages(rows)
    return ep+local,local


def stable_advantages(rows):
    """Legacy stability diagnostic; not used in current Actor or R/S updates."""
    unique,inverse=unique_turns(rows)
    if len(unique)!=len(rows):
        adv,reliable=stable_advantages([rows[i] for i in unique])
        return adv[inverse],reliable[inverse]
    step=step_advantages(rows)
    stable=_sign(step)!=0
    groups,cells=defaultdict(list),defaultdict(list)
    for i,r in enumerate(rows):
        groups[r['group']].append(i)
        cells[(r['group'],r['anchor'])].append(i)
    support=np.zeros(len(rows),bool)
    for ix in cells.values():
        tids={rows[i]['trajectory'] for i in ix}
        for i in ix:
            support[i]=bool(rows[i]['anchor']) and len(tids)>=3 and any(
                rows[j]['trajectory']!=rows[i]['trajectory'] and rows[j]['action']!=rows[i]['action'] for j in ix)
    stable &= support
    for ix in groups.values():
        for donor in {rows[i]['trajectory'] for i in ix}:
            kept=[i for i in ix if rows[i]['trajectory']!=donor]
            local=step_advantages([rows[i] for i in kept])
            stable[kept] &= (_sign(local)==_sign(step[kept]))&(_sign(local)!=0)
    return step,stable


def graph_advantages(rows, gamma=.95, scale_floor=.1):
    """Empirical observation+admissible Bellman Q-V, exactly as offline.

    Current-group natural transitions only; terminal/horizon bootstrap is zero.
    V follows empirical visit frequencies, not a max over admissible actions.
    Normalize by sqrt(Var(r+gamma*V(next)|state)+scale_floor**2).
    The caller removes padding; repeated real visits remain evidence.
    """
    if not np.isfinite(gamma) or not 0.<=gamma<1.:
        raise ValueError('Graph discount must be finite and in [0,1)')
    if not np.isfinite(scale_floor) or scale_floor<=0.:
        raise ValueError('Graph scale_floor must be finite and positive')
    groups=defaultdict(list);result=np.zeros(len(rows),dtype=np.float64)
    for i,r in enumerate(rows):groups[r['group']].append(i)
    for indices in groups.values():
        rs=[rows[i] for i in indices]
        states={};actions={};sid=[];aid=[];trajectories=defaultdict(list)
        for i,r in enumerate(rs):
            if any(k not in r for k in ('admissible','immediate','done','episode_length')):
                raise ValueError('Graph advantage requires complete natural transition metadata')
            state=(r['anchor'],tuple(sorted(r['admissible'])))
            sid.append(states.setdefault(state,len(states)))
            aid.append(actions.setdefault((state,r['action']),len(actions)))
            trajectories[r['trajectory']].append(i)
        sid=np.asarray(sid);aid=np.asarray(aid);ns=len(states);na=len(actions)
        nxt=np.full(len(rs),ns,dtype=int)
        for visits in trajectories.values():
            visits.sort(key=lambda i:rs[i]['turn'])
            if [rs[i]['turn'] for i in visits]!=list(range(len(visits))):
                raise ValueError('Graph advantage received an incomplete natural trajectory')
            if any(rs[i]['episode_length']!=len(visits) for i in visits):
                raise ValueError('Graph episode length differs from natural visits')
            for i,j in zip(visits,visits[1:]):
                if rs[i]['done']:raise ValueError('Graph trajectory contains rows after terminal')
                nxt[i]=sid[j]
        reward=np.asarray([r['immediate'] for r in rs],dtype=np.float64)
        if not np.isfinite(reward).all():raise ValueError('Nonfinite graph reward')
        sc=np.bincount(sid,minlength=ns);ac=np.bincount(aid,minlength=na)
        transition=np.zeros((ns,ns),dtype=np.float64)
        live=nxt<ns
        np.add.at(transition,(sid[live],nxt[live]),1.)
        transition/=sc[:,None]
        state_reward=np.bincount(sid,weights=reward,minlength=ns)/sc
        value=np.r_[np.linalg.solve(np.eye(ns)-gamma*transition,state_reward),0.]
        target=reward+gamma*value[nxt]
        mean=np.bincount(sid,weights=target,minlength=ns)/sc
        q=np.bincount(aid,weights=target,minlength=na)/ac
        raw=q[aid]-mean[sid]
        variance=np.bincount(sid,weights=(target-mean[sid])**2,minlength=ns)/sc
        result[indices]=raw/np.sqrt(variance[sid]+scale_floor**2)
    if not np.isfinite(result).all():raise ValueError('Nonfinite graph advantage')
    return result


def observation_state_keys(rows, embeddings, epsilon=.1):
    """One donor graph: normalized-centroid observation clustering inside G8/admissible.

    This is the recurrence used by the observation_last_adm_e0.1 offline
    experiment. Neither actions, Teacher scores nor rewards enter clustering.
    Sorting by the natural primary key preserves the tested visitation order.
    """
    embeddings=np.asarray(embeddings,dtype=np.float64)
    if embeddings.ndim!=2 or len(embeddings)!=len(rows):
        raise ValueError('Observation embedding row alignment')
    norms=np.linalg.norm(embeddings,axis=1)
    if not np.isfinite(embeddings).all() or np.any(norms<=0):
        raise ValueError('Nonfinite or zero observation embedding')
    if not np.isfinite(epsilon) or not 0.<=epsilon<1.:
        raise ValueError('Observation cosine distance must be in [0,1)')
    groups=defaultdict(list);keys=[None]*len(rows)
    for i,r in enumerate(rows):groups[r['group']].append(i)
    for indices in groups.values():
        indices=sorted(indices,key=lambda i:(rows[i]['trajectory'],rows[i]['turn']))
        h=embeddings[indices]/norms[indices,None];gram=h@h.T
        sims=[];counts=[];parts=[]
        for j,i in enumerate(indices):
            part=tuple(sorted(rows[i]['admissible']))
            allowed=[k for k,p in enumerate(parts) if p==part]
            k=max(allowed,key=lambda k:sims[k][j]) if allowed else None
            if k is not None and 1.-sims[k][j]<=epsilon:
                m=counts[k]+1;a=(m-1)/m;b=1/m
                norm=np.sqrt(max(1e-30,a*a+b*b+2*a*b*sims[k][j]))
                sims[k]=(a*sims[k]+b*gram[j])/norm;counts[k]=m
            else:
                k=len(sims);sims.append(gram[j].copy());counts.append(1);parts.append(part)
            keys[i]=k
    return keys


def bepo_action_advantages(rows, gamma=.95, state_keys=None):
    """Selected fine-state BEPO20/GAE(.8), gated by observed action diversity.

    Matches the tested observation+admissible/action-gate variant exactly.
    No history/inventory key, episode addend, reliability filter, max backup,
    or changes to the main policy objective. Terminal/horizon bootstrap is zero.
    This retains the tested 20-iteration solver rather than silently substituting
    the converged diagnostic variant. Padding is removed by rollout_advantages.
    """
    if not np.isfinite(gamma) or not 0.<=gamma<1.:
        raise ValueError('BEPO discount must be finite and in [0,1)')
    if state_keys is not None and len(state_keys)!=len(rows):
        raise ValueError('BEPO state key row alignment')
    groups=defaultdict(list);result=np.zeros(len(rows),dtype=np.float64)
    for i,r in enumerate(rows):groups[r['group']].append(i)
    for indices in groups.values():
        indices=sorted(indices,key=lambda i:(rows[i]['trajectory'],rows[i]['turn']))
        rs=[rows[i] for i in indices];states={};sid=[];trajectories=defaultdict(list)
        for i,r in enumerate(rs):
            if any(k not in r for k in ('admissible','immediate','rtg','done','episode_length')):
                raise ValueError('BEPO requires complete natural transition metadata')
            state=((r['anchor'],tuple(sorted(r['admissible']))) if state_keys is None
                   else state_keys[indices[i]])
            sid.append(states.setdefault(state,len(states)))
            trajectories[r['trajectory']].append(i)
        sid=np.asarray(sid);ns=len(states);n=len(rs);nxt=np.full(n,ns,dtype=int)
        for visits in trajectories.values():
            if [rs[i]['turn'] for i in visits]!=list(range(len(visits))):
                raise ValueError('BEPO received an incomplete natural trajectory')
            if any(rs[i]['episode_length']!=len(visits) for i in visits):
                raise ValueError('BEPO episode length differs from natural visits')
            for i,j in zip(visits,visits[1:]):
                if rs[i]['done']:raise ValueError('BEPO trajectory contains rows after terminal')
                nxt[i]=sid[j]
        reward=np.asarray([r['immediate'] for r in rs],dtype=np.float64)
        rtg=np.asarray([r['rtg'] for r in rs],dtype=np.float64)
        if not np.isfinite(reward).all() or not np.isfinite(rtg).all():
            raise ValueError('Nonfinite BEPO reward or return')
        counts=np.bincount(sid,minlength=ns)
        def state_mean(x):return np.bincount(sid,weights=x,minlength=ns)/counts
        value=np.r_[state_mean(rtg),0.]
        for _ in range(20):
            updated=state_mean(reward+gamma*value[nxt])
            difference=float(np.max(np.abs(updated-value[:ns])))
            value[:ns]=updated
            if difference<1e-6:break
        delta=reward+gamma*value[nxt]-value[sid]
        gae=np.zeros(n,dtype=np.float64)
        for visits in trajectories.values():
            running=0.
            for i in reversed(visits):
                running=delta[i]+gamma*.8*running;gae[i]=running
        # Preserve the tested BEPO float32 boundary and group standardization.
        gae=gae.astype(np.float32).astype(np.float64)
        sd=float(gae.std())
        normalized=(gae-gae.mean())/(sd+1e-6) if sd>1e-8 else np.zeros_like(gae)
        action_sets=defaultdict(set)
        for s,r in zip(sid,rs):action_sets[int(s)].add(r['action'])
        gate=np.asarray([len(action_sets[int(s)])>=2 for s in sid])
        result[indices]=normalized*gate
    if not np.isfinite(result).all():raise ValueError('Nonfinite BEPO advantage')
    return result


def rollout_advantages(rows, estimator='gigpo_step', gamma=.95, graph_scale_floor=.1,
                       observation_embeddings=None, observation_epsilon=.1):
    """Unfiltered auxiliary advantage, counting each rollout turn once."""
    unique,inverse=unique_turns(rows)
    visits=[rows[i] for i in unique]
    if estimator=='gigpo_step':step=step_advantages(visits)
    elif estimator=='graph_normalized':step=graph_advantages(visits,gamma,graph_scale_floor)
    elif estimator=='bepo_fine_action20':step=bepo_action_advantages(visits,gamma)
    elif estimator=='bepo_unified_obs_action20':
        if observation_embeddings is None:
            raise ValueError('Unified BEPO requires frozen reference observation embeddings')
        h=np.asarray(observation_embeddings)
        if h.ndim!=2 or len(h)!=len(rows):raise ValueError('Observation embedding row alignment')
        if not np.isfinite(h).all():raise ValueError('Nonfinite observation embedding')
        if not np.array_equal(h,h[unique][inverse]):
            raise ValueError('Conflicting duplicate observation embedding')
        keys=observation_state_keys(visits,h[unique],observation_epsilon)
        step=bepo_action_advantages(visits,gamma,state_keys=keys)
    else:raise ValueError(f'Unknown auxiliary advantage estimator: {estimator}')
    return step[inverse]


def hard_clip_rs_reward(reward, limit=DEFAULT_RS_REWARD_CLIP):
    """Bound final sequence rewards without centering or amplifying small values."""
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 0.:
        raise ValueError('rs_reward_clip must be finite and positive')
    reward = np.asarray(reward, dtype=np.float64)
    # Do not hide a nonfinite upstream reward by turning infinity into the cap.
    if not np.isfinite(reward).all():
        raise ValueError('nonfinite R/S reward before clipping')
    return np.clip(reward, -limit, limit)


def rs_credit(rows,reflection_keys,group_keys,teacher_gap,score_valid,
              reward_clip=DEFAULT_RS_REWARD_CLIP, advantage=None):
    """Raw A_step * (log p_T - log p_S), without stability/sign filtering.

    Reflection receives its trajectory's mean over valid scored turns (zeros
    included). Skill receives an equal-trajectory mean, excluding trajectories
    with no scored turn. Only the final R/S rewards are hard-clipped: Skill is
    computed from RAW trajectory means, never clipped Reflection rewards.
    Raw credits remain available for diagnostics. This is same-batch proxy
    credit, not causal attribution.
    """
    gap=np.asarray(teacher_gap,dtype=np.float64)
    valid=np.asarray(score_valid,dtype=bool)
    if gap.shape!=(len(rows),) or valid.shape!=(len(rows),):raise ValueError('R/S row alignment')
    if not np.isfinite(gap).all():raise ValueError('nonfinite Teacher-Student gap')
    adv=rollout_advantages(rows) if advantage is None else np.asarray(advantage,dtype=np.float64)
    if adv.shape!=(len(rows),) or not np.isfinite(adv).all():raise ValueError('Invalid R/S advantage')
    utility=np.where(valid,adv*gap,0.)
    if not np.isfinite(utility).all():raise ValueError('nonfinite R/S utility')
    unique,_=unique_turns(rows)
    trajectories=defaultdict(list)
    for i in unique:
        if valid[i]:trajectories[(str(rows[i]['group']),str(rows[i]['trajectory']))].append(i)
    means={key:float(np.mean(utility[ix])) for key,ix in trajectories.items()}
    reflection=np.zeros(len(reflection_keys));ref_counts=np.zeros(len(reflection_keys),int)
    for k,(group,trajectory) in enumerate(reflection_keys):
        key=(str(group),str(trajectory))
        reflection[k]=means.get(key,0.)
        ref_counts[k]=len(trajectories.get(key,[]))
    skill=np.zeros(len(group_keys));turn_counts=np.zeros(len(group_keys),int)
    trajectory_counts=np.zeros(len(group_keys),int)
    for k,key in enumerate(group_keys):
        keys=[t for t in means if t[0]==str(key)]
        trajectory_counts[k]=len(keys)
        turn_counts[k]=sum(len(trajectories[t]) for t in keys)
        if keys:skill[k]=float(np.mean([means[t] for t in keys]))
    return dict(action_credit=utility,reflection_credit=reflection,reflection_turn_counts=ref_counts,
                skill_credit=skill,skill_turn_counts=turn_counts,skill_trajectory_counts=trajectory_counts,
                reflection_reward=hard_clip_rs_reward(reflection,reward_clip),
                skill_reward=hard_clip_rs_reward(skill,reward_clip),reward_clip=float(reward_clip))


def skill_credit(rows,group_keys,teacher_gap,score_valid):
    """Convenience group-only view of the same unfiltered R/S estimator."""
    result=rs_credit(rows,[],group_keys,teacher_gap,score_valid)
    return result['skill_credit'],result['skill_turn_counts']


def token_route(teacher_lp,student_lp,action_mask,response_mask,advantage,row_valid,
                modulation_lambda=.75,modulation_tau=1.,*,actor_signal_scope='action_command',
                modulation_mode='asymmetric',modulation_max_factor=16.):
    """Soft, sign-preserving Teacher modulation by the same step A.

    The published symmetric_alignment rule uses the command's mean gap to
    compute one multiplier, then retains each selected token's own gap.
    action_command selects command content; response_token selects the valid
    response under the same command-derived multiplier. The summed command
    gap returned below remains unchanged for R/S credit.
    Both modes retain row validity and complete-action eligibility checks.
    Legacy asymmetric: positive gap uses 1+lambda*tanh(A/tau). Negative gap is weakened only
    when A>0, by 1-lambda*tanh(A/tau); A<=0 leaves negative gap unchanged.
    symmetric_alignment: multiply by 1+lambda*sign(mean_command_gap)*tanh(A/tau).
    Both agreeing signs strengthen, both conflicts weaken, with multiplier
    in [1-lambda,1+lambda]. Each token retains its individual sign.
    exponential_alignment: multiply by exp(log(M)*sign(gap)*tanh(A/tau)),
    where M=modulation_max_factor >= 1. Both agreeing signs are strengthened,
    both conflicts weakened, with multiplier in [1/M,M]. This mode does not
    use the legacy modulation_lambda; the outer loss coefficient is separate.
    Exact A=0 keeps either sign. The returned RAW action-command gap is identical
    in both modes and remains the R/S credit source, not the response-token gap.
    """
    import torch
    if not np.isfinite(modulation_lambda) or not 0.<=modulation_lambda<=1.:
        raise ValueError('modulation_lambda must be finite and in [0,1]')
    if not np.isfinite(modulation_tau) or modulation_tau<=0.:
        raise ValueError('modulation_tau must be finite and positive')
    if actor_signal_scope not in ('action_command','response_token'):
        raise ValueError('unknown actor_signal_scope')
    if modulation_mode not in ('asymmetric','symmetric_alignment','exponential_alignment'):
        raise ValueError('unknown modulation_mode')
    if not np.isfinite(modulation_max_factor) or modulation_max_factor<1.:
        raise ValueError('modulation_max_factor must be finite and >= 1')
    with torch.no_grad():
        teacher_lp=teacher_lp.detach();student_lp=student_lp.detach()
        if not torch.isfinite(teacher_lp).all() or not torch.isfinite(student_lp).all():
            raise ValueError('nonfinite teacher/student log probability')
        action_mask=action_mask.bool()&response_mask.bool()
        valid=torch.as_tensor(row_valid,device=teacher_lp.device,dtype=torch.bool)
        eligible=valid&action_mask.any(-1)
        delta=teacher_lp-student_lp
        command_gap=(delta*action_mask).sum(-1)
        mean_command_gap=command_gap/action_mask.sum(-1).clamp_min(1)
        # Preserve estimator precision through sign decisions and tanh.
        adv=torch.as_tensor(advantage,device=teacher_lp.device,dtype=torch.float64)
        if not torch.isfinite(adv).all():raise ValueError('nonfinite GiGPO advantage')
        if not torch.isfinite(command_gap).all():raise ValueError('nonfinite command gap')
        if adv.shape!=command_gap.shape:raise ValueError('advantage row alignment')
        if actor_signal_scope=='response_token':
            raw_actor_gap=delta
            actor_mask=response_mask.bool()
        else:
            raw_actor_gap=delta if modulation_mode=='symmetric_alignment' else command_gap[:,None]
            actor_mask=action_mask
        if modulation_mode=='exponential_alignment':
            factor=torch.exp(float(np.log(modulation_max_factor))
                             *raw_actor_gap.sign().to(adv.dtype)
                             *torch.tanh(adv/modulation_tau)[:,None])
        elif modulation_mode=='symmetric_alignment':
            factor=(1.+modulation_lambda*mean_command_gap.sign().to(adv.dtype)
                    *torch.tanh(adv/modulation_tau))[:,None]
        else:
            modulation=modulation_lambda*torch.tanh(adv/modulation_tau)
            if actor_signal_scope=='response_token':
                factor=torch.where(delta>0,1.+modulation[:,None],
                                   torch.where(adv[:,None]>0,1.-modulation[:,None],1.))
            else:
                factor=torch.where(command_gap>0,1.+modulation,
                                   torch.where(adv>0,1.-modulation,torch.ones_like(modulation)))[:,None]
        coefficient=raw_actor_gap*factor.to(raw_actor_gap.dtype)
        if not torch.isfinite(coefficient).all():raise ValueError('nonfinite calibrated gap')
        selected=eligible[:,None]&actor_mask&(coefficient!=0)
        weights=torch.where(selected,coefficient,torch.zeros_like(delta))
        return weights.detach(),selected.detach(),command_gap.detach()


def selective_route_loss_terms(log_prob,old_log_prob,teacher_log_prob,weights,selected,clip_low,clip_high):
    """Clipped signed PG only on selected tokens; zero distillation elsewhere."""
    import torch
    weights=weights.detach()
    legacy=weights*(teacher_log_prob.detach()-log_prob)
    ratio=torch.exp(torch.clamp(log_prob-old_log_prob.detach(),-20.,20.))
    signed=-torch.minimum(ratio*weights,torch.clamp(ratio,1.-clip_low,1.+clip_high)*weights)
    # Add a detached constant so the first-epoch value also agrees with legacy;
    # gradients, not the constant, define the signed update.
    signed=signed+(legacy-signed).detach()
    return torch.where(selected.bool(),signed,torch.zeros_like(signed))

"""Compose the four public profiles without changing the shared method."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('alfworld_3b', 'alfworld_7b', 'webshop_3b', 'webshop_7b')


def merge(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        if key == 'defaults':
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    common = yaml.safe_load((ROOT/'configs/rl_common.yaml').read_text(encoding='utf8'))
    overlay = yaml.safe_load((ROOT/f'configs/{profile}.yaml').read_text(encoding='utf8'))
    return merge(common, overlay)


def render(plan):
    cfg = load_config(plan['profile'])
    environment, _ = plan['profile'].split('_')
    model = plan['model']
    cfg['actor_rollout_ref']['model']['path'] = model
    cfg['data'].update(train_files=str(ROOT/f'data/rl/{environment}/train.parquet'),
                       val_files=str(ROOT/f'data/rl/{environment}/test.parquet'))
    cfg['algorithm']['grsd'].update(reflect_sample_dir=plan['output']+'/reflect_samples')
    if plan.get('vllm_memory_utilization') is not None:
        cfg['actor_rollout_ref']['rollout']['gpu_memory_utilization'] = plan['vllm_memory_utilization']
    cfg['trainer'].update(total_training_steps=plan['steps'], total_epochs=plan['steps'],
                          nnodes=1, n_gpus_per_node=plan['world_size'], gpus_per_node=[plan['world_size']],
                          controller_ip=plan.get('node_ip'), experiment_name=plan['run_id'],
                          default_local_dir=plan['output'], rollout_data_dir=plan['output']+'/rollouts',
                          validation_data_dir=plan['output']+'/validation',
                          resume_mode='resume_path' if plan.get('resume_from_path') else 'disable',
                          resume_from_path=plan.get('resume_from_path'))
    if environment == 'webshop':
        cfg['env']['webshop']['worker_node_ip'] = plan.get('node_ip')
    return cfg

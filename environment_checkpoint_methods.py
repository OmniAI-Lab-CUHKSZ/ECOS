"""Boundary state for ALFWorld's deterministic seeded game cycle and worker RNG."""
import random
import numpy as np
import torch

def worker_state(worker):
    return dict(seed=worker._checkpoint_seed,offset=worker._checkpoint_offset,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state())

def restore_worker(worker,state):
    worker.env.seed(int(state['seed']))
    worker.env.skip(int(state['offset']))
    worker._checkpoint_seed=int(state['seed']);worker._checkpoint_offset=int(state['offset'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])

def vector_state(env):
    import ray
    return dict(env_num=env.num_processes//env.group_n,group_n=env.group_n,is_train=env.is_train,seed=env.seed,
                workers=ray.get([w.state_dict.remote() for w in env.workers]))

def restore_vector(env,state):
    import ray
    assert state['env_num']*state['group_n']==env.num_processes and state['group_n']==env.group_n
    assert state['is_train']==env.is_train and state['seed']==env.seed
    assert len(state['workers'])==len(env.workers)
    ray.get([w.load_state_dict.remote(s) for w,s in zip(env.workers,state['workers'])])

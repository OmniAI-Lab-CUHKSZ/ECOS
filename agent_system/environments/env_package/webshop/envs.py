# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np
import random
import torch

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """

    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)

        env_kwargs = dict(env_kwargs)
        env_kwargs['seed'] = env_kwargs.pop('catalog_seed', seed)
        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)
        self._initial_rng = self.state_dict()
        self._terminal_result = None

    def state_dict(self):
        # Checkpoints are written only between complete rollout batches. The
        # next batch always resets sessions, so no in-episode simulator state
        # is needed. Preserve the process RNGs as well as the parent sampler.
        return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())

    def load_state_dict(self, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])

    def rewind_validation(self):
        self.load_state_dict(self._initial_rng)

    def step(self, action):
        """Execute a step in the environment"""
        if self._terminal_result is not None:
            obs, info = self._terminal_result
            return obs, 0.0, True, dict(info)
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        if done:
            self._terminal_result = (obs, dict(info))

        return obs, reward, done, info

    def reset(self, idx):
        """Reset the environment with given session index"""
        self._terminal_result = None
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        info['task_score'] = 0.0
        info['webshop_goal_index'] = int(idx)
        return obs, info

    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered

    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()

    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals

    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        worker_node_ip: str = None,
    ) -> None:
        super().__init__()
        self._workers = []

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.seed = int(seed)
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        if worker_node_ip:
            # Ray's default packing placed all 256 CPU environments on the
            # head and exhausted its RAM. Pin only these actors, not GPU roles.
            if not any(n['Alive'] and n['NodeManagerAddress'] == worker_node_ip for n in ray.nodes()):
                raise ValueError(f'WebShop CPU worker node is not alive: {worker_node_ip}')
            resources_per_worker = dict(resources_per_worker)
            resources = dict(resources_per_worker.get('resources', {}))
            resources['node:' + worker_node_ip] = 0.001
            resources_per_worker['resources'] = resources
            print(f'[WebShop] {self.num_processes} CPU environments on {worker_node_ip}; train={is_train}')
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))

        print(self.goal_idxs)

    def state_dict(self):
        return dict(seed=self.seed, env_num=self.env_num, group_n=self.group_n,
                    is_train=self.is_train, goal_idxs=list(self.goal_idxs),
                    sampler=self._rng.get_state(),
                    workers=ray.get([w.state_dict.remote() for w in self._workers]))

    def load_state_dict(self, state):
        for name in ('seed', 'env_num', 'group_n', 'is_train'):
            if state[name] != getattr(self, name):
                raise ValueError(f'WebShop resume changes {name}')
        if state['goal_idxs'] != list(self.goal_idxs) or len(state['workers']) != self.num_processes:
            raise ValueError('WebShop resume changes task pool or environment count')
        self._rng.set_state(state['sampler'])
        ray.get([w.load_state_dict.remote(s) for w, s in zip(self._workers, state['workers'])])

    def prepare_validation(self, task_offset=0):
        if self.is_train:
            raise ValueError('Cannot rewind the training task sampler for validation')
        if task_offset < 0 or task_offset % self.env_num:
            raise ValueError('Validation offset must be a whole environment batch')
        self._rng = np.random.RandomState(self.seed)
        for _ in range(task_offset // self.env_num):
            self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        ray.get([w.rewind_validation.remote() for w in self._workers])

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)

        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)

        # Wait for all workers to close
        ray.get(close_futures)

        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)

        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    worker_node_ip: str = None,
):
    """Construct the WebShop text environment workers."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        worker_node_ip=worker_node_ip,
    )

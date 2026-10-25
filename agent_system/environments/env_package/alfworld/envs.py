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

import os
import yaml
import gymnasium as gym
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.base_env = base_env
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
        self._checkpoint_seed=int(seed)
        self._checkpoint_offset=0
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        self._checkpoint_offset += 1
        infos['observation_text'] = obs
        return obs, infos

    def skip_games(self, n: int):
        """Advance the seeded game-cycle iterator without loading games.

        Used on resume so a restarted worker continues from where its
        checkpoint left off instead of replaying games from the start.
        """
        if n and n > 0:
            self.env.skip(n)
            self._checkpoint_offset += int(n)

    def rewind_evaluation(self, seed, offset):
        """Rebuild the seeded evaluation game cycle before a fixed panel."""
        self.env.seed(int(seed))
        self._checkpoint_seed=int(seed)
        self._checkpoint_offset=0
        self.skip_games(int(offset))

    def state_dict(self):
        from environment_checkpoint_methods import worker_state
        return worker_state(self)

    def load_state_dict(self, state):
        from environment_checkpoint_methods import restore_worker
        restore_worker(self, state)


class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = False
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = bool(is_train)
        self.seed = int(seed)
        start_offset = int(env_kwargs.get("start_offset", 0))

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            # Train: each group gets its own seeded permutation (same game for
            # the G=8 rollouts of a group). Val: all workers share ONE seed and
            # each skips i games, so worker i plays permutation[i] and the full
            # test pool is covered exactly once (no duplicates, no gaps).
            wseed = seed if not is_train else seed + (i // self.group_n)
            worker = env_worker.remote(config, wseed, base_env)
            self.workers.append(worker)
        if is_train:
            # Continue the game cycle across resume (offset = global step).
            if start_offset > 0:
                ray.get([worker.skip_games.remote(start_offset) for worker in self.workers])
        else:
            skip_calls = [
                worker.skip_games.remote(i)
                for i, worker in enumerate(self.workers)
                if i > 0
            ]
            if skip_calls:
                ray.get(skip_calls)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def state_dict(self):
        from environment_checkpoint_methods import vector_state
        return vector_state(self)

    def load_state_dict(self, state):
        from environment_checkpoint_methods import restore_vector
        restore_vector(self, state)

    def prepare_validation(self, task_offset=0):
        """Use the same disjoint task slots on every validation invocation.

        Only validation workers are rewound. Training's checkpointed game
        cycle and start_offset behavior remain unchanged.
        """
        if self.is_train:
            raise ValueError("Cannot rewind training environments for validation")
        ray.get([
            worker.rewind_evaluation.remote(self.seed, int(task_offset) + i)
            for i, worker in enumerate(self.workers)
        ])

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        image_obs_list = None

        return text_obs_list, image_obs_list, info_list


    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands


    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)

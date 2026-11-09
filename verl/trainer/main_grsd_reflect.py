"""Policy-native Reflection/Skill, action Teacher and same-Adam joint R/S.

The public profiles use environment outcomes and same-batch action credit.
An external judge is disabled; the legacy reflect module name is kept for
checkpoint/import compatibility.
"""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="../../configs", config_name="alfworld_3b", version_base=None)
def main(config):
    run_grsd_reflect(config)


def run_grsd_reflect(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    controller_ip = config.trainer.get('controller_ip')
    runner = (GRSDReflectTaskRunner.options(resources={f'node:{controller_ip}': 0.001}).remote()
              if controller_ip else GRSDReflectTaskRunner.remote())
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class GRSDReflectTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        grsd_cfg = config.algorithm.get("grsd", {})

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.actor.strategy != "fsdp":
            raise ValueError("The released Qwen2.5 profiles use FSDP")
        if config.actor_rollout_ref.rollout.name != "vllm" or config.actor_rollout_ref.rollout.mode != "sync":
            raise ValueError("The released pipeline uses synchronous vLLM rollout")
        if config.algorithm.adv_estimator != "grpo" or config.reward_model.enable:
            raise ValueError("This pipeline uses GRPO with environment rewards")
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        global_pool_id = "global_pool"
        configured_gpus_per_node = config.trainer.get("gpus_per_node", None)
        if configured_gpus_per_node is None:
            process_on_nodes = [int(config.trainer.n_gpus_per_node)] * int(config.trainer.nnodes)
        else:
            process_on_nodes = [int(count) for count in configured_gpus_per_node]
            if len(process_on_nodes) != int(config.trainer.nnodes):
                raise ValueError(
                    "trainer.gpus_per_node must contain exactly trainer.nnodes entries: "
                    f"got {process_on_nodes} for nnodes={config.trainer.nnodes}"
                )
            if any(count <= 0 for count in process_on_nodes):
                raise ValueError(f"trainer.gpus_per_node entries must be positive: {process_on_nodes}")
        resource_pool_spec = {
            global_pool_id: process_on_nodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.data import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.grsd_reflect_ray_trainer import GRSDReflectRayTrainer

        trainer = GRSDReflectRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

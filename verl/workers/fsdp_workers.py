# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import stat
import warnings
from contextlib import nullcontext

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import CPUOffloadPolicy, MixedPrecisionPolicy, apply_fsdp2, fsdp2_load_full_state_dict, fsdp_version, get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn, load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available


from codetiming import Timer

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _make_shared_tree_readable(tree_path, *, cross_uid_cleanup=False):
    """Make a rank-0 artifact accessible across differing node UIDs.

    ``cross_uid_cleanup`` is reserved for rank-0 checkpoint subdirectories that
    the trainer on another node must later remove.  Files remain read-only to
    other UIDs; only directories receive write permission, which is the POSIX
    permission required for unlinking their entries during checkpoint pruning.
    """
    if not os.path.isdir(tree_path):
        return

    directory_read_bits = stat.S_IRGRP | stat.S_IROTH
    directory_execute_bits = stat.S_IXGRP | stat.S_IXOTH
    directory_cleanup_bits = (
        stat.S_IWGRP | stat.S_IWOTH if cross_uid_cleanup else 0
    )
    file_read_bits = stat.S_IRGRP | stat.S_IROTH
    for root, directories, files in os.walk(tree_path):
        os.chmod(
            root,
            stat.S_IMODE(os.stat(root).st_mode)
            | directory_read_bits
            | directory_execute_bits
            | directory_cleanup_bits,
        )
        for directory in directories:
            directory_path = os.path.join(root, directory)
            os.chmod(
                directory_path,
                stat.S_IMODE(os.stat(directory_path).st_mode)
                | directory_read_bits
                | directory_execute_bits
                | directory_cleanup_bits,
            )
        for filename in files:
            file_path = os.path.join(root, filename)
            os.chmod(
                file_path,
                stat.S_IMODE(os.stat(file_path).st_mode) | file_read_bits,
            )


def _make_shared_hf_checkpoint_readable(local_path):
    _make_shared_tree_readable(os.path.join(local_path, "huggingface"))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        if self.config.actor.get('rs_joint_update', False):
            from verl.utils.network_fabric import configure_fabric
            print('[TrainingFabric] node fabric', configure_fabric(), flush=True)
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl", rank=rank, world_size=world_size)

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get('lora_rank', 0)
        self._is_lora = self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._is_lora or self.config.model.get("reflect_lora", {}).get("enabled", False):
            raise ValueError("Only full-parameter Actor/R-S training is included")
        self._reflect_lora_enabled = False
        self.reflect_lora_optimizer = None
        self.reflect_lora_lr_scheduler = None

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            logical_world_size=int(self.config.actor.get('elastic_logical_world_size',0))
            self.config.actor.ppo_mini_batch_size //= logical_world_size or (self.device_mesh.size() // self.ulysses_sequence_parallel_size)
            assert self.config.actor.ppo_mini_batch_size > 0, f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _full_actor_context(self, adapter_requested=False):
        if adapter_requested:
            raise ValueError("This release uses the shared full Actor without adapters")
        return nullcontext()

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
                
        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = "torch"
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            apply_reflect_lora = bool(
                role == "actor" and self._reflect_lora_enabled
            )
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # Preserve transformer-layer wrapping for the full-parameter Actor.
        use_lora_leaf_wrap = False
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=use_lora_leaf_wrap,
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        keep_reference_on_gpu = bool(role != "actor" and fsdp_config.get("keep_reference_on_gpu", False))
        cpu_offload = (None if role == "actor" or keep_reference_on_gpu
                       else CPUOffload(offload_params=True))
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                # Separate base and adapter optimizers require original
                # parameter identities.  Ordinary D/A/E paths retain the
                # historical flattened-parameter behavior.
                use_orig_params=apply_reflect_lora,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = (None if role == "actor" or keep_reference_on_gpu
                               else CPUOffloadPolicy(pin_memory=True))

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        reflect_optimizer = None
        reflect_lr_scheduler = None
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_parameters = list(actor_module_fsdp.parameters())
            reflect_parameters = []

            actor_optimizer = optim.AdamW(
                actor_parameters,
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")


            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
            reflect_optimizer,
            reflect_lr_scheduler,
        )

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        if rollout_name != "vllm" or self.config.rollout.mode != "sync":
            raise ValueError("This release supports synchronous vLLM rollout")
        if rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get('use_shm', False))
            lora_kwargs = {}
            # lora_kwargs = {}
            rollout = vLLMRollout(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
                **lora_kwargs)

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            if self.config.rollout.get('reuse_unchanged_weights', False) and self.config.rollout.mode != 'sync':
                raise ValueError('weight reuse is supported only for synchronous rollout')
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get('layered_summon', False),
                sync_base_with_peft=self._reflect_lora_enabled,
                reuse_unchanged_weights=self.config.rollout.get('reuse_unchanged_weights', False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.actor.get('rs_joint_update', False):
            # WorkerDict constructs roles inside patch.dict(os.environ), which
            # restores constructor changes. NCCL connects lazily in FSDP init;
            # apply the node-local fabric again outside that temporary context.
            from verl.utils.network_fabric import configure_fabric
            print('[TrainingFabric] model-init fabric', configure_fabric(), flush=True)
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get('use_shm', False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
                self.reflect_lora_optimizer,
                self.reflect_lora_lr_scheduler,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module


            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                reflect_optimizer=self.reflect_lora_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
                auxiliary_optimizers=None,
                auxiliary_lr_schedulers=None,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def update_actor_planned(self, data: DataProto):
        if 'elastic_loss_steps' not in data.meta_info:raise ValueError('Missing elastic plan')
        if not self.config.actor.get('elastic_logical_world_size',0) or data.meta_info.get('elastic_physical_world_size')!=self.world_size:
            raise ValueError('Loss packets do not match the active elastic topology')
        return self.update_actor(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def update_actor_joint_planned(self, data: DataProto, rs_data: DataProto):
        if not self.config.actor.get('rs_joint_update', False):
            raise ValueError('Joint Actor update not configured')
        for packet in (data, rs_data):
            if 'elastic_loss_steps' not in packet.meta_info or packet.meta_info.get('elastic_physical_world_size') != self.world_size:
                raise ValueError('Joint packet does not match the active topology')
        return self.update_actor(data, rs_data=rs_data)


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto, rs_data: DataProto = None):
        if hasattr(self, "rollout_sharding_manager") and hasattr(self.rollout_sharding_manager, "mark_weights_dirty"):
            self.rollout_sharding_manager.mark_weights_dirty()
        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        if rs_data is not None:
            rs_data = rs_data.to(get_torch_device().current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and not self.config.actor.get("stream_optimizer_states", False):
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_torch_device().current_device())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            if rs_data is not None:
                rs_data = self.ulysses_sharding_manager.preprocess_data(data=rs_data)
            # perform training
            with self._full_actor_context(False):
                with Timer(name="update_policy", logger=None) as timer:
                    metrics = self.actor.update_policy(data=data, rs_data=rs_data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
            manager = self.rollout_sharding_manager
            metrics["perf/rollout_weight_sync_total"] = float(getattr(manager, 'weight_sync_count', 0))
            metrics["perf/rollout_weight_reuse_total"] = float(getattr(manager, 'weight_reuse_count', 0))

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_torch_device().current_device())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            
            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        is_lora = data.meta_info.pop("is_lora", False)
        use_reflect_lora = bool(data.meta_info.get("use_reflect_lora", False))
        rs_response_only_logits = bool(
            data.meta_info.get("rs_response_only_logits", False)
        )
        rs_skip_entropy = bool(data.meta_info.get("rs_skip_entropy", False))
        if rs_skip_entropy and not rs_response_only_logits:
            raise ValueError(
                "rs_skip_entropy requires the explicit R/S response-only contract"
            )
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_torch_device().current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        if rs_response_only_logits:
            from verl.utils.rs_context_capacity import full_evidence_token_budget
            data.meta_info["max_token_len"] = full_evidence_token_budget(
                data.batch["attention_mask"], data.meta_info["max_token_len"],
                self.config.rollout.max_model_len)
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["response_only_logits"] = bool(self.config.actor.get("response_only_logits", False))
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(
                    data=data,
                    calculate_entropy=not rs_skip_entropy,
                )
            output_tensors = {"old_log_probs": output}
            if entropys is not None:
                output_tensors["entropys"] = entropys
            output = DataProto.from_dict(
                tensors=output_tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_observation_embeddings(self, data: DataProto):
        """Read-only original-model features, reusing the existing reference FSDP."""
        assert self._is_ref and not self._is_lora
        assert self.ulysses_sequence_parallel_size == 1
        from verl.utils.bepo_observation import reference_observation_features
        from verl.workers.actor.dp_actor import _command_model_forward
        data=data.to(get_torch_device().current_device())
        with self.ulysses_sharding_manager:
            data=self.ulysses_sharding_manager.preprocess_data(data)
            with torch.autocast(device_type=get_device_name(),dtype=torch.bfloat16):
                features=reference_observation_features(
                    self.ref_module_fsdp,dict(data.batch),_command_model_forward,
                    micro_batch_size=int(self.config.ref.get("observation_micro_batch_size", 4)))
            output=DataProto.from_dict(tensors={'observation_embeddings':features})
            output=self.ulysses_sharding_manager.postprocess_data(output)
        if self.world_size>1 and fsdp_version(self.ref_module_fsdp)==1:
            self.ref_module_fsdp._handle.reshard(True)
        return output.to('cpu')

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(get_torch_device().current_device())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["response_only_logits"] = bool(self.config.ref.get("response_only_logits", False))
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        if dist.get_rank() == 0:
            _make_shared_hf_checkpoint_readable(local_path)
        dist.barrier()


        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if hasattr(self, "rollout_sharding_manager") and hasattr(self.rollout_sharding_manager, "mark_weights_dirty"):
            self.rollout_sharding_manager.mark_weights_dirty()
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

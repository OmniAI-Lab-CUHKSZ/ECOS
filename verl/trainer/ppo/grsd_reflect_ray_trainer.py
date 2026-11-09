"""Policy-native analysis, action Teacher calibration and joint same-Adam R/S.

The released trainer uses only the selective_gigpo route and environment outcomes.
Native full-state save/recovery, exact action masks and loss denominators are retained.
"""

import json
import os
import socket
import shutil
from collections import defaultdict, deque
from contextlib import nullcontext
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.teacher_context import build_teacher_batch_with_text
from verl.trainer.ppo.route_loss import compute_action_mask
from verl.trainer.ppo.selective_gigpo import prepare_rows as selective_rows, rollout_advantages, token_route as selective_token_route, all_success_group_rows, mask_all_success_teacher, DEFAULT_RS_REWARD_CLIP
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


def _prepare_shared_actor_checkpoint_dir(default_local_dir, global_steps):
    """Make the shared Actor checkpoint directory writable by every node rank."""
    step_dir = os.path.join(default_local_dir, f"global_step_{global_steps}")
    actor_dir = os.path.join(step_dir, "actor")
    hf_dir = os.path.join(actor_dir, "huggingface")
    os.makedirs(hf_dir, mode=0o777, exist_ok=True)

    # ``makedirs(mode=...)`` is filtered by umask and does not widen an
    # existing directory.  The two training nodes expose the same account with
    # different numeric UID/GID values, so explicitly set both shared levels
    # before the remote FSDP ranks start writing their shards.
    os.chmod(step_dir, 0o777)
    os.chmod(actor_dir, 0o777)
    os.chmod(hf_dir, 0o777)
    return step_dir, actor_dir


class GRSDReflectRayTrainer(RayPPOTrainer):
    """Policy-native Actor, Reflection and Skill training with joint updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        grsd_cfg = self.config.algorithm.get("grsd", {})
        route_cfg = self.config.algorithm.get("route", {})
        self.route_mode = str(route_cfg.get("mode", "selective_gigpo"))
        if self.route_mode != "selective_gigpo":
            raise ValueError("This release supports the selective_gigpo route")
        if grsd_cfg.get("judge_enabled", False):
            raise ValueError("This trainer uses policy-generated analysis without an external judge")
        self.grsd_enabled = bool(grsd_cfg.get("enabled", True))
        self.selective_enabled = True
        self.route_lambda = float(route_cfg.get("lambda_route", 0.01))
        self.route_rms_normalize = bool(route_cfg.get("rms_normalize", False))
        self.rs_joint_update = bool(self.config.actor_rollout_ref.actor.get("rs_joint_update", True))
        if not self.rs_joint_update or not self.grsd_enabled or self.route_rms_normalize:
            raise ValueError("Joint R/S and policy analysis must be enabled, with raw Teacher-gap magnitude preserved")
        self.reflect_lora_enabled = bool(self.config.actor_rollout_ref.model.get("reflect_lora", {}).get("enabled", False))
        if self.reflect_lora_enabled:
            raise ValueError("The released method updates the shared full Actor without separate LoRA")
        self.grsd_teacher_max_prompt_length = int(grsd_cfg.get("teacher_max_prompt_length", self.config.data.max_prompt_length))
        if self.grsd_teacher_max_prompt_length < self.config.data.max_prompt_length:
            raise ValueError("Teacher prompt budget must cover data.max_prompt_length")
        # alpha in L = L_task + alpha * L_ref (paper Eq. total_loss).
        self.reflect_loss_coef = grsd_cfg.get("reflect_loss_coef", 0.1)
        # sample reflections/prior with the rollout temperature (diversity for GRPO).
        self.reflect_do_sample = grsd_cfg.get("reflect_do_sample", True)
        # prompt-shaping limits for the reflection/judge trajectory text.
        self.reflect_max_turns = grsd_cfg.get("reflect_max_turns", 50)
        self.reflect_max_chars_per_obs = grsd_cfg.get("reflect_max_chars_per_obs", 400)
        # Stage-A reflection and Stage-B prior generation can use a longer
        # context than normal task prompts. The vLLM max_model_len must cover
        # this prompt budget plus the generated response.
        self.reflect_max_prompt_length = int(
            grsd_cfg.get("reflect_max_prompt_length", self.config.data.max_prompt_length)
        )
        rollout_max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len", None)
        if rollout_max_model_len is not None:
            required_model_len = self.reflect_max_prompt_length + self.config.data.max_response_length
            if int(rollout_max_model_len) < required_model_len:
                raise ValueError(
                    "actor_rollout_ref.rollout.max_model_len must be at least "
                    "algorithm.grsd.reflect_max_prompt_length + data.max_response_length "
                    f"({required_model_len}), got {rollout_max_model_len}"
                )
        # Sampling dump: every `reflect_sample_freq` steps, persist one sampled
        # prompt's Stage-A reflections + Stage-B prior to jsonl for inspection.
        # Default dir lives under the checkpoint dir; None/"" disables dumping.
        self.reflect_sample_freq = int(grsd_cfg.get("reflect_sample_freq", 5))
        default_sample_dir = os.path.join(self.config.trainer.default_local_dir, "reflect_samples")
        self.reflect_sample_dir = grsd_cfg.get("reflect_sample_dir", default_sample_dir)

        self.selective_cfg = self.config.algorithm.get("selective_gigpo", {})
        self.selective_estimator = str(self.selective_cfg.get('advantage_estimator', 'bepo_unified_obs_action20'))
        self.selective_actor_signal_scope = str(self.selective_cfg.get('actor_signal_scope', 'action_command'))
        self.selective_modulation_mode = str(self.selective_cfg.get('modulation_mode', 'symmetric_alignment'))
        self.selective_max_factor = float(self.selective_cfg.get('modulation_max_factor', 16.))
        self.selective_lambda = float(self.selective_cfg.get('modulation_lambda', 1.))
        self.selective_tau = float(self.selective_cfg.get('modulation_tau', .5))
        self.selective_rs_reward_clip = float(self.selective_cfg.get('rs_reward_clip', DEFAULT_RS_REWARD_CLIP))
        self.skip_teacher_on_all_success = bool(self.selective_cfg.get('skip_teacher_on_all_success', False))
        if self.selective_actor_signal_scope not in ('action_command','response_token'):
            raise ValueError('Unknown selective actor_signal_scope')
        if self.selective_modulation_mode not in ('asymmetric','symmetric_alignment','exponential_alignment'):
            raise ValueError('Unknown selective modulation_mode')
        if not np.isfinite(self.selective_max_factor) or self.selective_max_factor < 1.:
            raise ValueError('selective_gigpo.modulation_max_factor must be finite and >= 1')
        if self.selective_estimator not in ('gigpo_step','graph_normalized','bepo_fine_action20','bepo_unified_obs_action20'):
            raise ValueError('Unknown selective advantage estimator')
        if self.selective_estimator=='bepo_unified_obs_action20':
            if not self.use_reference_policy or self.ref_in_actor:
                raise ValueError('Unified BEPO requires the existing frozen original reference policy')
            if self.config.actor_rollout_ref.actor.ulysses_sequence_parallel_size != 1:
                raise ValueError('Unified observation extraction currently supports sequence parallel size 1')
        if not np.isfinite(self.selective_lambda) or not 0. <= self.selective_lambda <= 1.:
            raise ValueError('selective_gigpo.modulation_lambda must be finite and in [0,1]')
        if not np.isfinite(self.selective_tau) or self.selective_tau <= 0.:
            raise ValueError('selective_gigpo.modulation_tau must be finite and positive')
        if not np.isfinite(self.selective_rs_reward_clip) or self.selective_rs_reward_clip <= 0.:
            raise ValueError('selective_gigpo.rs_reward_clip must be finite and positive')
        print(f'[SELECTIVE_GIGPO] A={self.selective_estimator}; Actor={self.selective_modulation_mode}, scope={self.selective_actor_signal_scope}, legacy_lambda={self.selective_lambda}, M={self.selective_max_factor}, tau={self.selective_tau}, beta={self.route_lambda}; R/S=trajectory_equal_state_centered_group_credit, shared_by_R8_S1, final_reward_clip=+/-{self.selective_rs_reward_clip}; no_crossfit; no_environment_queries')
        print(f'[TEACHER_GROUP] skip_all_success={self.skip_teacher_on_all_success}; success=all_natural_environment_outcomes; group_size={self.config.env.rollout.n}; Actor_Teacher_only; original_loss_denominator; RS_unchanged')
        self._health_step_times = deque(maxlen=10)
        self._health_success_rates = deque(maxlen=10)


    def _repair_incomplete_checkpoint_tracker(self) -> None:
        """Point auto-resume at the newest fully finalized checkpoint."""
        if not self.config.trainer.get("checkpoint_keep_latest_and_tens", False):
            return
        if self.config.trainer.resume_mode != "auto":
            return

        root = os.path.abspath(self.config.trainer.default_local_dir)
        tracker = os.path.join(root, "latest_checkpointed_iteration.txt")
        if not os.path.isfile(tracker):
            return

        try:
            with open(tracker, encoding="utf-8") as f:
                tracked_step = int(f.read().strip())
        except (OSError, ValueError):
            tracked_step = -1

        tracked_dir = os.path.join(root, f"global_step_{tracked_step}")
        if os.path.isfile(os.path.join(tracked_dir, "CHECKPOINT_COMPLETE")):
            return

        complete_steps = []
        if os.path.isdir(root):
            for name in os.listdir(root):
                if not name.startswith("global_step_"):
                    continue
                try:
                    step = int(name[len("global_step_") :])
                except ValueError:
                    continue
                marker = os.path.join(root, name, "CHECKPOINT_COMPLETE")
                if os.path.isfile(marker):
                    complete_steps.append(step)

        if not complete_steps:
            raise RuntimeError(
                f"Checkpoint tracker points to incomplete global_step_{tracked_step}, "
                "and no finalized checkpoint is available for rollback"
            )

        fallback_step = max(complete_steps)
        tracker_tmp = f"{tracker}.tmp.{os.getpid()}"
        with open(tracker_tmp, "w", encoding="utf-8") as f:
            f.write(str(fallback_step))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tracker_tmp, tracker)
        print(
            "[CheckpointRetention] incomplete latest checkpoint detected; "
            f"auto-resume rolled back from step {tracked_step} to complete step "
            f"{fallback_step}"
        )

    def _finalize_and_prune_checkpoints(self) -> None:
        """Always finalize saved state; optionally prune older complete state."""
        prune = self.config.trainer.get("checkpoint_keep_latest_and_tens", False)
        root = os.path.abspath(self.config.trainer.default_local_dir)
        step = int(self.global_steps)
        milestone_interval = int(
            self.config.trainer.get("checkpoint_milestone_interval", 20)
        ) if prune else None
        if prune and milestone_interval <= 0:
            raise ValueError(
                "trainer.checkpoint_milestone_interval must be a positive integer"
            )
        step_name = f"global_step_{step}"
        step_dir = os.path.abspath(os.path.join(root, step_name))
        if os.path.dirname(step_dir) != root or os.path.basename(step_dir) != step_name:
            raise RuntimeError(f"Unsafe checkpoint path: {step_dir}")

        marker = os.path.join(step_dir, "CHECKPOINT_COMPLETE")
        marker_tmp = f"{marker}.tmp.{os.getpid()}"
        with open(marker_tmp, "w", encoding="utf-8") as f:
            f.write(f"{step}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(marker_tmp, marker)

        durable_plan = os.environ.get("WEBSHOP_DURABLE_PLAN")
        if durable_plan:
            import json
            from pathlib import Path
            from durable_native_store import persist_checkpoint, write as durable_write
            plan = json.loads(Path(durable_plan).read_text())
            try:
                persist_checkpoint(plan, Path(step_dir))
            except Exception as error:
                durable_write(Path(plan['root']) / 'archival_error.json',
                              dict(step=step, error=repr(error), time=__import__('time').time()))
                raise

        # Rewrite the tracker only after Actor, optimizer, data, Probe and replay
        # have all been saved by this subclass.
        tracker = os.path.join(root, "latest_checkpointed_iteration.txt")
        tracker_tmp = f"{tracker}.tmp.{os.getpid()}"
        with open(tracker_tmp, "w", encoding="utf-8") as f:
            f.write(str(step))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tracker_tmp, tracker)

        if not prune:
            print(f"[CheckpointRetention] finalized step {step}; pruning disabled")
            return

        removed = []
        for name in os.listdir(root):
            if not name.startswith("global_step_"):
                continue
            try:
                old_step = int(name[len("global_step_") :])
            except ValueError:
                continue
            if old_step == step or old_step % milestone_interval == 0:
                continue
            old_dir = os.path.abspath(os.path.join(root, name))
            if os.path.dirname(old_dir) != root:
                raise RuntimeError(f"Unsafe checkpoint cleanup path: {old_dir}")
            if not os.path.isfile(os.path.join(old_dir, "CHECKPOINT_COMPLETE")):
                continue
            shutil.rmtree(old_dir)
            removed.append(old_step)

        kept = sorted(
            int(name[len("global_step_") :])
            for name in os.listdir(root)
            if name.startswith("global_step_")
            and name[len("global_step_") :].isdigit()
            and os.path.isdir(os.path.join(root, name))
        )
        print(
            f"[CheckpointRetention] finalized step {step}; "
            f"milestone_interval={milestone_interval}; removed={removed}; "
            f"kept={kept}"
        )

    def _load_checkpoint(self):
        """Load Actor state first, then the Probe saved at the same global step."""
        self._repair_incomplete_checkpoint_tracker()
        result = super()._load_checkpoint()
        if self.global_steps > 0 and 'webshop' in self.config.env.env_name.lower():
            source = self.config.trainer.resume_from_path
            if self.config.trainer.resume_mode == 'auto':
                source = os.path.join(self.config.trainer.default_local_dir, f'global_step_{self.global_steps}')
            env_state = torch.load(os.path.join(source, 'webshop_env.pt'), map_location='cpu', weights_only=False)
            self.envs.load_state_dict(env_state['train'])
            self.val_envs.load_state_dict(env_state['validation'])
        if self.global_steps > 0 and 'alfworld' in self.config.env.env_name.lower():
            source = self.config.trainer.resume_from_path
            if self.config.trainer.resume_mode == 'auto':
                source = os.path.join(self.config.trainer.default_local_dir, f'global_step_{self.global_steps}')
            saved = torch.load(os.path.join(source, 'alfworld_env.pt'), map_location='cpu', weights_only=False)
            self.envs.load_state_dict(saved['train'])
            self.val_envs.load_state_dict(saved['validation'])
        durable_plan = os.environ.get("WEBSHOP_DURABLE_PLAN")
        if durable_plan:
            import json
            from pathlib import Path
            plan = json.loads(Path(durable_plan).read_text())
            source = Path(plan['resume_from_path']) if plan.get('resume_from_path') else None
            if source is not None and self.global_steps == plan.get('resume_source_step') and (source / 'NATIVE_PORTABLE_VERIFIED.json').exists() and plan.get('resource_changes', {}).get('release_verified_tmpfs_portable_payloads_after_load', True):
                from durable_native_store import write as durable_write
                rollback = plan['durable_rollback_source']
                assert rollback['receipt']['step'] == self.global_steps
                assert rollback['receipt']['all_values_equal'] and rollback['receipt']['fsync']
                assert source.parent.resolve() == Path(plan['output']).resolve()
                assert not source.is_symlink()
                removed = []
                for name, size in rollback['files'].items():
                    assert name in ['elastic_model.pt', 'elastic_optim_all.pt']
                    file = source / 'actor' / name
                    if file.exists():
                        assert not file.is_symlink() and file.stat().st_size == size
                        file.unlink()
                    else:
                        phase = 'ELASTIC_MODEL_LOADED.json' if name == 'elastic_model.pt' else 'ELASTIC_OPTIMIZER_LOADED.json'
                        loaded = json.loads((source / phase).read_text())
                        assert loaded['loaded_all_ranks'] and loaded['source'] == rollback['source'] and loaded['step'] == self.global_steps
                    removed.append(name)
                durable_write(source / 'PORTABLE_PAYLOAD_RETIRED.json',
                              dict(step=self.global_steps, source_host=socket.gethostname(), source=rollback['source'], files=removed,
                                   reason='State and environment loaded; equivalent durable original retained'))
                print('[Elastic] released verified tmpfs portable payloads after complete state restore', flush=True)
        return result

    def _save_checkpoint(self):
        """Save full Actor, environment, method observations and durable receipts."""
        _prepare_shared_actor_checkpoint_dir(
            self.config.trainer.default_local_dir,
            self.global_steps,
        )
        super()._save_checkpoint()
        if 'alfworld' in self.config.env.env_name.lower():
            torch.save(dict(train=self.envs.state_dict(), validation=self.val_envs.state_dict()),
                       os.path.join(self.config.trainer.default_local_dir, f'global_step_{self.global_steps}', 'alfworld_env.pt'))
        if 'webshop' in self.config.env.env_name.lower():
            torch.save(dict(train=self.envs.state_dict(), validation=self.val_envs.state_dict()),
                       os.path.join(self.config.trainer.default_local_dir,
                                    f'global_step_{self.global_steps}', 'webshop_env.pt'))
        from durable_native_store import write as write_method_observation
        from pathlib import Path
        write_method_observation(Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}' / 'method_observed.json',
            dict(step=self.global_steps, rs_update_calls_cumulative=getattr(self, '_observed_rs_update_calls', 0),
                 rs_joint_update=True, route_mode='selective_gigpo', actor_signal_scope=self.selective_actor_signal_scope, modulation_mode=self.selective_modulation_mode, modulation_lambda=self.selective_lambda, modulation_tau=self.selective_tau, actor_gap_unit='per_response_token' if self.selective_actor_signal_scope in ('response_token',) else 'per_command_token', modulation_gap_unit='action_command_mean', modulation_advantage_unit='same_turn_action_advantage', selective_metrics=getattr(self,'_observed_selective_metrics',{}), actor_metrics=self._observed_actor_metrics))
        if True and self.global_steps==1:
            signal=Path(self.config.trainer.default_local_dir)/'unified_diagnostics'/'step1_natural_signals.npz'
            if signal.is_file():
                import shutil
                signal_name=('teacher_action_signals.npz' if self.selective_actor_signal_scope=='action_command'
                             else 'teacher_response_block_signals.npz')
                shutil.copyfile(signal,Path(self.config.trainer.default_local_dir)/'global_step_1'/signal_name)
        self._finalize_and_prune_checkpoints()

    # ------------------------------------------------------------------ #
    # Policy generation helpers (Stage A reflections + Stage B prior)
    # ------------------------------------------------------------------ #
    def _chat_prompt(self, system: str, user: str) -> str:
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(kwargs):
                kwargs = OmegaConf.to_container(kwargs, resolve=True)
        except Exception:
            pass
        return self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False, **(kwargs or {})
        )

    def _texts_to_gen_batch(self, prompt_texts) -> DataProto:
        """Tokenize a list of chat-templated prompt strings into a gen batch.

        Preserves every evidence token and uses only LEFT padding,
        because reflection prompts embed a full trajectory and may exceed
        max_prompt_length (task truncation is often 'error').
        """
        from verl.utils.dataset.rl_dataset import collate_fn

        max_prompt_length = self.reflect_max_prompt_length
        # Capacity is an error limit, not a target padding length. Only masked
        # left padding changes; every encoded evidence token is retained.
        padding_length = min(max_prompt_length, max(
            len(self.tokenizer.encode(text, add_special_tokens=False)) for text in prompt_texts))
        samples = []
        for text in prompt_texts:
            raw_prompt_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                raise ValueError(f'R/S evidence has {len(raw_prompt_ids)} tokens and exceeds {max_prompt_length}; no silent evidence truncation')
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=text,
                tokenizer=self.tokenizer,
                max_length=padding_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="error",
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            samples.append({
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
            })
        return DataProto.from_single_dict(data=collate_fn(samples))

    def _generate_with_policy(self, prompt_texts, do_sample=True):
        """Run the policy rollout engine on prompt_texts; returns unpadded DataProto."""
        if not prompt_texts:
            return None
        gen_batch = self._texts_to_gen_batch(prompt_texts)
        gen_batch.meta_info = {
            "do_sample": do_sample,
            "validate": False,
            "use_reflect_lora": bool(False),
        }
        padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.generate_sequences(padded)
        return unpad_dataproto(out_padded, pad_size=pad_size)

    # ------------------------------------------------------------------ #
    # Stage A + judge + Stage B (policy-native, replaces external API)
    # ------------------------------------------------------------------ #
    def _group_trajectories(self, batch: DataProto):
        """Group flattened turn rows by uid -> traj_uid -> ordered turns.

        Returns (uid_order, traj_list) where traj_list is a deterministic list of
        per-trajectory dicts: {uid, traj_uid, success, task_text, turns}.
        """
        bs = batch.batch["responses"].size(0)
        uids = batch.non_tensor_batch["uid"]
        traj_uids = batch.non_tensor_batch["traj_uid"]
        turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
        episode_rewards = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))

        group = {}
        evidence = {}
        traj_reward = {}
        traj_order = {}
        uid_order = []
        for i in range(bs):
            uid = uids[i]
            tj = traj_uids[i]
            obs, action = self._decode_turn(batch, i)
            metadata = batch.non_tensor_batch
            ev = dict(turn=int(turn_steps[i]), observation=str(metadata['anchor_obs'][i]),
                      action=str(metadata['executed_action'][i]), feedback=str(metadata['next_anchor_obs'][i]),
                      reward=float(metadata['rewards'][i]), done=bool(metadata['step_done'][i]),
                      raw_response=action, is_action_valid=bool(metadata['is_action_valid'][i]))
            ek = (str(uid),str(tj),int(turn_steps[i]))
            if ek in evidence:
                if evidence[ek] != ev:raise ValueError('Conflicting repeated R/S trajectory turn')
                continue
            evidence[ek] = ev
            group.setdefault(uid, {}).setdefault(tj, []).append((int(turn_steps[i]), obs, action))
            traj_reward[(uid, tj)] = float(episode_rewards[i])
            if uid not in traj_order:
                traj_order[uid] = []
                uid_order.append(uid)
            if tj not in traj_order[uid]:
                traj_order[uid].append(tj)

        def _task_text(uid):
            tj0 = traj_order[uid][0]
            turns = sorted(group[uid][tj0], key=lambda x: x[0])
            initial=evidence[(str(uid),str(tj0),turns[0][0])]['observation']
            import re
            match=re.search(r'Your task is to:\s*([^\n]+)',initial)
            if match is None:match=re.search(r'Your task is to:\s*([^\n]+)',turns[0][1])
            if match is None:raise ValueError('Cannot recover real task from first natural observation')
            return match.group(1).strip()

        traj_list = []
        for uid in uid_order:
            for tj in traj_order[uid]:
                turns_sorted = [
                    {"obs": o, "action": a}
                    for (_, o, a) in sorted(group[uid][tj], key=lambda x: x[0])
                ]
                traj_list.append({
                    "uid": uid,
                    "traj_uid": tj,
                    "success": traj_reward[(uid, tj)] > 0.0,
                    "task_text": _task_text(uid),
                    "turns": turns_sorted,
                    "evidence_rows": [dict(turn=e['turn'],observation=e['observation'],action=e['action'],
                        next_observation=e['feedback'],reward=e['reward'],done=e['done'],
                        raw_response=e['raw_response'],is_action_valid=e['is_action_valid'])
                        for k,e in sorted(evidence.items()) if k[:2]==(str(uid),str(tj))],
                })
        return uid_order, traj_list

    def _reflect_and_build_priors(self, batch: DataProto, timing_raw=None):
        """Policy-generated Reflections and a group Skill.

        Returns:
            row_skill_text: list[str] length bs (z_x per row, "" if m_x=0).
            row_valid: np.ndarray[bool] length bs (m_x per row).
            reflect_pack: dict with the Stage-A generation output + Norm_G
                          advantages, used to build the reflection train batch,
                          or None if no reflections were produced.
            metrics: dict.
        """
        def _t(name):
            return _timer(name, timing_raw) if timing_raw is not None else nullcontext()

        bs = batch.batch["responses"].size(0)
        uids = batch.non_tensor_batch["uid"]
        uid_order, traj_list = self._group_trajectories(batch)
        uid_to_int = {u: k for k, u in enumerate(uid_order)}

        # ---- Stage A: generate one reflection per trajectory with the policy. ----
        with _t("reflection_prompt_build"):
            reflect_prompts = []
            for tj in traj_list:
                from verl.trainer.ppo.rs_sft_protocol_seedtraj import reflection_messages
                domain = 'WebShop' if 'webshop' in self.config.env.env_name.lower() else 'ALFWorld'
                messages=reflection_messages(tj['task_text'],dict(rows=tj['evidence_rows'],success=tj['success']),environment=domain)
                reflect_prompts.append(self._chat_prompt(messages[0]['content'],messages[1]['content']))

        with _t("reflection_generate"):
            reflect_out = self._generate_with_policy(
                reflect_prompts, do_sample=self.reflect_do_sample
            )
        with _t("reflection_decode"):
            reflection_texts = self.tokenizer.batch_decode(
                reflect_out.batch["responses"], skip_special_tokens=True
            )

        # Judge removed (Action-Probe version).  The optional reflection
        # ablation uses trajectory-only D_hat credit assigned later in fit().
        # R_group is diagnostic and does not enter the Stage-A advantage.

        # ---- Stage B: contrastive prior z_x per group (stop-grad policy snapshot). ----
        uid_pos = defaultdict(list)
        uid_neg = defaultdict(list)
        uid_task_text = {}
        for k, tj in enumerate(traj_list):
            uid_task_text[tj["uid"]] = tj["task_text"]
            (uid_pos if tj["success"] else uid_neg)[tj["uid"]].append(reflection_texts[k])

        # Skill generated for EVERY group with evidence-matched output structure:
        # all-success -> DO only; all-failure -> AVOID only; mixed -> DO + AVOID.
        prior_uids = list(uid_order)
        with _t("skill_prompt_build"):
            from verl.trainer.ppo.rs_sft_protocol_seedtraj import skill_messages
            prior_prompts=[]
            for u in prior_uids:
                refs=[dict(trajectory=tj['traj_uid'],success=tj['success'],text=reflection_texts[k])
                      for k,tj in enumerate(traj_list) if tj['uid']==u]
                if len(refs)!=8:raise ValueError('New R/S requires all eight natural trajectories')
                domain = 'WebShop' if 'webshop' in self.config.env.env_name.lower() else 'ALFWorld'
                messages=skill_messages(uid_task_text[u],refs,environment=domain)
                prior_prompts.append(self._chat_prompt(messages[0]['content'],messages[1]['content']))
        uid_to_prior = {}
        prior_out = None
        if prior_prompts:
            with _t("skill_generate"):
                prior_out = self._generate_with_policy(
                    prior_prompts, do_sample=self.reflect_do_sample
                )
            with _t("skill_decode"):
                prior_texts = self.tokenizer.batch_decode(
                    prior_out.batch["responses"], skip_special_tokens=True
                )
            for u, z in zip(prior_uids, prior_texts):
                if z and z.strip():
                    uid_to_prior[u] = z.strip()

        row_skill_text = []
        row_valid = np.ones(bs, dtype=bool)
        for i in range(bs):
            z = uid_to_prior.get(uids[i])
            if z:
                row_skill_text.append(z)
            else:
                row_skill_text.append("")
                row_valid[i] = False

        # ---- Periodic sampling dump (Stage A reflections + Stage B prior). ----
        self._maybe_dump_reflect_sample(
            traj_list=traj_list,
            reflection_texts=reflection_texts,
            scores=np.zeros(len(traj_list), dtype=np.float64),
            valid=np.ones(len(traj_list), dtype=bool),
            uid_order=uid_order,
            uid_to_prior=uid_to_prior,
        )

        reflect_response_length = reflect_out.batch["responses"].size(1)
        reflect_token_counts = reflect_out.batch["attention_mask"][
            :, -reflect_response_length:
        ].sum(-1)
        if prior_out is not None:
            skill_response_length = prior_out.batch["responses"].size(1)
            skill_token_counts = prior_out.batch["attention_mask"][
                :, -skill_response_length:
            ].sum(-1)
        else:
            skill_token_counts = torch.zeros(0)

        metrics = {
            "grsd_reflect/n_trajectories": float(len(traj_list)),
            "grsd_reflect/n_valid_groups": float(len(uid_to_prior)),
            "grsd_reflect/reflection_tokens": float(reflect_token_counts.sum().item()),
            "grsd_reflect/reflection_tokens_mean": float(
                reflect_token_counts.float().mean().item()
            ) if len(reflect_token_counts) else 0.0,
            "grsd_reflect/skill_tokens": float(skill_token_counts.sum().item()),
            "grsd_reflect/skill_tokens_mean": float(
                skill_token_counts.float().mean().item()
            ) if len(skill_token_counts) else 0.0,
            "grsd_reflect/reflection_prompt_chars_mean": float(
                np.mean([len(x) for x in reflect_prompts])
            ) if reflect_prompts else 0.0,
            "grsd_reflect/skill_prompt_chars_mean": float(
                np.mean([len(x) for x in prior_prompts])
            ) if prior_prompts else 0.0,
        }
        reflect_ctx = {
            "traj_list": traj_list,
            "uid_order": uid_order,
            "uid_to_int": uid_to_int,
            "prior_uids": prior_uids,
            "uid_to_prior": uid_to_prior,
        }
        return row_skill_text, row_valid, reflect_out, prior_out, reflect_ctx, metrics

    def _maybe_dump_reflect_sample(
        self, traj_list, reflection_texts, scores, valid, uid_order, uid_to_prior
    ):
        """Every reflect_sample_freq steps, dump ONE sampled prompt's reflection
        artifacts to jsonl: all Stage-A per-rollout reflections for that prompt
        plus the Stage-B distilled privileged prior. No-op if disabled.
        """
        if not self.reflect_sample_dir or self.reflect_sample_freq <= 0:
            return
        if self.global_steps % self.reflect_sample_freq != 0:
            return

        # Pick one prompt (uid) with a generated Stage-B skill; fall back to the
        # first uid so we still capture Stage-A reflections after generation errors.
        chosen_uid = next((u for u in uid_order if u in uid_to_prior), None)
        if chosen_uid is None:
            chosen_uid = uid_order[0] if uid_order else None
        if chosen_uid is None:
            return

        stage_a = []
        task_text = ""
        for k, tj in enumerate(traj_list):
            if tj["uid"] != chosen_uid:
                continue
            task_text = tj["task_text"]
            stage_a.append({
                "traj_uid": str(tj["traj_uid"]),
                "success": bool(tj["success"]),
                "judge_score": float(scores[k]) if valid[k] else None,
                "judge_valid": bool(valid[k]),
                "reflection": reflection_texts[k],
            })

        record = {
            "global_step": int(self.global_steps),
            "uid": str(chosen_uid),
            "task_text": task_text,
            "n_rollouts": len(stage_a),
            "stage_a_reflections": stage_a,
            "stage_b_prior": uid_to_prior.get(chosen_uid),
            "stage_b_has_prior": chosen_uid in uid_to_prior,
        }

        try:
            os.makedirs(self.reflect_sample_dir, exist_ok=True)
            out_path = os.path.join(self.reflect_sample_dir, "reflect_samples.jsonl")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[GRSD] dumped reflection sample (step={self.global_steps}, "
                f"uid={chosen_uid}, n_rollouts={len(stage_a)}, "
                f"has_prior={chosen_uid in uid_to_prior}) -> {out_path}"
            )
        except Exception as e:  # noqa: BLE001 - dumping must never crash training
            print(f"[GRSD] reflection sample dump failed: {type(e).__name__}: {e}")


    def _selective_teacher_log_probs(self, batch, skills):
        """Current pre-update base Actor plus Skill; no adapter or second Teacher."""
        view = build_teacher_batch_with_text(
            batch, skills, self.tokenizer, self.grsd_teacher_max_prompt_length,
            truncation='error',
        )
        view.meta_info.update(use_reflect_lora=False, rs_response_only_logits=True, rs_skip_entropy=True)
        padded, pad = pad_dataproto_to_divisor(view, self.actor_rollout_wg.world_size)
        scored = self.actor_rollout_wg.compute_log_prob(padded)
        return unpad_dataproto(scored, pad_size=pad).batch['old_log_probs'].detach()

    def _selective_rs_train_batch(self, reflect_out, prior_out, context):
        """Train R/S text with bounded final rewards, preserving raw credit logs."""
        feedback=context['selective_feedback']
        metrics={'rs/same_batch_skill_feedback':1.,'rs/crossfit_enabled':0.,
                 'rs/active_reflection_rows':0.,'rs/active_skill_rows':0.,
                 'rs/reward_clip':float(feedback['reward_clip'])}
        outputs=[];credits=[];sequence_weights=[]
        for kind,out,values,keys in (
            ('reflection',reflect_out,feedback['reflection_reward'],[t['uid'] for t in context['traj_list']]),
            ('skill',prior_out,feedback['skill_reward'],context['prior_uids']),
        ):
            credit=np.asarray(values,dtype=np.float32)
            if not np.isfinite(credit).all():raise ValueError('Nonfinite R/S feedback')
            if out is None:continue
            if len(credit)!=len(out) or len(keys)!=len(out):raise ValueError('R/S output/feedback order mismatch')
            length=out.batch['responses'].shape[1]
            present=out.batch['attention_mask'][:,-length:].sum(-1).cpu().numpy()>0
            present &= np.asarray([u in context['uid_to_prior'] for u in keys],dtype=bool)
            raw=np.asarray(feedback[f'{kind}_credit'],dtype=np.float64)[present]
            used=np.asarray(values,dtype=np.float64)[present]
            metrics[f'rs/{kind}_reward_clip_fraction']=float(np.mean(np.abs(raw)>feedback['reward_clip'])) if len(raw) else 0.
            for stage,rewards in (('raw',raw),('used',used)):
                metrics[f'rs/{kind}_reward_{stage}_abs_mean']=float(np.abs(rewards).mean()) if len(rewards) else 0.
                metrics[f'rs/{kind}_reward_{stage}_abs_max']=float(np.abs(rewards).max()) if len(rewards) else 0.
            active=np.flatnonzero(present & (credit!=0.))
            metrics[f'rs/active_{kind}_rows']=float(len(active))
            if len(active):
                from verl.trainer.ppo.rs_sft_protocol import stage_weight
                outputs.append(out.select_idxs(active));credits.append(credit[active])
                sequence_weights.append(np.full(len(active),stage_weight(kind,len(context['prior_uids']))))
        if not outputs:return None,metrics
        # Reflection and Skill prompts/responses can have different lengths.
        # Left-pad prompts, right-pad responses, keeping the response boundary.
        plen=max(x.batch['input_ids'].shape[-1]-x.batch['responses'].shape[-1] for x in outputs)
        rlen=max(x.batch['responses'].shape[-1] for x in outputs)
        padded_outputs=[]
        for out in outputs:
            r=out.batch['responses'].shape[-1];p=out.batch['input_ids'].shape[-1]-r
            tensors={}
            for key in ('input_ids','attention_mask','position_ids'):
                value=self.tokenizer.pad_token_id if key=='input_ids' else 0
                src=out.batch[key]
                tensors[key]=torch.cat((torch.nn.functional.pad(src[...,:p],(plen-p,0),value=value),
                                        torch.nn.functional.pad(src[...,p:],(0,rlen-r),value=value)),dim=-1)
            tensors['responses']=torch.nn.functional.pad(out.batch['responses'],(0,rlen-r),value=self.tokenizer.pad_token_id)
            padded_outputs.append(DataProto.from_dict(tensors=tensors))
        packed=DataProto.concat(padded_outputs)
        credit=np.concatenate(credits)
        # Downstream sequence-mean reduction: restore the fixed .5 R / .5 S
        # objective, including zero-reward groups in its original denominator.
        credit=credit*len(credit)*np.concatenate(sequence_weights)
        padded, pad = pad_dataproto_to_divisor(packed, self.actor_rollout_wg.world_size)
        padded.meta_info.update(use_reflect_lora=bool(False), rs_response_only_logits=True, rs_skip_entropy=True)
        scored = self.actor_rollout_wg.compute_log_prob(padded)
        packed.batch['old_log_probs'] = unpad_dataproto(scored, pad_size=pad).batch['old_log_probs'].detach()
        response_mask = packed.batch['attention_mask'][:, -rlen:]
        packed.batch['advantages'] = torch.as_tensor(credit, dtype=torch.float32)[:,None] * response_mask
        metrics['rs/active_tokens'] = float(response_mask.sum())
        metrics['rs/active_positive_sequences']=float(np.sum(credit>0))
        metrics['rs/active_negative_sequences']=float(np.sum(credit<0))
        return packed, metrics

    def _selective_log(self, batch, advantage, mask, command_gap, feedback, context):
        path = os.path.join(self.config.trainer.default_local_dir, 'selective_gigpo', 'mechanism.jsonl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = dict(step=int(self.global_steps), same_batch_feedback=True, crossfit=False,
                      actor_rule=('step_exponential_teacher_gap' if self.selective_modulation_mode=='exponential_alignment'
                                  else 'step_symmetric_soft_teacher_gap' if self.selective_modulation_mode=='symmetric_alignment'
                                  else 'step_soft_teacher_gap'), actor_stability_filter=False, rs_stability_filter=False,
                      actor_signal_scope=self.selective_actor_signal_scope,
                      actor_gap_unit=('per_response_token' if self.selective_actor_signal_scope in ('response_token',) else 'per_command_token'),
                      modulation_gap_unit='action_command_mean', modulation_advantage_unit='same_turn_action_advantage',
                      modulation_mode=self.selective_modulation_mode, modulation_max_factor=self.selective_max_factor,
                      route_loss_coef=self.route_lambda,
                      teacher_student_gap_unit='raw_action_command_sum_for_rs',
                       advantage_source=self.selective_estimator, skill_advantage_source=self.selective_estimator,
                      modulation_lambda=self.selective_lambda, modulation_tau=self.selective_tau,
                      rs_reward_rule='weighted_state_center_linear_shared_group_clip3',
                      actor_zero_advantage='teacher_gap', actor_conflict='weaken',
                      advantage=np.asarray(advantage).tolist(), distilled=mask.any(-1).cpu().tolist(),
                      teacher_student_gap=np.asarray(command_gap).tolist(),
                      feedback={key:np.asarray(value).tolist() for key,value in feedback.items()},
                      prior_uids=[str(u) for u in context['prior_uids']],
                      reflection_keys=[[str(t['uid']),str(t['traj_uid'])] for t in context['traj_list']],
                      trajectory=[str(u) for u in batch.non_tensor_batch['traj_uid']])
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')

    def _response_loss_mask(self, batch):
        """Return the existing Actor loss mask aligned to response tokens."""

        response_length = batch.batch["responses"].size(1)
        if (
            self.config.actor_rollout_ref.rollout.multi_turn.enable
            and "loss_mask" in batch.batch
        ):
            return batch.batch["loss_mask"][:, -response_length:]
        return batch.batch["response_mask"]

    def _ensure_action_token_mask(self, batch):
        """Build and cache the exact action-content mask for this batch."""

        response_loss_mask = self._response_loss_mask(batch)
        responses = batch.batch["responses"]
        if "action_token_mask" in batch.batch:
            action_mask = batch.batch["action_token_mask"]
            if action_mask.shape != responses.shape:
                raise ValueError("action_token_mask must match responses")
            action_mask = action_mask.to(
                device=responses.device, dtype=torch.bool
            )
        else:
            action_mask = compute_action_mask(
                self.tokenizer,
                responses,
                response_loss_mask,
            )
        action_mask = action_mask & response_loss_mask.to(
            device=responses.device, dtype=torch.bool
        )
        batch.batch["action_token_mask"] = action_mask.detach()
        return batch.batch["action_token_mask"]


    # ------------------------------------------------------------------ #
    # fit(): base GRSD loop + policy-native prior + reflection update pass
    # ------------------------------------------------------------------ #
    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="GRSD Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                analysis_step_path = None
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):

                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self._score_physical(self.actor_rollout_wg.compute_log_prob,batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # Policy-generated analysis and environment-calibrated Teacher guidance
                    rs_train_batch = None
                    route_gate = None
                    route_token_mask = None
                    d_hat_rows = np.zeros(batch.batch["responses"].size(0), dtype=np.float32)
                    with _timer("grsd_reflect_construct", timing_raw):
                        row_skill_text, row_valid, reflect_out, prior_out, reflect_ctx, reflect_metrics = (
                            self._reflect_and_build_priors(batch, timing_raw=timing_raw)
                        )
                        metrics.update(reflect_metrics)


                    # GRSD and SEED supervise the complete assistant turn
                    # selected by a step-level signal.  Keep command-level
                    # D-hat as the routing decision, but apply its signed
                    # distillation signal to the complete generated turn
                    # instead of depending on a post-hoc <action> span.
                    # Passing the same response mask into the command trie
                    # also lets it return the already-computed Teacher
                    # log-probs and Student terminal hidden state for the
                    # sampled response.

                    # Probe features (student context, lagged checkpoint).
                    # Command mode reuses the exact same Student trie
                    # forward below; other modes retain the legacy pass.

                    # Build the Skill-prefixed Teacher view once.  In
                    # command+trie mode its candidate scorer also yields
                    # the sampled action's token log-probs, so the old full
                    # Teacher pass is skipped except for rare mapping
                    # fallback rows.  Other routes retain the old path.
                    teacher_dec = None
                    teacher_batch = None
                    with _timer('selective_teacher', timing_raw):
                        teacher_log_probs = self._selective_teacher_log_probs(batch, row_skill_text)
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs_with_priors(
                            batch, row_skill_text
                        )
                    batch.batch["teacher_log_probs"] = teacher_log_probs


                    # SDAR/legacy Probe gates can be built before reward
                    # credit.  Teacher calibration uses the ordinary GRPO rewards
                    # are available so strict trajectory-LOO can be exact.
                    r_ref = None
                    with _timer('selective_gigpo_route', timing_raw):
                        selective_data_rows = selective_rows(
                            batch.non_tensor_batch,
                            gamma=float(self.selective_cfg.get('gamma', .95)),
                            invalid_penalty=float(self.selective_cfg.get('invalid_penalty', .1)),
                            require_graph_metadata=self.selective_estimator in ('graph_normalized','bepo_fine_action20','bepo_unified_obs_action20'),
                        )
                        observation_embeddings=None
                        if self.selective_estimator=='bepo_unified_obs_action20':
                            from verl.utils.bepo_observation import observation_inputs
                            with _timer('bepo_observation_forward',timing_raw):
                                inputs,obs_inverse=observation_inputs(self.tokenizer,batch.non_tensor_batch['anchor_obs'])
                                features=self._score_physical(
                                    self.ref_policy_wg.compute_observation_embeddings,
                                    DataProto.from_dict(tensors=inputs))
                                observation_embeddings=features.batch['observation_embeddings'].numpy()[obs_inverse]
                            metrics['selective/observation_unique_count']=float(len(inputs['input_ids']))
                            metrics['selective/observation_input_tokens']=float(inputs['attention_mask'].sum())
                        selective_adv = rollout_advantages(
                            selective_data_rows, estimator=self.selective_estimator,
                            gamma=float(self.selective_cfg.get('gamma', .95)),
                            graph_scale_floor=float(self.selective_cfg.get('graph_scale_floor', .1)),
                            observation_embeddings=observation_embeddings,
                        )
                        action_mask = self._ensure_action_token_mask(batch)
                        route_token_mask = self._response_loss_mask(batch)
                        route_gate, selected_mask, command_gap = selective_token_route(
                            teacher_log_probs,batch.batch['old_log_probs'],action_mask,route_token_mask,
                            selective_adv,row_valid,
                            modulation_lambda=self.selective_lambda,modulation_tau=self.selective_tau,
                            actor_signal_scope=self.selective_actor_signal_scope,
                            modulation_mode=self.selective_modulation_mode,
                            modulation_max_factor=self.selective_max_factor,
                        )
                        natural_all_success,teacher_group_counts=all_success_group_rows(
                            selective_data_rows,int(self.config.env.rollout.n))
                        metrics.update({f'selective/teacher_groups_{key}':float(value)
                                        for key,value in teacher_group_counts.items()})
                        skipped_teacher_rows=np.zeros(len(selective_data_rows),dtype=bool)
                        if self.skip_teacher_on_all_success:
                            skipped_teacher_rows=natural_all_success
                            route_gate,selected_mask=mask_all_success_teacher(
                                route_gate,selected_mask,skipped_teacher_rows)
                        teacher_active_rows=~torch.as_tensor(skipped_teacher_rows,
                            device=route_gate.device,dtype=torch.bool)
                        batch.batch['selective_route_mask'] = selected_mask
                        eligible_mask=(torch.as_tensor(row_valid,device=route_gate.device,dtype=torch.bool)[:,None]
                                       & action_mask.bool() & route_token_mask.bool())
                        response_scope=self.selective_actor_signal_scope=='response_token'
                        actor_eligible_mask=(eligible_mask.any(-1)[:,None]&route_token_mask.bool()
                                             if response_scope else eligible_mask)
                        raw_actor_gap=teacher_log_probs-batch.batch['old_log_probs']
                        # Group removal is separate from BEPO amplitude modulation.
                        modulated_mask=actor_eligible_mask & teacher_active_rows[:,None] & (route_gate!=raw_actor_gap)
                        raw_gap=command_gap.detach().cpu().numpy()
                        # First batch: preserve natural inputs for an
                        # explicit old/new offline comparison, not a second live estimator.
                        if observation_embeddings is not None and not getattr(self,'_unified_snapshot_written',False):
                            diagnostic_dir=os.path.join(self.config.trainer.default_local_dir,'unified_diagnostics')
                            os.makedirs(diagnostic_dir,exist_ok=True)
                            np.savez_compressed(
                                os.path.join(diagnostic_dir,f'step{self.global_steps}_natural_signals.npz'),
                                rows_json=json.dumps(selective_data_rows),
                                raw_observations_json=json.dumps([str(x) for x in batch.non_tensor_batch['anchor_obs']]),
                                observation_embeddings=observation_embeddings,
                                advantage=selective_adv,raw_command_gap=raw_gap,
                                actor_signal_scope=self.selective_actor_signal_scope,
                                modulation_mode=self.selective_modulation_mode,
                                modulation_lambda=self.selective_lambda,
                                modulation_max_factor=self.selective_max_factor,
                                modulation_tau=self.selective_tau,route_loss_coef=self.route_lambda,
                                skip_teacher_on_all_success=self.skip_teacher_on_all_success,
                                teacher_skipped_rows=skipped_teacher_rows,
                                eligible_action_tokens=eligible_mask.sum(-1).cpu().numpy(),
                                all_action_tokens=action_mask.sum(-1).cpu().numpy(),
                                response_tokens=route_token_mask.sum(-1).cpu().numpy(),
                                raw_teacher_token_gap=(teacher_log_probs-batch.batch['old_log_probs']).detach().cpu().numpy(),
                                raw_actor_gap=(raw_actor_gap*actor_eligible_mask).detach().cpu().numpy(),
                                actor_token_weights=route_gate.detach().cpu().numpy(),
                                actor_selected_mask=selected_mask.cpu().numpy(),
                                actor_eligible_mask=actor_eligible_mask.cpu().numpy(),
                                actor_response_mask=route_token_mask.bool().cpu().numpy(),
                                action_token_mask=action_mask.bool().cpu().numpy(),
                                **(dict(raw_actor_token_gap=(raw_actor_gap*actor_eligible_mask).detach().cpu().numpy())
                                   if response_scope else {}))
                            self._unified_snapshot_written=True
                        from verl.trainer.ppo.selective_gigpo import observation_state_keys,unique_turns
                        from verl.trainer.ppo.rs_sft_protocol import centered_credit
                        unique,inverse=unique_turns(selective_data_rows)
                        metrics.update({
                            'selective/skip_teacher_on_all_success':float(self.skip_teacher_on_all_success),
                            'selective/teacher_skipped_unique_turns':float(skipped_teacher_rows[unique].sum()),
                            'selective/teacher_skipped_eligible_tokens':float((actor_eligible_mask & ~teacher_active_rows[:,None]).sum()),
                            'selective/all_success_teacher_nonzero_tokens':float(((route_gate!=0) & torch.as_tensor(
                                natural_all_success,device=route_gate.device,dtype=torch.bool)[:,None]).sum()),
                        })
                        state_keys=np.asarray(observation_state_keys([selective_data_rows[i] for i in unique],
                            observation_embeddings[unique],.1))[inverse]
                        feedback = centered_credit(selective_data_rows,selective_adv,raw_gap,
                            np.asarray(row_valid)&((action_mask.bool()&route_token_mask.bool()).any(-1).cpu().numpy()),
                            state_keys,group_keys=reflect_ctx['prior_uids'],
                            reflection_keys=[(t['uid'],t['traj_uid']) for t in reflect_ctx['traj_list']],
                            limit=self.selective_rs_reward_clip)
                        feedback={k:np.asarray(v) if isinstance(v,list) else v for k,v in feedback.items() if k!='groups'}
                        reflect_ctx['selective_feedback'] = feedback
                        metrics.update({
                            'selective/actor_stability_filter':0.,
                            'selective/rs_stability_filter':0.,
                            'selective/step_only_advantage':float(self.selective_estimator=='gigpo_step'),
                            'selective/graph_advantage':float(self.selective_estimator=='graph_normalized'),
                            'selective/bepo_fine_action20':float(self.selective_estimator=='bepo_fine_action20'),
                            'selective/bepo_unified_obs_action20':float(self.selective_estimator=='bepo_unified_obs_action20'),
                            'selective/actor_soft_modulation':1.,
                            'selective/actor_response_token_scope':float(response_scope),
                            'selective/action_advantage_shared_across_response':float(response_scope),
                            'selective/eligible_nonaction_tokens':float((actor_eligible_mask & ~action_mask.bool()).sum()),
                            'selective/distilled_nonaction_tokens':float((selected_mask & ~action_mask.bool()).sum()),
                            'selective/distilled_action_tokens':float((selected_mask & action_mask.bool()).sum()),
                            'selective/actor_exponential_alignment':float(self.selective_modulation_mode=='exponential_alignment'),
                            'selective/actor_symmetric_alignment':float(self.selective_modulation_mode=='symmetric_alignment'),
                            'selective/modulation_max_factor':self.selective_max_factor,
                            'selective/route_loss_coef':self.route_lambda,
                            'selective/modulation_lambda':self.selective_lambda,
                            'selective/modulation_tau':self.selective_tau,
                            'selective/distilled_turn_frac':float(selected_mask.any(-1).float().mean()),
                            'selective/distilled_response_token_frac':float(selected_mask.sum()/route_token_mask.sum().clamp_min(1)),
                            'selective/distilled_action_token_frac':float((selected_mask&action_mask.bool()).sum()/action_mask.sum().clamp_min(1)),
                            'selective/modulated_turn_frac':float(modulated_mask.any(-1).float().mean()),
                            'selective/modulated_action_token_frac':float((modulated_mask&action_mask.bool()).sum()/action_mask.sum().clamp_min(1)),
                            'selective/modulated_response_token_frac':float(modulated_mask.sum()/route_token_mask.sum().clamp_min(1)),
                            'selective/distilled_nonaction_token_frac':float((selected_mask&~action_mask.bool()).sum()/(route_token_mask.bool()&~action_mask.bool()).sum().clamp_min(1)),
                            'selective/selected_positive_tokens':float((route_gate>0).sum()),
                            'selective/selected_negative_tokens':float((route_gate<0).sum()),
                            'selective/raw_actor_abs_density':float((raw_actor_gap.abs()*actor_eligible_mask).sum()/route_token_mask.sum().clamp_min(1)),
                            'selective/mod_actor_abs_density':float(route_gate.abs().sum()/route_token_mask.sum().clamp_min(1)),
                            'selective/modification_abs_density':float(((route_gate-raw_actor_gap).abs()*actor_eligible_mask*teacher_active_rows[:,None]).sum()/route_token_mask.sum().clamp_min(1)),
                            'selective/nonzero_step_turn_frac':float(np.mean(np.asarray(selective_adv)!=0)),
                            'selective/rs_scored_turns':float(feedback['skill_turn_counts'].sum()),
                            'selective/skill_positive':float(np.sum(feedback['skill_credit']>0)),
                            'selective/skill_negative':float(np.sum(feedback['skill_credit']<0)),
                            'selective/reflection_positive':float(np.sum(feedback['reflection_credit']>0)),
                            'selective/reflection_negative':float(np.sum(feedback['reflection_credit']<0)),
                            'selective/fallback_exact_sdar':0.,
                            'selective/zero_advantage_uses_teacher_gap':1.,
                            'selective/conflict_weakened':1.,
                            'selective/zero_adv_teacher_turns':float(np.sum((np.asarray(selective_adv)==0)&selected_mask.any(-1).cpu().numpy())),
                            'selective/selected_positive_turns':float((route_gate>0).any(-1).sum()),
                            'selective/selected_negative_turns':float((route_gate<0).any(-1).sum()),
                        })
                        self._selective_log(batch,selective_adv,selected_mask,raw_gap,feedback,reflect_ctx)
                        self._observed_selective_metrics={key:float(value) for key,value in metrics.items() if key.startswith('selective/')}

                    if route_gate is not None:
                        batch.batch["route_gate"] = route_gate
                        batch.batch["route_token_mask"] = route_token_mask

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self._score_physical(self.ref_policy_wg.compute_ref_log_prob,batch)
                            else:
                                ref_log_prob = self._score_physical(self.actor_rollout_wg.compute_ref_log_prob,batch)
                            batch = batch.union(ref_log_prob)


                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )


                        rs_train_batch, selective_rs_metrics = self._selective_rs_train_batch(reflect_out,prior_out,reflect_ctx)
                        metrics.update(selective_rs_metrics)
                        if self.global_steps in self.config.trainer.get('analysis_sample_steps', []):
                            # Observe existing tensors before either optimizer update.
                            # Do not run another model forward, rollout or estimator.
                            with _timer('dump_analysis_assets', timing_raw):
                                try:
                                    from omegaconf import OmegaConf
                                    from verl.trainer.ppo.analysis_assets import save_analysis_step
                                    analysis_step_path, analysis_metrics = save_analysis_step(
                                        self.config.trainer.default_local_dir, self.global_steps,
                                        batch=batch, rows=selective_data_rows, advantage=selective_adv,
                                        observation_embeddings=observation_embeddings,
                                        command_gap=raw_gap, entropy=entropys,
                                        actor_eligible_mask=actor_eligible_mask,
                                        reflect_out=reflect_out, prior_out=prior_out,
                                        rs_train_batch=rs_train_batch, reflect_context=reflect_ctx,
                                        tokenizer=self.tokenizer,
                                        config=OmegaConf.to_container(self.config, resolve=True),
                                    )
                                    metrics.update(analysis_metrics)
                                    print(f'[Analysis] saved step {self.global_steps}: {analysis_step_path}', flush=True)
                                except Exception as error:
                                    metrics['analysis/save_failed'] = 1.
                                    print(f'[Analysis] save failed at step {self.global_steps}: {type(error).__name__}: {error}', flush=True)


                        if "route_gate" in batch.batch:
                            batch.meta_info["route_loss_coef"] = self.route_lambda
                        else:
                            batch.batch.pop("teacher_log_probs", None)


                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Main and R/S gradients accumulate into the same existing Adam slot.
                        joint_batch = rs_train_batch
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self._update_actor_dispatch(batch, rs_batch=joint_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        self._observed_actor_metrics = actor_output_metrics

                        # ---- Reflection GRPO update (separate alpha-weighted pass) ----

                        # ---- F arm: one existing Reflection+Skill block, no extra generation ----

                        # ---- Probe online update (score first, then train) ----
                        # One-batch diagnostic retained for continuity.  It is
                        # observational only: a noisy current batch must not
                        # freeze an otherwise continuously tracking estimator.

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                timing_raw["reflection_update_total"] = float(
                    timing_raw.get("grsd_reflect_logprob", 0.0)
                    + timing_raw.get("update_reflect_actor", 0.0)
                    + timing_raw.get("rs_old_logprob", 0.0)
                    + timing_raw.get("update_rs_actor", 0.0)
                    + timing_raw.get("update_rs_lora", 0.0)
                )
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics.update(self._health_metrics(metrics, timing_raw))

                logger.log(data=metrics, step=self.global_steps)

                if analysis_step_path is not None:
                    try:
                        from verl.trainer.ppo.analysis_assets import write_json
                        write_json(os.path.join(analysis_step_path, 'metrics.json'), metrics)
                    except Exception as error:
                        print(f'[Analysis] metrics save failed at step {self.global_steps}: {type(error).__name__}: {error}', flush=True)

                # Planned scale-up stops only after the portable snapshot and
                # the normal complete checkpoint (including data and environment state) exist.
                run_root = self.config.trainer.default_local_dir
                saved_step = os.path.join(run_root, f"global_step_{self.global_steps}")
                if (not is_last_step
                        and os.path.isfile(os.path.join(run_root, "SCALE_UP_REQUEST"))
                        and os.path.isfile(os.path.join(saved_step, "CHECKPOINT_COMPLETE"))
                        and os.path.isfile(os.path.join(saved_step, "actor", "elastic.json"))):
                    with open(os.path.join(run_root, "PAUSED_FOR_SCALE_UP"), "w") as stream:
                        stream.write(str(self.global_steps) + "\n")
                    print(f"[Elastic] paused after complete step {self.global_steps}", flush=True)
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    # ------------------------------------------------------------------ #
    # Action-Probe helpers (ours)
    # ------------------------------------------------------------------ #
    def _health_metrics(self, metrics, timing_raw) -> dict:
        """Observational timing and numerical-health metrics for long F runs."""
        step_time = float(timing_raw.get("step", 0.0))
        prior_median = (
            float(np.median(self._health_step_times))
            if self._health_step_times
            else step_time
        )
        if step_time > 0:
            self._health_step_times.append(step_time)
        success = metrics.get("episode/success_rate")
        if success is not None and np.isfinite(float(success)):
            self._health_success_rates.append(float(success))

        out = {
            "health/step_time_median_10": float(np.median(self._health_step_times))
            if self._health_step_times
            else 0.0,
            "health/step_time_spike_ratio": step_time / max(prior_median, 1e-6),
            "health/success_rate_mean_10": float(np.mean(self._health_success_rates))
            if self._health_success_rates
            else 0.0,
            "perf/eta_hours_current_step": max(
                0, self.total_training_steps - self.global_steps
            )
            * step_time
            / 3600.0,
        }
        if "route/token_weight_negative_frac" in metrics:
            out["health/route_weight_nonnegative"] = float(
                float(metrics["route/token_weight_negative_frac"]) == 0.0
            )
        if "rs/negative_advantage_frac" in metrics:
            negative_fraction = float(metrics["rs/negative_advantage_frac"])
            signed_update_enabled = bool(
                float(
                    metrics.get(
                        "rs/signed_update_enabled",
                        0.0,
                    )
                )
            )
            out["health/rs_advantage_contract_ok"] = float(
                np.isfinite(negative_fraction)
                and 0.0 <= negative_fraction <= 1.0
                and (
                    signed_update_enabled
                    or negative_fraction == 0.0
                )
            )
            out["health/rs_signed_update_enabled"] = float(
                signed_update_enabled
            )

        top_level = (
            "gen",
            "reward",
            "old_log_prob",
            "grsd_reflect_construct",
            "teacher_forward",
            "selective_teacher",
            "selective_gigpo_route",
            "rs_build",
            "ref",
            "adv",
            "update_actor",
            "update_reflect_actor",
            "update_rs_actor",
            "dump_rollout_generations",
            "testing",
            "save_checkpoint",
        )
        accounted = 0.0
        for name in top_level:
            value = float(timing_raw.get(name, 0.0))
            accounted += value
            out[f"timing_frac/{name}"] = value / max(step_time, 1e-6)
        out["timing_s/accounted_top_level"] = accounted
        out["timing_s/unaccounted"] = max(0.0, step_time - accounted)
        out["timing_frac/unaccounted"] = max(0.0, step_time - accounted) / max(
            step_time, 1e-6
        )

        critical_keys = (
            "actor/pg_loss",
            "actor/grad_norm",
            "actor/kl_loss",
            "actor/route_loss",
            "route/mean_abs_dhat",
            "route/mean_abs_gate",
            "rs/weight_mean",
        )
        nonfinite = 0
        for key in critical_keys:
            if key in metrics and not np.isfinite(float(metrics[key])):
                nonfinite += 1
        out["health/critical_nonfinite_count"] = float(nonfinite)
        out["health/critical_metrics_finite"] = float(nonfinite == 0)
        return out


    def _score_physical(self,method,batch):
        if not self.config.actor_rollout_ref.actor.get('elastic_logical_world_size',0):
            return method(batch)
        padded,count=pad_dataproto_to_divisor(batch,self.actor_rollout_wg.world_size)
        return unpad_dataproto(method(padded),pad_size=count)

    def _loss_packets(self,batch,reflection=False):
        from verl.trainer.ppo.elastic_dispatch import pack_loss_packets
        cfg=self.config.actor_rollout_ref.actor
        return pack_loss_packets(batch,logical_world_size=int(cfg.elastic_logical_world_size),
            physical_world_size=self.actor_rollout_wg.world_size,
            global_minibatch_size=int(cfg.ppo_mini_batch_size)*int(self.config.actor_rollout_ref.rollout.n),
            max_token_len=int(cfg.ppo_max_token_len_per_gpu),reflection=reflection)

    def _update_actor_dispatch(self,batch,rs_batch=None):
        if rs_batch is not None:
            from verl.trainer.ppo.elastic_dispatch import collect_loss_metrics, pack_joint_rs_packets
            cfg = self.config.actor_rollout_ref.actor
            if not cfg.get('rs_joint_update', False) or False:
                raise ValueError('Joint R/S must update the full Actor without LoRA')
            if not cfg.get('elastic_logical_world_size', 0):
                raise ValueError('Joint R/S uses the explicit elastic Actor optimizer plan')
            main_packets = self._loss_packets(batch)
            rs_batch.meta_info.update(
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                reflect_loss_coef=float(self.reflect_loss_coef),
                reflect_loss_agg_mode='seq-mean-token-mean',
                reflect_metric_prefix='rs', use_reflect_lora=False,
                rs_response_only_logits=True,
            )
            from verl.utils.rs_context_capacity import full_evidence_token_budget
            rs_token_budget = full_evidence_token_budget(
                rs_batch.batch['attention_mask'], cfg.ppo_max_token_len_per_gpu,
                self.config.actor_rollout_ref.rollout.max_model_len)
            rs_packets = pack_joint_rs_packets(rs_batch,
                physical_world_size=self.actor_rollout_wg.world_size,
                optimizer_steps=len(main_packets[0].meta_info['elastic_loss_steps']) * int(cfg.ppo_epochs),
                max_token_len=rs_token_budget)
            return collect_loss_metrics(self.actor_rollout_wg.update_actor_joint_planned(main_packets, rs_packets))
        if not self.config.actor_rollout_ref.actor.get('elastic_logical_world_size',0):
            return self.actor_rollout_wg.update_actor(batch)
        from verl.trainer.ppo.elastic_dispatch import collect_loss_metrics
        return collect_loss_metrics(self.actor_rollout_wg.update_actor_planned(self._loss_packets(batch)))


    def _decode_turn(self, batch: DataProto, i: int):
            """Decode (obs, action) text for flattened row i."""
            prompt_ids = batch.batch["prompts"][i]
            resp_ids = batch.batch["responses"][i]
            obs = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            action = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            return obs, action


    def _compute_teacher_log_probs_with_priors(self, batch: DataProto, row_skill_text):
            """Teacher forward where each row uses its group's online prior z_x."""
            teacher_batch = build_teacher_batch_with_text(
                batch=batch,
                row_skill_text=row_skill_text,
                tokenizer=self.tokenizer,
                max_prompt_length=self.grsd_teacher_max_prompt_length,
                truncation=self.config.data.get("truncation", "left"),
            )
            # Temporary inference copies are removed before any loss sees them.
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            padded, pad_size = pad_dataproto_to_divisor(teacher_batch, self.actor_rollout_wg.world_size)
            teacher_output = self.actor_rollout_wg.compute_log_prob(padded)
            return unpad_dataproto(teacher_output, pad_size=pad_size).batch["old_log_probs"]

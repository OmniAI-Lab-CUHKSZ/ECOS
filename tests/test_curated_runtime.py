"""Exercise dependencies affected by removing unused trainers and environments."""
import ast
from collections import deque
import os
from pathlib import Path
import types

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import yaml

from response_rs.config import load_config
from verl import DataProto

ROOT = Path(__file__).resolve().parents[1]


def source_nodes(relative):
    return ast.parse((ROOT / relative).read_text(encoding='utf8')).body


def trainer_class():
    cls = next(n for n in source_nodes('verl/trainer/ppo/grsd_reflect_ray_trainer.py')
               if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
               {'__init__', '_decode_turn', '_compute_teacher_log_probs_with_priors'}]

    class Base:
        def __init__(self, config):
            self.config = config
            self.use_reference_policy = True
            self.ref_in_actor = False

    namespace = dict(RayPPOTrainer=Base, np=np, os=os, deque=deque,
                     DataProto=DataProto, DEFAULT_RS_REWARD_CLIP=3.)
    # Keep the real initializer and inherited Teacher helper after hierarchy flattening.
    cls.body = methods
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 'curated_trainer_methods', 'exec'), namespace)
    return namespace[cls.name], namespace


def test_actual_initializer_retains_method_settings_and_context_checks():
    cls, _ = trainer_class()
    config = OmegaConf.create(load_config('alfworld_3b'))
    trainer = cls(config)
    assert trainer.route_lambda == .02 and trainer.reflect_loss_coef == .1
    assert trainer.selective_actor_signal_scope == 'action_command'
    assert trainer.selective_lambda == 1. and trainer.selective_tau == .5
    assert trainer.skip_teacher_on_all_success and trainer.rs_joint_update
    config.actor_rollout_ref.rollout.max_model_len = 4096
    with pytest.raises(ValueError, match='max_model_len'):
        cls(config)
    config = OmegaConf.create(load_config('alfworld_3b'))
    config.algorithm.route.mode = 'probe'
    with pytest.raises(ValueError, match='selective_gigpo'):
        cls(config)


def test_teacher_scoring_retains_padding_and_exact_response_rows():
    cls, namespace = trainer_class()
    trainer = cls(OmegaConf.create(load_config('alfworld_3b')))
    rows = DataProto.from_dict(tensors={'responses': torch.tensor([[3, 4], [5, 6], [7, 8]])})
    calls = []

    def build(batch, row_skill_text, **kwargs):
        assert row_skill_text == ['a', 'b', 'c']
        return batch

    def score(batch):
        calls.append(len(batch))
        return DataProto.from_dict(tensors={'old_log_probs': batch.batch['responses'].float()})

    namespace['build_teacher_batch_with_text'] = build
    trainer.tokenizer = object()
    trainer.actor_rollout_wg = types.SimpleNamespace(world_size=2, compute_log_prob=score)
    result = trainer._compute_teacher_log_probs_with_priors(rows, ['a', 'b', 'c'])
    assert calls == [4]
    torch.testing.assert_close(result, rows.batch['responses'].float())


def test_textworld_adapter_configuration_survives_pruning():
    config = yaml.safe_load((ROOT / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml').read_text())
    cls = next(n for n in source_nodes('agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py')
               if isinstance(n, ast.ClassDef) and n.name == 'AlfredTWEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_env')
    calls = []

    def register_games(files, infos, **kwargs):
        calls.append(kwargs)
        return 'test-env'

    namespace = dict(textworld=types.SimpleNamespace(
        EnvInfos=lambda **kw: types.SimpleNamespace(**kw),
        gym=types.SimpleNamespace(register_games=register_games, make=lambda name: name)),
        AlfredDemangler=lambda **kw: object(), AlfredInfos=object())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 'textworld_adapter_method', 'exec'), namespace)
    env = types.SimpleNamespace(config=config, train_eval='train', use_expert=False, game_files=['task.z8'])
    assert namespace['init_env'](env, batch_size=1) == 'test-env'
    assert calls[0]['max_episode_steps'] == 50

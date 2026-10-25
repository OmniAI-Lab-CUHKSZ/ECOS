"""Exercise the real paths affected by removing dormant framework experiments."""
import ast
from collections import OrderedDict
from enum import Enum
import inspect
from pathlib import Path
import types

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor

from response_rs.config import PROFILES, load_config
from verl import DataProto
from verl.trainer.ppo import core_algos

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    tree = ast.parse((ROOT/path).read_text(encoding='utf8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('profile', PROFILES)
def test_actual_dispatcher_uses_the_unchanged_grpo_outcome_function(profile):
    class Estimator(str, Enum):
        GRPO = 'grpo'

    namespace = dict(DataProto=DataProto, core_algos=core_algos, AdvantageEstimator=Estimator)
    load_definitions('verl/trainer/ppo/ray_trainer.py', {'compute_advantage'}, namespace)
    config = load_config(profile)
    rewards = torch.zeros(16, 5)
    rewards[:, 0] = torch.tensor([0., 1., 0., 1., 1., 0., 0., 1.] + [1.]*8)
    mask = (torch.arange(5)[None, :] <= torch.arange(16)[:, None] % 5).float()
    uid = np.repeat(['first', 'all_success'], 8)
    traj = np.array([f't{i}' for i in range(16)])
    data = DataProto.from_dict(tensors={'token_level_rewards': rewards.clone(), 'response_mask': mask})
    data.non_tensor_batch.update(uid=uid, traj_uid=traj)
    result = namespace['compute_advantage'](
        data, adv_estimator=config['algorithm']['adv_estimator'],
        norm_adv_by_std_in_grpo=config['algorithm']['norm_adv_by_std_in_grpo'])
    expected, expected_returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uid, traj_index=traj,
        norm_adv_by_std_in_grpo=True)
    torch.testing.assert_close(result.batch['advantages'], expected)
    torch.testing.assert_close(result.batch['returns'], expected_returns)
    assert torch.isfinite(expected).all()
    assert not result.batch['advantages'][mask == 0].any()


def test_full_model_sync_reuse_and_rng_boundaries_use_actual_methods():
    tree = ast.parse((ROOT/'verl/workers/sharding_manager/fsdp_vllm.py').read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
               {'__enter__', '__exit__', 'update_params', 'mark_weights_dirty'}]
    for method in methods:
        method.decorator_list = []
    events = []

    class Device:
        state = torch.tensor([9], dtype=torch.uint8)

        def empty_cache(self): events.append('empty_cache')
        def current_device(self): return torch.device('cpu')
        def get_rng_state(self): return self.state.clone()
        def set_rng_state(self, value): self.state = value.clone()

    device = Device()
    namespace = dict(torch=torch, DTensor=DTensor, inspect=inspect,
                     get_torch_device=lambda: device,
                     logger=types.SimpleNamespace(info=lambda *a: None),
                     log_gpu_memory_usage=lambda *a, **kw: None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), 'real_sharding_methods', 'exec'), namespace)
    params = OrderedDict(weight=torch.tensor([[1., 2.], [3., 4.]]), bias=torch.tensor([5., 6.]))
    loaded = []

    def load_weights(items):
        loaded.append({name: tensor.clone() for name, tensor in items})
        return set(loaded[-1])

    module = types.SimpleNamespace(state_dict=lambda: params, train=lambda: events.append('train'))
    engine = types.SimpleNamespace(
        wake_up=lambda tags=None: events.append(('wake', tuple(tags or []))),
        sleep=lambda level: events.append(('sleep', level)))
    obj = types.SimpleNamespace(module=module, inference_engine=engine,
                                model_runner=types.SimpleNamespace(model=types.SimpleNamespace(load_weights=load_weights)),
                                reuse_unchanged_weights=True, _weights_dirty=True,
                                device_mesh=object(), offload_param=False, full_params=False,
                                gen_random_states=torch.tensor([1], dtype=torch.uint8),
                                weight_sync_count=0, weight_reuse_count=0)
    for method in methods:
        setattr(obj, method.name, types.MethodType(namespace[method.name], obj))
    for _ in range(2):
        obj.__enter__()
        torch.testing.assert_close(device.state, torch.tensor([1], dtype=torch.uint8))
        obj.__exit__(None, None, None)
        torch.testing.assert_close(device.state, torch.tensor([9], dtype=torch.uint8))
    assert len(loaded) == 1 and obj.weight_sync_count == 1 and obj.weight_reuse_count == 1
    for key in params: torch.testing.assert_close(loaded[0][key], params[key])
    params['weight'] = params['weight'] + 1
    obj.mark_weights_dirty()
    obj.__enter__()
    obj.__exit__(None, None, None)
    assert len(loaded) == 2 and obj.weight_sync_count == 2
    torch.testing.assert_close(loaded[1]['weight'], params['weight'])
    assert events.count(('sleep', 1)) == 3
    assert events.count(('wake', ('weights',))) == 3
    assert events.count(('wake', ('kv_cache',))) == 3


def test_local_imported_symbols_exist_after_curation():
    files = [p for p in ROOT.rglob('*.py') if not {'.git', '__pycache__', '.pytest_cache'} & set(p.parts)]
    modules = {'.'.join(p.relative_to(ROOT).with_suffix('').parts).removesuffix('.__init__'): p for p in files}
    prefix = 'agent_system.environments.env_package.webshop.webshop.'
    modules.update({name[len(prefix):]: p for name, p in list(modules.items()) if name.startswith(prefix)})
    exports = {}

    def binding_names(nodes):
        names = set()
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Import):
                names.update(a.asname or a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names if a.name != '*')
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for field in ('body', 'orelse', 'finalbody'):
                    names.update(binding_names(getattr(node, field, [])))
                for handler in getattr(node, 'handlers', []): names.update(binding_names(handler.body))
        return names

    for name, path in modules.items(): exports[name] = binding_names(ast.parse(path.read_text(encoding='utf8')).body)
    missing = []
    for name, path in modules.items():
        package = name if path.name == '__init__.py' else name.rsplit('.', 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            if not isinstance(node, ast.ImportFrom): continue
            head = '.'.join(package.split('.')[:len(package.split('.'))-node.level+1]) if node.level else ''
            target = '.'.join(x for x in (head, node.module) if x)
            if target not in exports: continue
            for alias in node.names:
                if alias.name != '*' and alias.name not in exports[target] and target+'.'+alias.name not in modules:
                    missing.append((name, target, alias.name))
    assert missing == []

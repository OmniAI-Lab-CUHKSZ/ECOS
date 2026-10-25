import importlib.util
from pathlib import Path
import zipfile

import durable_native_store


def test_raw_archive_with_nested_output_never_includes_itself_or_native_shards(tmp_path):
    attempt=tmp_path/'attempt';output=attempt/'output';checkpoint=output/'global_step_1'
    (checkpoint/'actor').mkdir(parents=True)
    (checkpoint/'actor/model_world_size_2_rank_0.pt').write_bytes(b'native payload')
    (attempt/'archive_token').write_text('test-only-secret')
    (attempt/'train.log').write_text('raw log')
    (output/'rollouts').mkdir();(output/'rollouts/step1.jsonl').write_text('{}\n')
    plan=dict(root=str(attempt),output=str(output),short=str(tmp_path/'runtime'))
    durable_native_store.checkpoint_raw(plan,checkpoint)
    with zipfile.ZipFile(checkpoint/'attempt_raw.zip') as saved:
        assert saved.testzip() is None
        assert set(saved.namelist())=={'attempt/train.log','training/rollouts/step1.jsonl'}
        assert saved.read('attempt/train.log')==b'raw log'


def test_fabric_honors_explicit_interfaces_without_cluster_lookup(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('actual_fabric',root/'verl/utils/network_fabric.py')
    fabric=importlib.util.module_from_spec(spec);spec.loader.exec_module(fabric)
    monkeypatch.setenv('NCCL_SOCKET_IFNAME','lo');monkeypatch.setenv('GLOO_SOCKET_IFNAME','lo')
    def unexpected(*args,**kwargs):raise AssertionError('Explicit interface should not trigger subnet discovery')
    monkeypatch.setattr(fabric.subprocess,'check_output',unexpected)
    configured=fabric.configure_fabric()
    assert configured['NCCL_SOCKET_IFNAME']=='lo' and configured['GLOO_SOCKET_IFNAME']=='lo'

from pathlib import Path
import shutil
import subprocess

import pytest


def test_weight_ignore_rules_preserve_framework_model_source(tmp_path):
    git=shutil.which('git')
    if git is None:pytest.skip('Git is required to exercise its actual ignore rules')
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'.gitignore').write_bytes((root/'.gitignore').read_bytes())
    subprocess.run([git,'init','--quiet',str(tmp_path)],check=True,capture_output=True)
    sources=['verl/models/transformers/monkey_patch.py',
             'agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py']
    payloads=['models/model.safetensors','assets/catalog.json','runs/checkpoint/model.pt',
              'verl/models/model.safetensors']
    result=subprocess.run([git,'check-ignore','--no-index','-z','--stdin'],cwd=tmp_path,
                          input=('\0'.join(sources+payloads)+'\0').encode('utf8'),capture_output=True)
    assert result.returncode==0,result.stderr
    assert {name.decode('utf8') for name in result.stdout.split(b'\0') if name}==set(payloads)
    assert all((root/name).is_file() for name in sources)

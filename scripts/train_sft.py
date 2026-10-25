"""Launch one R8+S1 full-parameter SFT job on selected GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from response_rs.config import PROFILES
from response_rs.sft_data import load_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=PROFILES, required=True)
    parser.add_argument('--model', required=True, type=Path, help='Local Qwen model or prior HF-format model directory')
    parser.add_argument('--gpus', required=True, help='2..8 physical GPU indices, e.g. 0,1,2,3,4,5,6,7')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--material', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Two actual GPU updates; explicitly opt in')
    parser.add_argument('--resume-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if not 2 <= len(gpus) <= 8 or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Choose 2..8 distinct nonnegative physical GPU indices')
    material = (args.material or ROOT/'data/sft'/args.config).resolve()
    groups, info = load_groups(material)
    command = [sys.executable, '-X', 'utf8', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', str(len(gpus)), '-m', 'response_rs.sft',
               '--material', str(material), '--model', str(args.model.resolve()),
               '--output', str(args.output.resolve())]
    if args.resume_from: command += ['--resume-from', str(args.resume_from.resolve())]
    if args.smoke: command += ['--smoke']
    if args.resume_test: command += ['--resume-test']
    if args.dry_run:
        print(json.dumps(dict(command=command, gpus=gpus, groups=len(groups), recipe=info), indent=2))
        return 0
    model_config=json.loads((args.model/'config.json').read_text(encoding='utf8'))
    expected_hidden=2048 if args.config.endswith('3b') else 3584
    if model_config.get('hidden_size')!=expected_hidden:
        raise ValueError('Model size does not match --config')
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                            check=True, text=True, capture_output=True)
    uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in result.stdout.splitlines()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               TRAINING_GPU_UUIDS=json.dumps([uuids[g] for g in gpus]), PYTHONUTF8='1',
               PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''),
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
    args.output.mkdir(parents=True, exist_ok=True)
    import psutil
    from durable_native_store import write
    launch_path=args.output/'launch.json'
    # An existing or unknown launch is observed through its saved output. A
    # resume uses a new output directory, so it never overwrites the old attempt.
    with launch_path.open('x',encoding='utf8') as stream:
        json.dump(dict(state='PREPARING',command=command,time=time.time()),stream)
        stream.flush();os.fsync(stream.fileno())
    child=None;code=None;error=None
    try:
        with (args.output/'train.log').open('x',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(launch_path,dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),
                                  command=command,gpus=gpus,time=time.time(),state='RUNNING'))
            print(f'SFT started; log: {args.output.resolve()/"train.log"}',flush=True)
            code=child.wait()
    except BaseException as exc:
        error=repr(exc)
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise
    finally:
        write(args.output/'exit.json',dict(exit_code=child.poll() if child else code,error=error,time=time.time()))
    return code


if __name__ == '__main__':
    raise SystemExit(main())

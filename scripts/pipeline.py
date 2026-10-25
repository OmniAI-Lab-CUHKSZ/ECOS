"""Run SFT600 then the unified RL150 recipe for any of the four profiles."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from response_rs.config import PROFILES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,choices=PROFILES)
    parser.add_argument('--base-model',required=True,type=Path)
    parser.add_argument('--gpus',required=True)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--alfworld-data',type=Path)
    parser.add_argument('--webshop-assets',type=Path)
    parser.add_argument('--stage',choices=['both','sft','rl'],default='both')
    parser.add_argument('--resume-rl',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();run=args.run_dir.resolve()
    sft=[sys.executable,'-X','utf8',str(ROOT/'scripts/train_sft.py'),'--config',args.config,
         '--model',str(args.base_model.resolve()),'--gpus',args.gpus,'--output',str(run/'sft')]
    rl=[sys.executable,'-X','utf8',str(ROOT/'scripts/train_rl.py'),'--config',args.config,
        '--model',str(run/'sft/checkpoint600'),'--gpus',args.gpus,'--run-dir',str(run/'rl')]
    if args.alfworld_data:rl+=['--alfworld-data',str(args.alfworld_data.resolve())]
    if args.webshop_assets:rl+=['--webshop-assets',str(args.webshop_assets.resolve())]
    if args.resume_rl:rl+=['--resume']
    commands=([sft] if args.stage=='sft' else [rl] if args.stage=='rl' else [sft,rl])
    if args.dry_run:
        print(json.dumps(commands,indent=2));return 0
    for command in commands:
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())

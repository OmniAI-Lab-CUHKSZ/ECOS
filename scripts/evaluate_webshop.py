"""Evaluate a frozen3B/7B Actor on seeded128-goal WebShop panels."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--asset-root',required=True,type=Path)
    parser.add_argument('--gpu',required=True,type=int)
    parser.add_argument('--seeds',default='0,1,2')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      WEBSHOP_ASSET_ROOT=str(args.asset_root.resolve()),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    from response_rs.alfworld_evaluation import read,write
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Supply distinct evaluation seeds')
    settings=dict(model=str(args.model.resolve()),asset_root=str(args.asset_root.resolve()),seeds=seeds,
                  catalog_seed=0,temperature=.4,top_p=1.,top_k=-1,max_steps=15,history=2,prompt=4096,response=512,
                  panel_size=128,pool_size=500,panel_sampler='python.random.Random(seed).sample')
    if (output/'plan.json').exists():
        if read(output/'plan.json')!=settings:raise ValueError('Existing output uses different model or settings')
    else:write(output/'plan.json',settings)
    if (output/'summary.json').exists():print(json.dumps(read(output/'summary.json'),indent=2));return
    import fcntl
    lock=(output/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from response_rs.webshop_evaluation import evaluate
    error=None;code=1
    try:
        result=evaluate(settings['model'],output,seeds);code=0
        print(json.dumps(result,indent=2))
    except BaseException as exc:
        error=repr(exc);raise
    finally:write(output/'exit.json',dict(exit_code=code,error=error,time=time.time(),optimizer_steps=0))


if __name__=='__main__':main()

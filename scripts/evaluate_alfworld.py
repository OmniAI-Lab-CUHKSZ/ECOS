"""Evaluate one frozen exported Actor on all140 seen +134 unseen tasks."""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--alfworld-data',required=True,type=Path)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--seeds',default='0,1,2')
    args=parser.parse_args()
    os.environ.update(ALFWORLD_DATA=str(args.alfworld_data.resolve()),CUDA_VISIBLE_DEVICES=str(args.gpu),
                      CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    from response_rs.alfworld_evaluation import read,write,panel,worker
    exported=read(args.model/'export.json')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in args.seeds.split(',')]
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate evaluation seeds')
    plan=dict(root=str(root),repo=str(ROOT),checkpoint=exported['source'],model=str(args.model.resolve()),
              workers=[dict(host=socket.gethostname(),gpu=args.gpu,uuid=None,seeds=seeds)])
    saved=root/'plan.json'
    if saved.exists():
        if read(saved)!=plan:raise ValueError('Existing output belongs to a different model or evaluation setup')
    else:write(saved,plan)
    if (root/'summary.json').exists():
        print(json.dumps(read(root/'summary.json'),indent=2));return
    if not (root/'panel.json').exists():panel(plan)
    worker(plan,0)
    rows=[read(root/f'seed{seed}/summary.json') for seed in seeds]
    summary=dict(seeds=rows,optimizer_steps=0)
    for split in ['seen','unseen','total']:
        values=[r[split]['accuracy'] for r in rows]
        summary[split]=dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None)
    write(root/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

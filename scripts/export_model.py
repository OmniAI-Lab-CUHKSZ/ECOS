"""Export a complete native RL Model to frozen HF BF16 weights."""
import argparse
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();checkpoint=args.checkpoint.resolve()
    shards=list((checkpoint/'actor').glob('model_world_size_*_rank_*.pt'))
    identities=[tuple(map(int,re.fullmatch(r'model_world_size_(\d+)_rank_(\d+)\.pt',p.name).groups())) for p in shards]
    if not identities:raise FileNotFoundError('The native Model payloads are required, not only DIRECT_SAVED pointers')
    worlds={world for world,rank in identities}
    if len(worlds)!=1:raise ValueError('Mixed shard topology')
    world=worlds.pop()
    if {rank for _,rank in identities}!=set(range(world)):raise ValueError('Incomplete native Model ranks')
    step=int((checkpoint/'CHECKPOINT_COMPLETE').read_text().strip())
    from response_rs.alfworld_evaluation import export
    export(dict(checkpoint=str(checkpoint),model=str(args.output.resolve()),source_step=step,world_size=world))


if __name__=='__main__':main()

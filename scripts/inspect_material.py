"""Count and validate the four packaged SFT datasets without running a model."""
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from response_rs.config import PROFILES
from response_rs.sft_data import load_groups


def main():
    result={}
    for name in PROFILES:
        path=ROOT/'data/sft'/name
        groups, info=load_groups(path)
        natural=[]
        for file in sorted((path/'natural').glob('trajectories-*.jsonl.gz')):
            with gzip.open(file,'rt',encoding='utf8') as stream:
                natural.extend(json.loads(line) for line in stream)
        assert [n['group'] for n in natural]==[g['group'] for g in groups]
        for group, sample in zip(groups,natural):
            trajectories={int(t['trajectory']):t for t in sample['trajectories']}
            assert set(trajectories)==set(range(8))
            for r in group['reflections']:
                assert bool(r['success'])==bool(trajectories[int(r['trajectory'])]['success'])
        result[name]=dict(tasks=len(groups),reflections=sum(len(g['reflections']) for g in groups),
                          skills=sum(len(g['skills']) for g in groups),
                          natural_trajectories=sum(len(n['trajectories']) for n in natural),
                          max_sequence_tokens=info['max_observed_sequence_tokens'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

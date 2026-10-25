"""Read the retained, tokenized R8+S1 material; never silently retokenize it."""
import gzip
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_groups(material):
    material = Path(material)
    info = read_json(material/'metadata.json')
    groups = []
    for path in sorted(material.glob('groups-*.jsonl.gz')):
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            groups.extend(json.loads(line) for line in stream if line.strip())
    if len(groups) != info['groups'] or len({g['group'] for g in groups}) != len(groups):
        raise ValueError('Missing or duplicated SFT task groups')
    for group in groups:
        if len(group['reflections']) != 8 or len(group['skills']) != 1:
            raise ValueError('Each task needs exactly eight Reflections and one Skill')
        if {int(r['trajectory']) for r in group['reflections']} != set(range(8)):
            raise ValueError('Eight distinct natural trajectories are required')
        for kind, rows in [('reflection', group['reflections']), ('skill', group['skills'])]:
            for row in rows:
                if row['kind'] != kind or row['group'] != group['group']:
                    raise ValueError('SFT row alignment differs from its task group')
                prompt, response = row['prompt_ids'], row['response_ids']
                if not prompt or not response or len(prompt) + len(response) > 32768:
                    raise ValueError('Empty or overlength SFT sequence; truncation is disabled')
                if any(type(t) is not int or t < 0 for t in prompt + response):
                    raise ValueError('Invalid token ID')
                if row.get('exit_code', 0) or row.get('tool_calls', 0):
                    raise ValueError('The material contains an incomplete annotation')
    if len(groups) != 200:
        raise ValueError('The released 600-update recipe uses all 200 tasks')
    return groups, info


def slots(samples, rank, world):
    """Each real text appears once; zero-weight padding aligns FSDP collectives."""
    for start in range(0, len(samples), world):
        index = start + rank
        if index < len(samples):
            row = samples[index]
            yield row, .5 / (8 if row['kind'] == 'reflection' else 1)
        else:
            yield samples[-1], 0.

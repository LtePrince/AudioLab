"""Verify historical Eltaw outputs, then compare CPU-cache versus original E20."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/anchored_ar.py').exists())
sys.path.insert(0, str(ROOT))
import json
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json

ASSET = Path(__file__).resolve().parent
OUT = ROOT / 'out/ar_early_replay_20261102'


def read(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    paths = {
        'A0920': 'out/runtime_delivery_20261026/evaluation/4k-audio-e0920-greedy/00.json',
        'S1000': 'out/runtime_delivery_20261026/evaluation/4k-soft-e1000-greedy/00.json',
        'A1400': 'out/ar_saturation_20261023/continued1400/audio/evaluation-e1400/fixed15/00.json',
        'S1400': 'out/ar_saturation_20261023/continued1400/soft/evaluation-e1400/fixed15/00.json',
    }
    result = dict(schema='audiolab-early-replay-sanity-v1', eltaw={}, original_community_replay=[])
    for name, path in paths.items():
        old = read(ROOT / path)['prediction']
        current_path = OUT / name / 'Eltaw_IN.json'
        current = read(current_path)['prediction']
        fields = ('notes', 'onsets_seconds', 'status', 'types', 'termination')
        exact = {key: current[key] == old[key] for key in fields}
        result['eltaw'][name] = dict(previous=path, previous_sha256=sha256(ROOT / path),
            replay=str(current_path.relative_to(ROOT)), replay_sha256=sha256(current_path), exact=exact)
    plan = read(ASSET / 'selection.json')
    for name in ('A1400', 'S1400'):
        for song in plan['songs']:
            previous = ROOT / f'out/phira_generalization10_20261029/{song["id"]}/E14-{name[0]}/result.json'
            current = OUT / name / f'{song["id"]}.json'
            a, b = read(previous), read(current)
            result['original_community_replay'].append(dict(candidate=name, song_id=song['id'],
                exact_notes=a['prediction']['notes']==b['prediction']['notes'],
                original_f1=a['metrics']['onset_f1_50ms'], replay_f1=b['metrics']['onset_f1_50ms']))
    result['all_four_eltaw_exact'] = all(all(r['exact'].values()) for r in result['eltaw'].values())
    write_json(ASSET / 'sanity.json', result)
    print(json.dumps(dict(all_four_eltaw_exact=result['all_four_eltaw_exact'],
        exact_original_community=sum(r['exact_notes'] for r in result['original_community_replay']),
        community_checks=len(result['original_community_replay']))))

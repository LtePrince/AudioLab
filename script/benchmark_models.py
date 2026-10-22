"""CPU benchmark replay: import existing audits or score portable JSON inputs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from src.model_benchmark import SCHEMA, evaluate

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def probe_cases(manifest):
    result = []
    for source in manifest['rows']:
        c = {k: deepcopy(source[k]) for k in ('id', 'source_group', 'label_source', 'audio_sha256',
                                             'duration_seconds', 'coverage', 'events_seconds')}
        c.update(split='constructed_development', contract='exact_onsets' if c['events_seconds'] else 'silence',
                 clock=None, silent_regions=[], weak_events_seconds=[t for t, g in
                     zip(source['events_seconds'], source['gains']) if g < .1])
        # Fixed guard around deliberately edited one-second samples.
        cursor, duration = 0., c['duration_seconds']
        for a, b in source['active_intervals']:
            a, b = max(0., a-.1), min(duration, b+.1)
            if a > cursor:
                c['silent_regions'].append([cursor, a])
            cursor = max(cursor, b)
        if cursor < duration:
            c['silent_regions'].append([cursor, duration])
        result.append(c)
    return result


def decode_notes(generation, mode, bpm, offset):
    from src.data.chart_tokenizer import ChartTokenizer, bin_to_hit
    from src.pattern_diagnostics import LANE_X, KINDS
    tk = ChartTokenizer(4 if mode == '4k' else 64)
    tokens = generation['tokens']
    residuals = generation.get('residuals', {})
    offsets = {}
    for index, (token, tick) in enumerate(zip(tokens, tk.ticks_per_position(tokens))):
        if tk.kind(token) == 'NOTE':
            lane = tk.value(token)//4
            offsets[tick, lane] = float(residuals.get(str(index), .5))
    scale = 60/(32*bpm)
    return [dict(time_seconds=n.tick*scale+offset,
                 end_seconds=(n.tick+n.dur)*scale+offset,
                 position=n.lane, type=KINDS[n.type],
                 screen_x=LANE_X[n.lane] if mode == '4k' else bin_to_hit(n.lane, offsets.get((n.tick,n.lane), .5)))
            for n in tk.decode_tokens(tokens, strict=True)]


def import_audits(audit_root, cache_root):
    """Load only three cached references, never weights/features for inference."""
    import torch
    from src.data.chart_tokenizer import ChartTokenizer
    probes = probe_cases(read(audit_root/'onsets/fixtures/fixtures.json'))
    if probes != read(ROOT/'benchmarks/v1/probes.json'):
        raise ValueError('probe labels/audio hashes differ from frozen benchmark v1')
    cache = read(cache_root/'cache.json')
    music = {'4k': [], 'freex': []}
    for row in cache['rows']:
        if row['id'] not in ('Eltaw_IN', '-SURREALISM-_IN', 'ABYSS MOTION 192.333_IN'):
            continue
        if file_hash(cache_root/row['file']) != row['sha256']:
            raise ValueError('cache hash mismatch')
        sample = torch.load(cache_root/row['file'], map_location='cpu', weights_only=True)
        for mode in music:
            route = sample['routes'][mode]
            meta = route['reference']
            bpm, offset = meta['timebase_bpm'], meta['offset_seconds']
            tk = ChartTokenizer(4 if mode == '4k' else 64)
            notes = tk.decode_tokens(route['tokens'].tolist(), strict=True)
            music[mode].append(dict(id=row['id'], source_group=row['id'], label_source='chart_onset',
                split='seen_training', audio_sha256=row['audio']['sha256'],
                duration_seconds=sample['duration'], coverage=[[0., sample['duration']]],
                events_seconds=sorted({n.tick*60/(32*bpm)+offset for n in notes}),
                clock=dict(bpm=bpm, beat_zero_seconds=offset, source='reference-chart; not independently beat-reviewed', verified=False),
                source_cache_sha256=row['sha256']))
        del sample
    runs = []
    onset_path = audit_root/'onsets/report.json'
    onset = read(onset_path)
    runs.append(dict(id='onset_frontend', cases=deepcopy(probes),
        metadata=dict(stage='acoustic_onset', mode='shared', model=onset['model'],
                      selection=onset['selection'], artifact_sha256=file_hash(onset_path)),
        predictions={r['id']: dict(status='completed', onsets_seconds=r['selected_times_seconds']) for r in onset['rows']}))
    for mode in music:
        for variant in ('local', 'global', 'time_bias'):
            path = audit_root/f'{mode}-scratch'/variant/'report.json'
            report = read(path)
            for greedy in (True, False):
                cases = deepcopy(probes+music[mode])
                by_id = {c['id']: c for c in cases}
                predictions = {}
                for row in report['rows']:
                    if 'generation' in row:
                        gs = row['generation']
                    elif row.get('condition') == 'predicted':
                        gs = [row]
                    else:
                        continue  # Oracle/zero are separate ablations, not product inputs.
                    for g in gs:
                        if g['greedy'] != greedy:
                            continue
                        case = by_id[row['id']]
                        clock = case['clock']
                        pred = {k: g[k] for k in ('status', 'onsets_seconds', 'eos_clock_seconds', 'error') if k in g}
                        if g['status'] == 'completed':
                            pred['notes'] = decode_notes(g, mode, clock['bpm'] if clock else 120.,
                                                         clock['beat_zero_seconds'] if clock else 0.)
                        predictions[row['id']] = pred
                runs.append(dict(id=f'{mode}/{variant}/{"greedy" if greedy else "seed42"}', cases=cases,
                    metadata=dict(stage='raw_layout', mode=mode, attention=variant, greedy=greedy, seed=42,
                        temperature=1., top_p=.95, condition='predicted', checkpoint=report['checkpoint'],
                        artifact_sha256=file_hash(path),
                        evidence_gaps=['seeds 43/44 not run', 'no independent natural test',
                                       'no designated pattern continuations', 'no final Phira playtest']),
                    predictions=predictions))
    return dict(schema=SCHEMA, runs=runs)


def markdown(report):
    lines = ['# AudioLab benchmark v1', '',
        '节奏匹配仅加分，不设固定排布／比例；此入口只评分输入产物，不执行训练或推理。', '',
        '| 模型／采样 | 已评分／应评 | 基础检查 | 三音 F1@50ms | 弱音召回 | 静音误出音 | 自然曲 F1@50ms（训练曲） | 自然曲节奏奖励／起点 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    fmt = lambda x: '—' if x is None else f'{x:.3f}'
    for run in report['runs']:
        by_id = {r['id']: r for r in run['rows']}
        three = by_id.get('piano_three', {}).get('events', {}).get('at_50ms', {}).get('f1')
        weak = by_id.get('piano_weak_middle', {}).get('weak_recall_50ms')
        silence = by_id.get('silence', {}).get('events', {}).get('prediction_count_including_outside')
        natural = [r for r in run['rows'] if r['split'] == 'seen_training' and r['status'] == 'scored']
        f1 = sum(r['events']['at_50ms']['f1'] for r in natural)/len(natural) if natural else None
        bonuses = [r['rhythm'].get('bonus_per_onset') for r in natural]
        bonuses = [v for v in bonuses if v is not None]
        bonus = sum(bonuses)/len(bonuses) if bonuses else None
        scored = sum(r['status'] == 'scored' for r in run['rows'])
        lines.append(f'| {run["id"]} | {scored}/{len(run["rows"])} | {run["contract_status"]} | {fmt(three)} | {fmt(weak)} | {silence if silence is not None else "—"} | {fmt(f1)} | {fmt(bonus)} |')
    lines.extend(['', '空谱的节奏奖励为缺测，不是满分。自然曲的时基来自参考谱，尚未独立审核；',
                  '所有 300 首已训练，表中自然曲分数只能用于开发对照。缺项、错误及分段数值见 JSON。',
                  '基础检查通过不代表采样稳定性、独立测试或 Phira 试玩通过。',
                  f'协议 SHA256：`{report["protocol_sha256"]}`', ''])
    return '\n'.join(lines)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'benchmarks/v1/protocol.json')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path, help='portable cases/predictions JSON')
    group.add_argument('--audit-root', type=Path, help='existing onset/attention audit directory')
    parser.add_argument('--cache', type=Path, default=ROOT/'out/full_corpus_v1/cache')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='previous report; requires identical cases/configuration')
    parser.add_argument('--fail-on-contract', action='store_true', help='CI: fail on failed/incomplete base contracts')
    args = parser.parse_args()
    bundle = read(args.input) if args.input else import_audits(args.audit_root, args.cache)
    protocol = read(args.protocol)
    report = evaluate(bundle, protocol)
    report['scorer_source_sha256'] = {p: file_hash(ROOT/p) for p in (
        'script/benchmark_models.py', 'src/model_benchmark.py', 'src/pattern_diagnostics.py',
        'src/audio_understanding/evaluation.py', 'src/audio_understanding/annotations.py')}
    if args.baseline:
        old = read(args.baseline)
        if old['protocol_sha256'] != report['protocol_sha256']:
            raise ValueError('baseline protocol mismatch')
        before = {r['id']: r for r in old['runs']}
        for run in report['runs']:
            prev = before[run['id']]
            if prev['cases_sha256'] != run['cases_sha256']:
                raise ValueError(f'baseline cases mismatch: {run["id"]}')
            keys = ('stage', 'mode', 'greedy', 'seed', 'temperature', 'top_p', 'condition', 'selection')
            if any(prev['metadata'].get(k) != run['metadata'].get(k) for k in keys):
                raise ValueError(f'baseline generation/selection configuration mismatch: {run["id"]}')
            run['baseline_contract_status'] = prev['contract_status']
            prior = {r['id']: r for r in prev['rows']}
            for row in run['rows']:
                p = prior[row['id']]
                if p['status'] == row['status'] == 'scored':
                    row['delta_f1_50ms'] = row['events']['at_50ms']['f1']-p['events']['at_50ms']['f1']
    report['replay_resources'] = dict(device='cpu', elapsed_seconds=time.monotonic()-started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)/(1024**3),
        note='CPU scoring only, not model inference performance')
    write(args.output/'inputs.json', bundle)
    write(args.output/'report.json', report)
    (args.output/'report.md').write_text(markdown(report))
    print(markdown(report))
    if args.fail_on_contract and any(r['contract_status'] != 'pass' or r['coverage_status'] != 'complete' for r in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

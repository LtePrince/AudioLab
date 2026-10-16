"""Paired full-chart evaluation of original, NMS35 and global-rhythm outputs.

Requires all songs and completed generation reports. Raw outputs stay intact;
the same static-line repairs are applied to every variant. Metrics retain the
legacy multi-note greedy matching so they remain comparable with timing_v1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from script.diagnose_decoder import chart_stats, sha256_file
from script.eval_onset_f1 import f1, onsets
from script.repair_generated_charts import repair_chart
from script.train_models import write_json
from src.resource_budget import ResourceBudget, add_resource_args


METRICS = ['f1_25', 'f1_50', 'f1_93', 'position_f1_93', 'density_ratio',
           'precision_25', 'recall_25', 'shifted_f1_25']
VARIANTS = ['original', 'nms35', 'global_v2']
MODELS = ['decoder', 'decoder_freex']


def quality(generated, reference, free):
    scores = {f'f1_{key}': f1(generated, reference, tolerance, False)
              for key, tolerance in [(25, 25), (50, 50), (93, 92.9)]}
    scores['position_f1_93'] = f1(generated, reference, 92.9, not free, .05 if free else None)
    scores['density_ratio'] = len(generated) / max(len(reference), 1)
    hits = scores['f1_25'] * (len(generated) + len(reference)) / 2
    scores['precision_25'] = hits / max(len(generated), 1)
    scores['recall_25'] = hits / max(len(reference), 1)
    scores['shifted_f1_25'] = sum(f1([(t+s, lane, kind, x) for t, lane, kind, x in generated],
        reference, 25, False) for s in [500, -400]) / 2
    return scores


def audit_chart(chart, duration, free):
    if chart.get('formatVersion') != 3 or len(chart.get('judgeLineList', [])) != 1:
        raise ValueError('expected generated Phigros v3 single-line chart')
    line = chart['judgeLineList'][0]
    bpm, offset = line['bpm'], chart['offset']
    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(offset) or line['notesBelow']:
        raise ValueError('invalid timing or unsupported below-line output')
    result = dict(onsets_outside_audio=0, hold_ends_outside_audio=0,
                  invalid_positions=0, invalid_durations=0)
    times = []
    for note in line['notesAbove']:
        if note['type'] not in (1, 2, 3, 4) or not all(math.isfinite(note[k]) for k in
                                                    ['time', 'holdTime', 'positionX']):
            raise ValueError('invalid or nonfinite note fields')
        start = note['time'] * 60 / (32 * bpm) + offset
        times.append(start)
        result['onsets_outside_audio'] += not (0 <= start < duration)
        hold = note['type'] == 3
        result['invalid_durations'] += (note['holdTime'] < 1 if hold else note['holdTime'] != 0)
        result['hold_ends_outside_audio'] += hold and start + note['holdTime'] * 60 / (32*bpm) > duration + 1e-7
        x = .5 + note['positionX'] * .05625
        result['invalid_positions'] += (not .05-1e-8 <= x <= .95+1e-8 if free else
            min(abs(note['positionX'] - lane) for lane in [-3.75, -1.25, 1.25, 3.75]) > 1e-7)
    counts = Counter(times)
    result['max_chord_size'] = max(counts.values(), default=0)
    result['chords_over_two'] = sum(v > 2 for v in counts.values())
    times.sort()
    left = peak = 0
    for right, time in enumerate(times):
        while time - times[left] >= 1:
            left += 1
        peak = max(peak, right-left+1)
    result['peak_notes_per_1s'] = peak
    return result


def paired_delta(rows, metric, candidate='global_v2', baseline='nms35', seed=42):
    values = np.array([r['variants'][candidate]['raw'][metric] -
                       r['variants'][baseline]['raw'][metric] for r in rows])
    if not len(values):
        raise ValueError('paired comparison requires songs')
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return dict(mean_delta=float(values.mean()), median_delta=float(np.median(values)),
                ci95=np.quantile(boot, [.025, .975]).tolist(),
                wins=int((values > 1e-12).sum()), ties=int((abs(values) <= 1e-12).sum()),
                losses=int((values < -1e-12).sum()),
                worst=[dict(song=rows[i]['song'], delta=float(values[i]))
                       for i in np.argsort(values)[:5]])


def check_generation(report_path, songs, expected, seed):
    report = json.loads(Path(report_path).read_text())
    if not report.get('finished'):
        raise ValueError(f'generation is incomplete: {report_path}')
    config = report['config']
    required = dict(seed=seed, top_ps='0.95', sampling_policy='flat', teacher_only=False,
                    skeleton_source='transcriber', snap_ticks=0,
                    onset_threshold=expected[0], onset_min_interval_ms=expected[1])
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError(f'generation settings differ from the frozen comparison: {report_path}')
    for model in MODELS:
        records = report['models'].get(model, [])
        if len(records) != len(songs) or {r['song'] for r in records} != set(songs):
            raise ValueError(f'comparison requires exactly all listed songs for {model}')
        for record in records:
            chart = Path(report_path).parent / model / 'p0.95' / (record['song'] + '.json')
            expected_hash = record['samples']['0.95'].get('sha256')
            if not expected_hash or sha256_file(chart) != expected_hash:
                raise ValueError(f'generated chart hash missing or changed: {chart}')
    for path, digest in report['fingerprints'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'generation input changed: {path}')
    return report


def aggregate(rows, variant, stage):
    samples = [r['variants'][variant][stage] for r in rows]
    result = {key: float(np.mean([r[key] for r in samples])) for key in METRICS}
    for key in ['notes', 'hold_conflicts', 'close_chords']:
        result['total_' + key] = sum(r[key] for r in samples)
    result['types_diagnostic_only'] = [sum(r['types'][i] for r in samples) for i in range(4)]
    for key in ['onsets_outside_audio', 'hold_ends_outside_audio', 'invalid_positions', 'invalid_durations']:
        result[key] = sum(r['audit'][key] for r in samples)
    result['songs_beating_shifted_control'] = sum(r['f1_25'] > r['shifted_f1_25'] for r in samples)
    result['max_chord_size'] = max(r['audit']['max_chord_size'] for r in samples)
    result['max_peak_notes_per_1s'] = max(r['audit']['peak_notes_per_1s'] for r in samples)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', default='data/val.txt')
    p.add_argument('--original-root', default='checkpoints/timing_v1')
    p.add_argument('--nms-report', default='out/decoder_nms35_val30/report.json')
    p.add_argument('--global-report', default='out/decoder_global_v2_refined_val30/report.json')
    p.add_argument('--output', default='out/full_chart_comparison_val30')
    p.add_argument('--seed', type=int, default=42)
    add_resource_args(p)
    args = p.parse_args()
    output = Path(args.output)
    input_roots = [Path(args.original_root), Path(args.nms_report).parent, Path(args.global_report).parent]
    if any(output.resolve() == root.resolve() for root in input_roots):
        p.error('comparison output must differ from all generation directories')
    output.mkdir(parents=True, exist_ok=True)
    budget = ResourceBudget(args, torch.device('cpu'), output)
    base = Path(args.list).resolve().parent
    entries = []
    for row in Path(args.list).read_text().splitlines():
        if row.strip() and not row.startswith('#'):
            chart, audio, *_ = row.split(',')
            entries.append((Path(chart).parent.name, base / chart, base / audio))
    songs = [entry[0] for entry in entries]
    if not songs or len(set(songs)) != len(songs):
        raise ValueError('data list must contain unique song names')
    reports = {name: check_generation(path, songs, settings, args.seed)
               for name, path, settings in [('nms35', args.nms_report, (.7, 35)),
                   ('global_v2', args.global_report, (.25, 25))]}
    for report in reports.values():
        if Path(report['config']['data_list']).resolve() != Path(args.list).resolve():
            raise ValueError('comparison and generation data lists differ')
    # Decoder weights must match between variants; only the skeleton changes.
    for model in MODELS:
        hashes = []
        for report in reports.values():
            path = str((Path(report['config']['checkpoint_root']) / model / 'decoder_best.pt').resolve())
            hashes.append(report['fingerprints'][path])
        original_ckpt = Path(args.original_root) / model / 'decoder_best.pt'
        if hashes[0] != hashes[1] or sha256_file(original_ckpt) != hashes[0]:
            raise ValueError('decoder checkpoint differs between variants')
    original_manifest = Path(args.original_root) / 'run.json'
    legacy = json.loads(original_manifest.read_text())
    for model in MODELS:
        stage = legacy['stages']['generate_' + model]
        command = stage['command']
        if stage['status'] != 'complete' or command[command.index('--seed')+1] != str(args.seed):
            raise ValueError('original generation is incomplete or uses a different seed')
        if (command[command.index('--onset-threshold')+1] != '0.7' or
            Path(command[command.index('--list')+1]).resolve() != Path(args.list).resolve()):
            raise ValueError('original generation settings or data list differ')
        for path in [Path(args.original_root)/model/'decoder_best.pt',
                     Path(args.original_root)/'transcriber/transcriber_best.pt']:
            if path.stat().st_mtime > stage['started']:
                raise ValueError('original checkpoint changed after generation')
    directories = {model: dict(original=Path(args.original_root) / ('eval_' + model),
            nms35=Path(args.nms_report).parent / model / 'p0.95',
            global_v2=Path(args.global_report).parent / model / 'p0.95') for model in MODELS}
    # Fail before computing aggregates if even one expected chart is missing.
    for dirs in directories.values():
        for directory in dirs.values():
            for song in songs:
                if not (directory / (song + '.json')).is_file():
                    raise ValueError(f'missing chart: {directory / song}')
    result = dict(config=vars(args), complete=False, songs=len(songs), models={},
        provenance=dict(data_list_sha256=sha256_file(args.list),
            original_manifest_sha256=sha256_file(original_manifest),
            original_verification='legacy_manifest_and_checkpoint_mtime',
            generation_reports={name:dict(config=r['config'], fingerprints=r['fingerprints'], reuse=r['reuse'])
                                for name, r in reports.items()}))
    for model in MODELS:
        free = model == 'decoder_freex'
        rows = []
        result['models'][model] = dict(songs=rows)
        for song, reference, audio in entries:
            budget.check(force=True)
            if free:
                reference = Path('data/gameplay_charts') / (song + '.json')
            ref, duration = onsets(reference), sf.info(audio).duration
            row = dict(song=song, reference_notes=len(ref), reference_sha256=sha256_file(reference),
                       audio_seconds=duration, variants={})
            for variant, directory in directories[model].items():
                path = directory / (song + '.json')
                chart = json.loads(path.read_text())
                raw = onsets(path)
                raw_stats = {**chart_stats(path), **quality(raw, ref, free),
                             'audit': audit_chart(chart, duration, free)}
                repaired, repairs = repair_chart(chart, duration, 'freex' if free else '4k')
                target = output / 'playable' / variant / model / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                write_json(target, repaired)
                fixed = onsets(target)
                if [n[0] for n in raw] != [n[0] for n in fixed]:
                    raise ValueError('repair changed onset times or note counts')
                row['variants'][variant] = dict(path=str(path), sha256=sha256_file(path), raw=raw_stats,
                    repairs=repairs, playable_path=str(target), playable_sha256=sha256_file(target),
                    playable={**chart_stats(target), **quality(fixed, ref, free),
                              'audit': audit_chart(repaired, duration, free)})
            rows.append(row)
            result['resources'] = budget.log_state()
            write_json(output / 'report.json', result)
            print(f'[compare] {model} {len(rows)}/{len(entries)} {song}', flush=True)
        result['models'][model]['summary'] = {variant: {stage: aggregate(rows, variant, stage)
                         for stage in ['raw', 'playable']} for variant in VARIANTS}
        result['models'][model]['paired'] = {f'{candidate}_minus_{baseline}': {
            metric: paired_delta(rows, metric, candidate, baseline, args.seed)
            for metric in ['f1_25', 'f1_50', 'f1_93', 'position_f1_93', 'density_ratio']}
            for candidate, baseline in [('nms35', 'original'), ('global_v2', 'original'), ('global_v2', 'nms35')]}
    result['complete'] = True
    write_json(output / 'report.json', result)
    print(json.dumps({model: r['summary'] for model, r in result['models'].items()}), flush=True)


if __name__ == '__main__':
    main()

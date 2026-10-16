"""Compare explicitly selected contact profiles on every listed full song.

Separates decoding changes from export projection. Input reports and charts are
hash checked; generation remains immutable. Passing the engineering gate means
candidate for playtesting, not human-validated playability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from script.compare_full_charts import audit_chart, quality, paired_delta
from script.diagnose_decoder import chart_stats, sha256_file
from script.eval_onset_f1 import onsets
from script.repair_generated_charts import repair_chart
from script.train_models import write_json
from src.data.fourkey_contacts import constrain_4k_contacts
from src.data.free_contacts import constrain_free_contacts
from src.resource_budget import ResourceBudget, add_resource_args


def checked_directory(path, model, songs, data_list):
    path = Path(path)
    report = json.loads(path.read_text())
    if not report.get('finished'):
        raise ValueError('generation report must be finished')
    if Path(report['config']['data_list']).resolve() != Path(data_list).resolve():
        raise ValueError('generation data list differs')
    records = report['models'].get(model, [])
    if len(records) != len(songs) or {r['song'] for r in records} != set(songs):
        raise ValueError('generation must contain exactly all listed songs')
    if report['config']['top_ps'] != '0.95':
        raise ValueError('expected fixed top-p 0.95')
    for source, expected in report['fingerprints'].items():
        if sha256_file(source) != expected:
            raise ValueError(f'generation input changed: {source}')
    directory = path.parent / model / 'p0.95'
    for record in records:
        target = directory / (record['song']+'.json')
        if sha256_file(target) != record['samples']['0.95']['sha256']:
            raise ValueError(f'generated chart changed: {target}')
    return directory, report


def unique_times(chart):
    return {n['time']*60/(32*line['bpm'])+chart.get('offset', 0)
            for line in chart['judgeLineList'] for n in line['notesAbove']+line['notesBelow']}


def score(path, chart, reference, duration, free):
    return {**quality(onsets(path), reference, free), **chart_stats(path),
            'audit': audit_chart(chart, duration, free)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-report', default='out/decoder_nms35_val30/report.json')
    p.add_argument('--baseline-report', default='out/decoder_nms35_val30/report.json')
    p.add_argument('--list', default='data/val.txt')
    p.add_argument('--model', choices=['decoder', 'decoder_freex'], required=True)
    p.add_argument('--contacts', type=int, nargs='+', default=[2, 4, 6])
    p.add_argument('--output', required=True)
    add_resource_args(p)
    args = p.parse_args()
    if any(not 1 <= n <= 10 for n in args.contacts) or len(set(args.contacts)) != len(args.contacts):
        p.error('contacts must be distinct integers in [1, 10]')
    base = Path(args.list).resolve().parent
    entries = [line.strip().split(',') for line in Path(args.list).read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    songs = [Path(row[0]).parent.name for row in entries]
    if not songs or len(songs) != len(set(songs)):
        raise ValueError('nonempty unique song list required')
    source, generation = checked_directory(args.generation_report, args.model, songs, args.list)
    control, baseline = checked_directory(args.baseline_report, args.model, songs, args.list)
    settings = ['seed', 'sampling_policy', 'onset_threshold', 'onset_min_interval_ms',
                'skeleton_source', 'snap_ticks']
    if any(generation['config'][k] != baseline['config'][k] for k in settings):
        raise ValueError('comparison requires the same skeleton and sampling settings')
    if generation['fingerprints'] != baseline['fingerprints']:
        raise ValueError('comparison requires the same weights and list')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    budget = ResourceBudget(args, torch.device('cpu'), output)
    free = args.model == 'decoder_freex'
    profiles = args.contacts if free else [4]
    rows = []
    report = dict(config=vars(args), complete=False, started=time.time(), songs=rows,
        provenance={name: dict(path=str(path), sha256=sha256_file(path)) for name, path in
                    [('generation', args.generation_report), ('baseline', args.baseline_report), ('list', args.list)]})
    for song, (chart_path, audio_path, *_) in zip(songs, entries):
        budget.check(force=True)
        duration = sf.info(str(base/audio_path)).duration
        reference = Path('data/gameplay_charts', song+'.json') if free else base/chart_path
        ref = onsets(reference)
        raw_path, baseline_path = source/(song+'.json'), control/(song+'.json')
        raw, base_raw = [json.loads(path.read_text()) for path in [raw_path, baseline_path]]
        # Static format/line validation is shared with the previous comparison.
        legacy, _ = repair_chart(raw, duration, 'freex' if free else '4k')
        old, _ = repair_chart(base_raw, duration, 'freex' if free else '4k')
        row = dict(song=song, reference_sha256=sha256_file(reference), variants={})
        row['variants']['raw'] = dict(raw=score(raw_path, raw, ref, duration, free), sha256=sha256_file(raw_path))
        original_starts, decoded_starts = unique_times(base_raw), unique_times(raw)
        row['decode_onset_retention'] = len(original_starts & decoded_starts)/max(1, len(original_starts))
        for name, chart in [('baseline', old), ('legacy', legacy)]:
            path = output/name/(song+'.json'); path.parent.mkdir(exist_ok=True)
            write_json(path, chart)
            row['variants'][name] = dict(raw=score(path, chart, ref, duration, free))
        for count in profiles:
            line = raw['judgeLineList'][0]
            end_tick = (duration-raw.get('offset', 0))*32*line['bpm']/60
            if free:
                notes, changes = constrain_free_contacts(line['notesAbove'], end_tick,
                                                          line['bpm'], max_contacts=count)
            else:
                notes, changes = constrain_4k_contacts(line['notesAbove'], end_tick, line['bpm'])
            chart = {**raw, 'judgeLineList': [{**line, 'notesAbove': notes}]}
            name = f'contacts{count}'
            path = output/name/(song+'.json'); path.parent.mkdir(exist_ok=True)
            write_json(path, chart)
            write_json(path.with_suffix('.contacts.json'), changes)
            starts = unique_times(chart)
            row['variants'][name] = dict(raw=score(path, chart, ref, duration, free),
                changes={k:v for k,v in changes.items() if k not in ['contacts', 'source_indices', 'dropped_source_indices']},
                onset_retention_vs_baseline=len(starts & original_starts)/max(1, len(original_starts)),
                path=str(path), sha256=sha256_file(path))
        rows.append(row)
        print(f'[{len(rows)}/{len(songs)}] {song}', flush=True)
        write_json(output/'report.json', report)
    summary = {}
    report['summary'] = summary
    for name in rows[0]['variants']:
        metrics = [row['variants'][name]['raw'] for row in rows]
        data = {k:float(np.mean([m[k] for m in metrics])) for k in
                ['f1_25', 'f1_93', 'position_f1_93', 'density_ratio']}
        data['notes'] = sum(m['notes'] for m in metrics)
        data['max_chord_size'] = max(m['audit']['max_chord_size'] for m in metrics)
        data['static_errors'] = sum(m['hold_conflicts']+m['close_chords']+sum(m['audit'][k] for k in
            ['onsets_outside_audio', 'hold_ends_outside_audio', 'invalid_positions', 'invalid_durations']) for m in metrics)
        summary[name] = data
        if not name.startswith('contacts'):
            continue
        changes = [row['variants'][name]['changes'] for row in rows]
        for k in ['dropped_notes', 'dropped_onsets', 'moved_notes', 'shortened_holds', 'holds_to_taps']:
            data[k] = sum(c[k] for c in changes)
        data['mean_unique_onset_retention'] = float(np.mean([c['unique_onset_retention'] for c in changes]))
        data['mean_onset_retention_vs_baseline'] = float(np.mean([row['variants'][name]['onset_retention_vs_baseline'] for row in rows]))
        data['contact_violations'] = sum(sum(c['audit'].values()) for c in changes)
        data['paired_f1_25'] = paired_delta(rows, 'f1_25', candidate=name, baseline='baseline')
        data['engineering_gate_passed'] = (data['static_errors'] == 0 and data['contact_violations'] == 0
            and data['mean_onset_retention_vs_baseline'] >= .95 and data['paired_f1_25']['mean_delta'] >= -.02)
    report.update(complete=True, finished=time.time(), resources=budget.log_state())
    write_json(output/'report.json', report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

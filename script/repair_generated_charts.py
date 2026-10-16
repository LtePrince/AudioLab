"""Repair existing generated static-line charts without rerunning the models.

Inputs remain untouched. Use --mode 4k or freex to select the exported layout.
The output directory must be new, so raw and repaired evaluations stay separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from script.diagnose_decoder import chart_stats
from script.train_models import write_json
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes


def repair_chart(chart, audio_seconds, mode, min_gap=0.04):
    if len(chart['judgeLineList']) != 1:
        raise ValueError('only generated single static-line charts are supported')
    line = chart['judgeLineList'][0]
    for event in line['judgeLineMoveEvents']:
        if event['start'] != 0.5 or event['end'] != 0.5 or event['start2'] != event['end2']:
            raise ValueError('line must be static and centered')
    if any(e['start'] != 0 or e['end'] != 0 for e in line['judgeLineRotateEvents']):
        raise ValueError('rotating lines are not supported')
    if line['notesBelow']:
        raise ValueError('only generated notesAbove layouts are supported')
    end_tick = (audio_seconds - chart.get('offset', 0)) * 32 * line['bpm'] / 60
    notes = line['notesAbove']
    if mode == '4k':
        repaired, report = constrain_4k_notes(notes, end_tick)
    elif mode == 'freex':
        repaired, report = constrain_free_notes(notes, end_tick, min_gap)
    else:
        raise ValueError(f'unknown mode: {mode}')
    return {**chart, 'judgeLineList': [{**line, 'notesAbove': repaired}]}, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gen-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--list', default='data/val.txt')
    p.add_argument('--mode', choices=['4k', 'freex'], required=True)
    p.add_argument('--free-min-gap', type=float, default=0.04)
    args = p.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    base = Path(args.list).resolve().parent
    reports = []
    for row in Path(args.list).read_text().splitlines():
        if not row.strip() or row.startswith('#'):
            continue
        j, audio, *_ = row.split(',')
        song = Path(j).parent.name
        source = Path(args.gen_dir) / f'{song}.json'
        chart = json.loads(source.read_text())
        repaired, changes = repair_chart(chart, sf.info(str(base / audio)).duration,
                                          args.mode, args.free_min_gap)
        target = output / source.name
        write_json(target, repaired)
        reports.append(dict(song=song, changes=changes,
                            before=chart_stats(source), after=chart_stats(target)))
    report_path = output.with_name(output.name + '.report.json')
    write_json(report_path, dict(config=vars(args), charts=reports))
    print(f'{len(reports)} charts written to {output}; report: {report_path}')


if __name__ == '__main__':
    main()

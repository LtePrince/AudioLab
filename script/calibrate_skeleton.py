"""Calibrate shared onset skeletons on TRAIN; evaluate a frozen choice on VAL.

Scores unique onset times (chords count once), independently of note types and
positions. Parameters are chosen by mean F1@25ms, never by Hold proportions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import soundfile as sf
import torch

from script.eval_onset_f1 import onsets
from script.train_models import write_json
from src.data.dataset import PhigrosDataset
from src.data.timing import check_checkpoint_timing
from src.models.rhythm_skeleton import onset_candidates, candidates_to_ticks
from src.models.transcriber import TranscriberNet
from src.resource_budget import ResourceBudget, add_resource_args
from src.train_utils import pad_or_trim


def event_score(generated, reference, tolerance):
    """Maximum-cardinality monotone time matching; each onset counted once."""
    generated, reference = sorted(generated), sorted(reference)
    i = j = hits = 0
    while i < len(generated) and j < len(reference):
        if generated[i] < reference[j] - tolerance:
            i += 1
        elif reference[j] < generated[i] - tolerance:
            j += 1
        else:
            hits += 1; i += 1; j += 1
    precision = hits / max(len(generated), 1)
    recall = hits / max(len(reference), 1)
    return dict(precision=precision, recall=recall,
                f1=2 * hits / max(len(generated) + len(reference), 1))


def summarize(rows):
    return {key: sum(r[key] for r in rows) / len(rows)
            for key in ['f1_25', 'precision_25', 'recall_25', 'f1_50', 'f1_93',
                        'shifted_f1_25', 'density_ratio']}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['fit', 'evaluate'], required=True)
    p.add_argument('--data-list', default='data/train.txt')
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt', default='checkpoints/timing_v1/transcriber/transcriber_best.pt')
    p.add_argument('--output', required=True)
    p.add_argument('--calibration', help='frozen fit report; required for evaluate')
    p.add_argument('--thresholds', default='0.4,0.5,0.6,0.7,0.8,0.9')
    p.add_argument('--intervals-ms', default='0,15,25,35,45')
    p.add_argument('--device', default='cuda:0')
    add_resource_args(p)
    args = p.parse_args()
    if args.samples < 1:
        p.error('samples must be positive')
    baseline = dict(threshold=0.7, min_interval_ms=0.0)
    if args.phase == 'evaluate':
        if not args.calibration:
            p.error('evaluate needs a frozen --calibration report')
        calibration = json.loads(Path(args.calibration).read_text())
        if not calibration.get('complete') or calibration['config']['phase'] != 'fit':
            p.error('calibration must be a completed fit report')
        if Path(calibration['config']['data_list']).resolve() == Path(args.data_list).resolve():
            p.error('evaluation must use a different data list')
        choices = [baseline, calibration['selected']]
    else:
        choices = [dict(threshold=float(t), min_interval_ms=float(i))
                   for t in args.thresholds.split(',') for i in args.intervals_ms.split(',')]
        if baseline not in choices:
            choices.append(baseline)
    device = torch.device(args.device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    budget = ResourceBudget(args, device, output)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    check_checkpoint_timing(checkpoint)
    model = TranscriberNet(**checkpoint['config']).to(device).eval()
    model.load_state_dict(checkpoint.get('ema', checkpoint['model']))
    del checkpoint
    frame_ms = 512 / 22050 * 2000
    ds = PhigrosDataset(args.data_list, dict(frame_ms=frame_ms, max_frame=4096),
                        cache_dir='data/cache_mel', augment=False)
    if args.phase == 'evaluate':
        fit_songs = {r['song'] for r in calibration['choices'][0]['songs']}
        eval_songs = {Path(e[0]).parent.name for e in ds._entries}
        if fit_songs & eval_songs:
            p.error('fit and evaluation songs overlap')
    indices = sorted(random.Random(args.seed).sample(range(len(ds)), min(args.samples, len(ds))))
    report = dict(config=vars(args), choices=[dict(parameters=c, songs=[]) for c in choices])
    for index in indices:
        budget.check(force=True)
        source, audio, _ = ds._entries[index]
        chart = json.loads(Path(source).read_text())
        bpm, offset = chart['judgeLineList'][0]['bpm'], chart.get('offset', 0)
        duration_ms = sf.info(audio).duration * 1000
        mel = ds._load_mel(audio)
        frames = math.ceil(mel.shape[-1] / model.mel_per_frame)
        mel = pad_or_trim(mel, frames * model.mel_per_frame)[None].to(device)
        times, confidence = onset_candidates(model(mel), frame_ms)
        ref = sorted({round(x[0], 4) for x in onsets(source) if 0 <= x[0] < duration_ms})
        for choice in report['choices']:
            ticks = candidates_to_ticks(times, confidence, bpm, offset, **choice['parameters'])
            generated = [t * 60 / (32 * bpm) * 1000 + offset * 1000 for t in ticks]
            generated = [t for t in generated if 0 <= t < duration_ms]
            at25 = event_score(generated, ref, 25)
            row = dict(song=Path(source).parent.name, onsets=len(generated), reference_onsets=len(ref),
                       f1_25=at25['f1'], precision_25=at25['precision'], recall_25=at25['recall'],
                       f1_50=event_score(generated, ref, 50)['f1'],
                       f1_93=event_score(generated, ref, 92.9)['f1'],
                       shifted_f1_25=sum(event_score([t + s for t in generated], ref, 25)['f1']
                                         for s in [500, -400]) / 2,
                       density_ratio=len(generated) / max(len(ref), 1))
            choice['songs'].append(row)
        print(f'[{args.phase}] {len(report["choices"][0]["songs"])}/{len(indices)} {Path(source).parent.name}', flush=True)
        report['resources'] = budget.log_state()
        write_json(output / 'report.json', report)
    for choice in report['choices']:
        choice['summary'] = summarize(choice['songs'])
    if args.phase == 'fit':
        # Stable ties favor fewer changes: threshold nearer 0.7, smaller interval.
        best = max(report['choices'], key=lambda c: (c['summary']['f1_25'],
                   -abs(c['parameters']['threshold'] - 0.7), -c['parameters']['min_interval_ms']))
        report['selected'] = best['parameters']
    report['complete'] = True
    write_json(output / 'report.json', report)
    for choice in report['choices']:
        print(json.dumps(dict(parameters=choice['parameters'], **choice['summary'])), flush=True)


if __name__ == '__main__':
    main()

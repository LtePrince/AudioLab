"""Bounded teacher-forcing and sampling comparisons using existing checkpoints.

Calibration defaults to a seeded subset of TRAIN songs. Validation can be
passed explicitly for a final comparison, after fixing the sampling policy.
Writes only small JSON charts/reports; never modifies checkpoints or datasets.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from script.eval_onset_f1 import f1, onsets
from script.train_models import write_json
from src.data.chart2array import save_phigros_notes
from src.data.chart_tokenizer import ChartTokenizer, free_notes_to_phigros
from src.models.chart_decoder import generate_tokens
from src.models.transcriber import TranscriberNet
from src.models.rhythm_skeleton import resolve_skeleton_settings
from src.resource_budget import ResourceBudget, add_resource_args
from src.test_decoder import load_models, skeleton_from_chart, skeleton_from_transcriber
from src.train_decoder import TokenChartDataset
from src.train_utils import pad_or_trim


def truncate(probs, top_p):
    if not 0 < top_p < 1:
        return probs
    values, indices = probs.sort(descending=True, dim=-1)
    values = values * (values.cumsum(-1) - values < top_p)
    result = torch.zeros_like(probs).scatter(-1, indices, values)
    return result / result.sum(-1, keepdim=True)


def chart_stats(path):
    """Single static-line outputs: holds, physical conflicts and durations."""
    chart = json.loads(Path(path).read_text())
    events = []
    types = [0] * 4
    durations = []
    for line in chart['judgeLineList']:
        sec = 60 / (32 * float(line['bpm']))
        for n in line['notesAbove'] + line['notesBelow']:
            types[n['type'] - 1] += 1
            start = n['time'] * sec + chart.get('offset', 0)
            dur = n['holdTime'] * sec if n['type'] == 3 else 0
            events.append((start, 0.5 + n['positionX'] * 0.05625, dur))
            if n['type'] == 3:
                durations.append(dur)
    active = []
    hold_conflicts = close_chords = 0
    previous = []
    for t, x, dur in sorted(events):
        active = [(end, hx) for end, hx in active if end > t + 0.001]
        hold_conflicts += any(abs(x - hx) < 0.03 for _, hx in active)
        previous = [(pt, px) for pt, px in previous if t - pt <= 0.001]
        close_chords += any(abs(x - px) < 0.03 for _, px in previous)
        previous.append((t, x))
        if dur > 0:
            active.append((t + dur, x))
    return {'notes': sum(types), 'types': types, 'hold_conflicts': hold_conflicts,
            'close_chords': close_chords,
            'hold_duration_median_s': float(np.median(durations)) if durations else 0}


@torch.no_grad()
def teacher_stats(enc, dec, item, tk, device, top_ps):
    mel = pad_or_trim(item['mel'], 8192).unsqueeze(0).to(device)
    tokens = item['tokens'].unsqueeze(0).to(device)
    frames = item['frames'].unsqueeze(0).to(device)
    valid = torch.arange(4096, device=device)[None] < min(item['mel_frames'] // 2, 4096)
    memory = dec.prepare_memory(enc.features(mel))
    logits = dec(tokens[:, :-1], frames[:, :-1], memory, valid)[0]
    targets = tokens[0, 1:]
    first = tk.note(0, 1)
    mask = (targets >= first) & (targets < first + tk.n_lanes * 4)
    note_probs = logits[mask, first:first + tk.n_lanes * 4].softmax(-1)
    truth = (targets[mask] - first) % 4
    types = note_probs.reshape(-1, tk.n_lanes, 4).sum(1)
    predictions = types.argmax(-1)
    confusion = torch.bincount(truth * 4 + predictions, minlength=16).reshape(4, 4)
    result = {'notes': int(mask.sum()), 'true_types': torch.bincount(truth, minlength=4).tolist(),
              'predicted_type_mass': types.sum(0).tolist(), 'type_confusion': confusion.tolist(),
              'conditional_note_hold_mass': {str(p): float(truncate(note_probs, p)
                  .reshape(-1, tk.n_lanes, 4)[:, :, 2].sum()) for p in top_ps}}
    if tk.n_lanes == 64:
        predicted_bin = note_probs.reshape(-1, 64, 4).sum(-1).argmax(-1)
        true_bin = (targets[mask] - first) // 4
        error = (predicted_bin - true_bin).abs().float() * (0.9 / 64)
        result.update(position_mae_sum=float(error.sum()),
                      position_within_005=int((error <= 0.05).sum()))
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-list', default='data/train.txt')
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--models', nargs='+', choices=['decoder', 'decoder_freex'],
                   default=['decoder', 'decoder_freex'])
    p.add_argument('--checkpoint-root', default='checkpoints/timing_v1')
    p.add_argument('--skeleton-ckpt', help='optional separate transcriber/global rhythm checkpoint')
    p.add_argument('--skeleton-calibration', help='completed TRAIN fit report for this checkpoint')
    p.add_argument('--output', default='out/decoder_diagnostics')
    p.add_argument('--top-ps', default='0.95,1.0')
    p.add_argument('--teacher-only', action='store_true')
    p.add_argument('--sampling-policy', choices=['flat', 'type_preserving'], default='flat')
    p.add_argument('--onset-threshold', type=float, default=None)
    p.add_argument('--onset-min-interval-ms', type=float, default=None)
    p.add_argument('--skeleton-source', choices=['transcriber', 'reference'], default='transcriber',
                   help='reference is an oracle diagnostic, never an audio-only quality score')
    p.add_argument('--snap-ticks', type=int, default=0,
                   help='experimental transcriber grid snapping; 0 disables it')
    p.add_argument('--device', default='cuda:0')
    add_resource_args(p)
    args = p.parse_args()
    try:
        args.onset_threshold, args.onset_min_interval_ms = resolve_skeleton_settings(
            args.skeleton_ckpt or str(Path(args.checkpoint_root) / 'transcriber/transcriber_best.pt'),
            args.skeleton_calibration, args.onset_threshold, args.onset_min_interval_ms)
    except ValueError as error:
        p.error(str(error))
    if args.samples < 1 or args.snap_ticks < 0:
        p.error('samples must be positive and snap-ticks nonnegative')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    budget = ResourceBudget(args, device, output)
    top_ps = [float(v) for v in args.top_ps.split(',')]
    if any(not 0 < v <= 1 for v in top_ps):
        raise ValueError('top-ps must be in (0, 1]')
    ck = torch.load(args.skeleton_ckpt or Path(args.checkpoint_root) / 'transcriber/transcriber_best.pt',
                    map_location='cpu', weights_only=True)
    transcriber = TranscriberNet(**ck['config']).to(device).eval()
    transcriber.load_state_dict(ck.get('ema', ck['model']))
    del ck
    report = {'config': vars(args), 'started': time.time(), 'models': {}}
    for name in args.models:
        free = name == 'decoder_freex'
        enc, dec = load_models(str(Path(args.checkpoint_root) / name / 'decoder_best.pt'), device)
        tk = ChartTokenizer(n_lanes=dec.n_lanes)
        dataset = TokenChartDataset(args.data_list, 'data/cache_mel', frame_ms=512 / 22050 * 2000,
                    max_frame=4096, hop_length=512, sr=22050, n_mels=128,
                    gameplay_dir='data/gameplay' if free else None)
        indices = sorted(random.Random(args.seed).sample(range(len(dataset)), min(args.samples, len(dataset))))
        records = []
        report['models'][name] = records
        for index in indices:
            budget.check(force=True)
            source, audio, _ = dataset.base._entries[index]
            song = Path(source).parent.name
            item = dataset[index]
            reference = Path('data/gameplay_charts', song + '.json') if free else Path(source)
            record = {'song': song, 'reference': chart_stats(reference),
                      'teacher': teacher_stats(enc, dec, item, tk, device, top_ps), 'samples': {}}
            records.append(record)
            print(json.dumps({'model': name, 'song': song, 'teacher': record['teacher']}), flush=True)
            if not args.teacher_only:
                n_frames = math.ceil(item['mel'].shape[-1] / enc.mel_per_frame)
                mel = pad_or_trim(item['mel'], n_frames * enc.mel_per_frame).unsqueeze(0).to(device)
                memory = dec.prepare_memory(enc.features(mel))
                valid = torch.ones(1, n_frames, dtype=torch.bool, device=device)
                _, bpm, offset = dataset._notes(index)
                frame_ms = dataset.frame_ms
                skeleton = (skeleton_from_chart(str(reference), bpm, offset)
                            if args.skeleton_source == 'reference' else
                            skeleton_from_transcriber(mel, transcriber, bpm, frame_ms,
                                args.onset_threshold, offset, args.onset_min_interval_ms))
                if args.snap_ticks and args.skeleton_source == 'transcriber':
                    skeleton = sorted({round(t / args.snap_ticks) * args.snap_ticks for t in skeleton})
                record['skeleton_onsets'] = len(skeleton)
                for top_p in top_ps:
                    budget.check(force=True)
                    path = output / name / f'p{top_p}' / f'{song}.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    generated = generate_tokens(dec, memory, valid, bpm, frame_ms, n_frames,
                        top_p=top_p, generator=torch.Generator(device=device).manual_seed(args.seed),
                        tokenizer=tk, skeleton_ticks=skeleton, offset=offset,
                        sampling_policy=args.sampling_policy)
                    if free:
                        tokens, position_offsets = generated
                        ticks = tk.ticks_per_position(tokens)
                        offsets = {(ticks[i], tk.value(tokens[i]) // 4): o for i, o in position_offsets.items()}
                        notes = free_notes_to_phigros(tk, tk.decode_tokens(tokens), offsets, bpm)
                    else:
                        notes = tk.to_phigros_notes(tk.decode_tokens(generated), bpm)
                    save_phigros_notes(notes, bpm, str(path), offset=offset)
                    gen, ref = onsets(str(path)), onsets(str(reference))
                    stats = chart_stats(path)
                    stats.update(time_f1_25=f1(gen, ref, 25, False), time_f1_93=f1(gen, ref, 92.9, False),
                        position_f1_93=f1(gen, ref, 92.9, False, 0.05) if free else f1(gen, ref, 92.9, True))
                    record['samples'][str(top_p)] = stats
                    print(json.dumps({'model': name, 'song': song, 'top_p': top_p, **stats}), flush=True)
            report['resources'] = budget.log_state()
            write_json(output / 'report.json', report)
        del enc, dec, dataset
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    report['finished'] = time.time()
    write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()

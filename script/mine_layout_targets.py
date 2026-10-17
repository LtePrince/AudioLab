"""Mine unmodified TRAIN layouts with audio descriptors for conditional learning.

Does not turn an engineering contact limit into a universal training target.
Keeps multi-finger chords and original Hold durations, reports type counts only
as diagnostics. Existing mel caches are read-only and loaded one song at a time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import time
import warnings

import numpy as np
import soundfile as sf
import torch

from script.diagnose_decoder import sha256_file
from script.train_models import write_json
from src.data.chart_tokenizer import ChartTokenizer
from src.data.dataset import PhigrosDataset
from src.data.timing import TIMING_SCHEMA
from src.resource_budget import ResourceBudget, add_resource_args


def layout_events(notes, bpm, offset, mel, duration):
    """Descriptors use a ±2 mel-frame window, not a causal streaming feature."""
    sec = 60/(32*bpm)
    energy = np.mean(mel, axis=0)
    flux = np.r_[0.0, np.maximum(np.diff(mel, axis=1), 0).mean(axis=0)]
    active = []
    previous = None
    previous_positions = []
    for tick, group in groupby(sorted(notes, key=lambda n: (n['time'], n['positionX'])),
                               key=lambda n: n['time']):
        chord = list(group)
        t = tick*sec+offset
        if not 0 <= t < duration:
            continue
        frame = min(mel.shape[1]-1, max(0, round(t*22050/512)))
        lo, hi = max(0, frame-2), min(mel.shape[1], frame+3)
        active = [(end, x) for end, x in active if end > t+1e-7]
        xs = [.5+n['positionX']*.05625 for n in chord]
        yield dict(audio_s=t, tick=tick, chord_size=len(chord), screen_x=xs,
                   types=[n['type'] for n in chord], hold_s=[n['holdTime']*sec for n in chord],
                   previous_gap_s=None if previous is None else t-previous,
                   previous_screen_x=previous_positions,
                   active_holds_before=[dict(remaining_s=end-t, screen_x=x) for end, x in active],
                   logmel_mean=float(energy[lo:hi].mean()), positive_logmel_flux=float(flux[lo:hi].max()))
        active += [(t+n['holdTime']*sec, x) for n, x in zip(chord, xs) if n['type'] == 3]
        previous, previous_positions = t, xs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-list', default='data/train.txt')
    p.add_argument('--exclude-list', default='data/val.txt')
    p.add_argument('--output', default='out/layout_targets_train_v1')
    add_resource_args(p)
    args = p.parse_args()
    excluded = {Path(row.split(',')[0]).parent.name for row in Path(args.exclude_list).read_text().splitlines()
                if row.strip() and not row.startswith('#')}
    dataset = PhigrosDataset(data_list_path=args.train_list, cache_dir='data/cache_mel', augment=False,
                            convertor_params=dict(frame_ms=512/22050*2000, max_frame=4096))
    songs = [Path(j).parent.name for j, _, _ in dataset._entries]
    if len(songs) != len(set(songs)) or set(songs) & excluded:
        raise ValueError('training songs must be unique and disjoint from excluded validation songs')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    budget = ResourceBudget(args, torch.device('cpu'), output)
    tk = ChartTokenizer()
    report = dict(config=vars(args), complete=False, timing_schema=TIMING_SCHEMA, started=time.time(),
                  train_sha256=sha256_file(args.train_list), excluded_sha256=sha256_file(args.exclude_list),
                  note='original layouts; no two-finger cap or Hold quota; audio descriptors are noncausal',
                  songs=[], summary={})
    totals = {mode: Counter() for mode in ['4k', 'freex']}
    for source, audio, _ in dataset._entries:
        budget.check(force=True)
        song = Path(source).parent.name
        cache = dataset._cache_path(audio)
        if cache is None or not cache.is_file():
            raise ValueError(f'missing existing mel cache: {song}; refusing implicit audio rebuild')
        mel = np.load(cache, mmap_mode='r', allow_pickle=False)
        if mel.ndim != 2 or mel.shape[0] != 128 or not np.isfinite(mel).all():
            raise ValueError(f'invalid mel cache: {song}')
        chart = json.loads(Path(source).read_text())
        bpm, offset = chart['judgeLineList'][0]['bpm'], chart.get('offset', 0)
        duration = sf.info(audio).duration
        free_path = Path('data/gameplay_charts', song+'.json')
        free = json.loads(free_path.read_text())
        if free['judgeLineList'][0]['bpm'] != bpm or free.get('offset', 0) != offset:
            raise ValueError(f'gameplay timing mismatch: {song}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            four = tk.to_phigros_notes(tk.decode_tokens(tk.encode_chart(source)), bpm)
        record = dict(song=song, chart_sha256=sha256_file(source), gameplay_sha256=sha256_file(free_path),
                      mel_sha256=sha256_file(cache), audio_seconds=duration, modes={})
        for mode, notes in [('4k', four), ('freex', free['judgeLineList'][0]['notesAbove'])]:
            target = output/mode/(song+'.jsonl'); target.parent.mkdir(exist_ok=True)
            histogram, events, note_count = Counter(), 0, 0
            with target.open('w', encoding='utf-8') as stream:
                for event in layout_events(notes, bpm, offset, mel, duration):
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
                    histogram[event['chord_size']] += 1
                    events += 1; note_count += event['chord_size']
            totals[mode].update(histogram)
            record['modes'][mode] = dict(events=events, notes=note_count, chord_histogram=dict(histogram),
                                         targets_sha256=sha256_file(target))
        report['songs'].append(record)
        del mel
        if len(report['songs']) % 30 == 0:
            print(f"[mine] {len(report['songs'])}/{len(songs)} songs", flush=True)
        write_json(output/'report.json', report)
    report['summary'] = {mode:dict(chord_histogram=dict(hist), events=sum(hist.values()),
        notes=sum(k*v for k,v in hist.items()), events_over_two=sum(v for k,v in hist.items() if k>2))
        for mode,hist in totals.items()}
    report.update(complete=True, finished=time.time(), resources=budget.log_state())
    write_json(output/'report.json', report)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

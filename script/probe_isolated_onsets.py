"""Small, reproducible acoustic sanity probes; never training/benchmark labels.

Supply C4.mp3, Fs4.mp3 and A4.mp3 from Tonejs/audio/salamander. Samples are
Alexander Holm's Salamander Grand Piano, CC BY 3.0. No automatic downloads.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from src.audio_understanding.evaluation import evaluate_record
from src.audio_understanding.inference import code_version, infer_mel, load_audio, load_model, make_result, sha256
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget, add_resource_args

SR = 22050
SOURCE = 'https://github.com/Tonejs/audio/tree/master/salamander'


def prepare_fixtures(samples, output):
    output.mkdir(parents=True, exist_ok=True)
    clips, sources = [], []
    for name in ('C4', 'Fs4', 'A4'):
        path = samples / f'{name}.mp3'
        wave, rate = sf.read(path, dtype='float32', always_2d=True)
        wave = wave.mean(axis=1)
        divisor = math.gcd(rate, SR)
        wave = resample_poly(wave, SR//divisor, rate//divisor)
        # Fixed waveform-based origin, independent of the tested network.
        start = int(np.flatnonzero(abs(wave) > abs(wave).max()*1e-3)[0])
        clip = wave[start:start+SR].copy()
        clip *= .6/max(abs(clip).max(), 1e-8)
        clip[:44] *= np.linspace(0, 1, 44)
        clip[-2205:] *= np.linspace(1, 0, 2205)
        clips.append(clip)
        sources.append(dict(note=name, path=str(path.resolve()), sha256=sha256(path),
                            trimmed_seconds=start/SR, clip_seconds=len(clip)/SR))
    rows = []
    cases = [('piano_three', [1.,4.,7.], [1.,1.,1.], [0,1,2], 10.),
             ('piano_weak_middle', [1.,4.,7.], [1.,.03,1.], [0,1,2], 10.),
             ('piano_repeated', [1.,4.,7.], [1.,1.,1.], [0,0,0], 10.),
             ('piano_chunk_boundary', [1.,24.,29.], [1.,1.,1.], [0,1,2], 32.),
             ('silence', [], [], [], 10.)]
    for name, starts, gains, indices, duration in cases:
        wave = np.zeros(round(duration*SR), dtype=np.float32)
        for onset, gain, index in zip(starts, gains, indices):
            start = round(onset*SR)
            wave[start:start+len(clips[index])] += gain*clips[index]
        path = output / f'{name}.wav'
        sf.write(path, wave, SR, subtype='FLOAT')
        rows.append(dict(id=name, source_group=name, label_source='constructed_acoustic_probe',
                         audio=str(path.resolve()), audio_sha256=sha256(path),
                         events_seconds=starts, duration_seconds=duration, coverage=[[0.,duration]],
                         active_intervals=[[t,t+1.] for t in starts], gains=gains))
    manifest = dict(schema='audiolab-isolated-onset-probe-v1', rows=rows, sources=sources,
                    source_url=SOURCE, attribution='Alexander Holm, Salamander Grand Piano, CC BY 3.0',
                    license='https://creativecommons.org/licenses/by/3.0/',
                    limitations=['Edited piano samples, not an independent natural-music benchmark.',
                                 'Onsets are construction times after amplitude-based leading trim and 2ms fade.',
                                 'Samples shortened to 1s with 100ms release fade; no pitch evaluation.'])
    write_json(output/'fixtures.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', default='checkpoints/full300_history/transcriber/transcriber_best.pt')
    parser.add_argument('--device', default='cuda:0')
    add_resource_args(parser)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    budget = ResourceBudget(args, device, args.output)
    fixtures = prepare_fixtures(args.samples, args.output/'fixtures')
    model, metadata = load_model(args.checkpoint, device)
    report = dict(schema='audiolab-isolated-onset-probe-v1', model=metadata,
                  selection=dict(threshold=.7, min_interval_ms=35, tuned_on_probes=False),
                  implementation=code_version(), fixtures=fixtures, rows=[])
    with torch.inference_mode():
        for row in fixtures['rows']:
            mel, duration = load_audio(row['audio'], budget=budget)
            arrays = infer_mel(model, mel, duration, check=budget.check)
            result = make_result(arrays, duration, metadata)
            times = [e['time_seconds'] for e in result['events']]
            score = evaluate_record(row, times)
            score['selected_times_seconds'] = times
            p, t = arrays['event_probability'], arrays['event_times_seconds']
            score['probability_near_reference'] = [dict(time_seconds=onset,
                max_within_50ms=float(p[abs(t-onset)<=.05].max(initial=0))) for onset in row['events_seconds']]
            # Exclude 100ms around known sounding spans from pure-silence counts.
            silent = lambda x: not any(a-.1 <= x <= b+.1 for a,b in row['active_intervals'])
            score['events_in_silent_regions'] = sum(silent(x) for x in times)
            score['silence_max_probability'] = float(p[np.array([silent(x) for x in t])].max(initial=0))
            np.savez_compressed(args.output/f'{row["id"]}.npz', **{k:v for k,v in arrays.items() if k!='features'})
            report['rows'].append(score)
            write_json(args.output/'report.json', report)
            print(json.dumps(dict(id=row['id'], predictions=len(times), at_25ms=score['at_25ms'],
                                  near=score['probability_near_reference']), ensure_ascii=False), flush=True)
    report.update(status='completed', resources=budget.log_state())
    write_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()

"""Automatic versus provided timebases on the same three complete songs."""
from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from script.generate_product import parser
from src.audio_understanding.inference import code_version
from src.data.timing import tick_seconds
from src.product.artifacts import write_json
from src.product.pipeline import run_job


def describe(chart):
    line=chart['judgeLineList'][0];notes=line['notesAbove']
    chord=Counter(note['time'] for note in notes)
    return dict(note_count=len(notes),types=dict(Counter(str(n['type']) for n in notes)),
                unique_onsets=len(chord),max_chord=max(chord.values(),default=0),
                total_hold_seconds=sum(tick_seconds(n['holdTime'],line['bpm']) for n in notes if n['type']==3),
                screen_x_quantiles=np.quantile([.5+n['positionX']*.05625 for n in notes],[0,.25,.5,.75,1]).tolist())


def verify_control(previous, generated, original_events, current_events):
    """Reject reference runs with another cause for a layout change."""
    checks={}
    def compare(label,left,right):
        if left != right:raise ValueError(f'timebase control mismatch: {label}')
        checks[label]=left
    for key in ['modes','model_sha256']:
        compare(key,previous[key],generated[key])
    compare('audio_sha256',previous['audio']['sha256'],generated['audio']['sha256'])
    for key in ['seed','offset','timing_schema','difficulty','layout_encoder','max_seconds','timeout_seconds']:
        compare('config.'+key,previous['config'][key],generated['config'][key])
    for key in ['torch','device','threads','device_name']:
        compare('runtime.'+key,previous['runtime'][key],generated['runtime'][key])
    for mode in previous['modes']:
        for key in ['seed','temperature','top_p','sampling_policy','max_decoder_steps','encoder',
                    'chunk_frames','context_frames','chord_size_cap','termination_policy']:
            compare(f'{mode}.{key}',previous['routes'][mode][key],generated['routes'][mode][key])
    for key in ['schema','time_basis','duration_seconds','checkpoint_sha256','label_source',
                'feature_schema','selection','inference','feature_shape']:
        compare('audio_events.'+key,original_events[key],current_events[key])
    compare('audio_implementation_sources',original_events['implementation']['source_sha256'],
            current_events['implementation']['source_sha256'])
    if original_events['events'] != current_events['events']:
        raise ValueError('timebase control requires identical unsnapped audio events')
    return dict(status='matched',conditions=checks,audio_candidates_identical=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,help='completed provided-BPM audit directory')
    p.add_argument('--output',required=True);p.add_argument('--model-root',required=True)
    p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(2)
    reference=Path(args.reference).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((reference/'selection.json').read_text())
    report=dict(schema='audiolab-tempo-layout-control-v1',implementation=code_version(),results=[],
                interpretation='provided chart BPM is an encoding timebase, not verified musical ground truth')
    for i,song in enumerate(selection):
        job=parser().parse_args(['--audio',song['audio'],'--output',str(output/f'song-{i}'),
                                '--model-root',args.model_root,'--device',args.device])
        generated=run_job(job)
        previous=reference/f'song-{i}'
        manifest=json.loads((previous/'manifest.json').read_text())
        original_events=json.loads((previous/'audio_events.json').read_text())
        current_events=json.loads((Path(job.output)/'audio_events.json').read_text())
        control=verify_control(manifest,generated,original_events,current_events)
        row=dict(audio_sha256=generated['audio']['sha256'],duration_seconds=generated['audio']['duration_seconds'],
                 provided_timebase=manifest['config']['bpm'],tempo=generated['tempo'],
                 control=control,audio_candidates_identical=True,elapsed_seconds=generated['elapsed_seconds'],
                 resources=generated['resources'],routes={})
        for mode in ['4k','freex']:
            route={}
            for label,folder,meta in [('provided',previous,manifest),('automatic',Path(job.output),generated)]:
                route[label]=dict(raw=describe(json.loads((folder/f'{mode}.raw.json').read_text())),
                                  final=describe(json.loads((folder/f'{mode}.json').read_text())),
                                  repairs=meta['routes'][mode]['repairs'],
                                  quantization={k:v for k,v in meta['audio'].items() if 'quantization' in k})
            row['routes'][mode]=route
        report['results'].append(row);write_json(output/'audit.in_progress.json',report)
        print(json.dumps(dict(completed_song=i,estimated_bpm=generated['tempo']['timebase_bpm'],
                              elapsed_seconds=generated['elapsed_seconds'])),flush=True)
    report.update(status='engineering_control_completed',musical_tempo_accuracy='not_measured_no_reviewed_ground_truth',
                  learned_tempo_model=False,human_playability='unverified')
    write_json(output/'audit.json',report)


if __name__=='__main__':main()

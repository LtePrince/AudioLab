"""Three-song engineering smoke and one-song encoder-window control; no training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import soundfile as sf
import torch

from script.generate_product import parser as job_parser
from src.audio_understanding.inference import load_audio, result_to_ticks, sha256, code_version
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.data.timing import tick_seconds
from src.product.artifacts import chart_document, validate_chart, write_json
from src.product.pipeline import JobControl, decode_notes, run_job, quantize_candidates
from src.resource_budget import ResourceBudget


def notes_summary(chart):
    notes = chart['judgeLineList'][0]['notesAbove']
    return dict(notes=len(notes), types=dict(Counter(str(note['type']) for note in notes)),
                unique_onsets=len({note['time'] for note in notes}),
                total_hold_ticks=sum(note['holdTime'] for note in notes if note['type']==3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-list', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    listing = Path(args.data_list).resolve()
    songs = []
    for row in listing.read_text().splitlines():
        if not row or row.startswith('#'):
            continue
        chart, audio, *_ = row.split(',')
        audio = (listing.parent/audio).resolve()
        data = json.loads((listing.parent/chart).read_text())
        songs.append(dict(audio=str(audio), bpm=data['judgeLineList'][0]['bpm'],
                          duration=sf.info(audio).duration, source_chart=str((listing.parent/chart).resolve())))
    songs.sort(key=lambda song:song['duration'])
    if len(songs) < 3:
        raise ValueError('need at least three songs for the engineering smoke')
    selected = [songs[0], songs[len(songs)//2], songs[-1]]
    report = dict(schema='audiolab-product-engineering-smoke-v1', implementation=code_version(),
                  selection='shortest/median/longest of development list; not a frozen quality test',
                  data_list_sha256=sha256(listing), results=[], window_controls=[])
    write_json(output/'selection.json', selected)
    for index, song in enumerate(selected):
        job = job_parser().parse_args(['--audio', song['audio'], '--bpm', str(song['bpm']),
                                      '--model-root', args.model_root, '--output', str(output/f'song-{index}'),
                                      '--device', args.device])
        result = run_job(job)
        report['results'].append(dict(song=song, job=f'song-{index}/manifest.json',
                                       elapsed_seconds=result['elapsed_seconds'], routes=result['routes'],
                                       resources=result['resources']))
        write_json(output/'audit.in_progress.json', report)
        print(json.dumps(dict(completed_song=index, elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False), flush=True)
    # Same first-song mel, candidates, seed and weights: only encoder attention
    # differs. Raw and static-repaired outcomes are kept separately.
    song = selected[0]; completed = output/'song-0'
    control_dir = output/'full_encoder_control'; control_dir.mkdir()
    job.output = str(control_dir)
    device = torch.device(args.device)
    budget = ResourceBudget(job, device, control_dir)
    control = JobControl(control_dir, budget, job.timeout_seconds)
    mel, duration = load_audio(song['audio'], budget=budget)
    evidence = json.loads((completed/'audio_events.json').read_text())
    skeleton, _ = quantize_candidates(evidence, song['bpm'], 0)
    for mode, folder in [('4k','decoder'), ('freex','decoder_freex')]:
        start = time.monotonic()
        checkpoint = Path(args.model_root)/folder/'decoder_best.pt'
        raw, decoding = decode_notes(mel,duration,skeleton,checkpoint,mode,song['bpm'],0,42,
                                     device,control,max_tokens=12000,encoder_mode='full')
        raw_chart = chart_document(raw,song['bpm'],0,control_dir/f'{mode}.raw.json')
        end_tick = duration/tick_seconds(1,song['bpm'])
        final, repairs = (constrain_4k_notes(raw,end_tick) if mode=='4k' else constrain_free_notes(raw,end_tick))
        final_chart = chart_document(final,song['bpm'],0,control_dir/f'{mode}.json')
        _, checks = validate_chart(final_chart,mode,duration,skeleton)
        comparison = dict(mode=mode,audio_sha256=evidence['audio_sha256'],
                          checkpoint_sha256=sha256(checkpoint),seed=42,decoding=decoding,
                          full=dict(raw=notes_summary(raw_chart),final=notes_summary(final_chart),repairs=repairs),
                          chunked={},same_note_multiset={},elapsed_seconds=time.monotonic()-start,checks=checks)
        for kind, suffix, full in [('raw','.raw.json',raw_chart),('final','.json',final_chart)]:
            chunked = json.loads((completed/f'{mode}{suffix}').read_text())
            comparison['chunked'][kind] = notes_summary(chunked)
            def signature(chart):
                return Counter((n['time'],n['type'],round(n['positionX'],7),n['holdTime'])
                               for n in chart['judgeLineList'][0]['notesAbove'])
            a,b = signature(full),signature(chunked)
            comparison['same_note_multiset'][kind] = dict(common=sum((a&b).values()),
                    full_only=sum((a-b).values()),chunked_only=sum((b-a).values()))
        comparison['chunked']['repairs'] = report['results'][0]['routes'][mode]['repairs']
        report['window_controls'].append(comparison)
        write_json(output/'audit.in_progress.json', report)
    report['status'] = 'engineering_smoke_completed'
    report['quality_acceptance'] = 'unverified; these comparisons are not acoustic or human-playability scores'
    write_json(output/'audit.json',report)
    print(json.dumps(dict(status=report['status'],output=str(output))),flush=True)


if __name__ == '__main__':
    main()

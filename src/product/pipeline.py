"""Bounded, serial, reproducible audio -> chart -> offline preview jobs."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.audio_understanding.inference import (code_version, infer_mel, load_audio,
                                             load_model, make_result, result_to_ticks, sha256)
from src.audio_understanding import FRAME_SECONDS, FEATURE_SCHEMA
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.data.chart_tokenizer import ChartTokenizer, free_notes_to_phigros
from src.data.timing import TIMING_SCHEMA, tick_seconds
from src.models.chart_decoder import generate_tokens
from src.resource_budget import GIB, ResourceBudget
from src.test_decoder import load_models
from . import JOB_SCHEMA
from .artifacts import chart_document, repair_log, validate_chart, write_json, write_preview


class JobCancelled(Exception):
    pass


def quantize_candidates(evidence, bpm, offset):
    """Use the nearest legal tick without losing a final sub-tick event."""
    result_to_ticks(evidence, bpm, offset)  # shared schema/time/confidence validation
    duration = evidence['duration_seconds']
    times = [event['time_seconds'] for event in evidence['events']]
    if not times:
        raise ValueError('no audio candidates; no playable chart was generated')
    if any(t < offset for t in times):
        raise ValueError('chart offset excludes selected audio events; use a smaller offset')
    tick = tick_seconds(1, bpm)
    last = math.ceil((duration-offset)/tick)-1
    while last >= 0 and last*tick+offset >= duration:
        last -= 1
    if last < 0:
        raise ValueError('no legal chart ticks within the audio duration')
    nearest = [round((t-offset)/tick) for t in times]
    bounded = [min(last, max(0, value)) for value in nearest]
    skeleton = sorted(set(bounded))
    return skeleton, dict(quantization='nearest-legal-tick-v1',
                          quantization_merged_events=len(times)-len(skeleton),
                          quantization_clamped_events=sum(a!=b for a,b in zip(nearest,bounded)),
                          max_quantization_error_ms=max(abs(t-(v*tick+offset))*1000 for t,v in zip(times,bounded)))


class JobControl:
    def __init__(self, directory, budget, timeout, cancel_file=None):
        self.directory, self.budget = directory, budget
        self.start = time.monotonic()
        self.timeout = timeout
        self.cancel_file = Path(cancel_file) if cancel_file else directory / 'CANCEL'
        self.state = dict(schema=JOB_SCHEMA, status='running', stage='preflight', progress=None)

    def check(self):
        if self.cancel_file.exists():
            raise JobCancelled('cancel requested')
        if time.monotonic() - self.start > self.timeout:
            raise TimeoutError(f'generation exceeded {self.timeout:g} second time budget')
        self.budget.check()

    def update(self, stage=None, **extra):
        if stage:
            self.state['stage'] = stage
        self.state.update(extra, elapsed_seconds=time.monotonic()-self.start)
        write_json(self.directory / 'job.json', self.state)

    def progress(self, steps, selected, total):
        self.check()
        self.update(progress=dict(decoder_steps=steps, candidates_reached=selected, candidates_total=total))


@torch.no_grad()
def decode_notes(mel, duration, skeleton, checkpoint, mode, bpm, offset, seed, device, control,
                 max_tokens=6000, encoder_mode='chunked'):
    """Keep the AR state for the entire song; bound only encoder attention.

    Encoder windows are explicitly versioned and require quality calibration.
    Natural EOS and exact candidate coverage are independently required.
    """
    encoding_start = time.monotonic()
    enc, dec = load_models(str(checkpoint), device)
    expected_lanes = 4 if mode == '4k' else 64
    if dec.n_lanes != expected_lanes:
        raise ValueError(f'{mode} requires a {expected_lanes}-position decoder checkpoint')
    if encoder_mode == 'chunked':
        encoded = infer_mel(enc, mel, duration, check=control.check)
        features = torch.from_numpy(encoded['features']).unsqueeze(0).to(device)
        del encoded
    elif encoder_mode == 'full':
        # Explicit diagnostic only: the same mel, duration, candidates and seed
        # isolate the effect of encoder attention windows from audio decoding.
        total = math.ceil(mel.shape[-1]/enc.mel_per_frame)*enc.mel_per_frame
        features = enc.features(F.pad(mel, (0,total-mel.shape[-1])).unsqueeze(0).to(device))
    else:
        raise ValueError('unknown encoder mode')
    memory = dec.prepare_memory(features)
    encoding_seconds = time.monotonic()-encoding_start
    del features, enc
    valid = torch.ones(1, memory.shape[1], dtype=torch.bool, device=device)
    tk = ChartTokenizer(n_lanes=dec.n_lanes)
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.monotonic()
    generated = generate_tokens(dec, memory, valid, bpm, FRAME_SECONDS*2000,
                                duration / (FRAME_SECONDS*2), tokenizer=tk, offset=offset,
                                skeleton_ticks=skeleton, generator=generator, top_p=.95,
                                max_tokens=max_tokens, progress_callback=control.progress,
                                require_complete=True)
    if dec.offset_head is not None:
        tokens, offsets = generated
        decoded = tk.decode_tokens(tokens, strict=True)
        ticks = tk.ticks_per_position(tokens)
        positions = {(ticks[i], tk.value(tokens[i])//4): value for i, value in offsets.items()}
        notes = free_notes_to_phigros(tk, decoded, positions, bpm)
    else:
        tokens = generated
        notes = tk.to_phigros_notes(tk.decode_tokens(tokens, strict=True), bpm)
    control.check()
    return notes, dict(token_count=len(tokens), max_decoder_steps=max_tokens,
                       model_load_and_encoding_seconds=encoding_seconds,
                       encoded_frames=memory.shape[1],
                       decoding_seconds=time.monotonic()-started,
                       encoder='disjoint-core-with-context-v1' if encoder_mode == 'chunked' else 'full-attention-v1',
                       chunk_frames=1024 if encoder_mode == 'chunked' else None,
                       context_frames=128 if encoder_mode == 'chunked' else None,
                       encoder_quality_calibration='pending', seed=seed, temperature=1.0,
                       top_p=.95, sampling_policy='flat', chord_size_cap=None,
                       termination_policy='natural_eos_required')


def _copy_audio(source, target, control):
    size = source.stat().st_size
    if size > 128 * 1024**2:
        raise ValueError('input audio file exceeds 128 MiB job artifact limit')
    control.budget.check(force=True, extra_bytes=size + 32*1024**2)
    # A copied snapshot makes the preview portable and prevents later source
    # edits from silently changing a completed job's soundtrack.
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        while block := incoming.read(1024*1024):
            control.check()
            outgoing.write(block)


def run_job(args):
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    control = None
    try:
        if not math.isfinite(args.bpm) or not 1 <= args.bpm <= 1000:
            raise ValueError('BPM must be finite in [1, 1000]; it is supplied, not inferred')
        if not math.isfinite(args.offset) or args.offset < 0:
            raise ValueError('chart offset must be finite and nonnegative')
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise ValueError('timeout must be positive and finite')
        if type(args.max_decoder_steps) is not int or args.max_decoder_steps <= 0:
            raise ValueError('max decoder steps must be positive')
        source = Path(args.audio).resolve()
        if source.suffix.lower() not in ('.wav', '.ogg'):
            raise ValueError('first local workflow supports WAV and OGG; other formats are not validated')
        modes = ['4k', 'freex'] if args.mode == 'both' else [args.mode]
        model_root = Path(args.model_root).resolve()
        audio_checkpoint = Path(args.audio_checkpoint) if args.audio_checkpoint else model_root/'transcriber/transcriber_best.pt'
        checkpoints = {'4k': model_root/'decoder/decoder_best.pt',
                       'freex': model_root/'decoder_freex/decoder_best.pt'}
        for path in [source, audio_checkpoint] + [checkpoints[mode] for mode in modes]:
            if not path.is_file():
                raise FileNotFoundError(f'missing input: {path}')
        device = torch.device(args.device)
        budget = ResourceBudget(args, device, directory)
        control = JobControl(directory, budget, args.timeout_seconds, args.cancel_file)
        control.update()
        control.check()
        audio_path = directory / ('audio'+source.suffix.lower())
        _copy_audio(source, audio_path, control)
        source_hash = sha256(audio_path)
        report = dict(schema=JOB_SCHEMA, status='running', title=source.stem, modes=modes,
                      audio=dict(file=audio_path.name, sha256=source_hash, original_name=source.name),
                      implementation=code_version(), runtime=dict(torch=str(torch.__version__),
                      device=str(device), threads=torch.get_num_threads()),
                      config=dict(bpm=args.bpm, timing_source='provided', offset=args.offset,
                                  timing_schema=TIMING_SCHEMA, difficulty='baseline', seed=args.seed,
                                  max_seconds=args.max_seconds, timeout_seconds=args.timeout_seconds),
                      model_sha256={mode: sha256(checkpoints[mode]) for mode in modes},
                      stages={}, routes={})
        report['implementation']['product_source_sha256'] = {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*')) if path.is_file()}
        report['runtime']['device_name'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
        control.update('audio_understanding', progress=None)
        phase = time.monotonic()
        mel, duration = load_audio(audio_path, args.max_seconds, budget=budget)
        control.check()
        audio_model, metadata = load_model(audio_checkpoint, device)
        arrays = infer_mel(audio_model, mel, duration, check=control.check)
        evidence = make_result(arrays, duration, metadata)
        evidence.update(audio_sha256=source_hash, audio_name=source.name,
                        inference=dict(version='disjoint-core-with-context-v1', chunk_frames=1024,
                                       context_frames=128, device=str(device), max_seconds=args.max_seconds),
                        dense_file='audio_events.npz', feature_shape=list(arrays['features'].shape),
                        implementation=report['implementation'])
        budget.check(force=True, extra_bytes=sum(a.nbytes for a in arrays.values())+1024*1024)
        np.savez_compressed(directory/'audio_events.npz', **arrays)
        evidence['dense_sha256'] = sha256(directory/'audio_events.npz')
        write_json(directory/'audio_events.json', evidence)
        report['stages']['audio_understanding_seconds'] = time.monotonic()-phase
        report['audio'].update(duration_seconds=duration, feature_schema=FEATURE_SCHEMA,
                               checkpoint_sha256=metadata['checkpoint_sha256'],
                               label_source=metadata['label_source'])
        skeleton, quantization = quantize_candidates(evidence, args.bpm, args.offset)
        report['audio']['candidates'] = len(evidence['events'])
        report['audio']['candidate_ticks'] = len(skeleton)
        report['audio'].update(quantization)
        del arrays, audio_model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        routes = {}
        for mode in modes:
            phase = time.monotonic()
            budget.check(force=True, extra_bytes=16*1024**2)
            control.update('layout_'+mode, progress=None)
            raw, decoding = decode_notes(mel, duration, skeleton, checkpoints[mode], mode,
                                         args.bpm, args.offset, args.seed, device, control,
                                         max_tokens=args.max_decoder_steps)
            # Raw artifacts remain diagnostic only; an over-budget decoder may
            # contain an early EOS and must never receive a completion marker.
            raw_name, chart_name, events_name = f'{mode}.raw.json', f'{mode}.json', f'{mode}.events.json'
            chart_document(raw, args.bpm, args.offset, directory/raw_name)
            end_tick = (duration-args.offset)/tick_seconds(1, args.bpm)
            final, repairs = (constrain_4k_notes(raw, end_tick) if mode == '4k'
                              else constrain_free_notes(raw, end_tick))
            chart = chart_document(final, args.bpm, args.offset, directory/chart_name)
            events, checks = validate_chart(chart, mode, duration, skeleton)
            events['source_audio_sha256'] = source_hash
            write_json(directory/events_name, events)
            changes = repair_log(raw, final)
            write_json(directory/f'{mode}.repairs.json', dict(profile='static-v1', summary=repairs,
                                                            changes=changes))
            report['routes'][mode] = dict(**checks, **decoding, repairs=repairs,
                                         changed_notes=len(changes), chart_file=chart_name,
                                         raw_file=raw_name, events_file=events_name,
                                         elapsed_seconds=time.monotonic()-phase)
            routes[mode] = dict(events=events['events'], chart_file=chart_name, events_file=events_name)
            control.check()
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        control.update('preview', progress=None)
        budget.check(force=True, extra_bytes=16*1024**2)
        write_preview(directory/'preview.html', title=source.stem, audio_name=audio_path.name,
                      routes=routes, duration=duration)
        control.check()
        report.update(status='completed', elapsed_seconds=time.monotonic()-started,
                      preview='preview.html', resources=budget.log_state())
        report['resources']['process_peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/GIB
        report['artifacts'] = {path.name: dict(sha256=sha256(path), bytes=path.stat().st_size)
                               for path in sorted(directory.iterdir()) if path.is_file() and path.name != 'job.json'}
        write_json(directory/'manifest.json', report)  # sole completion marker
        control.update('complete', status='completed', progress=None, manifest='manifest.json')
        return report
    except BaseException as error:
        (directory/'manifest.json').unlink(missing_ok=True)
        status = 'cancelled' if isinstance(error, (KeyboardInterrupt, JobCancelled)) else 'failed'
        message = str(error) or type(error).__name__
        if control:
            control.update(status=status, error=dict(type=type(error).__name__, message=message))
        else:
            write_json(directory/'job.json', dict(schema=JOB_SCHEMA, status=status,
                       stage='preflight', error=dict(type=type(error).__name__, message=message)))
        raise

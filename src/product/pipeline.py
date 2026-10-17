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
from src.resource_budget import GIB, ResourceBudget, resource_snapshot
from src.test_decoder import load_models
from . import JOB_SCHEMA
from .artifacts import chart_document, repair_log, validate_chart, write_json, write_preview
from .tempo import estimate_tempo, provided_tempo, TempoRequired
from .phira import build_phira_package
from .limits import validate_ram_reserve


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
                 max_tokens=6000, encoder_mode='full'):
    """Keep the AR state for the entire song; bound only encoder attention.

    Encoder windows are explicitly versioned and require quality calibration.
    Natural EOS and exact candidate coverage are independently required.
    """
    encoding_start = time.monotonic()
    enc, dec = load_models(str(checkpoint), device)
    expected_lanes = 4 if mode == '4k' else 64
    if dec.n_lanes != expected_lanes:
        raise ValueError(f'{mode} requires a {expected_lanes}-position decoder checkpoint')
    frames = math.ceil(mel.shape[-1]/enc.mel_per_frame)
    attention_frames = frames if encoder_mode == 'full' else min(frames,640)
    hidden = enc.conv_in.out_channels
    # Conservative math-attention fallback bound: two score matrices in one
    # layer, plus feature/MLP workspaces and the complete decoder KV cache.
    # Actual SDPA backends may use much less. Never silently change encoders.
    attention_bytes = 2*enc.num_heads*attention_frames**2*4
    work_bytes = attention_frames*hidden*4*16 + frames*hidden*4*4
    from src.data.chart_tokenizer import N_DTB, N_DTT
    audit_tk=ChartTokenizer(n_lanes=dec.n_lanes)
    extra_forced_steps=0;previous_tick=0
    for tick in skeleton:
        if tick > previous_tick:
            chain=[]
            audit_tk._emit_span(chain,tick-previous_tick,audit_tk.dtb,audit_tk.dtt,N_DTB,N_DTT)
            extra_forced_steps+=max(0,len(chain)-1)
        previous_tick=tick
    decoder_hidden=dec.tok_emb.embedding_dim
    cache_tokens=max_tokens+extra_forced_steps+3
    cache_bytes = 2*(cache_tokens+frames)*decoder_hidden*4*len(dec.blocks)
    model_bytes = Path(checkpoint).stat().st_size
    cpu_extra = model_bytes*2 + frames*hidden*4*4
    device_extra = attention_bytes + work_bytes + cache_bytes
    if device.type == 'cpu':
        cpu_extra += device_extra
    snapshot = resource_snapshot(control.directory)
    if snapshot['rss_gib']+cpu_extra/GIB > control.budget.args.max_rss_gib or \
            snapshot['ram_available_gib']-cpu_extra/GIB < control.budget.args.min_available_ram_gib:
        raise RuntimeError('Resource budget: full-song layout estimate exceeds RAM reserve/RSS; '
                           'use a shorter input or explicitly test the experimental chunked encoder')
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        permitted = total*control.budget.args.gpu_memory_fraction-torch.cuda.memory_allocated(device)
        if device_extra > min(free,permitted):
            raise RuntimeError('Resource budget: layout attention estimate exceeds available GPU budget')
    preflight = dict(cpu_extra_gib=cpu_extra/GIB, device_extra_gib=device_extra/GIB,
                     decoder_hidden=decoder_hidden,cache_token_bound=cache_tokens,
                     extra_forced_dt_steps=extra_forced_steps,
                     method='conservative-two-score-matrices-and-forced-dt-kv-v2')
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
                       encoding_preflight=preflight,
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
    resource_limits = None
    try:
        args.min_available_ram_gib = validate_ram_reserve(args.min_available_ram_gib)
        resource_limits = dict(min_available_ram_gib=args.min_available_ram_gib,
                               min_free_disk_gib=args.min_free_disk_gib,
                               max_rss_gib=args.max_rss_gib,gpu_memory_fraction=args.gpu_memory_fraction)
        if args.bpm is not None and (not math.isfinite(args.bpm) or not 1 <= args.bpm <= 1000):
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
        control.update(resource_limits=resource_limits)
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
                                  layout_encoder=args.layout_encoder,
                                  resource_limits=resource_limits,
                                  max_seconds=args.max_seconds, timeout_seconds=args.timeout_seconds),
                      model_sha256={mode: sha256(checkpoints[mode]) for mode in modes},
                      stages={}, routes={})
        report['implementation']['product_source_sha256'] = {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*')) if path.is_file()}
        report['runtime']['device_name'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
        control.update('audio_features', progress=None)
        phase = time.monotonic()
        mel, duration = load_audio(audio_path, args.max_seconds, budget=budget)
        control.check()
        report['stages']['audio_features_seconds']=time.monotonic()-phase
        control.update('tempo',progress=None)
        phase=time.monotonic()
        tempo=estimate_tempo(mel,duration) if args.bpm is None else provided_tempo(args.bpm,duration)
        tempo['audio_sha256']=source_hash
        write_json(directory/'tempo.json',tempo)
        if tempo['status']=='needs_tempo':
            raise TempoRequired('tempo could not be estimated; enter BPM in optional settings and retry')
        bpm=tempo['timebase_bpm']
        report['tempo']=tempo
        report['config'].update(bpm=bpm,timebase_bpm=bpm,timing_source=tempo['source'])
        report['stages']['tempo_seconds']=time.monotonic()-phase
        control.check()
        control.update('audio_understanding',progress=None)
        phase=time.monotonic()
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
        skeleton, quantization = quantize_candidates(evidence, bpm, args.offset)
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
                                         bpm, args.offset, args.seed, device, control,
                                         max_tokens=args.max_decoder_steps, encoder_mode=args.layout_encoder)
            # Raw artifacts remain diagnostic only; an over-budget decoder may
            # contain an early EOS and must never receive a completion marker.
            raw_name, chart_name, events_name = f'{mode}.raw.json', f'{mode}.json', f'{mode}.events.json'
            chart_document(raw, bpm, args.offset, directory/raw_name)
            end_tick = (duration-args.offset)/tick_seconds(1, bpm)
            final, repairs = (constrain_4k_notes(raw, end_tick) if mode == '4k'
                              else constrain_free_notes(raw, end_tick))
            chart = chart_document(final, bpm, args.offset, directory/chart_name)
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
        control.update('packages',progress=None)
        report['packages']={}
        for mode in modes:
            package=build_phira_package(directory/report['routes'][mode]['chart_file'],audio_path,
                directory/f'{mode}.phira.zip',mode=mode,title=source.stem,duration_seconds=duration,
                source=dict(kind='model_generated',generation_commit=report['implementation']['git_commit'],
                            model_sha256=report['model_sha256'][mode],
                            audio_checkpoint_sha256=metadata['checkpoint_sha256'],
                            timebase_source=tempo['source'],seed=args.seed,repair_profile='static-v1'),
                check=control.check,min_available_ram_gib=args.min_available_ram_gib)
            report['packages'][mode]=package
            report['routes'][mode]['package_file']=package['file']
            routes[mode]['package_file']=package['file']
        control.update('preview', progress=None)
        budget.check(force=True, extra_bytes=16*1024**2)
        write_preview(directory/'preview.html', title=source.stem, audio_name=audio_path.name,
                      routes=routes, duration=duration,tempo=tempo)
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
        for mode in ('4k','freex'):
            (directory/f'{mode}.phira.zip').unlink(missing_ok=True)
        status = ('cancelled' if isinstance(error, (KeyboardInterrupt, JobCancelled)) else
                  'needs_tempo' if isinstance(error,TempoRequired) else 'failed')
        message = str(error) or type(error).__name__
        if control:
            control.update(status=status, error=dict(type=type(error).__name__, message=message))
        else:
            write_json(directory/'job.json', dict(schema=JOB_SCHEMA, status=status,
                       stage='preflight', resource_limits=resource_limits,
                       error=dict(type=type(error).__name__, message=message)))
        raise

"""Bounded attention windows, absolute audio time, and inspectable evidence."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from src.data.audio2mel import AudioGPUprocessor
from src.data.timing import check_checkpoint_timing
from src.models.transcriber import TranscriberNet
from src.models.rhythm_skeleton import onset_candidates, select_onsets, candidates_to_ticks
from src.resource_budget import resource_snapshot, GIB
from . import SCHEMA, FEATURE_SCHEMA, FRAME_SECONDS, SAMPLE_RATE


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_version():
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, text=True, capture_output=True)
    return dict(git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                working_tree_dirty=bool(dirty.stdout.strip()),
                source_sha256={str(path.relative_to(root)): sha256(path) for path in
                               sorted((root/'src/audio_understanding').glob('*.py'))})


def load_audio(path, max_seconds=600, budget=None):
    """Reuse the checkpoint's GPU-processor feature convention, on CPU.

    Inspect headers before decoding; attention is chunked but mel is whole-song
    so dB normalization remains identical across window boundaries.
    """
    info = sf.info(path)
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be positive and finite')
    if not 0 < info.duration <= max_seconds or info.channels > 8 or info.samplerate > 192000:
        raise ValueError('audio exceeds supported duration/channels/sample-rate budget')
    if budget:
        budget.check(force=True)
        # Conservative input/resampling/STFT working-set estimate before decode.
        source_bytes = info.frames * info.channels * 4
        resampled_bytes = math.ceil(info.duration*SAMPLE_RATE)*info.channels*4
        spectral_bytes = math.ceil(info.duration/FRAME_SECONDS)*1025*24
        extra_gib = (3*source_bytes + 4*resampled_bytes + spectral_bytes)/GIB
        state = resource_snapshot(budget.directory)
        if state['ram_available_gib']-extra_gib < budget.args.min_available_ram_gib \
                or state['rss_gib']+extra_gib > budget.args.max_rss_gib:
            raise RuntimeError('Resource budget: audio decode estimate exceeds RAM reserve/RSS limit')
    proc = AudioGPUprocessor(device='cpu')
    waveform = proc.load_from_path(path)
    if not torch.isfinite(waveform).all():
        raise ValueError('non-finite audio samples')
    duration = waveform.shape[-1] / SAMPLE_RATE
    # reflect padding in the existing STFT needs at least n_fft/2+1 samples.
    waveform = F.pad(waveform, (0, max(0, 1025 - waveform.shape[-1])))
    mel = proc(waveform).squeeze(0)
    frames = math.ceil(duration / FRAME_SECONDS)
    if budget:
        budget.check(force=True)
    return mel[:, :frames], duration


def load_model(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    check_checkpoint_timing(ck)
    if ck.get('audio_schema') not in (None, SCHEMA):
        raise ValueError('unsupported audio checkpoint schema')
    if ck.get('audio_schema') and ck.get('feature_schema') != FEATURE_SCHEMA:
        raise ValueError('incompatible audio feature schema')
    model = TranscriberNet(**ck['config'])
    model.load_state_dict(ck.get('ema', ck['model']))
    if model.conv_in.in_channels != 128 or model.mel_per_frame != 2:
        raise ValueError('audio interface requires 128 mel bins and mel_per_frame=2')
    domain = ck.get('label_source', 'chart_onset')
    if domain not in ('reviewed_acoustic', 'pseudo_acoustic', 'chart_onset'):
        raise ValueError('unknown checkpoint label source')
    if domain != 'chart_onset' and (not ck.get('audio_schema') or model.rhythm_head is None):
        raise ValueError('acoustic checkpoint requires a versioned event head')
    metadata = dict(checkpoint_sha256=sha256(path), label_source=domain,
                    feature_schema=FEATURE_SCHEMA, training=ck.get('provenance', {}))
    return model.to(device).eval(), metadata


@torch.no_grad()
def infer_mel(model, mel, duration, *, chunk_frames=1024, context_frames=128, check=None):
    """Disjoint core windows with context halos, no duplicate or missing tail.

    Returned features remain at the encoder's rate; probabilities/offsets keep
    their native head rate. All arrays stay on CPU between forward passes.
    Chunked bidirectional attention is intentionally not full-song attention.
    """
    mpf = model.mel_per_frame
    if chunk_frames <= 0 or context_frames < 0 or chunk_frames % mpf or context_frames % mpf:
        raise ValueError('window sizes must be nonnegative multiples of mel_per_frame; core > 0')
    if mel.ndim != 2 or not torch.isfinite(mel).all() or not math.isfinite(duration) or duration <= 0:
        raise ValueError('finite mel and positive duration required')
    actual = math.ceil(duration / FRAME_SECONDS)
    if mel.shape[-1] < actual:
        raise ValueError('mel does not cover complete audio')
    total = math.ceil(actual / mpf) * mpf
    mel = F.pad(mel[:, :actual].cpu(), (0, total - actual))
    device = next(model.parameters()).device
    global_head = model.rhythm_head is not None
    step = 1 if global_head else mpf
    features, logits = [], []
    model.eval()
    for start in range(0, total, chunk_frames):
        if check:
            check()
        end = min(start + chunk_frames, total)
        lo, hi = max(0, start - context_frames), min(total, end + context_frames)
        h = model.features(mel[:, lo:hi].unsqueeze(0).to(device))
        pred = model.rhythm_from_features(h) if global_head else model.head(h).transpose(1, 2)
        features.append(h[0, (start-lo)//mpf:(end-lo)//mpf].cpu())
        logits.append(pred[:, :, (start-lo)//step:(end-lo)//step].cpu())
    logits = torch.cat(logits, -1)[:, :, :math.ceil(actual / step)]
    times, confidence = onset_candidates(logits, FRAME_SECONDS * step * 1000)
    times = np.asarray(times, dtype=np.float64) / 1000
    confidence = np.asarray(confidence, dtype=np.float32)
    frame_times = np.arange(len(times), dtype=np.float64) * FRAME_SECONDS * step
    return dict(features=torch.cat(features).numpy(),
                feature_times_seconds=np.arange(total//mpf) * FRAME_SECONDS * mpf,
                frame_times_seconds=frame_times, event_times_seconds=times,
                event_probability=confidence, valid_event_time=times < duration,
                time_offset_seconds=times-frame_times)


def make_result(arrays, duration, metadata, *, threshold=.7, min_interval_ms=35):
    valid = arrays['valid_event_time']
    times = arrays['event_times_seconds'][valid]
    scores = arrays['event_probability'][valid]
    selected = select_onsets((times * 1000).tolist(), scores.tolist(), threshold, min_interval_ms)
    by_time = dict(zip((times * 1000).tolist(), scores.tolist()))
    return dict(schema=SCHEMA, time_basis='audio_seconds', duration_seconds=duration,
                **metadata, selection=dict(threshold=threshold, min_interval_ms=min_interval_ms,
                                           calibration='manual_unvalidated'),
                events=[dict(id=f'event-{i:06d}', time_seconds=t / 1000, confidence=by_time[t])
                        for i, t in enumerate(selected)],
                capabilities=dict(events=metadata['label_source'], continuous_features=True,
                                  beat='unavailable', downbeat='unavailable',
                                  sustain='unavailable', pitch='unavailable', sections='unavailable'),
                tempo_map=None, status='candidates_available' if selected else 'no_candidates')


def result_to_ticks(result, bpm, offset=0.0):
    """Explicit downstream selection baseline, shared by 4K and free-x.

    Constant BPM is supplied by the caller; this never pretends to infer tempo.
    Difficulty-aware event selection remains the next product stage.
    """
    if result.get('schema') != SCHEMA or result.get('time_basis') != 'audio_seconds':
        raise ValueError('incompatible audio result schema/time basis')
    duration = result['duration_seconds']
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid result duration')
    events = result['events']
    if any(not math.isfinite(e['time_seconds']) or not 0 <= e['time_seconds'] < duration
           or not math.isfinite(e['confidence']) or not 0 <= e['confidence'] <= 1 for e in events):
        raise ValueError('invalid event time or confidence')
    ticks = candidates_to_ticks([e['time_seconds']*1000 for e in events],
                               [e['confidence'] for e in events], bpm, offset, threshold=0)
    return [t for t in ticks if 0 <= t * 60 / (32*bpm) + offset < duration]

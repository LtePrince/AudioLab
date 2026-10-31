"""Bounded group-clock input noise; clean next-event and layout labels stay fixed.

This teaches recovery from small timing errors, not missing/extra event recovery.
All clock-dependent model inputs are rebuilt. No clean-clock side channel is
added, and the next clean target must remain reachable under the output protocol.
"""
from __future__ import annotations

import math
import random

import torch

from src.anchored_ar import MIN_PROGRESS_SECONDS
from src.data.chart_tokenizer import BOS, ChartTokenizer
from src.models.chart_history import history_features
from src.soft_onset import MEMORY_SECONDS

SCHEMA = 'audiolab-bounded-group-history-time-noise-v1'


def perturb_history_times(target, *, n_lanes: int, max_seconds: float, rng: random.Random):
    """Return a new input view and a noise audit, without mutating clean targets.

    Sample uniformly on the legal intersection of the +/- noise interval,
    monotone history and current output window. Report restricted intervals;
    unlike clipping a draw, this creates no point mass at the next label.
    """
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('history noise amplitude must be finite and nonnegative')
    audit = dict(groups=0, restricted_groups=0, unchanged_groups=0,
                 absolute_noise_seconds=0., max_absolute_noise_seconds=0.)
    if max_seconds == 0:
        return target, audit
    tk = ChartTokenizer(n_lanes)
    tokens = target['tokens'].tolist()
    frames = target['frames'].clone()
    bounds = target.get('bounds')
    anchors = [i for i, token in enumerate(tokens) if token == tk.cond(0)]
    reference = target['reference_times']
    if len(anchors) != len(reference):
        raise ValueError('one clean reference time per ANCHOR group required')
    # Enough headroom for the float32 clock round trip even in long recordings.
    margin = max(1e-5, float(target['duration']) * 4 * torch.finfo(frames.dtype).eps)
    for group, start in enumerate(anchors):
        stop = start + 1
        while stop < len(tokens) and tokens[stop-1] != tk.cond(1):
            stop += 1
        if tokens[stop-1] != tk.cond(1):
            raise ValueError('history group has no END_GROUP')
        original = float(reference[group])
        previous = float(frames[start-1]) * MEMORY_SECONDS
        window_start, window_end = (map(float, bounds[start]) if bounds is not None
                                    else (0., float(target['duration'])))
        lo = max(original-max_seconds, window_start, previous+margin)
        hi = min(original+max_seconds, window_end-margin)
        if group+1 < len(reference) and float(reference[group+1]) < window_end:
            hi = min(hi, float(reference[group+1])-margin)
        audit['groups'] += 1
        audit['restricted_groups'] += int(lo > original-max_seconds or hi < original+max_seconds)
        if lo >= hi:
            # At very tight boundaries use the original frame if it is legal;
            # do not invent a minimum musical inter-onset interval.
            candidate = target['frames'][start]
            audit['unchanged_groups'] += 1
        else:
            candidate = frames.new_tensor(rng.uniform(lo, hi) / MEMORY_SECONDS)
        frames[start:stop] = candidate
        actual = float(candidate) * MEMORY_SECONDS
        if abs(actual-original) > max_seconds + margin:
            raise ValueError('history clock outside its numerical noise budget')
        delta = abs(actual-original)
        audit['absolute_noise_seconds'] += delta
        audit['max_absolute_noise_seconds'] = max(audit['max_absolute_noise_seconds'], delta)
    if (frames[1:] < frames[:-1]).any():
        raise ValueError('history perturbation violated causal clock order')
    result = dict(target)
    result['frames'] = frames
    result['history'] = history_features(tokens, frames, n_lanes=n_lanes,
        frames_per_beat=60/target['bpm']/MEMORY_SECONDS,
        offsets=target.get('offsets') if n_lanes == 64 else None)
    lower = []
    for i in target['decisions'].tolist():
        if tokens[i] in (BOS, tk.cond(2)):
            value = float(bounds[i, 0]) if bounds is not None else 0.
        else:
            value = float(frames[i])*MEMORY_SECONDS+MIN_PROGRESS_SECONDS
            if bounds is not None:
                value = max(value, float(bounds[i, 0]))
        lower.append(value)
    result['lower_seconds'] = torch.tensor(lower, dtype=torch.float64)
    positive = target['stop_targets'] == 0
    if (result['lower_seconds'][positive] > target['next_seconds'][positive]).any():
        raise ValueError('history perturbation made a clean time target unreachable')
    return result, audit

"""Audio onset candidates and optional sub-frame confidence suppression."""
from __future__ import annotations

import bisect
import math

import torch

from src.models.transcriber import H_ONSET, H_SOFF


def onset_candidates(pred: torch.Tensor, frame_ms: float) -> tuple[list[float], list[float]]:
    """Transcriber (1,32,T) -> audio milliseconds and any-lane confidence."""
    lane_p = pred[0, H_ONSET:H_ONSET + 4].sigmoid()
    confidence = 1 - (1 - lane_p).prod(0)
    offsets = pred[0, H_SOFF:H_SOFF + 4].sigmoid()
    peak_lane = lane_p.argmax(0, keepdim=True)
    subframe = offsets.gather(0, peak_lane)[0].double()
    times = (torch.arange(len(subframe), device=pred.device) + subframe) * frame_ms
    return times.tolist(), confidence.tolist()


def select_onsets(times_ms, confidence, threshold=0.7, min_interval_ms=0.0):
    """Keep confident peaks; optional NMS retains the strongest nearby onset.

    Work in sub-frame audio time, not chart-frame indices. Stable tie-breaking
    favors the earlier onset. A zero interval reproduces threshold-only mode.
    """
    if not 0 <= threshold <= 1 or not math.isfinite(min_interval_ms) or min_interval_ms < 0:
        raise ValueError('threshold must be in [0,1] and min interval nonnegative')
    if len(times_ms) != len(confidence):
        raise ValueError('time/confidence lengths differ')
    candidates = [(t, p) for t, p in zip(times_ms, confidence) if p >= threshold]
    if not min_interval_ms:
        return sorted({t for t, _ in candidates})
    kept = []
    for t, _ in sorted(candidates, key=lambda v: (-v[1], v[0])):
        i = bisect.bisect_left(kept, t)
        if i and t - kept[i - 1] < min_interval_ms:
            continue
        if i < len(kept) and kept[i] - t < min_interval_ms:
            continue
        kept.insert(i, t)
    return kept


def candidates_to_ticks(times_ms, confidence, bpm, offset=0.0, threshold=0.7, min_interval_ms=0.0):
    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(offset):
        raise ValueError('bpm must be positive and offset finite')
    if len(times_ms) != len(confidence):
        raise ValueError('time/confidence lengths differ')
    pairs = [(t, p) for t, p in zip(times_ms, confidence) if t >= offset * 1000]
    times = select_onsets([t for t, _ in pairs], [p for _, p in pairs], threshold, min_interval_ms)
    return sorted({round((t / 1000 - offset) * bpm * 32 / 60) for t in times})

"""Deterministic output constraints for generated static-line charts.

These repairs improve exported charts, not the learned distribution. Callers
retain raw generation for quality comparisons and report every changed note.
Times and durations use the chart's normalized tick clock (no audio offset).
"""
from __future__ import annotations

import math


def constrain_4k_notes(notes: list[dict], end_tick: float) -> tuple[list[dict], dict]:
    """Clip a Hold to the next onset in its lane and to the audio endpoint.

    Keep onset, lane and note count unchanged. A Hold without a whole tick of
    room becomes a Tap. Inputs are copied; no source or model data is modified.
    """
    if not math.isfinite(end_tick):
        raise ValueError('end_tick must be finite')
    result = [dict(n) for n in notes]
    next_onset = {}
    report = dict(shortened_holds=0, holds_to_taps=0)
    for n in sorted(result, key=lambda n: n['time'], reverse=True):
        lane = n['positionX']  # tokenizer emits one exact anchor per 4K lane
        if n['type'] == 3:
            limit = min(end_tick, next_onset.get(lane, end_tick))
            dur = max(0, min(n['holdTime'], math.floor(limit - n['time'] + 1e-7)))
            if dur != n['holdTime']:
                report['shortened_holds'] += 1
            if dur < 1:
                n['type'] = 1
                dur = 0
                report['holds_to_taps'] += 1
            n['holdTime'] = dur
        next_onset[lane] = n['time']
    return result, report

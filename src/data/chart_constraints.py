"""Deterministic output constraints for generated static-line charts.

These repairs improve exported charts, not the learned distribution. Callers
retain raw generation for quality comparisons and report every changed note.
Times and durations use the chart's normalized tick clock (no audio offset).
"""
from __future__ import annotations

import math
from itertools import groupby


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
                n['speed'] = 1.0
                dur = 0
                report['holds_to_taps'] += 1
            n['holdTime'] = dur
        next_onset[lane] = n['time']
    return result, report


def _space_chord(positions: list[float], gap: float, low: float, high: float) -> list[float]:
    """Least-squares projection of sorted x onto bounds and minimum spacing.

    Subtract i*gap, then apply pooled adjacent violators (isotonic regression).
    Clipping the fitted blocks to common bounds preserves their order.
    """
    upper = high - (len(positions) - 1) * gap
    if upper < low - 1e-9:
        raise ValueError('chord cannot fit minimum spacing; reduce free-min-gap or use raw-output')
    blocks = []
    for i, x in enumerate(positions):
        blocks.append([x - i * gap, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    fitted = [min(max(total / count, low), max(low, upper))
              for total, count in blocks for _ in range(count)]
    return [z + i * gap for i, z in enumerate(fitted)]


def constrain_free_notes(notes: list[dict], end_tick: float,
                         min_gap: float = 0.04) -> tuple[list[dict], dict]:
    """Space static-line chords, then release Holds before nearby future hits.

    Screen x is bounded to [0.05, 0.95]. The gap is an adjustable readability
    heuristic, not a guarantee of human playability or support for moving lines.
    Note count, onset and left-to-right chord order are preserved. Infeasible
    chords raise an error instead of silently discarding model output.
    """
    if not math.isfinite(end_tick) or not math.isfinite(min_gap) or not 0 < min_gap <= 0.9:
        raise ValueError('end_tick must be finite and min_gap must be in (0, 0.9]')
    unit = 0.05625  # native Phigros positionX -> screen width on a centered static line
    result = [dict(n) for n in notes]
    report = dict(moved_notes=0, max_displacement=0.0, shortened_holds=0, holds_to_taps=0)
    for _, group in groupby(sorted(result, key=lambda n: n['time']), key=lambda n: n['time']):
        chord = sorted(group, key=lambda n: n['positionX'])
        original = [0.5 + n['positionX'] * unit for n in chord]
        positions = _space_chord(original, min_gap, 0.05, 0.95)
        for n, before, after in zip(chord, original, positions):
            if abs(after - before) > 1e-9:
                n['positionX'] = (after - 0.5) / unit
                report['moved_notes'] += 1
                report['max_displacement'] = max(report['max_displacement'], abs(after - before))
    future = []
    for n in sorted(result, key=lambda n: n['time'], reverse=True):
        x = 0.5 + n['positionX'] * unit
        if n['type'] == 3:
            limit = min((t for t, fx in future if abs(x - fx) < min_gap - 1e-9), default=end_tick)
            dur = max(0, min(n['holdTime'], math.floor(min(end_tick, limit) - n['time'] + 1e-7)))
            if dur != n['holdTime']:
                report['shortened_holds'] += 1
            if dur < 1:
                n['type'] = 1
                n['speed'] = 1.0
                dur = 0
                report['holds_to_taps'] += 1
            n['holdTime'] = dur
        future.append((n['time'], x))
    return result, report

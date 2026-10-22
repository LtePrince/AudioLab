"""Validated chart artifacts and a portable, offline visual preview.

The preview is not a gameplay judge. Exported notes retain the existing static
Phigros v3 convention and are independently checked in audio seconds.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

from src.data.chart2array import LANE_CENTERS, save_phigros_notes
from src.data.timing import tick_seconds
from . import EVENT_SCHEMA


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                             encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def chart_document(notes, bpm, offset, path):
    # The historical writer scales floorPosition in place: never give it raw
    # model notes or the notes used to construct the repair log.
    save_phigros_notes(deepcopy(notes), bpm, str(path), offset=offset)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_chart(chart, mode, duration, skeleton_ticks):
    """Reject incomplete/empty output and hard format/time/position errors.

    This deliberately makes no claims about finger count, difficulty, or fun.
    Skeleton coverage is an interface invariant, not musical quality evidence.
    """
    if mode not in ('4k', 'freex') or not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid mode or duration')
    if chart.get('formatVersion') != 3 or len(chart.get('judgeLineList', [])) != 1:
        raise ValueError('expected one static Phigros v3 line')
    line = chart['judgeLineList'][0]
    bpm, offset = line['bpm'], chart['offset']
    tick = tick_seconds(1, bpm)
    if not math.isfinite(offset):
        raise ValueError('non-finite chart offset')
    notes = line['notesAbove'] + line['notesBelow']
    if not notes:
        raise ValueError('empty chart is not a successful generation')
    observed = set()
    events = []
    counts = {name: 0 for name in ('tap', 'drag', 'hold', 'flick')}
    names = {1: 'tap', 2: 'drag', 3: 'hold', 4: 'flick'}
    for index, note in enumerate(notes):
        kind = note['type']
        if type(kind) is not int or kind not in names:
            raise ValueError('invalid note type')
        if any(not math.isfinite(note[key]) for key in
               ('time', 'holdTime', 'positionX', 'floorPosition', 'speed')):
            raise ValueError('non-finite note')
        if note['time'] < 0 or note['holdTime'] < 0 or note['speed'] <= 0:
            raise ValueError('invalid note time, duration, or speed')
        if (kind == 3) != (note['holdTime'] > 0):
            raise ValueError('Hold duration does not match note type')
        start = note['time'] * tick + offset
        end = (note['time'] + note['holdTime']) * tick + offset
        if start < -1e-8 or start >= duration or end > duration + 1e-8:
            raise ValueError('note lies outside audio timeline')
        x = .5 + note['positionX'] * .05625
        if mode == '4k' and not any(abs(note['positionX'] - lane) < 1e-7 for lane in LANE_CENTERS):
            raise ValueError('4K note is not on a fixed lane')
        if mode == 'freex' and not .05 - 1e-8 <= x <= .95 + 1e-8:
            raise ValueError('free-x note is outside the visible interval')
        speed = line['speedEvents'][0]['value']
        if abs(note['floorPosition'] - speed * note['time'] * tick) > 1e-5:
            raise ValueError('visual onset differs from musical onset')
        if kind == 3 and abs(note['speed'] - speed) > 1e-7:
            raise ValueError('Hold tail speed differs from line speed')
        observed.add(note['time'])
        counts[names[kind]] += 1
        events.append(dict(id=f'{mode}-note-{index:06d}', source_index=index,
                           type=names[kind], time_seconds=start, end_seconds=end,
                           screen_x=x, lane=min(range(4), key=lambda lane: abs(LANE_CENTERS[lane]-note['positionX']))
                           if mode == '4k' else None))
    expected = set(skeleton_ticks) if skeleton_ticks is not None else None
    if expected is not None and observed != expected:
        raise ValueError(f'incomplete candidate coverage: {len(expected-observed)} missing, '
                         f'{len(observed-expected)} unexpected onsets; output may be truncated')
    return dict(schema=EVENT_SCHEMA, time_basis='audio_seconds', mode=mode,
                duration_seconds=duration, difficulty='baseline', events=events), dict(
                    note_count=len(notes), note_types=counts, unique_onsets=len(observed),
                    candidate_ticks=len(expected) if expected is not None else None,
                    complete_candidate_coverage=True if expected is not None else None,
                    hard_checks='passed', human_playability='unverified',
                    target_player_compatibility='unverified')


def repair_log(raw, final):
    if len(raw) != len(final):
        raise ValueError('static repair must preserve note identity/count')
    changes = []
    for index, (before, after) in enumerate(zip(raw, final)):
        fields = {key: dict(before=before[key], after=after[key]) for key in
                  ('type', 'time', 'holdTime', 'positionX') if before[key] != after[key]}
        if fields:
            changes.append(dict(source_index=index, changes=fields))
    return changes


def write_preview(path, *, title, audio_name, routes, duration, tempo=None):
    """Embed chart data, not executable user text; requires no HTTP/CDN/fetch."""
    payload = json.dumps(dict(title=title, audio=audio_name, duration=duration, routes=routes,tempo=tempo),
                         ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e')
    template = Path(__file__).with_name('preview.html').read_text(encoding='utf-8')
    Path(path).write_text(template.replace('__AUDIOLAB_DATA__', payload), encoding='utf-8')

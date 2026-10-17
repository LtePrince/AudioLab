"""Reviewed coverage is explicit: unlabeled audio is never a negative label."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from . import ANNOTATION_SCHEMA, FRAME_SECONDS


def intervals(values, duration):
    result = []
    for start, end in sorted(values):
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= duration:
            raise ValueError('intervals must satisfy 0 <= start < end <= duration')
        if result and start < result[-1][1]:
            raise ValueError('intervals must not overlap')
        if result and start == result[-1][1]:
            result[-1][1] = end
        else:
            result.append([start, end])
    return result


def reviewed_regions(record):
    regions = intervals(record['coverage'], record['duration_seconds'])
    for lo, hi in intervals(record.get('uncertain', []), record['duration_seconds']):
        regions = [[a, b] for start, end in regions
                   for a, b in ((start, min(end, lo)), (max(start, hi), end)) if a < b]
    return regions


def contains(time, regions):
    return any(start <= time < end for start, end in regions)


def read_manifest(path):
    path = Path(path)
    records, ids = [], set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('schema') != ANNOTATION_SCHEMA:
            raise ValueError('unsupported annotation schema')
        for key in ('id', 'source_group', 'audio', 'provenance'):
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f'annotation requires {key}')
        if r['id'] in ids:
            raise ValueError('duplicate annotation id')
        ids.add(r['id'])
        if r.get('label_source') not in ('reviewed_acoustic', 'pseudo_acoustic', 'chart_onset'):
            raise ValueError('explicit label_source required')
        duration = r['duration_seconds']
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('positive finite duration required')
        source_start = r.get('source_start_seconds', 0.)
        if not math.isfinite(source_start) or source_start < 0:
            raise ValueError('source_start_seconds must be finite and nonnegative')
        coverage = intervals(r['coverage'], duration)
        if not reviewed_regions(r):
            raise ValueError('nonempty reviewed coverage required')
        if any(not math.isfinite(t) or not contains(t, coverage) for t in r['events_seconds']):
            raise ValueError('events must be finite audio times inside coverage')
        # Unique onsets at microsecond precision; no hidden chart-offset shift.
        r['events_seconds'] = sorted(set(round(t, 6) for t in r['events_seconds']))
        if any(not contains(t, coverage) for t in r['events_seconds']):
            raise ValueError('rounded event falls outside coverage')
        r['audio'] = str((path.parent / r['audio']).resolve())
        records.append(r)
    if not records:
        raise ValueError('empty annotation manifest')
    return records


def event_targets(record, frames, frame_seconds=FRAME_SECONDS):
    """Only fully reviewed bins train negatives; collisions retain earliest onset.

    One offset cannot represent two distinct events in a bin. Report the lost
    count and mask its offset loss instead of inventing an averaged event.
    """
    regions = reviewed_regions(record)
    starts = torch.arange(frames, dtype=torch.float64) * frame_seconds
    ends = torch.minimum(starts + frame_seconds, starts.new_tensor(record['duration_seconds']))
    valid = torch.zeros(frames, dtype=torch.bool)
    for a, b in regions:
        valid |= (starts >= a - 1e-10) & (ends <= b + 1e-10) & (starts < ends)
    target = torch.zeros(2, frames)
    counts = torch.zeros(frames, dtype=torch.long)
    for time in sorted(set(record['events_seconds'])):
        i = math.floor(time / frame_seconds)
        if 0 <= i < frames and valid[i] and contains(time, regions):
            if counts[i] == 0:
                target[1, i] = time / frame_seconds - i
            counts[i] += 1
    target[0] = (counts > 0).float()
    return target, valid, counts == 1, int((counts - 1).clamp_min(0).sum())

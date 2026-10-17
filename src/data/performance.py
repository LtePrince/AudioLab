"""Query original PGR curves without quantizing their native line clocks.

This is a declared mathematical interpretation, not an official renderer.
Ambiguous interior gaps/overlaps remain unsupported. Source data survives the
canonical round trip; no generated static chart is used as a dynamic label.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


SCHEMA = 'audiolab-original-performance-v1'
INTERPRETER = 'native-clock-right-continuous-segments-v1'
X_UNIT_W = .05625
X_UNIT_H = .1
TRACK_FIELDS = {'move': 'judgeLineMoveEvents', 'rotate': 'judgeLineRotateEvents',
                'alpha': 'judgeLineDisappearEvents', 'speed': 'speedEvents'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def number(value):
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError('source fields must be finite numbers')
    return result


def unpack_v1(value):
    """Signed remainder, explicit for negative packed coordinates (Rust/PGR)."""
    value = number(value)
    remainder = math.fmod(value, 1000.)
    return ((value - remainder) / 1000. / 880., remainder / 520.)


@dataclass(frozen=True)
class Segment:
    identifier: str
    start: float
    end: float
    before: tuple[float, ...]
    after: tuple[float, ...]

    def value(self, tick):
        if self.start == self.end:
            return self.after
        fraction = (tick - self.start) / (self.end - self.start)
        return tuple(a + fraction * (b - a) for a, b in zip(self.before, self.after))


class Track:
    """Half-open nonzero spans, right-hand span owns a shared endpoint.

    A zero span supplies its end value at that instant if no right-hand span
    exists. Outside the outermost endpoints, endpoint values are held. This
    does not invent a bridge across an interior gap. Conflicting coverage is
    unsupported; original order/fields are preserved separately in source.
    """
    def __init__(self, kind, source, version, identifier):
        self.kind, self.identifier = kind, identifier
        self.segments = []
        self.issues = []
        self.malformed = False
        for index, event in enumerate(source):
            event_id = f'{identifier}/event-{index}'
            try:
                start, end = number(event['startTime']), number(event['endTime'])
                if kind == 'speed':
                    before = after = (number(event['value']),)
                elif kind == 'move' and version == 1:
                    before, after = unpack_v1(event['start']), unpack_v1(event['end'])
                elif kind == 'move':
                    before = (number(event['start']), number(event['start2']))
                    after = (number(event['end']), number(event['end2']))
                else:
                    before, after = (number(event['start']),), (number(event['end']),)
                self.segments.append(Segment(event_id, start, end, before, after))
                if start >= end:
                    self.issues.append(dict(code='zero_duration' if start == end else 'reversed_interval',
                                            event_ids=[event_id], native_range=[min(start, end), max(start, end)]))
            except (KeyError, ValueError, TypeError, OverflowError) as error:
                self.malformed = True
                self.issues.append(dict(code='malformed_event', event_ids=[event_id], message=str(error)))
        if not source:
            self.issues.append(dict(code='empty_track', event_ids=[]))
        self.valid = sorted((s for s in self.segments if s.end > s.start), key=lambda s: s.start)
        self.by_identifier = {segment.identifier: segment for segment in self.segments}
        original_order = [s.start for s in self.segments if s.end >= s.start]
        if any(b < a for a, b in zip(original_order, original_order[1:])):
            self.issues.append(dict(code='source_event_order_unsorted', event_ids=[s.identifier for s in self.segments]))
        self.starts = [s.start for s in self.valid]
        self.prefix_end = []
        maximum = -math.inf
        for segment in self.valid:
            maximum = max(maximum, segment.end)
            self.prefix_end.append(maximum)
        self.zeros = {}
        for segment in self.segments:
            if segment.start == segment.end:
                self.zeros.setdefault(segment.start, []).append(segment)
        self.reversed = [s for s in self.segments if s.start > s.end]
        self.endings = {}
        for segment in self.valid:
            self.endings.setdefault(segment.end, []).append(segment)
        if self.valid:
            previous = self.valid[0]
            for segment in self.valid[1:]:
                if segment.start < previous.end:
                    self.issues.append(dict(code='overlapping_intervals',
                                            event_ids=[previous.identifier, segment.identifier],
                                            native_range=[segment.start, min(previous.end, segment.end)]))
                elif segment.start > previous.end:
                    self.issues.append(dict(code='interior_gap', event_ids=[previous.identifier, segment.identifier],
                                            native_range=[previous.end, segment.start]))
                if segment.end > previous.end:
                    previous = segment

    def query(self, tick):
        tick = number(tick)
        def answer(value, status, segments=()):
            return dict(value=list(value) if value is not None else None, status=status,
                        event_ids=[s.identifier for s in segments])
        if self.malformed:
            return answer(None, 'unsupported_malformed_track')
        invalid = [s for s in self.reversed if s.end <= tick <= s.start]
        if invalid:
            return answer(None, 'unsupported_reversed_interval', invalid)
        active = []
        index = bisect_right(self.starts, tick) - 1
        while index >= 0 and self.prefix_end[index] > tick:
            segment = self.valid[index]
            if tick < segment.end:
                active.append(segment)
            index -= 1
        if len(active) > 1:
            return answer(None, 'unsupported_overlap', active)
        zeros = self.zeros.get(tick, [])
        if zeros and not any(segment.start == tick for segment in active):
            if any(s.after != zeros[-1].after for s in zeros):
                return answer(None, 'unsupported_conflicting_zero_spans', zeros)
            return answer(zeros[-1].after, 'zero_duration_point', zeros)
        if active:
            segment = active[0]
            return answer(segment.value(tick), 'interpolated', [segment])
        endings = self.endings.get(tick, [])
        if endings:
            if any(s.after != endings[-1].after for s in endings):
                return answer(None, 'unsupported_conflicting_endpoints', endings)
            return answer(endings[-1].after, 'endpoint', endings)
        outer = [s for s in self.segments if s.end >= s.start]
        if not outer:
            return answer(None, 'unsupported_empty_track')
        first = min(outer, key=lambda s: s.start)
        last = max(outer, key=lambda s: s.end)
        if tick < first.start:
            return answer(first.before, 'outer_endpoint_hold', [first])
        if tick > last.end:
            return answer(last.after, 'outer_endpoint_hold', [last])
        return answer(None, 'unsupported_interior_gap')

    def boundaries(self, start, end):
        return sorted({value for segment in self.segments for value in (segment.start, segment.end)
                       if start < value < end})

    def interval_values(self, start, end):
        """One-sided limits on a boundary-free open interval (start, end).

        query(end) belongs to the next span and cannot supply this limit.
        Zero-duration points have no extent and are examined independently.
        """
        if end <= start or self.boundaries(start, end):
            raise ValueError('interval must be positive and split at every source boundary')
        state = self.query((start + end) / 2)
        if state['value'] is None:
            return dict(before=None, after=None, status=state['status'])
        if state['status'] == 'interpolated':
            segment = self.by_identifier[state['event_ids'][0]]
            return dict(before=list(segment.value(start)), after=list(segment.value(end)), status='supported_one_sided_limits')
        return dict(before=state['value'], after=state['value'], status=state['status'])

    def integral(self, start, end, seconds_per_tick):
        if self.kind != 'speed':
            raise ValueError('only piecewise-constant source speed has this integral')
        if start == end:
            return dict(value=0., status='integrated', unsupported_intervals=[])
        direction = 1 if end > start else -1
        low, high = sorted((start, end))
        points = [low, *self.boundaries(low, high), high]
        parts, invalid = [], []
        for a, b in zip(points, points[1:]):
            state = self.query((a + b) / 2)
            if state['value'] is None:
                invalid.append(dict(native_range=[a, b], status=state['status']))
            else:
                parts.append((b - a) * seconds_per_tick * state['value'][0])
        return dict(value=None if invalid else direction * math.fsum(parts),
                    status='unsupported_integral' if invalid else 'integrated', unsupported_intervals=invalid)


class PerformanceChart:
    def __init__(self, source, duration, provenance=None):
        self.source = copy.deepcopy(source)
        self.duration = number(duration)
        if self.duration <= 0:
            raise ValueError('audio duration must be positive')
        self.version = source['formatVersion']
        if self.version not in (1, 3):
            raise ValueError('only original PGR format versions 1 and 3 are supported')
        self.offset = number(source.get('offset', 0))
        self.provenance = copy.deepcopy(provenance or {})
        self.lines = []
        self.notes = []
        for line_index, raw in enumerate(source['judgeLineList']):
            line_id = f'line-{line_index}'
            bpm = number(raw['bpm'])
            if bpm <= 0:
                raise ValueError(f'{line_id}: BPM must be positive')
            line = dict(id=line_id, source_index=line_index, bpm=bpm, seconds_per_tick=60 / (32 * bpm),
                        tracks={kind: Track(kind, raw[field], self.version, f'{line_id}/{kind}')
                                for kind, field in TRACK_FIELDS.items()}, notes=[])
            line['boundaries'] = sorted({tick for track in line['tracks'].values() for segment in track.segments
                                         for tick in (segment.start, segment.end)})
            for side, field in [('above', 'notesAbove'), ('below', 'notesBelow')]:
                for index, note in enumerate(raw[field]):
                    kind = number(note['type'])
                    if kind not in (1, 2, 3, 4):
                        raise ValueError(f'{line_id}: unsupported note type {kind}')
                    native_time = number(note['time'])
                    native_duration = number(note['holdTime']) if kind == 3 else 0.
                    if native_duration < 0:
                        raise ValueError(f'{line_id}: negative Hold duration')
                    event = dict(id=f'{line_id}/{side}/note-{index}', line_id=line_id, source_index=index,
                                 side=side, type=int(kind), native_time=native_time, native_duration=native_duration,
                                 time_seconds=self.offset + native_time * line['seconds_per_tick'],
                                 end_seconds=self.offset + (native_time + native_duration) * line['seconds_per_tick'],
                                 position_x=number(note['positionX']), source_speed=number(note['speed']),
                                 source_floor_position=number(note['floorPosition']))
                    line['notes'].append(event)
                    self.notes.append(event)
            self.lines.append(line)
        self.by_id = {line['id']: line for line in self.lines}
        self.notes_by_id = {note['id']: note for note in self.notes}

    @classmethod
    def from_file(cls, path, audio_path, split='unknown'):
        import soundfile as sf
        path, audio_path = Path(path).resolve(), Path(audio_path).resolve()
        if path.name == 'generate.json':
            raise ValueError('static generate.json is not an original performance label')
        return cls(json.loads(path.read_text()), sf.info(audio_path).duration,
                   dict(chart_path=str(path), source_sha256=file_hash(path), audio_path=str(audio_path),
                        audio_sha256=file_hash(audio_path), split=split))

    def to_document(self):
        lines = []
        for line in self.lines:
            tracks = {}
            for kind, track in line['tracks'].items():
                tracks[kind] = [dict(id=segment.identifier, native_start=segment.start, native_end=segment.end,
                                     start_seconds=self.offset + segment.start * line['seconds_per_tick'],
                                     end_seconds=self.offset + segment.end * line['seconds_per_tick'],
                                     before=list(segment.before), after=list(segment.after))
                                for segment in track.segments]
            lines.append(dict(id=line['id'], bpm=line['bpm'], seconds_per_tick=line['seconds_per_tick'], tracks=tracks))
        return dict(schema=SCHEMA, interpreter=INTERPRETER, duration_seconds=self.duration,
                    coordinate_system='16:9-normalized-bottom-left', provenance=self.provenance,
                    source_content_sha256=content_hash(self.source), source_document=copy.deepcopy(self.source),
                    lines=lines, notes=copy.deepcopy(self.notes))

    @classmethod
    def from_document(cls, document):
        if document['schema'] != SCHEMA or document['interpreter'] != INTERPRETER:
            raise ValueError('unsupported performance representation')
        if content_hash(document['source_document']) != document['source_content_sha256']:
            raise ValueError('source content hash mismatch')
        chart = cls(document['source_document'], document['duration_seconds'], document['provenance'])
        if chart.notes != document['notes'] or chart.to_document()['lines'] != document['lines']:
            raise ValueError('canonical events disagree with retained source')
        return chart

    def native_tick(self, line, audio_seconds):
        audio_seconds = number(audio_seconds)
        tick = (audio_seconds - self.offset) / line['seconds_per_tick']
        # Recover a source boundary after its float64 seconds round trip.
        # This is a handful of representable values, not musical quantization.
        tolerance = 8 * max(math.ulp(tick), math.ulp(audio_seconds) / line['seconds_per_tick'])
        index = bisect_right(line['boundaries'], tick)
        for boundary in line['boundaries'][max(0, index - 1):index + 1]:
            if abs(boundary - tick) <= tolerance:
                return boundary
        return tick

    def line_state(self, line_id, audio_seconds, include_scroll=True):
        line = self.by_id[line_id]
        tick = self.native_tick(line, audio_seconds)
        states = {kind: track.query(tick) for kind, track in line['tracks'].items()}
        value = {kind: state['value'] for kind, state in states.items()}
        scalar = lambda kind: value[kind][0] if value[kind] is not None else None
        result = dict(line_id=line_id, audio_seconds=audio_seconds, native_tick=tick,
                      within_audio=0 <= audio_seconds <= self.duration,
                      position=value['move'], rotation_degrees=scalar('rotate'),
                      alpha_source=scalar('alpha'), speed_source=scalar('speed'), tracks=states)
        result['alpha_nonnegative'] = None if scalar('alpha') is None else max(0., scalar('alpha'))
        result['alpha_interpretation'] = 'nonnegative-display-mask-without-upper-clamp'
        if include_scroll:
            result['scroll_integral'] = line['tracks']['speed'].integral(0., tick, line['seconds_per_tick'])
            result['scroll_units'] = 'source-speed-times-chart-seconds; origin-native-tick-zero'
        result['current_state_supported'] = all(state['value'] is not None for state in states.values())
        result['scroll_supported'] = result['scroll_integral']['value'] is not None if include_scroll else None
        result['supported'] = result['current_state_supported'] and (not include_scroll or result['scroll_supported'])
        result['effective_supported'] = result['supported'] and result['within_audio']
        return result

    def note_geometry(self, note_id, audio_seconds, include_scroll=True):
        note = self.notes_by_id[note_id]
        state = self.line_state(note['line_id'], audio_seconds, include_scroll=include_scroll)
        point = None
        if state['position'] is not None and state['rotation_degrees'] is not None:
            radians = math.radians(state['rotation_degrees'])
            point = [state['position'][0] + note['position_x'] * X_UNIT_W * math.cos(radians),
                     state['position'][1] + note['position_x'] * X_UNIT_H * math.sin(radians)]
        return dict(note_id=note_id, audio_seconds=audio_seconds, target_world=point,
                    in_viewport=None if point is None else all(0 <= coordinate <= 1 for coordinate in point),
                    line_state=state, trajectory_meaning='judgment-geometry; not a touch-following requirement')

    def sample_hold(self, note_id, max_step_seconds=.25, max_rotation_step_degrees=22.5, max_points=20000,
                    include_scroll=True):
        note = self.notes_by_id[note_id]
        if note['type'] != 3:
            raise ValueError('trajectory sampling requires a Hold')
        if max_step_seconds <= 0 or max_rotation_step_degrees <= 0:
            raise ValueError('trajectory sample steps must be positive')
        line = self.by_id[note['line_id']]
        start, end = note['native_time'], note['native_time'] + note['native_duration']
        boundaries = sorted({start, end, *(tick for track in line['tracks'].values()
                                          for tick in track.boundaries(start, end))})
        ticks = set(boundaries)
        jumps = []
        for a, b in zip(boundaries, boundaries[1:]):
            count = max(1, math.ceil((b - a) * line['seconds_per_tick'] / max_step_seconds))
            rotation = line['tracks']['rotate'].interval_values(a, b)
            if rotation['before'] is not None:
                count = max(count, math.ceil(abs(rotation['after'][0] - rotation['before'][0]) / max_rotation_step_degrees))
            for kind, track in line['tracks'].items():
                interval, endpoint = track.interval_values(a, b), track.query(b)
                if interval['after'] is not None and endpoint['value'] is not None and interval['after'] != endpoint['value']:
                    jumps.append(dict(track=kind, native_tick=b, audio_seconds=self.offset + b * line['seconds_per_tick'],
                                      left_limit=interval['after'], at_boundary=endpoint['value']))
            if len(ticks) + count > max_points:
                raise ValueError('Hold trajectory exceeds explicit query sample budget')
            ticks.update(a + (b - a) * i / count for i in range(1, count))
            # Check one-sided states as well as right-continuous boundaries.
            epsilon = min((b - a) / 4, 1e-7 / line['seconds_per_tick'])
            ticks.update([a + epsilon, b - epsilon])
        if len(ticks) > max_points:
            raise ValueError('Hold trajectory exceeds explicit query sample budget')
        samples = [self.note_geometry(note_id, self.offset + tick * line['seconds_per_tick'], include_scroll=include_scroll)
                   for tick in sorted(ticks)]
        return dict(note_id=note_id, samples=samples, max_step_seconds=max_step_seconds,
                    max_rotation_step_degrees=max_rotation_step_degrees,
                    rotation_step_scope='supported_continuous_spans_only; jumps_are_separate', jumps=jumps,
                    scroll_integral_sampled=include_scroll,
                    continuous_safety_proof=False, source_event_boundaries_included=True)

    def classify_line(self, line_id):
        line = self.by_id[line_id]
        effective = [note for note in line['notes'] if note['end_seconds'] >= 0 and note['time_seconds'] <= self.duration]
        base = dict(source_note_count=len(line['notes']), effective_note_count=len(effective))
        if effective:
            return dict(classification='gameplay', **base)
        lo, hi = self.native_tick(line, 0.), self.native_tick(line, self.duration)
        boundaries = sorted({lo, hi, *(tick for track in line['tracks'].values()
                                      for tick in track.boundaries(lo, hi))})
        points = {*boundaries, *((a + b) / 2 for a, b in zip(boundaries, boundaries[1:]))}
        alpha_intervals = []
        for a, b in zip(boundaries, boundaries[1:]):
            interval = line['tracks']['alpha'].interval_values(a, b)
            alpha_intervals.append(interval)
            if interval['before'] is None:
                continue
            first, last = interval['before'][0], interval['after'][0]
            # Add a strictly positive witness when positivity only occurs near
            # an endpoint; the following right-hand span may jump below zero.
            if first <= 0 < last:
                crossing = a + (b - a) * (-first / (last - first))
                points.add((crossing + b) / 2)
            elif last <= 0 < first:
                crossing = a + (b - a) * (-first / (last - first))
                points.add((a + crossing) / 2)
        points = sorted(points)
        states = [self.line_state(line_id, self.offset + tick * line['seconds_per_tick'], include_scroll=False)
                  for tick in points]
        alpha_supported = all(state['alpha_source'] is not None for state in states) and all(
            interval['before'] is not None for interval in alpha_intervals)
        # Alpha is linear on every partition; extrema are at its endpoints.
        if alpha_supported and all(state['supported'] and state['alpha_source'] <= 0 for state in states) and all(
                max(interval['before'][0], interval['after'][0]) <= 0 for interval in alpha_intervals):
            return dict(classification='idle', reason='alpha_nonpositive_over_supported_piecewise_linear_domain',
                        **base, sampled_points=len(points))
        potential = 0
        unknown = not alpha_supported or any(not state['supported'] for state in states)
        for state in states:
            if state['position'] is None or state['rotation_degrees'] is None or state['alpha_source'] is None:
                unknown = True
                continue
            if state['alpha_source'] <= 0:
                continue
            x, y = state['position']; angle = math.radians(state['rotation_degrees'])
            # Infinite line/rectangle intersection is only a potential mask;
            # texture length, thickness and occlusion are not in this format.
            nx, ny = -math.sin(angle), math.cos(angle) * 9 / 16
            values = [nx * (cx - x) + ny * (cy - y) for cx in (0, 1) for cy in (0, 1)]
            if min(values) <= 0 <= max(values):
                potential += 1
        return dict(classification='unknown' if unknown else 'visible_decoration_candidate' if potential else 'sampled_not_visible',
                    **base, sampled_points=len(points), potential_visible_samples=potential,
                    actual_texture_visibility='unknown', whole_interval_visibility_proof=False)

    def diagnostics(self):
        issues = []
        for line in self.lines:
            for note in line['notes']:
                audio_range = [note['time_seconds'], note['end_seconds']]
                intersects = audio_range[1] >= 0 and audio_range[0] <= self.duration
                if note['type'] == 3 and note['native_duration'] == 0:
                    issues.append(dict(code='zero_hold_duration', line_id=line['id'], note_id=note['id'],
                                       audio_range=audio_range, intersects_audio=intersects))
                if note['time_seconds'] < 0 or note['end_seconds'] > self.duration:
                    issues.append(dict(code='note_outside_audio', line_id=line['id'], note_id=note['id'],
                                       audio_range=audio_range, intersects_audio=intersects))
            for kind, track in line['tracks'].items():
                for issue in track.issues:
                    row = dict(issue, line_id=line['id'], track=kind)
                    if 'native_range' in issue:
                        audio_range = [self.offset + tick * line['seconds_per_tick'] for tick in issue['native_range']]
                        row.update(audio_range=audio_range, intersects_audio=audio_range[1] >= 0 and audio_range[0] <= self.duration)
                    else:
                        row['intersects_audio'] = True
                    issues.append(row)
        return issues

"""Small diagnostic adapter derived from the pinned Phira PGR/Anim source.

This does not execute Phira and does not produce training labels. It models
keyframe conversion in float64, not bit-exact Rust f32 arithmetic. Ambiguous
cursor-history cases are reported instead of guessing a renderer result.
See docs/phira_reference.json for original files, hashes and fixed URLs.
"""
from bisect import bisect_right
import math

PHIRA_COMMIT = 'a22ea8958410ac940adec83b54bbf4e6b293d191'
HEIGHT_RATIO = .83175


class Keys:
    def __init__(self, points):
        self.points = points
        self.times = [point[0] for point in points]
        self.unsorted = any(b < a for a, b in zip(self.times, self.times[1:]))
        self.duplicates = {a for a, b in zip(self.times, self.times[1:]) if a == b}

    def query(self, time):
        if not self.points:
            return dict(value=None, status='unsupported_reference_empty_keys')
        if self.unsorted:
            return dict(value=None, status='unsupported_reference_unsorted_keys')
        if time in self.duplicates:
            return dict(value=None, status='unsupported_reference_duplicate_time_cursor_history')
        index = max(0, bisect_right(self.times, time) - 1)
        at, value, linear = self.points[index]
        if index == len(self.points) - 1 or not linear:
            return dict(value=value, status='reference_keyframe_value')
        next_time, next_value, _ = self.points[index + 1]
        if next_time == at:
            return dict(value=None, status='unsupported_reference_zero_key_interval')
        return dict(value=value + (next_value - value) * (time - at) / (next_time - at),
                    status='reference_keyframe_linear')


def convert_curve(events, ratio, kind, version):
    axes = [[], []] if kind == 'move' else [[]]
    for event in events:
        if event['startTime'] > event['endTime']:
            continue  # This reference ignores reversed spans; canonical data retains them.
        start_time = max(0., event['startTime'] * ratio)
        end_time = event['endTime'] * ratio
        if kind == 'move' and version == 1:
            # Signed Rust remainder; preserve the parser's raw packed-value
            # comparisons, even though these differ from decoded axis values.
            first_x = math.trunc(event['start'] / 1000.)
            last_x = math.trunc(event['end'] / 1000.)
            before = [first_x, event['start'] - first_x * 1000.]
            after = [last_x, event['end'] - last_x * 1000.]
            compare = [event['start'], event.get('start2', 0.)]
        elif kind == 'move':
            before, after = [event['start'], event.get('start2', 0.)], [event['end'], event.get('end2', 0.)]
            compare = before
        else:
            before, after = [event['start']], [event['end']]
            compare = before
        for index, keys in enumerate(axes):
            if not keys or keys[-1][1] != compare[index]:
                keys.append((start_time, before[index], True))
            keys.append((end_time, after[index], True))
    # The parser assumes the last endpoint is redundant/sentinel. A standalone
    # nonconstant last event consequently differs from its mathematical curve.
    for keys in axes:
        if keys:
            keys.pop()
    if kind == 'move' and version == 1:
        axes = [[(time, value / scale, linear) for time, value, linear in axis]
                for axis, scale in zip(axes, [880., 520.])]
    return [Keys(axis) for axis in axes]


class PhiraSourceReference:
    def __init__(self, source):
        self.offset = source.get('offset', 0.)
        self.lines = []
        heads = [max((note['time'] * 60 / (32 * line['bpm']) for key in ['notesAbove', 'notesBelow']
                      for note in line[key]), default=0.) for line in source['judgeLineList']]
        self.max_time = max(heads, default=0.) + 1.
        for line in source['judgeLineList']:
            ratio = 60 / (32 * line['bpm'])
            tracks = {kind: convert_curve(line[field], ratio, kind, source['formatVersion'])
                      for kind, field in [('move', 'judgeLineMoveEvents'), ('rotate', 'judgeLineRotateEvents'),
                                          ('alpha', 'judgeLineDisappearEvents')]}
            events = [dict(event) for event in line['speedEvents'] if event['startTime'] <= event['endTime']]
            speed, floor = [], []
            if events:
                events[0]['startTime'] = 0.
                accumulated = 0.
                for event in events[:-1]:
                    floor.append((event['startTime'] * ratio, accumulated, True))
                    accumulated += (event['endTime'] - event['startTime']) * ratio * event['value']
                last = events[-1]
                floor.append((last['startTime'] * ratio, accumulated, True))
                floor.append((self.max_time, accumulated + (self.max_time - last['startTime'] * ratio) * last['value'], False))
                speed = [(event['startTime'] * ratio, event['value'], False) for event in events]
            # Keep source scroll units here; the renderer subsequently divides
            # the height by HEIGHT_RATIO and viewport aspect.
            tracks.update(speed=[Keys(speed)], floor=[Keys(floor)])
            self.lines.append(tracks)

    def line_state(self, line_index, audio_seconds):
        time = audio_seconds - self.offset
        result = {kind: [axis.query(time) for axis in axes]
                  for kind, axes in self.lines[line_index].items()}
        values = {kind: [axis['value'] for axis in axes] for kind, axes in result.items()}
        return dict(position=values['move'] if all(value is not None for value in values['move']) else None,
                    rotation_degrees=values['rotate'][0], alpha_source=values['alpha'][0],
                    speed_source=values['speed'][0], scroll_source=values['floor'][0], tracks=result,
                    profile='source-derived-float64-adapter; native-player-not-executed')

    @staticmethod
    def note_interpretation(note):
        """Expose parser policy without claiming a complete note renderer."""
        return dict(effective_note_speed=1. if note['type'] == 3 else note['speed'],
                    source_note_speed=note['speed'], source_floor_position=note['floorPosition'],
                    original_floor_used=False)

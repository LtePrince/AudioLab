"""Causal chart-prefix features shared by teacher forcing and incremental decode.

Read features BEFORE observing the current input token. A NOTE's residual is
available only after its hidden state predicts it; it first affects the NEXT
input token. These are descriptive inputs, never masks or pattern quotas.
"""
from collections import deque
import math

from src.data.chart_tokenizer import ChartTokenizer, PAD

HISTORY_SCHEMA = 'causal-chart-history-v1'
HISTORY_NAMES = (
    'has_previous_note', 'previous_x', 'previous_previous_x', 'note_delta_x',
    'previous_note_gap_beats', 'age_beats', 'same_onset', 'notes_at_current_onset',
    'repeat_run', 'alternating_run', 'monotonic_run', 'direction',
    'group_delta_x', 'group_gap_beats', 'group_gap_change', 'group_span',
    'previous_type_tap', 'previous_type_drag', 'previous_type_hold', 'previous_type_flick',
    'group_0_x', 'group_0_singleton', 'group_1_x', 'group_1_singleton',
    'group_2_x', 'group_2_singleton', 'group_3_x', 'group_3_singleton',
)
HISTORY_DIM = len(HISTORY_NAMES)


def _scaled(value, maximum=64):
    return math.log1p(min(max(float(value), 0.), maximum))/math.log1p(maximum)


class CausalChartHistory:
    def __init__(self, n_lanes, frames_per_beat):
        if n_lanes not in (4, 64):
            raise ValueError('history supports 4 lanes or 64 free-x bins')
        if not math.isfinite(frames_per_beat) or frames_per_beat <= 0:
            raise ValueError('frames_per_beat must be finite and positive')
        self.tk = ChartTokenizer(n_lanes)
        self.frames_per_beat = float(frames_per_beat)
        self.notes = deque(maxlen=2)
        self.groups = deque(maxlen=64)
        self.last_frame = None

    def features(self, frame):
        """Prefix-only features; `frame` is the current input token's clock."""
        frame = float(frame)
        if not math.isfinite(frame) or (self.last_frame is not None and frame < self.last_frame-1e-5):
            raise ValueError('finite, nondecreasing token frames required')
        if not self.notes:
            return [0.] * HISTORY_DIM
        last = self.notes[-1]
        previous = self.notes[-2] if len(self.notes) == 2 else last
        gap = (last['frame']-previous['frame'])/self.frames_per_beat
        groups = list(self.groups)
        recent = []
        for group in reversed(groups):
            if group['count'] != 1:
                break  # A chord interrupts the singleton sequence; never skip it.
            recent.append(group)
        repeat = 0
        for group in recent:
            if group['lane'] != recent[0]['lane']:
                break
            repeat += 1
        alternating = 0
        if len(recent) >= 2 and recent[0]['lane'] != recent[1]['lane']:
            alternating = 2
            for i in range(2, len(recent)):
                if recent[i]['lane'] != recent[i % 2]['lane']:
                    break
                alternating += 1
        direction = monotonic = 0
        if len(recent) >= 2:
            delta = recent[0]['lane']-recent[1]['lane']
            direction = (delta > 0)-(delta < 0)
            if direction:
                monotonic = 2
                for a, b in zip(recent[1:], recent[2:]):
                    delta = a['lane']-b['lane']
                    if (delta > 0)-(delta < 0) != direction:
                        break
                    monotonic += 1
        latest = groups[-1]
        older = groups[-2] if len(groups) >= 2 else latest
        group_gap = (latest['frame']-older['frame'])/self.frames_per_beat
        old_gap = ((older['frame']-groups[-3]['frame'])/self.frames_per_beat
                   if len(groups) >= 3 else group_gap)
        values = [1., last['x'], previous['x'], last['x']-previous['x'],
                  _scaled(gap), _scaled((frame-last['frame'])/self.frames_per_beat),
                  float(len(self.notes) == 2 and gap == 0),
                  min(latest['count'], self.tk.n_lanes)/self.tk.n_lanes
                  if abs(frame-latest['frame']) < 1e-5 else 0.,
                  _scaled(repeat), _scaled(alternating), _scaled(monotonic), float(direction),
                  latest['x']-older['x'], _scaled(group_gap),
                  max(-1., min(1., math.log((group_gap+1e-6)/(old_gap+1e-6))/4)),
                  latest['hi']-latest['lo'],
                  *[float(last['type'] == kind) for kind in (1, 2, 3, 4)]]
        for i in range(4):
            group = groups[-i-1] if i < len(groups) else None
            values.extend([group['x'], float(group['count'] == 1)] if group else [0., 0.])
        return values

    def observe(self, token, frame, offset=None):
        """Call AFTER current token forward/offset prediction, never before it."""
        token, frame = int(token), float(frame)
        self.features(frame)  # Validate clock even for non-NOTE tokens.
        kind = self.tk.kind(token)
        if kind == 'PAD':
            return
        self.last_frame = frame
        if kind != 'NOTE':
            return
        lane, type_index = divmod(self.tk.value(token), 4)
        if self.tk.n_lanes == 64:
            if offset is None or not math.isfinite(float(offset)) or not 0 <= float(offset) <= 1:
                raise ValueError('free-x NOTE needs its finite predicted/reference residual in [0,1]')
            x = (lane+float(offset))/64  # Normalized within the playable horizontal interval.
        else:
            x = lane/3
        self.notes.append(dict(lane=lane, type=type_index+1, x=x, frame=frame))
        if self.groups and abs(self.groups[-1]['frame']-frame) < 1e-5:
            group = self.groups[-1]
            group['x'] = (group['x']*group['count']+x)/(group['count']+1)
            group['count'] += 1
            group['lo'], group['hi'] = min(group['lo'], x), max(group['hi'], x)
        else:
            self.groups.append(dict(lane=lane, x=x, lo=x, hi=x, count=1, frame=frame))


def history_features(tokens, frames, *, n_lanes, frames_per_beat, offsets=None):
    """CPU tensor (L,D); each row depends only on earlier completed tokens.

    Teacher forcing uses earlier reference residuals, just as earlier reference
    tokens. Incremental decoding supplies earlier predictions instead.
    """
    import torch
    tokens, frames = list(tokens), list(frames)
    if len(tokens) != len(frames) or (offsets is not None and len(offsets) != len(tokens)):
        raise ValueError('tokens, frames and offsets must have matching lengths')
    state = CausalChartHistory(n_lanes, frames_per_beat)
    rows = []
    for i, (token, frame) in enumerate(zip(tokens, frames)):
        if int(token) == PAD:
            rows.append([0.] * HISTORY_DIM)
            continue
        rows.append(state.features(frame))
        state.observe(token, frame, None if offsets is None else offsets[i])
    return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), HISTORY_DIM)

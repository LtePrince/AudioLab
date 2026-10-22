import math

import pytest
import torch

from src.data.chart_tokenizer import ChartTokenizer, TokenNote
from src.models.chart_history import CausalChartHistory, HISTORY_NAMES, history_features


def build(lanes, *, positions=4, offsets=None):
    tk = ChartTokenizer(positions)
    tokens = tk.encode_notes([TokenNote(i*32, lane, 1, 0) for i, lane in enumerate(lanes)])
    frames = tk.ticks_per_position(tokens)
    residuals = [.5]*len(tokens) if offsets is None else offsets
    return tk, tokens, frames, history_features(tokens, frames, n_lanes=positions,
        frames_per_beat=32, offsets=residuals)


@pytest.mark.parametrize('lanes,feature', [([1]*6, 'repeat_run'), ([0,2,0,2,0,2], 'alternating_run'),
                                          ([0,1,2,3], 'monotonic_run')])
def test_prefix_carries_longitudinal_patterns(lanes, feature):
    _, _, _, values = build(lanes)
    assert values[-1, HISTORY_NAMES.index(feature)] == pytest.approx(math.log1p(len(lanes))/math.log(65))


def test_chord_interrupts_singleton_patterns():
    tk = ChartTokenizer(4)
    notes = [TokenNote(0,0,1,0), TokenNote(32,0,1,0), TokenNote(32,1,1,0), TokenNote(64,0,1,0)]
    tokens = tk.encode_notes(notes)
    values = history_features(tokens, tk.ticks_per_position(tokens), n_lanes=4, frames_per_beat=32)
    assert values[-1, HISTORY_NAMES.index('repeat_run')] == pytest.approx(math.log(2)/math.log(65))
    assert values[-1, HISTORY_NAMES.index('alternating_run')] == 0


def test_free_residual_is_available_only_after_its_note_and_streaming_matches():
    tk, tokens, frames, original = build([3,18,3], positions=64)
    offsets = [.5]*len(tokens)
    first = next(i for i, token in enumerate(tokens) if tk.kind(token) == 'NOTE')
    offsets[first] = .9
    changed = history_features(tokens, frames, n_lanes=64, frames_per_beat=32, offsets=offsets)
    assert torch.equal(original[:first+1], changed[:first+1])
    assert not torch.equal(original[first+1], changed[first+1])
    state = CausalChartHistory(64, 32)
    streamed = []
    for token, frame, residual in zip(tokens, frames, offsets):
        streamed.append(state.features(frame))
        state.observe(token, frame, residual)
    assert torch.equal(changed, torch.tensor(streamed))


def test_future_notes_and_hold_duration_cannot_leak_into_current_state():
    tk = ChartTokenizer(64)
    a = tk.encode_notes([TokenNote(0,5,3,32), TokenNote(64,8,1,0)])
    b = tk.encode_notes([TokenNote(0,5,3,64), TokenNote(96,40,1,0)])
    x = history_features(a, tk.ticks_per_position(a), n_lanes=64, frames_per_beat=32, offsets=[.5]*len(a))
    y = history_features(b, tk.ticks_per_position(b), n_lanes=64, frames_per_beat=32, offsets=[.5]*len(b))
    assert torch.equal(x[:3], y[:3])  # BOS, HOLD, and first duration prediction input.


def test_invalid_or_missing_previous_coordinates_fail():
    tk = ChartTokenizer(64)
    state = CausalChartHistory(64, 32)
    with pytest.raises(ValueError, match='residual'):
        state.observe(tk.note(3,1), 0)
    with pytest.raises(ValueError, match='nondecreasing'):
        state.features(float('nan'))

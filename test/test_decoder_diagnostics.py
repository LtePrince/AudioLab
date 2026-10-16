"""Checks for calibration statistics; no checkpoints or CUDA required."""
import json

import pytest
import torch

from script.diagnose_decoder import chart_stats, truncate


def test_nucleus_keeps_boundary_and_mask():
    p = torch.tensor([[0.6, 0.3, 0.1, 0], [0, 0.1, 0.2, 0.7]])
    result = truncate(p, 0.8)
    assert torch.allclose(result.sum(-1), torch.ones(2))
    assert torch.allclose(result[0], torch.tensor([2/3, 1/3, 0, 0]))
    assert (result[p == 0] == 0).all()
    assert torch.equal(truncate(p, 1), p)


def test_chart_stats_counts_conflicts_but_not_released_hold(tmp_path):
    def note(t, x, kind=1, duration=0):
        return dict(time=t, positionX=x, type=kind, holdTime=duration)
    path = tmp_path / 'chart.json'
    path.write_text(json.dumps(dict(offset=-0.1, judgeLineList=[dict(
        bpm=120, notesBelow=[], notesAbove=[note(0, 0, 3, 32), note(16, 0),
        note(32, 0), note(32, 0.1), note(32, 4)])])))
    stats = chart_stats(path)
    assert stats['notes'] == 5
    assert stats['types'] == [4, 0, 1, 0]
    assert stats['hold_conflicts'] == 1
    assert stats['close_chords'] == 1
    assert stats['hold_duration_median_s'] == pytest.approx(0.5)

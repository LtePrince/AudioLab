import json

import pytest
import torch

from script.calibrate_skeleton import event_score
from src.models.rhythm_skeleton import (select_onsets, onset_candidates, candidates_to_ticks,
                                       resolve_skeleton_settings)
from src.models.transcriber import H_ONSET, H_SOFF


def test_subframe_nms_keeps_strongest_and_exact_boundary():
    assert select_onsets([10, 20, 35, 100], [.7, .9, .8, .6], .7, 15) == [20, 35]
    assert select_onsets([10, 20, 35], [.7, .9, .8], .7, 0) == [10, 20, 35]
    assert select_onsets([20, 10], [.8, .8], .7, 15) == [10]


def test_audio_offset_filter_precedes_suppression():
    # A strong onset before chart start must not suppress a valid following one.
    assert candidates_to_ticks([95, 105], [.99, .8], 120, .1, .7, 15) == [0]
    assert candidates_to_ticks([100, 600], [.8, .9], 120, .1) == [0, 32]


def test_onset_candidates_uses_peak_lane_subframe_offset():
    logits = torch.full((1, 32, 3), -100.0)
    logits[0, H_ONSET + 2, 1] = 100
    logits[0, H_SOFF + 2, 1] = 0
    times, probability = onset_candidates(logits, 40)
    assert times[1] == 60 and probability[1] == 1


def test_event_matching_maximizes_hits_instead_of_nearest_greedy():
    # Nearest-first would consume reference=50 for generated=35 and miss 80.
    assert event_score([35, 80], [0, 50], 40) == dict(precision=1, recall=1, f1=1)
    assert event_score([], [0], 25)['f1'] == 0
    assert event_score([0, 10], [0], 25)['precision'] == .5


def test_invalid_suppression_settings():
    with pytest.raises(ValueError):
        select_onsets([1], [.8], .7, -1)
    with pytest.raises(ValueError):
        select_onsets([1], [], .7)


def test_generation_uses_frozen_training_calibration(tmp_path):
    checkpoint = tmp_path / 'rhythm.pt'
    report = tmp_path / 'fit.json'
    report.write_text(json.dumps(dict(complete=True, config=dict(phase='fit', ckpt=str(checkpoint)),
                                     selected=dict(threshold=.25, min_interval_ms=25))))
    assert resolve_skeleton_settings(checkpoint, report) == (.25, 25)
    assert resolve_skeleton_settings(None, default_threshold=.5) == (.5, 0)
    assert resolve_skeleton_settings(None) == (.7, 0)
    assert resolve_skeleton_settings(checkpoint, threshold=0, min_interval_ms=0) == (0, 0)
    with pytest.raises(ValueError, match='differs'):
        resolve_skeleton_settings(tmp_path / 'other.pt', report)
    with pytest.raises(ValueError, match='requires'):
        resolve_skeleton_settings(None, report)
    with pytest.raises(ValueError, match='manual'):
        resolve_skeleton_settings(checkpoint, report, threshold=.7)
    with pytest.raises(ValueError, match='manual'):
        resolve_skeleton_settings(checkpoint, report, min_interval_ms=0)


@pytest.mark.parametrize('complete,phase,selected', [
    (False, 'fit', dict(threshold=.25, min_interval_ms=25)),
    (True, 'evaluate', dict(threshold=.25, min_interval_ms=25)),
    (True, 'fit', {}),
    (True, 'fit', dict(threshold=.25, min_interval_ms=-1)),
])
def test_generation_rejects_invalid_calibration(tmp_path, complete, phase, selected):
    report = tmp_path / 'report.json'
    checkpoint = tmp_path / 'rhythm.pt'
    report.write_text(json.dumps(dict(complete=complete, config=dict(phase=phase, ckpt=str(checkpoint)),
                                     selected=selected)))
    with pytest.raises(ValueError):
        resolve_skeleton_settings(checkpoint, report)

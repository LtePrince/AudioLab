"""Checks for calibration statistics; no checkpoints or CUDA required."""
import json

import pytest
import torch

from script.diagnose_decoder import chart_stats, truncate, reuse_records, sha256_file


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


def test_reuse_checks_settings_hashes_and_completion(tmp_path):
    chart = tmp_path / 'decoder/p0.95/song.json'
    chart.parent.mkdir(parents=True)
    chart.write_text(json.dumps(dict(offset=0, judgeLineList=[dict(bpm=120,
        notesBelow=[], notesAbove=[dict(time=0, positionX=0, type=1, holdTime=0)])])))
    config = dict(seed=42, top_ps='0.95', teacher_only=False, sampling_policy='flat',
                  onset_threshold=.25, onset_min_interval_ms=25, skeleton_source='transcriber',
                  snap_ticks=0, data_list='data/val.txt', checkpoint_root='checkpoints/timing_v1')
    fingerprints = {'weights.pt': 'test-hash'}
    stats = {**chart_stats(chart), 'sha256': sha256_file(chart)}
    report = dict(config=config, finished=1, fingerprints=fingerprints,
                  models={'decoder': [dict(song='song', samples={'0.95': stats})]})
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    records, root, provenance = reuse_records(path, {**config, 'samples': 30}, [], fingerprints)
    assert ('decoder', 'song') in records and root == tmp_path
    assert provenance['verification'] == 'sha256'
    with pytest.raises(ValueError, match='settings differ'):
        reuse_records(path, {**config, 'seed': 7}, [], fingerprints)
    with pytest.raises(ValueError, match='fingerprint differs'):
        reuse_records(path, config, [], {'weights.pt': 'changed'})
    report.pop('finished')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='must be complete'):
        reuse_records(path, config, [], fingerprints)
    assert reuse_records(path, config, [], fingerprints, resume=True)[0]
    # A semantically invisible edit still invalidates a recorded chart hash.
    chart.write_text(chart.read_text() + ' ')
    with pytest.raises(ValueError, match='hash changed'):
        reuse_records(path, config, [], fingerprints, resume=True)


def test_legacy_reuse_refuses_modified_weights(tmp_path):
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'weights')
    config = dict(seed=42, top_ps='0.95', teacher_only=False, sampling_policy='flat',
                  onset_threshold=.7, onset_min_interval_ms=35, skeleton_source='transcriber',
                  snap_ticks=0, data_list='data/val.txt', checkpoint_root='checkpoints/timing_v1')
    report = dict(config=config, started=checkpoint.stat().st_mtime-1, finished=1, models={})
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='checkpoint changed'):
        reuse_records(path, config, [checkpoint], {})

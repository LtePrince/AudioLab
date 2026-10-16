import json

import pytest

from script.compare_full_charts import audit_chart, paired_delta, quality, check_generation


def test_quality_counts_chord_notes_and_reports_precision_recall():
    reference = [(0, 0, 1, .2), (0, 1, 1, .4), (100, 2, 1, .6)]
    generated = [(0, 0, 1, .2), (100, 2, 1, .6)]
    scores = quality(generated, reference, False)
    assert scores['f1_25'] == pytest.approx(.8)
    assert scores['precision_25'] == 1
    assert scores['recall_25'] == pytest.approx(2/3)
    assert scores['density_ratio'] == pytest.approx(2/3)
    assert scores['position_f1_93'] == pytest.approx(.8)
    assert quality([], reference, True)['f1_25'] == 0


def test_audit_uses_audio_offset_actual_duration_and_lane_anchors():
    chart = dict(formatVersion=3, offset=-.1, judgeLineList=[dict(bpm=120, notesBelow=[],
        notesAbove=[dict(type=1, time=0, holdTime=0, positionX=-3.75),
                    dict(type=3, time=32, holdTime=64, positionX=1.25),
                    dict(type=1, time=32, holdTime=0, positionX=0)])])
    audit = audit_chart(chart, 1, False)
    assert audit['onsets_outside_audio'] == 1
    assert audit['hold_ends_outside_audio'] == 1
    assert audit['invalid_positions'] == 1
    assert audit['invalid_durations'] == 0
    assert audit['max_chord_size'] == 2
    assert audit['peak_notes_per_1s'] == 3
    assert audit_chart(chart, 1, True)['invalid_positions'] == 0


def test_paired_bootstrap_preserves_song_pairs_and_lists_regressions():
    rows = [dict(song=str(i), variants=dict(
        global_v2=dict(raw=dict(f1_25=a)), nms35=dict(raw=dict(f1_25=b))))
        for i, (a, b) in enumerate([(.2, .3), (.6, .4), (.9, .9)])]
    result = paired_delta(rows, 'f1_25')
    assert result['mean_delta'] == pytest.approx(.1/3)
    assert (result['wins'], result['ties'], result['losses']) == (1, 1, 1)
    assert result['worst'][0]['song'] == '0'
    assert result['ci95'][0] < 0 < result['ci95'][1]
    assert paired_delta(rows, 'f1_25') == result


def test_comparison_refuses_partial_or_wrong_parameter_runs(tmp_path):
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(dict(models={})))
    with pytest.raises(ValueError, match='incomplete'):
        check_generation(path, ['one', 'two'], (.7, 35), 42)
    config = dict(seed=42, top_ps='0.95', sampling_policy='flat', teacher_only=False,
        skeleton_source='transcriber', snap_ticks=0, onset_threshold=.7, onset_min_interval_ms=35)
    path.write_text(json.dumps(dict(finished=1, config=config, models={})))
    with pytest.raises(ValueError, match='exactly all'):
        check_generation(path, ['one', 'two'], (.7, 35), 42)
    with pytest.raises(ValueError, match='settings differ'):
        check_generation(path, ['one', 'two'], (.25, 25), 42)

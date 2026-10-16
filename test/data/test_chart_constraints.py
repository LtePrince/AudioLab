import copy

import pytest

from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes, _space_chord


def note(t, lane, kind=1, dur=0):
    return dict(time=t, positionX=lane, type=kind, holdTime=dur, speed=1)


def test_4k_holds_release_at_next_lane_note_and_audio_end():
    notes = [note(32, 0), note(0, 0, 3, 128), note(8, 1), note(48, 2, 3, 32)]
    before = copy.deepcopy(notes)
    result, report = constrain_4k_notes(notes, 64.9)
    assert notes == before
    assert [n['time'] for n in result] == [n['time'] for n in notes]
    assert result[1]['holdTime'] == 32
    assert result[3]['holdTime'] == 16
    assert report == dict(shortened_holds=2, holds_to_taps=0)
    assert constrain_4k_notes(result, 64.9)[0] == result


def test_4k_subtick_room_becomes_tap_and_valid_holds_unchanged():
    result, report = constrain_4k_notes([note(0, 0, 3, 8), note(9, 1, 3, 4)], 9.5)
    assert result[0] == note(0, 0, 3, 8)
    assert result[1] == note(9, 1)
    assert report['holds_to_taps'] == 1


def test_reject_nonfinite_audio_end():
    with pytest.raises(ValueError):
        constrain_4k_notes([], float('nan'))


def free_note(t, x, kind=1, dur=0):
    return note(t, (x - 0.5) / 0.05625, kind, dur)


def test_free_chord_projection_spreads_both_sides_and_keeps_bounds():
    assert _space_chord([0.5, 0.51], 0.04, 0.05, 0.95) == pytest.approx([0.485, 0.525])
    assert _space_chord([0.05, 0.051, 0.052], 0.04, 0.05, 0.95) == pytest.approx([0.05, 0.09, 0.13])
    assert _space_chord([0.949, 0.95], 0.04, 0.05, 0.95) == pytest.approx([0.91, 0.95])


def test_free_holds_use_repaired_future_positions():
    notes = [free_note(0, 0.48, 3, 64), free_note(16, 0.51), free_note(16, 0.512)]
    before = copy.deepcopy(notes)
    result, report = constrain_free_notes(notes, 100)
    assert result[0]['holdTime'] == 16
    assert report['moved_notes'] == 2
    assert report['shortened_holds'] == 1
    assert notes == before
    assert [n['time'] for n in result] == [0, 16, 16]
    assert constrain_free_notes(result, 100)[0] == result


def test_free_same_chord_hold_does_not_cut_neighbor_at_exact_spacing():
    result, report = constrain_free_notes([free_note(0, 0.5, 3, 16), free_note(0, 0.54)], 50)
    assert result[0]['holdTime'] == 16
    assert report['shortened_holds'] == 0


def test_infeasible_chord_fails_without_dropping_notes():
    with pytest.raises(ValueError, match='cannot fit'):
        constrain_free_notes([free_note(0, 0.5)] * 24, 100)


@pytest.mark.parametrize('gap', [0, -0.1, float('nan'), 1.0])
def test_invalid_free_gap(gap):
    with pytest.raises(ValueError):
        constrain_free_notes([], 100, gap)


def test_batch_repair_preserves_render_metadata_and_audio_offset():
    from script.repair_generated_charts import repair_chart
    chart = dict(offset=0.25, judgeLineList=[dict(bpm=120, notesBelow=[],
        notesAbove=[{**free_note(0, 0.5, 3, 128), 'floorPosition': 0, 'speed': 3}],
        judgeLineMoveEvents=[dict(start=0.5, end=0.5, start2=0.2, end2=0.2)],
        judgeLineRotateEvents=[dict(start=0, end=0)], speedEvents=[dict(value=3)])])
    original = copy.deepcopy(chart)
    result, _ = repair_chart(chart, 1.25, 'freex')
    assert result['judgeLineList'][0]['notesAbove'][0]['holdTime'] == 64
    assert result['judgeLineList'][0]['notesAbove'][0]['speed'] == 3
    assert result['judgeLineList'][0]['speedEvents'] == chart['judgeLineList'][0]['speedEvents']
    assert chart == original
    chart['judgeLineList'][0]['judgeLineRotateEvents'][0]['end'] = 45
    with pytest.raises(ValueError, match='rotating'):
        repair_chart(chart, 1.25, 'freex')

import copy

import pytest

from src.data.chart_constraints import constrain_4k_notes


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

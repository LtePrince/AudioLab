import copy

from src.data.fourkey_contacts import constrain_4k_contacts
from src.data.contact_audit import LANES


def note(t, lane, kind=1, dur=0):
    return dict(time=t, positionX=LANES[lane], type=kind, holdTime=dur, speed=1)


def test_fast_jacks_move_to_available_keys_without_changing_onsets():
    notes = [note(t, 0) for t in [0, 2, 4, 6, 8]]
    before = copy.deepcopy(notes)
    result, report = constrain_4k_contacts(notes, 100, 120)
    assert [n['time'] for n in result] == [0, 2, 4, 6, 8]
    assert report['moved_notes'] > 0 and report['dropped_notes'] == 0
    assert not any(report['audit'].values()) and notes == before


def test_four_chord_kept_and_holds_release_before_restrike():
    notes = [note(0, i, 3, 100) for i in range(4)]+[note(8, i) for i in range(4)]
    result, report = constrain_4k_contacts(notes, 128, 120)
    assert len(result) == 8 and report['moved_notes'] == 0
    assert all(n['holdTime'] == 6 for n in result[:4])
    assert report['shortened_holds'] == 4 and not any(report['audit'].values())


def test_impossible_dense_chord_is_reported_not_time_shifted():
    notes = [note(0, i) for i in range(4)] + [note(1, i) for i in range(4)]
    result, report = constrain_4k_contacts(notes, 100, 120)
    assert len(result) == 4 and report['dropped_onsets'] == 1
    assert report['dropped_notes'] == 4

import copy
import random

import pytest

from src.data.free_contacts import constrain_free_contacts
from src.data.contact_audit import audit_contacts


def note(t, x, kind=1, dur=0):
    return dict(time=t, positionX=(x-.5)/.05625, type=kind, holdTime=dur, speed=1)


def test_multifinger_chords_are_preserved_when_budget_allows():
    notes = [note(0, x) for x in [.15, .35, .55, .75]]
    original = copy.deepcopy(notes)
    for count in [4, 6, 10]:
        result, report = constrain_free_contacts(notes, 128, 120, max_contacts=count)
        assert len(result) == 4
        assert report['dropped_notes'] == 0
        assert report['moved_notes'] == 0
        assert not any(report['audit'].values())
    assert notes == original
    result, report = constrain_free_contacts(notes, 128, 120, max_contacts=2)
    assert len(result) == 2 and report['dropped_notes'] == 2


def test_reserves_contact_for_next_onset_and_releases_hold_for_travel():
    notes = [note(0, .2, 3, 80), note(0, .8), note(3, .5), note(8, .9)]
    result, report = constrain_free_contacts(notes, 100, 120, max_contacts=2)
    assert {n['time'] for n in result} == {0, 3, 8}
    assert report['unique_onset_retention'] == 1
    assert not any(audit_contacts(result, report['contacts'], 120, 'freex', max_contacts=2).values())


@pytest.mark.parametrize('count', [2, 4, 6])
def test_adversarial_dense_holds_have_valid_witness_and_unchanged_onsets(count):
    rng = random.Random(42)
    notes = [note(rng.randrange(100), rng.uniform(.05, .95), 3, rng.randrange(1, 120)) for _ in range(160)]
    result, report = constrain_free_contacts(notes, 130.3, 180, max_contacts=count)
    assert not any(report['audit'].values())
    assert len(result) + report['dropped_notes'] == len(notes)
    assert all(n['time'] == notes[i]['time'] for n, i in zip(result, report['source_indices']))
    assert all(n['time']+n['holdTime'] <= 130.3 for n in result)


def test_no_implicit_contact_budget():
    with pytest.raises(TypeError):
        constrain_free_contacts([], 1, 120)
    with pytest.raises(ValueError):
        constrain_free_contacts([], 1, 120, max_contacts=0)

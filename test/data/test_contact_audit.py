import pytest

from src.data.contact_audit import audit_contacts


def note(t, x=0, kind=1, dur=0):
    return dict(time=t, positionX=x, type=kind, holdTime=dur, speed=1)


def test_detects_simultaneous_contacts_and_held_finger_teleport():
    notes = [note(0, -4, 3, 64), note(64, 4), note(64, 0)]
    report = audit_contacts(notes, [0, 0, 0], 120, 'freex')
    assert report['repeat_violations'] == 1
    assert report['release_violations'] == 2
    assert report['movement_violations'] == 2


def test_4k_contact_must_match_physical_key():
    assert audit_contacts([note(0, -3.75)], [1], 120, '4k')['invalid_contacts'] == 1
    assert not any(audit_contacts([note(0, -3.75), note(8, -3.75)], [0, 0], 120, '4k').values())
    with pytest.raises(ValueError):
        audit_contacts([note(0)], [], 120, '4k')

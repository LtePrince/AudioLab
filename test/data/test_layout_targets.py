import numpy as np
import pytest

from script.mine_layout_targets import layout_events


def test_mining_preserves_multifinger_chord_and_audio_offset():
    notes = [dict(time=32, positionX=x, type=3, holdTime=16) for x in [-3, -1, 1, 3]]
    notes += [dict(time=40, positionX=0, type=1, holdTime=0)]
    mel = np.zeros((128, 200), dtype=np.float32)
    mel[:, 32:] = 1
    rows = list(layout_events(notes, 120, .25, mel, 2))
    assert rows[0]['audio_s'] == .75
    assert rows[0]['chord_size'] == 4 and rows[0]['hold_s'] == [.25]*4
    assert rows[0]['positive_logmel_flux'] == 1
    assert len(rows[1]['active_holds_before']) == 4
    assert rows[1]['previous_gap_s'] == pytest.approx(.125)
    assert not list(layout_events(notes, 120, .25, mel, .5))

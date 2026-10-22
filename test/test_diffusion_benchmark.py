import hashlib
import json
import zipfile

import pytest

from script.benchmark_diffusion import chart_prediction, uploaded_prediction


def chart():
    return dict(formatVersion=3, offset=.25, judgeLineList=[dict(bpm=120, notesBelow=[],
        notesAbove=[dict(time=32, holdTime=16, positionX=-3.75, type=3),
                    dict(time=32, holdTime=0, positionX=3.75, type=1)])])


def test_chart_adapter_preserves_seconds_chords_types_and_duration():
    result = chart_prediction(chart())
    assert result['onsets_seconds'] == [.75]
    assert len(result['notes']) == 2
    assert result['notes'][0]['end_seconds'] == 1.
    assert result['notes'][0]['type'] == 'hold'
    assert result['notes'][0]['position'] == 0
    assert result['notes'][1]['position'] == 3


def test_zip_audio_identity_required_and_no_time_fitting(tmp_path):
    path = tmp_path/'upload.zip'; audio = b'fixture-audio'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('nested/chart.json', json.dumps(chart()))
        z.writestr('nested/music.ogg', audio)
    result, meta = uploaded_prediction(path, hashlib.sha256(audio).hexdigest())
    assert result['onsets_seconds'] == [.75]
    assert meta['offset'] == .25
    assert not (tmp_path/'nested').exists()
    with pytest.raises(ValueError, match='differs'):
        uploaded_prediction(path, '0'*64)


def test_ambiguous_chart_not_silently_selected(tmp_path):
    path = tmp_path/'upload.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a.json', json.dumps(chart()))
        z.writestr('b.json', json.dumps(chart()))
        z.writestr('music.ogg', b'audio')
    with pytest.raises(ValueError, match='exactly one'):
        uploaded_prediction(path, hashlib.sha256(b'audio').hexdigest())

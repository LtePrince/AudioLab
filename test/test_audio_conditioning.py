import math

import pytest
import torch

from script.audit_audio_conditioning import alter_features, token_scores
from src.data.chart_tokenizer import ChartTokenizer, EOS


@pytest.mark.parametrize('lanes', [4, 64])
def test_uniform_logits_separate_type_position_and_time(lanes):
    tk = ChartTokenizer(n_lanes=lanes)
    target = torch.tensor([tk.note(1, 3), tk.durb(2), tk.dtt(4), EOS])
    scores = token_scores(torch.zeros(4, tk.vocab_size), target, tk)
    assert scores['all']['bits'] == pytest.approx(math.log2(tk.vocab_size))
    assert scores['note_type'] == dict(count=1, bits=2)
    assert scores['position']['bits'] == pytest.approx(math.log2(lanes))
    assert scores['duration']['count'] == 1
    assert scores['time']['count'] == 1


def test_acoustic_ablations_keep_length_padding_and_input():
    features = torch.arange(12).reshape(1, 6, 2).float()
    original = features.clone()
    reversed_ = alter_features(features, 4, 'reversed')
    assert torch.equal(reversed_[:, :4], original[:, :4].flip(1))
    assert torch.equal(reversed_[:, 4:], original[:, 4:])
    wrong = alter_features(features, 4, 'wrong_song', torch.ones(1, 3, 2), 2)
    assert torch.equal(wrong[:, :4], torch.ones(1, 4, 2))
    assert torch.equal(wrong[:, 4:], original[:, 4:])
    assert not alter_features(features, 4, 'no_audio').any()
    assert torch.equal(features, original)
    assert torch.equal(alter_features(features, 4, 'matched'), original)


def test_wrong_song_requires_another_recording():
    with pytest.raises(ValueError, match='other audio'):
        alter_features(torch.zeros(1, 4, 2), 4, 'wrong_song')

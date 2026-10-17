import pytest
import torch

from src.models.chord_count import ChordCountHead, event_features
from src.train_chord_count import song_split, evaluate


def test_event_windows_use_audio_clock_and_keep_context_separate():
    encoded=torch.arange(10.).unsqueeze(1)
    result=event_features(encoded,[.25,.75],.1)
    assert torch.equal(result[:,:3],torch.tensor([[0.,2.,4.],[5.,7.,9.]]))
    assert torch.allclose(result[:,-2:],torch.log1p(torch.tensor([[0.,5.],[5.,0.]])))
    with pytest.raises(ValueError):event_features(encoded,[.2,.2],.1)
    assert event_features(encoded,[],.1).shape==(0,5)


def test_song_split_is_disjoint_and_deterministic():
    songs=[f'song{i}' for i in range(30)]
    fit,tune=song_split(songs,42,3)
    assert len(fit)==27 and len(tune)==3 and not set(fit)&set(tune)
    assert (fit,tune)==song_split(songs,42,3)


def test_prior_retains_rare_multifinger_labels_and_wrong_audio_is_nonmutating():
    torch.manual_seed(42)
    head=ChordCountHead(5,64)
    histogram=torch.tensor([1000,200,5,3,1,0,0,1]+[0]*56)
    head.initialize_prior(histogram)
    prior=(histogram.float()+.05)/(histogram.sum()+64*.05)
    rows=[dict(song='a',x=torch.randn(4,5).half(),y=torch.tensor([0,1,3,7])),
          dict(song='b',x=torch.randn(3,5).half(),y=torch.tensor([0,1,0]))]
    original=[r['x'].clone() for r in rows]
    result=evaluate(head,rows,prior,'cpu',wrong_audio=True)
    assert result['songs'][0]['target_histogram'][7]==1
    assert all(torch.equal(r['x'],x) for r,x in zip(rows,original))
    assert torch.isfinite(torch.tensor(result['macro']['nll']))

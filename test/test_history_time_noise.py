import random

import pytest
import torch

from src.history_time_noise import perturb_history_times
from src.models.chart_history import history_features
from src.soft_onset import MEMORY_SECONDS
from src.windowed_ar import WindowedAR, encode_window_target, window_intervals
from src.anchored_ar import normalized_loss
from src.data.chart_tokenizer import ChartTokenizer, TokenNote


def target(n_lanes=4):
    tk=ChartTokenizer(n_lanes)
    notes=[TokenNote(0,0,1),TokenNote(32,1,4),TokenNote(32,2,2),
           TokenNote(608,3,3,128),TokenNote(639,1,1),TokenNote(640,2,1),TokenNote(641,0,1)]
    tokens=tk.encode_notes(notes)
    route=dict(tokens=torch.tensor(tokens),reference=dict(timebase_bpm=120.,offset_seconds=0.))
    if n_lanes==64:
        route['offset_mask']=torch.tensor([tk.kind(t)=='NOTE' for t in tokens])
        route['offsets']=torch.tensor([.37 if tk.kind(t)=='NOTE' else 0. for t in tokens])
    return encode_window_target(route,25.,n_lanes=n_lanes)


@pytest.mark.parametrize('lanes',[4,64])
def test_group_noise_keeps_labels_chords_boundaries_and_full_holds(lanes):
    clean=target(lanes);snapshot={k:v.clone() for k,v in clean.items() if torch.is_tensor(v)}
    noisy,audit=perturb_history_times(clean,n_lanes=lanes,max_seconds=.02,rng=random.Random(42))
    for k,v in snapshot.items():assert torch.equal(clean[k],v)
    for k in ('tokens','layout_targets','next_seconds','stop_targets','bounds','decisions'):
        assert torch.equal(clean[k],noisy[k])
    assert not torch.equal(noisy['frames'],clean['frames'])
    assert not torch.equal(noisy['history'],clean['history'])
    tk=ChartTokenizer(lanes);anchors=torch.where(clean['tokens']==tk.cond(0))[0]
    assert audit['groups']==len(anchors) and audit['restricted_groups']>0
    assert audit['max_absolute_noise_seconds']<=.02001
    for start in anchors.tolist():
        end=start+1
        while int(clean['tokens'][end])!=tk.cond(1):end+=1
        assert (noisy['frames'][start:end+1]==noisy['frames'][start]).all()
    boundary=(clean['tokens']==tk.cond(2))|(clean['tokens']==1)
    assert torch.equal(clean['frames'][boundary],noisy['frames'][boundary])
    assert (noisy['frames'][1:]>=noisy['frames'][:-1]).all()
    rebuilt=history_features(noisy['tokens'],noisy['frames'],n_lanes=lanes,
        frames_per_beat=60/120/MEMORY_SECONDS,offsets=noisy.get('offsets'))
    assert torch.equal(noisy['history'],rebuilt)
    if lanes==64:
        assert torch.equal(noisy['offsets'],clean['offsets'])
        assert torch.equal(noisy['offset_mask'],clean['offset_mask'])


def test_every_noised_target_is_still_reachable_and_has_finite_gradients():
    clean=target();length=int(25/MEMORY_SECONDS)+1;valid=torch.ones(1,length,dtype=torch.bool)
    for seed in range(40):
        noisy,_=perturb_history_times(clean,n_lanes=4,max_seconds=.02,rng=random.Random(seed))
        positive=clean['stop_targets']==0;indices=clean['decisions'][positive]
        lo,hi,allowed=window_intervals(noisy['lower_seconds'][positive],clean['bounds'][indices,1],length,valid)
        truth=clean['next_seconds'][positive];bins=(truth/MEMORY_SECONDS).long();rows=torch.arange(len(bins))
        assert allowed[rows,bins].all()
        fraction=(truth-lo[rows,bins])/(hi[rows,bins]-lo[rows,bins])
        assert ((fraction>=0)&(fraction<1)).all()
    model=WindowedAR(dict(vocab_size=129,hidden_dim=32,depth=2,num_heads=4,mem_dim=32,
        time_dim=16,dropout=0.,token_dropout=0.,history_dim=28,soft_onset_dim=2))
    condition=dict(features=torch.randn(1,length,32),soft_onsets=torch.rand(1,length,2),valid=valid,duration=25.)
    normalized_loss(model.components(noisy,condition,use_soft=False),noisy).backward()
    assert torch.isfinite(model.time_query.weight.grad).all() and model.time_query.weight.grad.abs().sum()>0


def test_zero_noise_is_the_exact_original_control_and_does_not_draw_rng():
    clean=target();rng=random.Random(42);state=rng.getstate()
    noisy,audit=perturb_history_times(clean,n_lanes=4,max_seconds=0.,rng=rng)
    assert noisy is clean and rng.getstate()==state and audit['groups']==0


def test_noisy_clock_is_reproducible_independently_of_training_sampler():
    clean=target()
    a,_=perturb_history_times(clean,n_lanes=4,max_seconds=.02,rng=random.Random(123))
    b,_=perturb_history_times(clean,n_lanes=4,max_seconds=.02,rng=random.Random(123))
    assert torch.equal(a['frames'],b['frames'])
    for invalid in (-1.,float('inf'),float('nan')):
        with pytest.raises(ValueError):perturb_history_times(clean,n_lanes=4,max_seconds=invalid,rng=random.Random(1))

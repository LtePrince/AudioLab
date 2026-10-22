import math
import random

import pytest
import torch

from src.audio_understanding import SAMPLE_RATE
from src.conditional_training import (choose_pair, shifted_waveform, shifted_targets,
    empty_targets, paired_loss, time_log_probability)
from src.data.chart_tokenizer import ChartTokenizer, TokenNote
from src.models.chart_decoder import ChartDecoder
from src.soft_onset import MEMORY_SECONDS


def test_waveform_and_all_four_targets_follow_same_sample_shift():
    tk=ChartTokenizer(4);bpm=120.;scale=60/(32*bpm)
    notes=[TokenNote(32,0,1,0),TokenNote(48,1,2,0),TokenNote(64,2,3,16),TokenNote(96,3,4,0)]
    plan=dict(start_sample=0,content_samples=2*SAMPLE_RATE,total_samples=5*SAMPLE_RATE,
              shift_samples=[SAMPLE_RATE,2*SAMPLE_RATE])
    wave=torch.zeros(1,1,3*SAMPLE_RATE)
    for n in notes:wave[0,0,round(n.tick*scale*SAMPLE_RATE)]=n.type
    for shift in plan['shift_samples']:
        output=shifted_waveform(wave,plan,shift)
        target=shifted_targets(notes,bpm,0.,plan,shift)
        recovered=tk.decode_tokens(target['tokens'].tolist())
        assert [n.type for n in recovered]==[1,2,3,4]
        for original,n in zip(notes,recovered):
            assert n.tick*scale==pytest.approx(original.tick*scale+shift/SAMPLE_RATE)
            assert output[0,0,round(n.tick*scale*SAMPLE_RATE)]==n.type
        assert recovered[2].dur==16
        assert torch.allclose(target['frames']*MEMORY_SECONDS,
            torch.tensor(tk.ticks_per_position(target['tokens'].tolist()))*scale)


def test_cropped_hold_is_rejected_instead_of_relabelled():
    note=TokenNote(32,0,3,128)
    plan=dict(start_sample=0,content_samples=SAMPLE_RATE,total_samples=5*SAMPLE_RATE)
    with pytest.raises(ValueError,match='Hold crosses'):
        shifted_targets([note],120,0.,plan,SAMPLE_RATE)
    plan['start_sample']=SAMPLE_RATE
    with pytest.raises(ValueError,match='Hold crosses'):
        shifted_targets([note],120,0.,plan,SAMPLE_RATE)


def test_half_tick_translation_preserves_short_hold_type_and_duration():
    tk=ChartTokenizer(4)
    plan=dict(start_sample=0,content_samples=SAMPLE_RATE,total_samples=2*SAMPLE_RATE)
    target=shifted_targets([TokenNote(1,0,3,1)],120,60/32/120/2,plan,0)
    decoded=tk.decode_tokens(target['tokens'].tolist())
    assert len(decoded)==1 and decoded[0].type==3 and decoded[0].dur==1


def test_counterfactual_pair_is_same_length_and_not_equivalent():
    notes=[TokenNote(i*73,i%4,1,0) for i in range(1,70)]
    plan=choose_pair(notes,120,0.,85.,12.,random.Random(42))
    assert plan['overlap_25ms']<.5
    assert (plan['shift_samples'][1]-plan['shift_samples'][0])/SAMPLE_RATE>=.35
    wave=torch.randn(1,2,85*SAMPLE_RATE)
    views=[shifted_waveform(wave,plan,s) for s in plan['shift_samples']]
    assert views[0].shape==views[1].shape==(1,2,12*SAMPLE_RATE)
    assert torch.equal(views[0][...,plan['shift_samples'][0]:plan['shift_samples'][0]+plan['content_samples']],
                       views[1][...,plan['shift_samples'][1]:plan['shift_samples'][1]+plan['content_samples']])


def test_time_score_excludes_key_types_and_uses_next_token():
    tk=ChartTokenizer(4);tokens=torch.tensor(tk.encode_notes([TokenNote(8,0,1,0),TokenNote(16,1,4,0)]))
    logits=torch.zeros(len(tokens)-1,129,requires_grad=True)
    score,count=time_log_probability(logits,tokens)
    assert count==2
    (-score).backward()
    for i,t in enumerate(tokens[1:]):
        if tk.kind(int(t)) not in ('DTT','DTB'):assert logits.grad[i].abs().sum()==0


def test_identical_conditions_have_zero_gap_despite_dropout_and_preserve_rng():
    torch.set_num_threads(2);torch.manual_seed(17)
    decoder=ChartDecoder(hidden_dim=32,depth=1,num_heads=4,mem_dim=32,time_dim=16,
                         history_dim=28,soft_onset_dim=2,token_dropout=.4,dropout=.3).train()
    tk=ChartTokenizer(4)
    target=empty_targets(120)
    target['tokens']=torch.tensor(tk.encode_notes([TokenNote(8,0,1,0),TokenNote(16,1,4,0)]))
    target['frames']=torch.tensor(tk.ticks_per_position(target['tokens'].tolist()))*(60/32/120)/MEMORY_SECONDS
    condition=dict(features=torch.randn(1,32,32),soft_onsets=torch.rand(1,32,2),valid=torch.ones(1,32,dtype=torch.bool))
    state=torch.get_rng_state()
    ce,rank,details=paired_loss(decoder,target,condition,condition,torch.device('cpu'),rank_weight=.2)
    after_rank=torch.get_rng_state()
    assert details['gap']==pytest.approx(0.,abs=1e-7)
    assert rank.item()==pytest.approx(.2*math.log(2))
    (ce/details['supervised_tokens']+rank).backward()
    assert decoder.mem_proj.weight.grad.abs().sum()>0
    torch.set_rng_state(state)
    paired_loss(decoder,target,condition,condition,torch.device('cpu'),rank_weight=0.)
    assert torch.equal(torch.get_rng_state(),after_rank)

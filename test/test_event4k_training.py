"""Actual fitting and checkpoint provenance, beyond shape-only model checks."""
from types import SimpleNamespace
import math
import random

import pytest
import torch

from script.train_alignment_scratch import restore_training
from script.train_event4k import save_version
from src.audio_understanding.inference import sha256
from src.event4k import Event4K, encode_events, normalized_loss, generate
from src.soft_onset import MEMORY_SECONDS, state_hash
from test.test_event4k import ENC, DEC, note


def example():
    return torch.randn(128,44),encode_events(
        [note(.1,0),note(.1,3,'hold',.7),note(.8,1,'flick')],1.,150.)


def update(model,optimizer,mel,target):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=normalized_loss(model.components(target,mel,1.),target)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    return float(loss.detach())


def test_tiny_case_learns_atomic_chord_hold_and_one_song_eos():
    torch.set_num_threads(2);torch.manual_seed(73)
    model=Event4K(ENC,DEC);mel,target=example()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    losses=[update(model,optimizer,mel,target) for _ in range(250)]
    output=generate(model,mel,1.,150.,max_events=20)
    assert losses[-1]<.01 and losses[-1]<losses[0]/100
    assert output['status']=='completed' and output['learned_song_eos']
    assert output['termination']=='learned_song_eos' and 'windows' not in output
    assert len(output['notes'])==3 and output['event_count']==3
    for actual,expected in zip(output['notes'],target['reference'],strict=True):
        assert (actual['position'],actual['type'])==(expected['position'],expected['type'])
        assert abs(actual['time_seconds']-expected['time_seconds'])<.002
        assert abs(actual['end_seconds']-expected['end_seconds'])<.002


def test_weight_versions_are_distinct_and_exact_resume_matches_next_update(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(23)
    model=Event4K(ENC,DEC);mel,target=example()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001);sampler=random.Random(42)
    class Budget:
        def save(self,payload,path):torch.save(payload,path)
    args=SimpleNamespace(output=tmp_path,bpm_condition=True,epochs=2)
    objective=update(model,optimizer,mel,target)
    curves=[dict(epoch=1,objective=objective)]
    version,best=save_version(args,Budget(),model,optimizer,sampler,{},1,1,curves,float('inf'))
    assert version=='AL-Event4K@0.3.0'
    path=tmp_path/'model_latest.pt';before=sha256(path)
    saved=torch.load(path,weights_only=True)
    update(model,optimizer,mel,target);expected=state_hash(model.state_dict())
    with pytest.raises(RuntimeError,match='immutable'):
        save_version(args,Budget(),model,optimizer,sampler,{},1,1,curves,best)
    assert sha256(path)==before
    restored=Event4K(ENC,DEC);restored.load_state_dict(saved['model'])
    other=torch.optim.AdamW(restored.parameters(),lr=.001);sample2=random.Random(99)
    restore_training(saved['training_state'],other,sample2)
    assert sample2.getstate()==sampler.getstate()
    objective2=update(restored,other,mel,target)
    assert state_hash(restored.state_dict())==expected
    version,_=save_version(args,Budget(),restored,other,sample2,{},2,2,curves+[dict(epoch=2,objective=objective2)],best)
    assert version=='AL-Event4K@0.3.1' and sha256(path)!=before


@pytest.mark.parametrize('event_budget',[4,3])
def test_rollout_keeps_entire_history_across_old_boundaries_and_eos_is_terminal(monkeypatch,event_budget):
    torch.set_num_threads(2);model=Event4K(ENC,DEC)
    duration=21.;frames=math.ceil(duration/MEMORY_SECONDS)
    mel=torch.randn(128,2*frames);choices=[int(t/MEMORY_SECONDS) for t in (.1,9.9,10.1,20.1)]+[frames]
    chosen=[];steps=[];real_step=model.chart.step
    def step(event,seconds,position,caches,bpm):
        h=real_step(event,seconds,position,caches,bpm)
        steps.append(dict(position=position,seconds=seconds,cache_ids=[id(c) for c in caches],
                          history=[c['k'].shape[2] for c in caches],audio=[c['kx'].shape[2] for c in caches]))
        return h
    def time_logits(h,memory,allowed):
        index=choices[len(chosen)];chosen.append(index)
        assert allowed[0,-1]  # Entire remaining song stays selectable, including the first decision.
        if index<frames:assert allowed[0,index]
        logits=h.new_full((1,frames+1),-100.)
        logits[0,index]=100.;return logits
    def event_logits(h,audio,seconds):
        logits=h.new_full((1,4,5),-100.);logits[0,0,0]=100.;return logits
    monkeypatch.setattr(model.chart,'step',step)
    monkeypatch.setattr(model.chart,'time_logits',time_logits)
    monkeypatch.setattr(model.chart,'fractions',lambda h,audio:h.new_tensor([.5]))
    monkeypatch.setattr(model.chart,'event_logits',event_logits)
    output=generate(model,mel,duration,150.,max_events=event_budget)
    assert output['event_count']==event_budget
    assert all(s['cache_ids']==steps[0]['cache_ids'] and s['history']==[s['position']+1]*len(steps[0]['history'])
               and s['audio']==[frames]*len(steps[0]['audio']) for s in steps)
    assert [s['position'] for s in steps]==list(range(event_budget+1))
    if event_budget==4:
        assert output['status']=='completed' and output['learned_song_eos']
        assert output['event_times_seconds'][-1]>20.
        assert len(chosen)==5  # Song EOS ends decoding immediately; no later-window restart.
    else:
        assert output['status']=='failed' and not output['learned_song_eos']
        assert output['termination']=='event_budget_exhausted'


def test_physical_audio_exhaustion_is_not_counted_as_learned_eos(monkeypatch):
    torch.set_num_threads(2);model=Event4K(ENC,DEC)
    def final_frame(h,memory,allowed):
        logits=h.new_full((1,memory.shape[1]+1),-100.);logits[0,-2]=100.;return logits
    def tap(h,audio,seconds):
        logits=h.new_full((1,4,5),-100.);logits[0,0,0]=100.;return logits
    monkeypatch.setattr(model.chart,'time_logits',final_frame)
    monkeypatch.setattr(model.chart,'fractions',lambda h,audio:h.new_tensor([1.]))
    monkeypatch.setattr(model.chart,'event_logits',tap)
    result=generate(model,torch.randn(128,44),1.,150.)
    assert result['event_count']==1 and result['event_times_seconds'][0]<1.
    assert not result['learned_song_eos'] and result['termination']=='forced_audio_end'
    assert result['status']=='failed'

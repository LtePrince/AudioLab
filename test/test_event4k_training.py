"""Actual fitting and checkpoint provenance, beyond shape-only model checks."""
from types import SimpleNamespace
import random

import pytest
import torch

from script.train_alignment_scratch import restore_training
from script.train_event4k import save_version
from src.audio_understanding.inference import sha256
from src.event4k import Event4K, encode_events, normalized_loss, generate
from src.soft_onset import state_hash
from test.test_event4k import ENC, DEC, note


def example():
    return torch.randn(128,44),encode_events(
        [note(.1,0),note(.1,3,'hold',.7),note(.8,1,'flick')],1.,150.,width=.5)


def update(model,optimizer,mel,target):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=normalized_loss(model.components(target,mel,1.),target)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    return float(loss.detach())


def test_tiny_case_learns_atomic_chord_cross_window_hold_and_window_ends():
    torch.set_num_threads(2);torch.manual_seed(73)
    model=Event4K(ENC,DEC);mel,target=example()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    losses=[update(model,optimizer,mel,target) for _ in range(250)]
    output=generate(model,mel,1.,150.,width=.5,max_events=20)
    assert losses[-1]<.01 and losses[-1]<losses[0]/100
    assert output['status']=='completed' and output['forced_windows']==0
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
    assert version=='AL-Event4K@0.2.0'
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
    assert version=='AL-Event4K@0.2.1' and sha256(path)!=before

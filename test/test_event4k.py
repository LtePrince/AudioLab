import math

import pytest
import torch

from src.event4k import Event4K, EventDecoder, encode_events, event_intervals, events_to_notes, normalized_loss, sample_event, source_notes
from src.joint_audio_ar import gradient_audit
from src.pattern_diagnostics import KINDS, LANE_X
from src.soft_onset import MEMORY_SECONDS, state_hash

ENC=dict(n_mels=128,hidden_dim=32,conv_blocks=1,depth=1,num_heads=4,dropout=0.)
DEC=dict(hidden_dim=32,depth=2,num_heads=4,mem_dim=32,time_dim=16,dropout=0.,pointer_dim=16,bpm_condition=True)

def note(t,lane,kind='tap',end=None):
    return dict(time_seconds=t,end_seconds=t if end is None else end,position=lane,type=kind,screen_x=LANE_X[lane])

def test_atomic_roundtrip_chords_mixed_holds_reopen_and_final_endpoint():
    notes=[note(0.,i) for i in range(4)]+[note(.5,0,'hold',10.),note(.5,3,'flick'),
            note(10.,0,'hold',20.),note(10.,2,'drag')]
    target=encode_events(notes,20.,150.)
    assert target['event_count']==4  # 0, .5, 10, 20; Hold ends are actual events.
    assert target['events'][0].sum()==4
    assert target['events'][2,0,3:5].tolist()==[1.,1.]
    assert events_to_notes(target['times'],target['events'])==notes
    assert target['event_indices'].tolist()==[0,1,2,3,-1]
    assert target['prefix'].tolist()==[0,1,2,3,4]
    assert target['eos_count']==1
    assert not {'width','upper','bounds','stop_count'} & target.keys()
    assert not any('history' in k or 'embedding' in k for k in target)

def test_no_silent_merge_of_unsupported_notes():
    with pytest.raises(ValueError,match='duplicate'):encode_events([note(0,0),note(0,0)],2.,120.)
    with pytest.raises(ValueError,match='overlapping'):encode_events([note(0,0,'hold',1.5),note(.5,0,'hold',1.)],2.,120.)
    t=encode_events([],2.,120.)
    assert t['event_count']==0 and t['eos_count']==1 and t['event_indices'].tolist()==[-1]
    with pytest.raises(ValueError,match='HoldEnd'):events_to_notes([0.],torch.tensor([[[0,0,0,0,1]]*4]))

def test_incremental_equals_teacher_forcing_and_no_future_leak():
    torch.set_num_threads(2);torch.manual_seed(10);decoder=EventDecoder(**DEC).eval()
    e=torch.zeros(1,4,4,5);e[0,1,0,0]=1;e[0,2,1,3]=1;e[0,3,1,4]=1
    times=torch.tensor([[0.,.2,.2,1.]],dtype=torch.float64)
    memory=decoder.memory(torch.randn(1,30,32));full=decoder.hidden(e,times,memory,150.)
    caches=[b.init_cache(memory) for b in decoder.blocks]
    steps=torch.stack([decoder.step(e[0,i],float(times[0,i]),i,caches,150.)[0] for i in range(4)])[None]
    torch.testing.assert_close(full,steps,atol=1e-6,rtol=1e-5)
    other=e.clone();other[:,3]=1
    torch.testing.assert_close(full[:,:3],decoder.hidden(other,times,memory,150.)[:,:3],atol=0,rtol=0)
    assert not any('history' in n or 'bounds' in n or 'window' in n for n,_ in decoder.named_parameters())

def test_bpm_ablation_common_initialization_and_no_hidden_bpm_path():
    torch.manual_seed(42);a=EventDecoder(**DEC)
    torch.manual_seed(42);b=EventDecoder(**dict(DEC,bpm_condition=False))
    assert state_hash({k:v for k,v in a.state_dict().items() if not k.startswith('bpm_proj.')})==state_hash(b.state_dict())
    e=torch.zeros(1,1,4,5);t=torch.zeros(1,1);m=torch.randn(1,20,32)
    a.eval();b.eval()
    torch.testing.assert_close(b.hidden(e,t,m,60),b.hidden(e,t,m,240),atol=0,rtol=0)
    assert not torch.allclose(a.hidden(e,t,m,60),a.hidden(e,t,m,240))
    with pytest.raises(ValueError,match='BPM'):a.hidden(e,t,m,0)

def test_all_parameters_learn_from_atomic_targets_and_time_support():
    torch.set_num_threads(2);model=Event4K(ENC,DEC).train()
    target=encode_events([note(.1,0),note(.1,3,'hold',.9),note(.3,1,'drag')],1.,150.)
    mel=torch.randn(128,2*math.ceil(1./MEMORY_SECONDS),requires_grad=True)
    parts=model.components(target,mel,1.);loss=normalized_loss(parts,target);loss.backward()
    audit=gradient_audit(model)
    assert audit['all_connected'] and mel.grad.abs().sum()>0
    assert model.chart.bpm_proj.weight.grad.abs().sum()>0
    # No remaining physical time leaves only EOS, without NaNs or a boundary embedding.
    h=torch.randn(2,32);memory=torch.randn(1,5,32);mask=torch.zeros(2,5,dtype=torch.bool)
    p=model.chart.time_logits(h,memory,mask).softmax(-1)
    assert p[:,-1].tolist()==[1.,1.]

def test_full_song_labels_and_support_cross_old_window_boundaries():
    times=[0.,9.9,10.,10.1,19.9,20.,20.1]
    target=encode_events([note(t,0) for t in times],21.,150.)
    assert target['prefix'].tolist()==list(range(8))
    assert target['event_indices'].tolist()==list(range(7))+[-1]
    assert target['eos_count']==1
    length=math.ceil(21./MEMORY_SECONDS)
    lo,hi,allowed=event_intervals(torch.tensor([0.,10.001],dtype=torch.float64),21.,length)
    assert allowed[0].all() and hi[0,-1]==21.
    assert not allowed[1,:int(10./MEMORY_SECONDS)].any()
    assert allowed[1,-1] and lo[1,int(10.001/MEMORY_SECONDS)]==10.001
    # Every event can use all preceding events; neither 10 nor 20 adds a prefix/control row.
    assert len(target['prefix'])==len(times)+1

def test_legacy_window_labels_cannot_silently_train_full_song_eos():
    model=Event4K(ENC,DEC);target=encode_events([note(.1,0)],1.,150.)
    mel=torch.zeros(128,2*math.ceil(1./MEMORY_SECONDS))
    for changed in ({'schema':'audiolab-synchronous-event4k-v1'}, {'stop_count':2}, {'eos_count':2}):
        with pytest.raises(ValueError,match='full-song'):
            model.components(dict(target,**changed),mel,1.)

def test_silent_view_has_finite_loss_and_connected_heads():
    torch.set_num_threads(2);model=Event4K(ENC,DEC).train();target=encode_events([],1.,150.)
    mel=torch.zeros(128,2*math.ceil(1./MEMORY_SECONDS))
    loss=normalized_loss(model.components(target,mel,1.),target);assert torch.isfinite(loss)
    loss.backward();assert all(p.grad is not None for p in model.parameters())

def test_matrix_sampling_allows_chords_and_paired_hold_restart():
    logits=torch.full((4,5),-20.);logits[0,0]=20;logits[3,3:5]=20
    rng=torch.Generator().manual_seed(10)
    event=sample_event(logits,{3:0},greedy=True,rng=rng)
    assert event[0,0] and event[3,3] and event[3,4] and event.sum()==3
    for greedy in (False,True):
        event=sample_event(torch.full((4,5),-5.),{},greedy=greedy,rng=rng)
        assert event.any() and not event[:,4].any()
    # Preserve mask -inf even when every nonempty option is extremely unlikely.
    event=sample_event(torch.full((4,5),-1000.),{},greedy=True,rng=rng)
    assert event.any() and not event[:,4].any()

def test_cache_migration_never_reads_handcrafted_history():
    from src.data.chart_tokenizer import TokenNote
    from src.windowed_ar import encode_window_target
    from test.test_windowed_ar import route
    old=encode_window_target(route([TokenNote(8,0,1),TokenNote(8,3,3,8)]),1.)
    before=source_notes(old);old['history'].fill_(float('nan'))
    assert source_notes(old)==before


def test_roundtrip_checks_actual_error_at_decimal_rounding_boundary():
    from script.train_event4k import roundtrip_error
    a=note(.1234567851,0);b=note(.123456785,0)
    assert round(a['time_seconds'],8)!=round(b['time_seconds'],8)
    assert roundtrip_error([a],[b])<1e-9
    with pytest.raises(ValueError,match='1 ns'):roundtrip_error([a],[note(.12345678,0)])

"""Free-x protocol, coordinate causality, and physical waveform translation."""
import pytest
import torch

from src.anchored_ar import AnchoredAR, encode_target, generate, normalized_loss, residuals_by_note
from src.conditional_training import shifted_targets, empty_targets
from src.data.chart_tokenizer import BOS, ChartTokenizer, TokenNote, bin_to_hit, hit_to_bin
from src.models.chart_history import CausalChartHistory
from src.soft_onset import MEMORY_SECONDS


def route(notes, residuals):
    tk=ChartTokenizer(64)
    tokens=tk.encode_notes(notes)
    offsets=[];mask=[]
    for token,tick in zip(tokens,tk.ticks_per_position(tokens)):
        is_note=tk.kind(token)=='NOTE';mask.append(is_note)
        offsets.append(residuals[tick,tk.value(token)//4] if is_note else 0.)
    return dict(tokens=torch.tensor(tokens),offsets=torch.tensor(offsets),offset_mask=torch.tensor(mask),
                reference=dict(timebase_bpm=120.,offset_seconds=0.))


def model():
    torch.set_num_threads(2);torch.manual_seed(42)
    return AnchoredAR(dict(vocab_size=369,n_lanes=64,offset_head=True,hidden_dim=32,depth=2,
        num_heads=4,mem_dim=32,time_dim=16,dropout=0.,token_dropout=0.,history_dim=28,soft_onset_dim=2))


def condition():
    frames=round(4./MEMORY_SECONDS)+1
    return dict(features=torch.randn(1,frames,32),soft_onsets=torch.rand(1,frames,2),
                valid=torch.ones(1,frames,dtype=torch.bool),duration=4.)


def test_reserved_ids_and_residual_labels_follow_reencoded_notes():
    tk=ChartTokenizer(64)
    source=route([TokenNote(32,15,4),TokenNote(32,63,3,16),TokenNote(64,16,1)],
                 {(32,15):.2,(32,63):.8,(64,16):.4})
    t=encode_target(source,4.,n_lanes=64)
    assert tk.note(15,4)==113 and tk.note(16,1)==114  # 4K's reserved IDs are real free-x notes.
    assert [int(t['tokens'][i]) for i in t['decisions']]==[BOS,354,354]
    assert t['coordinate_count']==3 and t['time_count']==2
    assert t['offsets'][t['offset_mask']].tolist()==pytest.approx([.2,.8,.4])
    assert t['tokens'][t['offset_mask']].tolist()==[113,tk.note(63,3),114]
    assert int(tk.durt(16)) in t['tokens'].tolist()
    assert torch.all(t['layout_targets'][t['decisions']]==-100)


def test_residual_changes_history_only_after_consuming_that_note():
    notes=[TokenNote(32,15,4),TokenNote(64,16,1)]
    a=encode_target(route(notes,{(32,15):.2,(64,16):.4}),4.,n_lanes=64)
    b=encode_target(route(notes,{(32,15):.9,(64,16):.4}),4.,n_lanes=64)
    first=int(torch.where(a['offset_mask'])[0][0])
    assert torch.equal(a['history'][:first+1],b['history'][:first+1])
    assert not torch.equal(a['history'][first+1:],b['history'][first+1:])
    state=CausalChartHistory(64,60/120/MEMORY_SECONDS)
    for token,frame,offset,expected in zip(a['tokens'],a['frames'],a['offsets'],a['history']):
        assert torch.tensor(state.features(frame)).allclose(expected)
        state.observe(token,frame,offset)


def test_shift_preserves_subbin_geometry_and_complete_hold():
    notes=[TokenNote(32,63,3,16),TokenNote(64,15,4)]
    source=route(notes,{(32,63):.8,(64,15):.2})
    from src.audio_understanding import SAMPLE_RATE
    plan=dict(start_sample=0,content_samples=2*SAMPLE_RATE,total_samples=4*SAMPLE_RATE)
    moved=shifted_targets(notes,120.,0.,plan,SAMPLE_RATE,n_lanes=64,residuals=residuals_by_note(source))
    assert residuals_by_note(moved)==pytest.approx({(96,63):.8,(128,15):.2})
    target=encode_target(moved,4.,n_lanes=64)
    assert target['reference_times']==[1.5,2.]
    decoded=ChartTokenizer(64).decode_tokens(moved['tokens'].tolist(),strict=True)
    assert decoded[0].dur==16
    for note in decoded:
        offset=residuals_by_note(moved)[note.tick,note.lane]
        x=bin_to_hit(note.lane,offset)
        lane,back=hit_to_bin(x)
        assert lane==note.lane and back==pytest.approx(offset)


def test_coordinate_head_gets_gradients_and_silence_has_no_coordinate_loss():
    m=model();c=condition()
    t=encode_target(route([TokenNote(32,63,3,8)],{(32,63):.9}),4.,n_lanes=64)
    totals={k:t[k] for k in ('time_count','layout_count','stop_count','continue_count','coordinate_count')}
    parts=m.components(t,c,use_soft=True);normalized_loss(parts,totals).backward()
    for layer in (m.decoder.offset_head[0],m.time_query,m.stop_head):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum()>0
    silence=encode_target(empty_targets(120.),4.,n_lanes=64)
    assert silence['coordinate_count']==0
    assert m.components(silence,c,use_soft=False)['coordinate_mse']==0


def test_generated_residual_uses_current_note_hidden_and_previous_note_history(monkeypatch):
    m=model().eval();c=condition();tk=ChartTokenizer(64)
    sequence=[BOS,353,tk.note(15,4),tk.note(63,3),tk.durt(4),354,353,tk.note(16,1),354]
    observed=[]

    class Stop(torch.nn.Module):
        def forward(self,h):return torch.where(h[:,:1]>=8,40.,-40.)

    def step(token,frame,pos,caches,length,valid,history):
        observed.append((int(token[0]),float(frame[0]),history[0].clone()))
        logits=torch.full((1,369),-40.)
        logits[0,sequence[min(pos+1,len(sequence)-1)]]=40.
        return logits,torch.full((1,32),float(pos))

    def pointer(h,memory,allowed):
        scores=torch.full(allowed.shape,-40.);scores[:,21]=40.
        return scores.masked_fill(~allowed,float('-inf'))

    monkeypatch.setattr(m.decoder,'step',step)
    monkeypatch.setattr(m.decoder,'predict_offset',lambda h:h[...,0]/10)
    monkeypatch.setattr(m,'frame_logits',pointer)
    monkeypatch.setattr(m,'fraction',lambda h,k:torch.full((len(h),),.5))
    m.stop_head=Stop()
    result=generate(m,c,120,use_soft=True,greedy=True)
    assert result['status']=='completed' and result['note_count']==3
    assert [n['residual'] for n in result['notes']]==pytest.approx([.2,.3,.7])
    assert [n['screen_x'] for n in result['notes']]==pytest.approx([
        bin_to_hit(15,.2),bin_to_hit(63,.3),bin_to_hit(16,.7)])
    assert result['notes'][1]['end_seconds']-result['notes'][1]['time_seconds']==pytest.approx(4/64)
    state=CausalChartHistory(64,60/120/MEMORY_SECONDS)
    for i,(token,frame,features) in enumerate(observed):
        assert torch.tensor(state.features(frame)).allclose(features,atol=2e-5)
        state.observe(token,frame,i/10 if tk.kind(token)=='NOTE' else None)
    partial=generate(m,c,120,use_soft=True,greedy=True,max_tokens=2)
    assert partial['status']=='failed' and partial['notes']==[]
    assert partial['incomplete_coordinate_notes']==1


def test_missing_coordinate_labels_are_rejected():
    source=route([TokenNote(32,63,1)],{(32,63):.8})
    source['offset_mask'].fill_(False)
    with pytest.raises(ValueError,match='mask'):
        encode_target(source,4.,n_lanes=64)

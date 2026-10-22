import pytest
import torch

from src.anchored_ar import (ANCHOR, END_GROUP, AnchoredAR, encode_target, time_intervals,
                            generate, normalized_loss)
from src.data.chart_tokenizer import BOS, ChartTokenizer, TokenNote
from src.soft_onset import MEMORY_SECONDS


def target(notes, duration=4.):
    return encode_target(dict(tokens=torch.tensor(ChartTokenizer(4).encode_notes(notes)),
        reference=dict(timebase_bpm=120.,offset_seconds=0.)),duration)


def model():
    torch.set_num_threads(2);torch.manual_seed(42)
    return AnchoredAR(dict(vocab_size=129,hidden_dim=32,depth=2,num_heads=4,mem_dim=32,
        time_dim=16,dropout=0.,token_dropout=0.,history_dim=28,soft_onset_dim=2))


def condition(duration=4.):
    length=int(duration/MEMORY_SECONDS)+1
    return dict(features=torch.randn(1,length,32),soft_onsets=torch.rand(1,length,2),
                valid=torch.ones(1,length,dtype=torch.bool),duration=duration)


def test_target_grouping_and_pointer_before_future_time_or_note():
    a=target([TokenNote(32,0,1),TokenNote(32,2,3,16),TokenNote(33,1,4),TokenNote(64,3,2)])
    b=target([TokenNote(40,0,1),TokenNote(40,2,3,16),TokenNote(41,1,4),TokenNote(72,3,2)])
    assert a['time_count']==3 and a['stop_targets'].tolist()==[0,0,0,1]
    first=int(a['decisions'][0]);assert first==0
    for key in ('tokens','frames','history'):
        assert torch.equal(a[key][:first+1],b[key][:first+1])
    assert a['next_seconds'][0]!=b['next_seconds'][0]
    for index in a['decisions']:
        assert int(a['tokens'][index]) in (BOS,END_GROUP)
        assert a['layout_targets'][index]==-100
    assert list(a['tokens']).count(ANCHOR)==3


def test_same_frame_successors_and_tail_intervals_roundtrip():
    # Two real successors inside one memory frame, including close to song end.
    duration=2*MEMORY_SECONDS-.001
    lower=torch.tensor([.01,.025,duration-1e-6],dtype=torch.float64)
    lo,hi,allowed=time_intervals(lower,3,duration,torch.tensor([[True,True,False]]))
    assert allowed[0,0] and allowed[1,0] and allowed[2,1]
    assert not allowed[:,2].any()
    truth=torch.tensor([.02,.03,duration-.5e-6],dtype=torch.float64)
    bins=torch.floor(truth/MEMORY_SECONDS).long();rows=torch.arange(3)
    fraction=(truth-lo[rows,bins])/(hi[rows,bins]-lo[rows,bins])
    restored=lo[rows,bins]+fraction*(hi[rows,bins]-lo[rows,bins])
    assert torch.allclose(restored,truth,atol=1e-12,rtol=0)
    assert (restored>lower).all()


def test_next_anchor_change_cannot_change_previous_group_hidden():
    m=model().eval();c=condition();d=m.decoder
    a=target([TokenNote(32,0,1),TokenNote(64,2,4)])
    b=target([TokenNote(32,0,1),TokenNote(80,3,2)])
    boundary=int(a['decisions'][1])
    with torch.no_grad():
        memory=d.prepare_memory(c['features'],c['soft_onsets'])
        def hidden(t):
            return d(t['tokens'][None],t['frames'][None],memory,c['valid'],
                     history=t['history'][None],return_hidden=True)[1]
        ha,hb=hidden(a),hidden(b)
    assert torch.allclose(ha[:,:boundary+1],hb[:,:boundary+1],atol=1e-7)
    assert not torch.allclose(ha[:,boundary+1:],hb[:,boundary+1:])


def test_full_forward_equals_cached_step_including_group_history():
    m=model().eval();t=target([TokenNote(32,0,3,16),TokenNote(32,2,1),TokenNote(33,1,4)])
    c=condition();d=m.decoder
    memory=d.prepare_memory(c['features'],c['soft_onsets'])
    with torch.no_grad():
        logits,h=d(t['tokens'][None],t['frames'][None],memory,c['valid'],history=t['history'][None],return_hidden=True)
        caches=d.init_cache(memory);rows=[];states=[]
        for i in range(len(t['tokens'])):
            out,state=d.step(t['tokens'][i:i+1],t['frames'][i:i+1],i,caches,memory.shape[1],c['valid'],history=t['history'][i:i+1])
            rows.append(out[0]);states.append(state[0])
    assert torch.allclose(torch.stack(rows),logits[0],atol=2e-6)
    assert torch.allclose(torch.stack(states),h[0],atol=2e-6)


def test_supervised_heads_receive_gradients_and_zero_soft_is_exact_control():
    m=model();t=target([TokenNote(32,0,1),TokenNote(33,2,3,8)])
    c=condition();counts={k:t[k] for k in ('time_count','layout_count','stop_count','continue_count')}
    parts=m.components(t,c,use_soft=True);normalized_loss(parts,counts).backward()
    for head in (m.time_query,m.time_key,m.stop_head,m.subframe_head[0],m.decoder.soft_onset_proj):
        assert head.weight.grad is not None and head.weight.grad.abs().sum()>0
    m.eval()
    with torch.no_grad():
        a=m.components(t,c,use_soft=False)
        zero=dict(c,soft_onsets=torch.zeros_like(c['soft_onsets']))
        b=m.components(t,zero,use_soft=True)
    assert a['frame_ce']==b['frame_ce'] and a['layout_ce']==b['layout_ce']


def test_silence_can_stop_at_bos_without_any_forced_note():
    m=model().eval();c=condition();t=target([])
    with torch.no_grad():m.stop_head.weight.zero_();m.stop_head.bias.fill_(40)
    result=generate(m,c,120,use_soft=True,greedy=True)
    assert result['status']=='completed' and result['notes']==[] and result['trace']==[[BOS,0.]]
    p=m.components(t,c,use_soft=True)
    assert p['frame_ce']==0 and p['layout_ce']==0 and torch.isfinite(p['stop_bce'])


def test_invalid_reference_hold_is_reported_not_shortened():
    with pytest.raises(ValueError,match='bounds'):
        target([TokenNote(32,0,3,256)],duration=2.)


def test_free_protocol_exports_two_same_frame_groups_and_complete_hold(monkeypatch):
    m=model().eval();c=condition();tk=ChartTokenizer(4)
    sequence=[BOS,ANCHOR,tk.note(0,3),tk.durt(4),tk.note(2,1),END_GROUP,ANCHOR,tk.note(1,4),END_GROUP]

    class ScriptedStop(torch.nn.Module):
        def forward(self,hidden):
            return torch.where(hidden[:,:1]>=8,40.,-40.)

    def step(token,frame,pos,caches,length,valid,history):
        logits=torch.full((1,129),-40.)
        logits[0,sequence[min(pos+1,len(sequence)-1)]]=40.
        return logits,torch.full((1,32),float(pos))

    def pointer(hidden,memory,allowed):
        logits=torch.full(allowed.shape,-40.)
        logits[:,21]=40.
        return logits.masked_fill(~allowed,float('-inf'))

    monkeypatch.setattr(m.decoder,'step',step)
    monkeypatch.setattr(m,'frame_logits',pointer)
    monkeypatch.setattr(m,'fraction',lambda h,k:torch.full((len(h),),.5))
    m.stop_head=ScriptedStop()
    result=generate(m,c,120,use_soft=True,greedy=True)
    assert result['status']=='completed' and result['note_count']==3 and result['generated_onsets']==2
    a,b=result['onsets_seconds']
    assert a<b and int(a/MEMORY_SECONDS)==int(b/MEMORY_SECONDS)==21
    assert result['notes'][0]['time_seconds']==result['notes'][1]['time_seconds']==a
    assert result['notes'][0]['end_seconds']-a==pytest.approx(4*60/32/120)
    assert {n['type'] for n in result['notes']}=={'tap','hold','flick'}


@pytest.mark.parametrize('fraction,frame_index',[(0.,0),(1.,86)])
def test_saturated_offsets_keep_first_and_last_intervals_valid(monkeypatch,fraction,frame_index):
    m=model().eval();c=condition();tk=ChartTokenizer(4)
    sequence=[BOS,ANCHOR,tk.note(0,1),END_GROUP]

    class Stop(torch.nn.Module):
        def forward(self,hidden):return torch.where(hidden[:,:1]>=3,40.,-40.)

    def step(token,frame,pos,caches,length,valid,history):
        logits=torch.full((1,129),-40.)
        logits[0,sequence[min(pos+1,3)]]=40.
        return logits,torch.full((1,32),float(pos))

    def pointer(hidden,memory,allowed):
        scores=torch.full(allowed.shape,-40.);scores[:,frame_index]=40.
        return scores.masked_fill(~allowed,float('-inf'))

    monkeypatch.setattr(m.decoder,'step',step);monkeypatch.setattr(m,'frame_logits',pointer)
    monkeypatch.setattr(m,'fraction',lambda h,k:torch.full((len(h),),fraction))
    m.stop_head=Stop()
    result=generate(m,c,120,use_soft=True,greedy=True)
    assert result['status']=='completed' and result['note_count']==1
    assert 0<=result['onsets_seconds'][0]<c['duration']
    if fraction==0:assert result['onsets_seconds']==[0.]
    # An anchor selected on the final budget step is not an exported note.
    partial=generate(m,c,120,use_soft=True,greedy=True,max_tokens=1)
    assert partial['status']=='failed' and partial['onsets_seconds']==[]

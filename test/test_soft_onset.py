import copy

import numpy as np
import pytest
import torch

from src.data.chart_tokenizer import BOS, EOS, ChartTokenizer, TokenNote
from src.data.timing import TIMING_SCHEMA
from src.models.chart_decoder import ChartDecoder, generate_tokens
from src.models.chart_history import CausalChartHistory, HISTORY_DIM, history_features
from src.soft_onset import (MEMORY_SECONDS, condition_input, dense_condition,
                            forward_loss, initialize_decoder, load_decoder, state_hash)
from script.train_soft_onset import onset_metrics, sample_order


def tiny_config():
    return dict(hidden_dim=16,depth=1,num_heads=2,mlp_ratio=2,dropout=0.,mem_dim=8,time_dim=8)


def test_dense_condition_keeps_low_probability_and_native_offset():
    frames=np.arange(3)*MEMORY_SECONDS
    delta=np.array([.1,.7,.9])*MEMORY_SECONDS
    arrays=dict(frame_times_seconds=frames,event_probability=np.array([1e-6,.2,.99]),
                time_offset_seconds=delta,event_times_seconds=frames+delta)
    result=dense_condition(arrays,3)
    torch.testing.assert_close(result,torch.tensor([[[1e-6,.1],[.2,.7],[.99,.9]]]))
    assert torch.count_nonzero(condition_input(result,'none'))==0
    assert condition_input(result,'soft') is result
    arrays['frame_times_seconds']=frames/2
    with pytest.raises(ValueError,match='every encoder-memory frame'):dense_condition(arrays,3)


def test_invalid_probability_and_clock_rejected():
    arrays=dict(frame_times_seconds=np.array([0.]),event_probability=np.array([1.1]),
                time_offset_seconds=np.array([0.]),event_times_seconds=np.array([0.]))
    with pytest.raises(ValueError,match='invalid dense'):dense_condition(arrays,1)
    arrays['event_probability'][0]=.2;arrays['event_times_seconds'][0]=.1
    with pytest.raises(ValueError,match='absolute event clock'):dense_condition(arrays,1)


def test_zero_initialization_legacy_equality_and_explicit_contract(tmp_path):
    torch.manual_seed(12);legacy=ChartDecoder(**tiny_config()).eval()
    path=tmp_path/'old.pt'
    torch.save(dict(timing_schema=TIMING_SCHEMA,dec_config=tiny_config(),
                    model={'decoder.'+k:v for k,v in legacy.state_dict().items()}),path)
    a,_,ia=initialize_decoder(path,42);b,_,ib=initialize_decoder(path,42)
    a.eval();b.eval();assert ia==ib
    assert state_hash(a.state_dict())==state_hash(b.state_dict())
    feats=torch.randn(1,5,8);soft=torch.rand(1,5,2)
    torch.testing.assert_close(a.prepare_memory(feats,torch.zeros_like(soft)),legacy.prepare_memory(feats),rtol=0,atol=0)
    torch.testing.assert_close(b.prepare_memory(feats,soft),legacy.prepare_memory(feats),rtol=0,atol=0)
    with pytest.raises(ValueError,match='requires aligned'):b.prepare_memory(feats)
    with pytest.raises(ValueError,match='legacy decoder'):legacy.prepare_memory(feats,soft)
    with pytest.raises(ValueError,match='experiment checkpoint'):load_decoder(path,'cpu')


def test_soft_branch_learns_while_zero_control_cannot_read_onsets():
    torch.manual_seed(12);a=ChartDecoder(**tiny_config(),soft_onset_dim=2)
    b=copy.deepcopy(a);tk=ChartTokenizer()
    tokens=tk.encode_notes([TokenNote(1,0,1,0),TokenNote(4,2,3,2)])
    sample=dict(valid=torch.ones(1,5,dtype=torch.bool),soft_onsets=torch.rand(1,5,2),routes={
        '4k':dict(tokens=torch.tensor(tokens),frames=torch.tensor(tk.ticks_per_position(tokens),dtype=torch.float32),
                  features=torch.randn(1,5,8),offsets=torch.zeros(len(tokens)),
                  offset_mask=torch.zeros(len(tokens),dtype=torch.bool))})
    before=sample['soft_onsets'].clone()
    for decoder,condition in ((a,'none'),(b,'soft')):
        opt=torch.optim.SGD(decoder.parameters(),lr=.1)
        loss,_=forward_loss(decoder,sample,'4k',condition,'cpu');loss.backward()
        norm=float(decoder.soft_onset_proj.weight.grad.norm())
        assert norm==0 if condition=='none' else norm>0
        opt.step()
    b.eval();feats=sample['routes']['4k']['features']
    assert not torch.equal(b.prepare_memory(feats,before),b.prepare_memory(feats,torch.zeros_like(before)))
    assert torch.equal(sample['soft_onsets'],before)
    assert not sample['routes']['4k']['features'].requires_grad


def test_free_decoder_can_skip_candidate_and_add_different_time(monkeypatch):
    tk=ChartTokenizer();dec=ChartDecoder(**tiny_config()).eval()
    # A hypothetical posterior event is at tick 2. Free decoder chooses tick 4;
    # this proof exercises the existing real grammar, without a forced skeleton.
    sequence=iter([tk.dtt(4),tk.note(1,1),EOS])
    def step(*args):
        logits=torch.full((1,tk.vocab_size),-100.)
        logits[0,next(sequence)]=100.
        return logits,torch.zeros(1,16)
    monkeypatch.setattr(dec,'step',step)
    out=generate_tokens(dec,torch.zeros(1,8,16),torch.ones(1,8,dtype=torch.bool),120,
                        MEMORY_SECONDS*1000,100,tokenizer=tk,skeleton_ticks=None,max_tokens=4,require_complete=True)
    assert [n.tick for n in tk.decode_tokens(out,strict=True)]==[4]


def test_fixed_training_order_independent_of_condition():
    rows=list(range(7))
    assert sample_order(rows,20,42)==sample_order(rows,20,42)
    assert set(sample_order(rows,7,42))==set(rows)
    assert sample_order(rows,20,43)!=sample_order(rows,20,42)


def test_onset_score_reuses_optimal_one_to_one_matcher():
    from types import SimpleNamespace
    # At 60 BPM, one tick is 1/32 seconds: truth [0,.06], prediction [.04,.08].
    result=onset_metrics([dict(time=.04*32,holdTime=0),dict(time=.08*32,holdTime=0)],
                         [SimpleNamespace(tick=0),SimpleNamespace(tick=.06*32)],60,0,1)
    assert result['matched']==2 and result['f1']==1.


@pytest.mark.parametrize('lanes',[4,64])
def test_history_teacher_step_agree_and_current_residual_is_not_visible(lanes):
    torch.manual_seed(7);tk=ChartTokenizer(lanes)
    dec=ChartDecoder(**tiny_config(),vocab_size=tk.vocab_size,n_lanes=lanes,
                     offset_head=lanes==64,history_dim=HISTORY_DIM).eval()
    # Nonzero learned history branch ensures the test can catch omission/misalignment.
    torch.nn.init.normal_(dec.history_proj.weight,std=.1)
    tokens=torch.tensor(tk.encode_notes([TokenNote(1,0,1,0),TokenNote(5,2,3,2),TokenNote(9,1,4,0)]))
    frames=torch.tensor(tk.ticks_per_position(tokens),dtype=torch.float32)*.1
    residual=torch.full((len(tokens),),.2)
    kwargs=dict(n_lanes=lanes,frames_per_beat=3.2,offsets=residual if lanes==64 else None)
    hist=history_features(tokens,frames,**kwargs)
    memory=dec.prepare_memory(torch.randn(1,6,8));valid=torch.ones(1,6,dtype=torch.bool)
    full=dec(tokens[None],frames[None],memory,valid,history=hist[None])
    cache=dec.init_cache(memory);state=CausalChartHistory(lanes,3.2);stepped=[]
    for i,token in enumerate(tokens):
        prefix=torch.tensor([state.features(frames[i])])
        torch.testing.assert_close(prefix[0],hist[i])
        logits,_=dec.step(token[None],frames[i:i+1],i,cache,6,valid,history=prefix)
        stepped.append(logits[0]);state.observe(token,frames[i],residual[i] if lanes==64 else None)
    torch.testing.assert_close(torch.stack(stepped),full[0],atol=1e-5,rtol=1e-5)
    if lanes==64:
        note=next(i for i,t in enumerate(tokens) if tk.kind(int(t))=='NOTE')
        changed=residual.clone();changed[note]=.9
        other=history_features(tokens,frames,n_lanes=lanes,frames_per_beat=3.2,offsets=changed)
        altered=dec(tokens[None],frames[None],memory,valid,history=other[None])
        torch.testing.assert_close(full[:,:note+1],altered[:,:note+1],atol=0,rtol=0)
        assert not torch.equal(full[:,note+1:],altered[:,note+1:])


def test_incremental_free_history_observes_predicted_offset_after_note(monkeypatch):
    tk=ChartTokenizer(64)
    dec=ChartDecoder(**tiny_config(),vocab_size=tk.vocab_size,n_lanes=64,
                     offset_head=True,history_dim=HISTORY_DIM).eval()
    sequence=iter([tk.dtt(4),tk.note(8,1),tk.dtt(4),EOS]);seen=[]
    def step(token,frame,pos,caches,t,valid,history=None):
        seen.append((int(token[0]),history.clone()))
        logits=torch.full((1,tk.vocab_size),-100.);logits[0,next(sequence)]=100.
        return logits,torch.zeros(1,16)
    monkeypatch.setattr(dec,'step',step)
    monkeypatch.setattr(dec,'predict_offset',lambda h:torch.full(h.shape[:2],.75))
    tokens,offsets=generate_tokens(dec,torch.zeros(1,8,16),torch.ones(1,8,dtype=torch.bool),120,
        MEMORY_SECONDS*1000,100,tokenizer=tk,max_tokens=5,require_complete=True)
    current=next(h for token,h in seen if token==tk.note(8,1))
    assert current[0,0]==0  # Its own offset is unavailable to its hidden state.
    later=seen[-1][1]
    assert later[0,1]==pytest.approx((8+.75)/64)
    assert list(offsets.values())==[.75]

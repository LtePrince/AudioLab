import math

import torch

from src.anchored_ar import AnchoredAR, encode_target, normalized_loss
from src.data.chart_tokenizer import BOS, ChartTokenizer, TokenNote
from src.soft_onset import MEMORY_SECONDS
from src.windowed_ar import WindowedAR, encode_window_target, generate, window_intervals


CONFIG=dict(vocab_size=129,hidden_dim=32,depth=2,num_heads=4,mem_dim=32,time_dim=16,
            dropout=0.,token_dropout=0.,history_dim=28,soft_onset_dim=2)


def make_model():
    torch.set_num_threads(2);torch.manual_seed(42)
    return WindowedAR(CONFIG).eval()


def route(notes):
    return dict(tokens=torch.tensor(ChartTokenizer(4).encode_notes(notes)),
                reference=dict(timebase_bpm=120.,offset_seconds=0.))


def condition(duration=25.):
    length=math.ceil(duration/MEMORY_SECONDS)
    return dict(features=torch.randn(1,length,32),soft_onsets=torch.rand(1,length,2),
                valid=torch.ones(1,length,dtype=torch.bool),duration=duration)


def test_common_backbone_initialization_and_zero_window_condition():
    torch.manual_seed(42);baseline=AnchoredAR(CONFIG)
    model=make_model()
    for name,tensor in baseline.state_dict().items():assert torch.equal(tensor,model.state_dict()[name])
    assert not model.window_condition(torch.tensor([[0.,10.],[10.,20.]])).any()


def test_window_target_keeps_every_original_group_and_hold_span_once():
    r=route([TokenNote(608,0,3,128),TokenNote(640,1,4),TokenNote(1280,3,2)])
    original=encode_target(r,25.);target=encode_window_target(r,25.)
    eow=ChartTokenizer(4).cond(2)
    assert target['tokens'][target['tokens']!=eow].tolist()==original['tokens'].tolist()
    assert target['time_count']==3 and target['stop_count']==3
    assert target['stop_targets'].tolist()==[0.,1.,0.,1.,0.,1.]
    assert target['next_seconds'][target['stop_targets']==0].tolist()==[9.5,10.,20.]
    assert target['windows'][0]['reference_groups']==1
    assert target['windows'][1]['reference_groups']==1
    assert target['windows'][2]['reference_groups']==1
    for i,token in enumerate(target['tokens']):
        if int(token) in (BOS,ChartTokenizer(4).cond(1),eow):assert target['layout_targets'][i]==-100


def test_pointer_boundaries_are_half_open_without_masking_global_memory():
    model=make_model();c=condition()
    lower=torch.tensor([0.,9.999,10.,19.999],dtype=torch.float64)
    upper=torch.tensor([10.,10.,20.,20.],dtype=torch.float64)
    lo,hi,allowed=window_intervals(lower,upper,c['features'].shape[1],c['valid'])
    assert ((lo>=lower[:,None])&((hi<=upper[:,None]))).all()
    assert torch.all(hi[allowed]>lo[allowed])
    assert not allowed[0,math.ceil(10/MEMORY_SECONDS):].any()
    mask=model.decoder.cross_mask(torch.tensor([[0.,10.,20.]]),c['features'].shape[1],c['valid'])
    assert mask.all() and mask.shape[-1]==c['features'].shape[1]
    truth=torch.tensor([.01,9.9995,10.,19.9999],dtype=torch.float64)
    bins=torch.floor(truth/MEMORY_SECONDS).long();rows=torch.arange(4)
    fractions=(truth-lo[rows,bins])/(hi[rows,bins]-lo[rows,bins])
    assert allowed[rows,bins].all() and ((fractions>=0)&(fractions<1)).all()
    assert torch.allclose(lo[rows,bins]+fractions*(hi[rows,bins]-lo[rows,bins]),truth,rtol=0.,atol=1e-12)


def test_teacher_forcing_and_incremental_steps_have_same_window_context():
    model=make_model();c=condition()
    target=encode_window_target(route([TokenNote(608,0,3,128),TokenNote(640,2,1),TokenNote(1280,3,4)]),25.)
    # Exercise nonzero learned boundary conditioning, not just zero-init parity.
    torch.nn.init.normal_(model.bounds_proj.weight,std=.03)
    d=model.decoder;memory=d.prepare_memory(c['features'],c['soft_onsets'])
    bias=model.window_condition(target['bounds'])
    with torch.no_grad():
        logits,hidden=d(target['tokens'][None],target['frames'][None],memory,c['valid'],
            history=target['history'][None],input_condition=bias[None],return_hidden=True)
        caches=d.init_cache(memory);outputs=[];states=[]
        for i in range(len(target['tokens'])):
            out,state=d.step(target['tokens'][i:i+1],target['frames'][i:i+1],i,caches,memory.shape[1],
                c['valid'],history=target['history'][i:i+1],input_condition=bias[i:i+1])
            outputs.append(out[0]);states.append(state[0])
    assert torch.allclose(torch.stack(outputs),logits[0],atol=3e-6)
    assert torch.allclose(torch.stack(states),hidden[0],atol=3e-6)
    assert caches[0]['k'].shape[-2]==len(target['tokens'])


def test_future_audio_visible_but_future_chart_cannot_leak():
    model=make_model();c=condition();d=model.decoder
    a=encode_window_target(route([TokenNote(32,0,1),TokenNote(1280,1,4)]),25.)
    b=encode_window_target(route([TokenNote(32,0,1),TokenNote(1344,2,2)]),25.)
    boundary=int(a['decisions'][1])
    def hidden(target,features):
        memory=d.prepare_memory(features,c['soft_onsets'])
        return d(target['tokens'][None],target['frames'][None],memory,c['valid'],
                 history=target['history'][None],input_condition=model.window_condition(target['bounds'])[None],
                 return_hidden=True)[1]
    with torch.no_grad():
        ha=hidden(a,c['features']);hb=hidden(b,c['features'])
        changed=c['features'].clone();changed[:,int(20/MEMORY_SECONDS):]+=torch.randn_like(changed[:,int(20/MEMORY_SECONDS):])*10
        hc=hidden(a,changed)
    assert torch.allclose(ha[:,:boundary+1],hb[:,:boundary+1],atol=1e-7)
    assert not torch.allclose(ha[:,0],hc[:,0],atol=1e-5)


def test_window_boundaries_receive_supervised_gradients():
    model=make_model().train();c=condition()
    target=encode_window_target(route([TokenNote(64,0,1),TokenNote(700,2,3,32)]),25.)
    parts=model.components(target,c,use_soft=False)
    normalized_loss(parts,target).backward()
    assert model.bounds_proj.weight.grad.abs().sum()>0
    assert model.time_query.weight.grad.abs().sum()>0
    assert model.stop_head.weight.grad.abs().sum()>0


def test_silent_windows_are_legal_without_claiming_song_eos():
    model=make_model();c=condition()
    with torch.no_grad():model.stop_head.weight.zero_();model.stop_head.bias.fill_(40)
    out=generate(model,c,120.,use_soft=False,greedy=True)
    assert out['status']=='completed' and out['notes']==[]
    assert out['scheduler_traversal_fraction']==out['learned_window_end_fraction']==1.
    assert out['learned_song_eos'] is False and out['eos_clock_seconds'] is None
    assert [row[1] for row in out['trace']]==[0.,10.,20.]
    assert [w['note_count'] for w in out['windows']]==[0,0,0]


def test_forced_boundaries_are_reported_separately_and_keep_later_windows():
    model=make_model();c=condition()
    with torch.no_grad():
        model.stop_head.weight.zero_();model.stop_head.bias.fill_(-40)
        model.subframe_head[-1].weight.zero_();model.subframe_head[-1].bias.fill_(100)
        model.decoder.lm_head.weight.zero_();model.decoder.lm_head.bias.zero_()
        model.decoder.lm_head.bias[ChartTokenizer(4).note(0,1)]=100
        model.decoder.lm_head.bias[ChartTokenizer(4).cond(1)]=50
    model.frame_logits=lambda h,memory,allowed:torch.arange(memory.shape[1],device=h.device).float()[None].masked_fill(~allowed,float('-inf'))
    out=generate(model,c,120.,use_soft=False,greedy=True,max_tokens=50)
    assert out['status']=='failed' and out['forced_windows']==3
    assert out['scheduler_traversal_fraction']==1. and out['learned_window_end_fraction']==0.
    assert len(out['notes'])==3
    for note,window in zip(out['notes'],out['windows']):
        assert window['start']<=note['time_seconds']<window['end']

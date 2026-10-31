import torch

from script.train_pattern_continuation import PREFIX, FAMILIES, lanes_for, prefix_target, continuation, loss_for
from src.data.chart_tokenizer import ChartTokenizer, TokenNote
from src.windowed_variants import VariantAR
from test.test_windowed_ar import CONFIG, condition, route


def fixture(family='staircase'):
    lanes=lanes_for(family,(0,1,2,3));ticks=[32+i*32 for i in range(24)]
    return dict(lanes=lanes,prefix=lanes[:PREFIX],times=[t/64 for t in ticks],duration=13.,
        route=route([TokenNote(t,lane,1) for t,lane in zip(ticks,lanes)]),condition=condition(13.))


def test_prefix_unambiguously_selects_the_pattern_and_is_excluded_from_loss():
    assert len({tuple(lanes_for(f,(0,1,2,3))[:PREFIX]) for f in FAMILIES})==5
    row=fixture();target,indices,labels=prefix_target(row)
    assert len(indices)==16
    assert labels.tolist()==row['lanes'][PREFIX:]
    # The position head is queried at ANCHOR, before the corresponding NOTE.
    assert (target['tokens'][indices]==ChartTokenizer(4).cond(0)).all()
    assert (target['frames'][indices]*1024/22050>=row['times'][PREFIX]-.00001).all()


def test_rollout_only_copies_prefix_and_then_feeds_its_own_predictions():
    torch.set_num_threads(2);row=fixture();model=VariantAR(CONFIG).eval()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.decoder.lm_head.bias[ChartTokenizer(4).note(3,1)]=10.
    seen=[]
    handle=model.decoder.tok_emb.register_forward_pre_hook(lambda _m,args:seen.extend(args[0].flatten().tolist()))
    result=continuation(model,row['condition'],row['times'],row['prefix'])
    handle.remove()
    assert result[:PREFIX]==row['prefix']
    assert result[PREFIX:]==[3]*16
    note_ids=[t for t in seen if 50<=t<66]
    assert note_ids==[ChartTokenizer(4).note(l,1) for l in result]
    assert ChartTokenizer(4).cond(2) in seen  # preserve pattern history across output windows


def test_suffix_prediction_has_no_future_chart_leak():
    torch.manual_seed(1);row=fixture();model=VariantAR(CONFIG).eval()
    target,indices,labels=prefix_target(row)
    other=dict(row,route=route([TokenNote(round(t*64),lane if i<=PREFIX else 3,1)
        for i,(t,lane) in enumerate(zip(row['times'],row['lanes']))]))
    changed,changed_indices,_=prefix_target(other)
    d=model.decoder;c=row['condition'];mem=d.prepare_memory(c['features'],c['soft_onsets']*0)
    def forward(t):
        return d(t['tokens'][None],t['frames'][None],mem,c['valid'],history=t['history'][None],
            input_condition=model.window_condition(t['bounds'])[None])
    with torch.no_grad():a,b=forward(target),forward(changed)
    assert torch.equal(indices,changed_indices)
    assert torch.allclose(a[:,indices[0]],b[:,indices[0]],atol=1e-7)
    loss_for(model,row,target,indices,labels).backward()
    assert model.decoder.blocks[0].qkv.weight.grad.abs().sum()>0

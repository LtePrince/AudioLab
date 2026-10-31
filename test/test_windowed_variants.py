import torch
import torch.nn.functional as F

from src.anchored_ar import normalized_loss
from src.models.transcriber import TranscriberNet
from src.windowed_ar import WindowedAR, encode_window_target
from src.windowed_variants import AcousticTail, ConditionalNoteHead, VariantAR
from test.test_windowed_ar import CONFIG, condition, route
from src.data.chart_tokenizer import TokenNote


def test_conditional_head_is_normalized_and_each_note_has_both_attributes():
    torch.manual_seed(1)
    for lanes in (4,64):
        head=ConditionalNoteHead(32,113+4*lanes,lanes)
        h=torch.randn(2,5,32);out=head(h)
        assert out.shape==(2,5,113+4*lanes)
        assert torch.allclose(out.logsumexp(-1),torch.zeros(2,5),atol=1e-6)
        action=head.action(h).log_softmax(-1);kind=head.kind(h).log_softmax(-1)
        position=head.position(h[...,None,:]+head.type_condition.weight).log_softmax(-1)
        for lane in (0,lanes-1):
            for typ in range(4):
                assert torch.allclose(out[...,50+4*lane+typ],action[...,-1]+kind[...,typ]+position[...,typ,lane])
        F.nll_loss(out.reshape(-1,out.shape[-1]),torch.tensor([50+i%(4*lanes) for i in range(10)])).backward()
        assert head.kind.weight.grad.abs().sum()>0
        assert head.type_condition.weight.grad.abs().sum()>0
        assert head.position[0].weight.grad.abs().sum()>0


def test_shared_initial_tensors_unchanged_for_representation_arm():
    torch.manual_seed(42);a=WindowedAR(CONFIG)
    torch.manual_seed(42);b=VariantAR(CONFIG,variant='kind_position')
    for k,v in a.state_dict().items():
        if not k.startswith('decoder.lm_head.'):assert torch.equal(v,b.state_dict()[k])
    assert b.decoder.cross_window is None
    assert b.decoder.cross_time_scale is None


def test_factorized_output_identical_in_teacher_and_cached_rollout():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=VariantAR(CONFIG,variant='kind_position').eval();c=condition()
    target=encode_window_target(route([TokenNote(64,0,1),TokenNote(700,2,3,32)]),25.)
    d=model.decoder;memory=d.prepare_memory(c['features'],c['soft_onsets']);bias=model.window_condition(target['bounds'])
    with torch.no_grad():
        logits=d(target['tokens'][None],target['frames'][None],memory,c['valid'],
            history=target['history'][None],input_condition=bias[None])
        caches=d.init_cache(memory);pred=[]
        for i in range(len(target['tokens'])):
            out,_=d.step(target['tokens'][i:i+1],target['frames'][i:i+1],i,caches,memory.shape[1],
                c['valid'],history=target['history'][i:i+1],input_condition=bias[i:i+1])
            pred.append(out[0])
    assert torch.allclose(torch.stack(pred),logits[0],atol=4e-6)


def test_real_audio_tail_reconstruction_and_training_gradient():
    torch.set_num_threads(2);torch.manual_seed(7)
    encoder=TranscriberNet(n_mels=8,hidden_dim=32,conv_blocks=1,depth=2,num_heads=4,dropout=0.).eval().requires_grad_(False)
    captured=[]
    hook=encoder.blocks[-1].register_forward_pre_hook(lambda _,args:captured.append(args[0].detach()))
    mel=torch.randn(1,8,432)
    with torch.no_grad():full=encoder.features(mel)
    hook.remove();tail=AcousticTail.from_encoder(encoder).eval()
    assert torch.equal(tail(captured[0]),full)
    model=VariantAR(CONFIG,variant='encoder_tail',tail_config=tail.config)
    model.audio_tail.load_state_dict(tail.state_dict())
    c=dict(features=full.half(),prefix_features=captured[0],soft_onsets=torch.zeros(1,216,2),
        valid=torch.ones(1,216,dtype=torch.bool),duration=10.)
    assert torch.equal(model.eval().audio_condition(c)['features'],full.half().float())
    model.train();target=encode_window_target(route([TokenNote(64,1,1),TokenNote(128,2,4)]),10.)
    normalized_loss(model.components(target,c,use_soft=False),target).backward()
    assert model.audio_tail.block.qkv.weight.grad.abs().sum()>0
    assert model.audio_tail.norm.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in encoder.parameters())


def test_variant_checkpoint_roundtrip_including_adapted_tail():
    config=dict(hidden_dim=32,num_heads=4,mlp_ratio=4.,dropout=0.,rope_theta=10000.)
    for variant in ('kind_position','encoder_tail','capacity'):
        model=VariantAR(CONFIG,variant=variant,tail_config=config)
        copy=VariantAR(CONFIG,variant=variant,tail_config=config)
        copy.load_state_dict(model.state_dict(),strict=True)
        assert all(torch.equal(v,copy.state_dict()[k]) for k,v in model.state_dict().items())
        assert copy.decoder.cross_window is None

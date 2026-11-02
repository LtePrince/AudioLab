from copy import deepcopy

import pytest
import torch

from src.anchored_ar import normalized_loss
from src.data.chart_tokenizer import TokenNote
from src.joint_audio_ar import AcousticBackbone, JointAudioWindowAR, gradient_audit
from src.models.transcriber import TranscriberNet
from src.windowed_ar import WindowedAR, encode_window_target
from test.test_windowed_ar import CONFIG, route


ENC = dict(n_mels=128,hidden_dim=32,conv_blocks=2,depth=2,num_heads=4,dropout=0.)


def parents():
    torch.set_num_threads(2);torch.manual_seed(7)
    original=TranscriberNet(**ENC)
    chart=WindowedAR(CONFIG)
    parent=dict(config=dict(architecture='window',variant='audio',decoder_config=CONFIG),model=chart.state_dict())
    frontend=dict(enc_config=ENC,model={f'encoder.{k}':v for k,v in original.state_dict().items()})
    joint,_=JointAudioWindowAR.from_parents(parent,frontend)
    return original,chart,joint


def example():
    # One-second waveform: 44 Mel frames -> 22 audio tokens. Include Hold and
    # multiple positions so all A output heads receive the real training loss.
    mel=torch.randn(128,44)
    target=encode_window_target(route([TokenNote(8,0,1),TokenNote(20,2,3,12),TokenNote(40,3,4)]),1.)
    return mel,target


def test_migration_preserves_pretrained_features_and_pure_a_outputs():
    original,chart,joint=parents();original.eval();chart.eval();joint.eval()
    mel,target=example()
    with torch.no_grad():
        expected=original.features(mel[None])
        torch.testing.assert_close(joint.acoustic.features(mel[None]),expected,rtol=0,atol=0)
        condition=joint.condition(mel,1.)
        old=chart.components(target,condition,use_soft=False)
        new=joint.components(target,mel,1.)
        for key in old:
            torch.testing.assert_close(torch.as_tensor(new[key]),torch.as_tensor(old[key]),rtol=0,atol=0)
    assert joint.chart.decoder.soft_onset_proj is None
    assert not any('head.' in name for name,_ in joint.acoustic.named_parameters())
    with pytest.raises(ValueError,match='no soft-onset'):
        joint.chart.components(target,condition,use_soft=True)


def test_real_chart_loss_updates_every_audio_block_and_all_parameters_connected():
    _,_,model=parents();model.train();mel,target=example()
    mel.requires_grad_(True)
    groups=model.optimizer_groups(decoder_lr=3e-5,encoder_lr=3e-6)
    assert {id(p) for group in groups for p in group['params']}=={id(p) for p in model.parameters()}
    assert sum(len(g['params']) for g in groups)==len(list(model.parameters()))
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(groups)
    normalized_loss(model.components(target,mel,1.),target).backward()
    audit=gradient_audit(model)
    assert audit['all_connected'] and mel.grad.abs().sum()>0
    optimizer.step()
    for prefix in audit['group_gradient_norms']:
        assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items() if k.startswith(prefix))
    # Saving only the decoder would lose the learned frontend. Round-trip both.
    clone=deepcopy(model);clone.load_state_dict(model.state_dict(),strict=True)
    model.eval();clone.eval()
    torch.testing.assert_close(model.condition(mel.detach(),1.)['features'],clone.condition(mel.detach(),1.)['features'])


def test_recomputation_preserves_full_encoder_outputs_and_gradients():
    _,_,model=parents();a=model.acoustic.train();b=deepcopy(a)
    b.gradient_checkpointing=False
    mel=torch.randn(1,128,44)
    xa=a.features(mel);xb=b.features(mel)
    torch.testing.assert_close(xa,xb,rtol=0,atol=0)
    weights=torch.randn_like(xa)
    (xa*weights).sum().backward();(xb*weights).sum().backward()
    for pa,pb in zip(a.parameters(),b.parameters(),strict=True):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)


def test_physical_clock_and_missing_gradient_are_rejected():
    _,_,model=parents();mel,target=example()
    with pytest.raises(ValueError,match='different clocks'):model.condition(mel,2.)
    with pytest.raises(RuntimeError,match='disconnected'):gradient_audit(model)

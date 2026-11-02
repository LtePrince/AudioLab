"""Scratch initialization must not silently inherit an acoustic checkpoint."""
import json
from types import SimpleNamespace

import torch

from script.train_joint_audio_a import load_initial, set_learning_rates
from src.soft_onset import state_hash
from test.test_joint_audio_ar import ENC
from test.test_windowed_ar import CONFIG


def scratch_args(tmp_path,seed=42):
    architecture=tmp_path/'architecture.json'
    architecture.write_text(json.dumps(dict(encoder_config=ENC,decoder_config=dict(CONFIG,soft_onset_dim=0))))
    cache=tmp_path/'mel';cache.mkdir(exist_ok=True)
    (cache/'manifest.json').write_text('{}')
    return SimpleNamespace(initialization='scratch',architecture_config=architecture,seed=seed,run_id='test-scratch',
        epochs=300,lr=3e-4,encoder_lr=3e-4,warmup_steps=1500,baseline_eval='focus',eval_every=20,full_eval_every=100,
        output=tmp_path,mel_cache=cache,parent=tmp_path/'DO_NOT_READ.pt',frontend=tmp_path/'DO_NOT_READ_ENCODER.pt')


def test_scratch_does_not_load_weights_is_reproducible_and_has_no_parent(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    def forbidden(*args,**kwargs):raise AssertionError('scratch initialization attempted to load a checkpoint')
    monkeypatch.setattr(torch,'load',forbidden)
    args=scratch_args(tmp_path)
    a,config=load_initial(args,torch.device('cpu'))
    b,other=load_initial(args,torch.device('cpu'))
    assert config==other and state_hash(a.state_dict())==state_hash(b.state_dict())
    assert config['parent'] is None and config['frontend'] is None and config['inherited_parameter_tensors']==0
    assert config['initial_encoder_sha256']==state_hash(a.acoustic.state_dict())
    args.seed+=1
    c,_=load_initial(args,torch.device('cpu'))
    assert state_hash(c.acoustic.state_dict())!=state_hash(a.acoustic.state_dict())
    assert state_hash(c.chart.state_dict())!=state_hash(a.chart.state_dict())
    optimizer=torch.optim.AdamW(a.optimizer_groups(decoder_lr=args.lr,encoder_lr=args.encoder_lr))
    assert not optimizer.state
    assert all(p.requires_grad for p in a.parameters())


def test_warmup_uses_absolute_optimizer_step_so_resume_keeps_learning_rate(tmp_path):
    args=scratch_args(tmp_path);model,_=load_initial(args,torch.device('cpu'))
    optimizer=torch.optim.AdamW(model.optimizer_groups(decoder_lr=args.lr,encoder_lr=args.encoder_lr))
    for step,expected in [(0,2e-7),(749,1.5e-4),(1499,3e-4),(9000,3e-4)]:
        set_learning_rates(optimizer,args,step)
        assert all(abs(group['lr']-expected)<1e-15 for group in optimizer.param_groups)

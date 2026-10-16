import pytest
import torch

from src.models.transcriber import TranscriberNet
from src.models.rhythm_skeleton import predict_onsets
from src.train_rhythm import onset_targets, rhythm_loss, trainable_backbone


def test_global_targets_deduplicate_chords_and_average_subframe_collisions():
    target, collisions = onset_targets([-1, 5, 5, 15, 25, 200], 20, 4)
    assert target[0].tolist() == [1, 1, 0, 0]
    assert target[1].tolist() == [0.5, 0.25, 0, 0]
    assert collisions == 1


def test_rhythm_loss_masks_padding_and_empty_songs():
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    target = torch.zeros(1, 2, 4)
    valid = torch.tensor([[True, True, False, False]])
    a, _ = rhythm_loss(logits, target, valid)
    target[:, :, 2:] = 1
    b, _ = rhythm_loss(logits, target, valid)
    assert a.item() == pytest.approx(b.item())
    b.backward()
    assert torch.equal(logits.grad[:, :, 2:], torch.zeros(1, 2, 2))


def test_frozen_encoder_and_fine_onset_head_backward_compatibility():
    cfg = dict(n_mels=8, hidden_dim=32, conv_blocks=1, depth=1, num_heads=2)
    old = TranscriberNet(**cfg).eval()
    model = TranscriberNet(**cfg, rhythm_head=True).eval()
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('rhythm_head.') for k in missing)
    model.requires_grad_(False); model.rhythm_head.requires_grad_(True)
    mel = torch.randn(1, 8, 20)
    assert torch.allclose(model(mel), old(mel))
    logits = model.rhythm_logits(mel)
    assert logits.shape == (1, 2, 20)
    logits.sum().backward()
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith('rhythm_head.'))
    times, scores = predict_onsets(model, mel, 40)
    assert len(times) == len(scores) == 20
    assert all(i*20 <= t <= (i+1)*20 for i, t in enumerate(times))


def test_staged_finetuning_only_enables_backbone_and_rhythm_head():
    model = TranscriberNet(n_mels=8, hidden_dim=32, conv_blocks=1, depth=1,
                           num_heads=2, rhythm_head=True)
    assert not trainable_backbone(model, 0)
    assert all(p.requires_grad == n.startswith('rhythm_head.') for n, p in model.named_parameters())
    assert trainable_backbone(model, 1e-5)
    assert all(p.requires_grad == (not n.startswith('head.')) for n, p in model.named_parameters())

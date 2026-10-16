from types import SimpleNamespace

import pytest
import torch

import src.resource_budget as rb
from script.train_models import build_stages


def args(**overrides):
    values = dict(min_free_disk_gib=8, min_available_ram_gib=8,
                  max_rss_gib=10, gpu_memory_fraction=0.65, keep_checkpoints=1)
    return SimpleNamespace(**(values | overrides))


def snapshot(monkeypatch, disk=14, ram=30, rss=1):
    monkeypatch.setattr(rb, 'resource_snapshot', lambda _: dict(
        disk_free_gib=disk, ram_available_gib=ram, rss_gib=rss))


@pytest.mark.parametrize('resource', ['disk', 'ram', 'rss'])
def test_resource_limits_stop_before_work(monkeypatch, tmp_path, resource):
    snapshot(monkeypatch, **{resource: 11 if resource == 'rss' else 7})
    with pytest.raises(RuntimeError, match='Resource budget'):
        rb.ResourceBudget(args(), torch.device('cpu'), tmp_path)


def test_checkpoint_space_accounts_for_full_temporary_file(monkeypatch, tmp_path):
    snapshot(monkeypatch, disk=8.01)
    guard = rb.ResourceBudget(args(), torch.device('cpu'), tmp_path)
    target = tmp_path / 'best.pt'
    target.write_bytes(b'previous checkpoint')
    with pytest.raises(RuntimeError, match='disk reserve'):
        guard.save({'model': torch.ones(1)}, target)
    assert target.read_bytes() == b'previous checkpoint'
    assert not target.with_suffix('.pt.tmp').exists()


def test_checkpoint_failure_preserves_previous_file(monkeypatch, tmp_path):
    snapshot(monkeypatch)
    guard = rb.ResourceBudget(args(), torch.device('cpu'), tmp_path)
    target = tmp_path / 'best.pt'
    target.write_bytes(b'previous checkpoint')

    def fail(payload, fh):
        fh.write(b'partial write')
        raise OSError('simulated write failure')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        guard.save({'model': torch.ones(1)}, target)
    assert target.read_bytes() == b'previous checkpoint'
    assert not target.with_suffix('.pt.tmp').exists()


def test_checkpoint_atomic_success(monkeypatch, tmp_path):
    snapshot(monkeypatch)
    guard = rb.ResourceBudget(args(), torch.device('cpu'), tmp_path)
    target = tmp_path / 'best.pt'
    guard.save({'model': torch.ones(3)}, target)
    assert torch.equal(torch.load(target, weights_only=True)['model'], torch.ones(3))
    assert not target.with_suffix('.pt.tmp').exists()


def test_pipeline_uses_small_batches_and_isolated_outputs():
    config = args(run_dir='checkpoints/test_run', transcriber_epochs=20, decoder_epochs=30)
    stages = build_stages(config)
    assert [s[0] for s in stages[:3]] == ['transcriber', 'decoder', 'decoder_freex']
    for _, _, command in stages[:3]:
        assert command[command.index('--num-workers') + 1] == '0'
        assert command[command.index('--keep-checkpoints') + 1] == '1'
        assert command[command.index('--ckpt-dir') + 1] == config.run_dir
    for _, module, command in stages:
        if module == 'script.eval_onset_f1' and 'freex' in command[1]:
            assert command[command.index('--ref-dir') + 1] == 'data/gameplay_charts'


def test_decoder_nll_excludes_auxiliary_losses(monkeypatch):
    import math
    import src.train_decoder as training
    monkeypatch.setattr(training, '_step_loss', lambda *a, **kw:
                        (torch.tensor(50.0), torch.tensor(2), 4, torch.tensor(2.0)))
    model = torch.nn.Linear(1, 1).train()
    nll, accuracy, total = training._validate(model, [{}], torch.device('cpu'))
    assert nll == pytest.approx(2.0 / math.log(2))
    assert total == 50.0 and accuracy == 0.5
    assert model.training

from copy import deepcopy
import random
import time

import pytest
import torch

from docs.evaluations.assets.windowed_ar_20261031.continue_natural import equal_tree, extend_checkpoint
from docs.evaluations.assets.windowed_ar_20261031 import continue_natural as continuation
from script.train_alignment_scratch import restore_training
from script.train_windowed_corpus import SCHEMA


def step(model, optimizer, sampler):
    optimizer.zero_grad(set_to_none=True)
    x = torch.rand(4, 3) * sampler.random()
    loss = torch.nn.functional.dropout(model(x), .2, training=True).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach().clone()


def checkpoint():
    torch.manual_seed(25)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    sampler = random.Random(42)
    step(model, optimizer, sampler)
    return dict(schema=SCHEMA,
        config=dict(architecture='window', epochs=1, lr=3e-5, schedule='constant_lr; shared control'),
        model=deepcopy(model.state_dict()), epoch=1, steps=1, curves=[dict(epoch=1, steps=1)],
        training_state=dict(optimizer=deepcopy(optimizer.state_dict()), sampler=sampler.getstate(),
                            torch_rng=torch.get_rng_state(), cuda_rng=[]))


def test_extended_run_reproduces_exact_next_training_step_after_disk_roundtrip(tmp_path):
    original = checkpoint()
    path = tmp_path / 'extended.pt'
    torch.save(extend_checkpoint(original, 100), path)
    extended = torch.load(path, map_location='cpu', weights_only=True)
    assert original['config']['epochs'] == 1
    assert extended['config']['epochs'] == 100
    assert equal_tree(original, dict(extended, config=original['config']))
    results = []
    for state in (original, extended):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.2)
        sampler = random.Random(999)
        model.load_state_dict(state['model'])
        restore_training(deepcopy(state['training_state']), optimizer, sampler)
        loss = step(model, optimizer, sampler)
        results.append(dict(loss=loss, model=deepcopy(model.state_dict()),
                            optimizer=deepcopy(optimizer.state_dict()), sampler=sampler.getstate(),
                            torch_rng=torch.get_rng_state()))
    assert equal_tree(results[0], results[1])


@pytest.mark.parametrize('change', ['schedule', 'lr', 'rng', 'steps', 'unfinished'])
def test_extension_rejects_non_equivalent_or_incomplete_resume(change):
    state = checkpoint()
    if change == 'schedule':
        state['config']['schedule'] = 'cosine'
    elif change == 'lr':
        state['config']['lr'] = .1
    elif change == 'rng':
        del state['training_state']['sampler']
    elif change == 'steps':
        state['steps'] = 2
    else:
        state['config']['epochs'] = 2
    with pytest.raises(ValueError):
        extend_checkpoint(state, 100)


def test_running_file_does_not_override_missing_processes(monkeypatch):
    state = dict(status='running', pid=123, worker_pid=456, updated=time.time(), epoch=26)
    monkeypatch.setattr(continuation, 'process_alive', lambda *args: False)
    assert continuation.runtime_status(state)['effective_status'] == 'interrupted'
    # A worker may outlive its supervisor; it must not be mistaken for idle.
    monkeypatch.setattr(continuation, 'process_alive', lambda pid, marker: pid == 456)
    health = continuation.runtime_status(state)
    assert health['effective_status'] == 'orphaned_worker' and health['worker_alive']


def test_interrupted_checkpoint_keeps_full_training_state_and_rejects_changed_recipe():
    state = checkpoint()
    state.update(epoch=26, steps=7800,
                 curves=[dict(epoch=i, steps=i * 300) for i in range(1, 27)])
    state['config']['epochs'] = 100
    state['training_state']['cuda_rng'] = [torch.zeros(8, dtype=torch.uint8) for _ in range(2)]
    before = deepcopy(state)
    assert continuation.validate_interrupted_checkpoint(state, deepcopy(state['config'])) == 26
    assert equal_tree(state, before)
    with pytest.raises(ValueError, match='config changed'):
        continuation.validate_interrupted_checkpoint(state, dict(state['config'], lr=.01))
    del state['training_state']['sampler']
    with pytest.raises(ValueError, match='complete optimizer and RNG'):
        continuation.validate_interrupted_checkpoint(state, state['config'])

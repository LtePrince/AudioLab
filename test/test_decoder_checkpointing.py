"""Full-context activation recomputation preserves training outputs/gradients."""
import copy

import pytest
import torch

from src.models.chart_decoder import ChartDecoder
from src.models.chart_history import HISTORY_DIM


@pytest.mark.parametrize('positions', [4, 64])
def test_block_recomputation_preserves_dropout_and_parameter_gradients(positions):
    torch.manual_seed(71)
    original = ChartDecoder(vocab_size=4*positions+113, hidden_dim=32, depth=2,
        num_heads=4, mem_dim=16, time_dim=16, n_lanes=positions,
        soft_onset_dim=2, history_dim=HISTORY_DIM, relation_auxiliary=True,
        offset_head=positions==64, dropout=.2, token_dropout=.1, cross_window=3)
    recomputed = copy.deepcopy(original)
    recomputed.gradient_checkpointing = True
    tokens = torch.tensor([[0, 2, 50, 19, 54, 20, 58]])
    frames = torch.tensor([[0., 0., 1., 1., 3., 3., 5.]])
    acoustic = torch.randn(1, 11, 16)
    condition = torch.rand(1, 11, 2)
    valid = torch.arange(11)[None] < 10
    history = torch.randn(1, len(tokens[0]), HISTORY_DIM)
    outputs = []
    for model in (original, recomputed):
        model.train()
        torch.manual_seed(83)
        memory = model.prepare_memory(acoustic, condition)
        logits, hidden = model(tokens, frames, memory, valid,
                               history=history, return_hidden=True)
        loss = logits.square().mean()+model.relation_head(hidden).square().mean()
        if positions==64:
            loss = loss+model.predict_offset(hidden).square().mean()
        loss.backward()
        outputs.append(logits.detach())
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for (name, left), (other, right) in zip(original.named_parameters(), recomputed.named_parameters()):
        assert name == other
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)
    # The runtime option must neither change the serialized architecture nor
    # introduce recomputation into inference.
    original.eval(); recomputed.eval()
    with torch.no_grad():
        first = original(tokens, frames, original.prepare_memory(acoustic, condition), valid, history=history)
        second = recomputed(tokens, frames, recomputed.prepare_memory(acoustic, condition), valid, history=history)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

from types import SimpleNamespace

import pytest
import torch

from script.train_soft_onset import check_size, load_sample, store_features


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_cache_roundtrip_only_quantizes_audio_features(tmp_path, dtype):
    source = torch.tensor([[[.123456789, -.987654321]]])
    stored, contract = store_features(source, dtype)
    # Low scores, subframe offsets and clocks stay float32, even below fp16 range.
    soft = torch.tensor([[[1e-10, .123456789]]])
    frames = torch.tensor([.123456789, 13.987654321])
    offsets = torch.tensor([.123456789, .987654321])
    sample = dict(soft_onsets=soft, routes={'4k':dict(features=stored, frames=frames,
                                                   offsets=offsets, feature_storage=contract)})
    torch.save(sample, tmp_path/'sample.pt')
    restored = load_sample(tmp_path, {'file':'sample.pt'})
    route = restored['routes']['4k']
    assert stored.dtype == getattr(torch, dtype)
    assert route['features'].dtype == torch.float32
    torch.testing.assert_close(route['features'], source.to(getattr(torch, dtype)).float(), rtol=0, atol=0)
    for key, expected in [('frames', frames), ('offsets', offsets)]:
        torch.testing.assert_close(route[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(restored['soft_onsets'], soft, rtol=0, atol=0)
    assert contract['max_abs_roundtrip_error'] == float((source-route['features']).abs().max())


def test_cache_rejects_float16_overflow():
    with pytest.raises(ValueError, match='overflow'):
        store_features(torch.tensor([[[70000.]]]), 'float16')


def test_artifact_budget_is_explicit_and_legacy_default_is_700(tmp_path):
    path = tmp_path/'tiny';path.write_bytes(b'abcd')
    checks = []
    budget = SimpleNamespace(args=SimpleNamespace(max_artifact_mib=3/1024**2),
                             check=lambda **kw:checks.append(kw))
    with pytest.raises(RuntimeError, match='artifact budget'):
        check_size(tmp_path,budget)
    budget.args.max_artifact_mib=4/1024**2
    assert check_size(tmp_path,budget)['bytes']==4
    del budget.args.max_artifact_mib
    assert check_size(tmp_path,budget)['bytes']==4
    assert all(x=={'force':True} for x in checks)

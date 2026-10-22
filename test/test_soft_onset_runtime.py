"""Portable experimental weights preserve conditioning and free generation."""
from pathlib import Path

import numpy as np
import pytest
import torch

from script.export_soft_onset_bundle import portable_checkpoint
from src.models.chart_history import HISTORY_SCHEMA
from src.product import pipeline, runtime
from src.product.experimental_models import load_models, runtime_contract
from src.product.model_bundle import sha256
from src.soft_onset import SCHEMA, initialize_decoder, load_decoder
from src.data.timing import TIMING_SCHEMA
from test.test_product_pipeline import inputs, arguments, controlled_package_budget


@pytest.fixture
def packed_inputs(tmp_path, inputs):
    audio, base = inputs
    root = tmp_path/'packed'
    onset = base/'transcriber/transcriber_best.pt'
    (root/'transcriber').mkdir(parents=True)
    (root/'transcriber/transcriber_best.pt').write_bytes(onset.read_bytes())
    sources = {}
    for mode, folder in [('4k', 'decoder'), ('freex', 'decoder_freex')]:
        old = base/folder/'decoder_best.pt'
        decoder, config, _ = initialize_decoder(old, 42, with_history=True)
        # Make soft evidence affect logits, so conditioning cannot be dropped silently.
        with torch.no_grad():
            decoder.soft_onset_proj.weight.fill_(.1)
        ck = dict(soft_onset_schema=SCHEMA, timing_schema=TIMING_SCHEMA,
            condition='soft_history', history_schema=HISTORY_SCHEMA, relation_schema=None,
            relation_weight=None, model=decoder.state_dict(), dec_config=config,
            mode=mode, steps=3000, seed=42, training_scope='all', evaluation_scope='training_diagnostic',
            frozen_sources={mode:dict(sha256=sha256(old)), 'onset':dict(sha256=sha256(onset))},
            inference_contract=dict(audio_features='each route frozen legacy encoder, full attention',
                onset_encoder='1024 mel core + 128 context, same train and inference',
                cache_feature_dtype='float16'))
        source = tmp_path/f'{mode}.pt'
        torch.save(ck, source)
        packed = portable_checkpoint(source, old, onset, mode)
        (root/folder).mkdir()
        torch.save(packed, root/folder/'decoder_best.pt')
        sources[mode] = source
    return (audio, root), sources


@pytest.mark.parametrize('mode,folder', [('4k','decoder'), ('freex','decoder_freex')])
def test_portable_weights_match_trained_decoder_and_reject_mixed_frontend(packed_inputs, mode, folder):
    (_, root), sources = packed_inputs
    path = root/folder/'decoder_best.pt'
    _, loaded = load_models(path, torch.device('cpu'))
    trained, _ = load_decoder(sources[mode], 'cpu')
    assert loaded.history_dim == 28
    assert set(loaded.state_dict()) == set(trained.state_dict())
    for key, value in trained.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    assert runtime_contract(path, audio_sha256=sha256(root/'transcriber/transcriber_best.pt'), mode=mode)
    with pytest.raises(ValueError, match='does not match training'):
        runtime_contract(path, audio_sha256='0'*64, mode=mode)
    with pytest.raises(ValueError, match='export the experiment'):
        load_models(sources[mode], torch.device('cpu'))


def test_both_routes_use_all_weak_onsets_without_forced_skeleton(tmp_path, packed_inputs, monkeypatch):
    pair, _ = packed_inputs
    infer = pipeline.infer_mel
    def weak(*args, **kwargs):
        arrays = infer(*args, **kwargs)
        arrays['event_probability'][:] = .001
        return arrays
    monkeypatch.setattr(pipeline, 'infer_mel', weak)
    result = pipeline.make_result
    def no_selected_events(*args, **kwargs):
        evidence = result(*args, **kwargs)
        evidence.update(events=[], status='no_candidates')
        return evidence
    monkeypatch.setattr(pipeline, 'make_result', no_selected_events)
    def forbidden(*args, **kwargs):
        pytest.fail('thresholded skeleton entered the experimental route')
    monkeypatch.setattr(pipeline, 'quantize_candidates', forbidden)
    from src import soft_onset
    dense = soft_onset.dense_condition
    observed = []
    def observe(arrays, frames):
        observed.append(arrays['event_probability'].copy())
        return dense(arrays, frames)
    monkeypatch.setattr(soft_onset, 'dense_condition', observe)
    generate = pipeline.generate_tokens
    def free(*args, **kwargs):
        assert kwargs['skeleton_ticks'] is None
        return generate(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'generate_tokens', free)
    report = pipeline.run_job(arguments(tmp_path, pair))
    assert report['status'] == 'completed'
    assert len(observed) == 2 and all(np.allclose(p, .001) for p in observed)
    for mode, route in report['routes'].items():
        assert route['note_count'] > 0
        assert route['complete_candidate_coverage'] is None
        assert route['runtime_contract']['forced_skeleton'] is False
        assert route['generation_policy'] == 'dense-soft-onset-free-ar-v1'
        assert route['human_playability'] == 'unverified'
        assert Path(tmp_path/'job'/route['package_file']).is_file()


def test_runtime_preflight_loads_packaged_models(tmp_path, packed_inputs, monkeypatch):
    (_, root), _ = packed_inputs
    monkeypatch.setattr(runtime, 'check_resources', lambda *a, **k: None)
    result = runtime.probe_models(root, tmp_path)
    assert result['4k']['quality_status'] == result['freex']['quality_status'] == 'experimental'


def test_export_refuses_frontend_not_used_in_training(tmp_path, inputs, packed_inputs):
    (_, _), sources = packed_inputs
    bad = tmp_path/'wrong.pt'
    bad.write_bytes(b'not the trained onset model')
    with pytest.raises(ValueError, match='differs from the training checkpoint'):
        portable_checkpoint(sources['4k'], inputs[1]/'decoder/decoder_best.pt', bad, '4k')

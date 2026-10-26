"""Runtime packaging preserves the trained protocol, seconds and coordinates."""
import json
import math
from copy import deepcopy

import pytest
import torch

from script.export_anchored_bundle import portable_checkpoint
from src.anchored_ar import AnchoredAR, SCHEMA, FREEX_SCHEMA
from src.product import pipeline, runtime
from src.product.anchored_models import phigros_notes, export_prediction, PARTIAL_TERMINATIONS
from src.product.artifacts import chart_document, validate_chart
from src.product.experimental_models import load_models, runtime_contract
from src.product.model_bundle import sha256
from test.test_product_pipeline import inputs, arguments, controlled_package_budget


@pytest.fixture
def packed_inputs(tmp_path, inputs):
    audio, base = inputs
    manifest = tmp_path/'cache.json'
    folders = {'4k': 'decoder', 'freex': 'decoder_freex'}
    onset = base/'transcriber/transcriber_best.pt'
    manifest.write_text(json.dumps(dict(models={
        **{m: dict(sha256=sha256(base/f/'decoder_best.pt')) for m, f in folders.items()},
        'onset': dict(sha256=sha256(onset))})))
    root = tmp_path/'portable'
    (root/'transcriber').mkdir(parents=True)
    (root/'transcriber/transcriber_best.pt').write_bytes(onset.read_bytes())
    sources = {}
    for mode, folder in folders.items():
        old = base/folder/'decoder_best.pt'
        c = torch.load(old, weights_only=True)['dec_config']
        c.update(history_dim=28, soft_onset_dim=2, cross_window=None, cross_time_scale=None)
        model = AnchoredAR(c).eval()
        ck = dict(schema=SCHEMA if mode == '4k' else FREEX_SCHEMA,
                  config=dict(decoder_config=c, variant='audio' if mode == '4k' else 'soft',
                              full_cache_sha256=sha256(manifest)),
                  model=model.state_dict(), epoch=980, steps=294000)
        source = tmp_path/f'{mode}.pt'
        torch.save(ck, source)
        packed = portable_checkpoint(source, old, onset, mode, manifest)
        (root/folder).mkdir()
        torch.save(packed, root/folder/'decoder_best.pt')
        sources[mode] = source
    return (audio, root), sources, manifest


@pytest.mark.parametrize('mode,folder', [('4k', 'decoder'), ('freex', 'decoder_freex')])
def test_packaged_state_and_frontend_are_exact(packed_inputs, mode, folder):
    (_, root), sources, _ = packed_inputs
    path = root/folder/'decoder_best.pt'
    _, model = load_models(path, torch.device('cpu'))
    source = torch.load(sources[mode], weights_only=True)
    assert model.n_lanes == (4 if mode == '4k' else 64)
    assert set(source['model']) == set(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in source['model'].items())
    assert runtime_contract(path, audio_sha256=sha256(root/'transcriber/transcriber_best.pt'), mode=mode)
    with pytest.raises(ValueError, match='does not match training'):
        runtime_contract(path, audio_sha256='0'*64, mode=mode)
    with pytest.raises(ValueError, match='export the experiment'):
        load_models(sources[mode], torch.device('cpu'))


def test_phigros_conversion_preserves_fractional_seconds_and_continuous_x(tmp_path):
    notes = [dict(time_seconds=.12345678, end_seconds=.57345678, screen_x=.513579,
                  position=32, type='hold')]
    predicted = dict(status='completed', termination='learned_stop', notes=notes)
    raw = phigros_notes(predicted, 'freex', 193., .02, 1.)
    chart = chart_document(raw, 193., .02, tmp_path/'chart.json')
    events, _ = validate_chart(chart, 'freex', 1., None)
    n = events['events'][0]
    for key in ('time_seconds', 'end_seconds', 'screen_x'):
        assert n[key] == pytest.approx(notes[0][key], abs=1e-12)
    predicted.update(status='failed', termination='failed', error='token budget exhausted')
    with pytest.raises(ValueError, match='did not finish'):
        phigros_notes(predicted, 'freex', 193., .02, 1.)


def test_both_routes_use_anchored_protocol_and_preserve_output(tmp_path, packed_inputs, monkeypatch):
    pair, _, _ = packed_inputs
    from src import anchored_ar
    seen = []
    def predict(model, condition, bpm, **kw):
        seen.append((model.n_lanes, kw['use_soft']))
        assert kw['greedy'] and condition['features'].dtype == torch.float32
        assert torch.equal(condition['features'], condition['features'].half().float())
        assert condition['soft_onsets'].shape[-1] == 2
        x = .2890625 if model.n_lanes == 4 else .513579
        return dict(status='completed', termination='learned_stop', eos_clock_seconds=.5, seconds=.01,
            notes=[dict(time_seconds=.12345678, end_seconds=.57345678,
                        screen_x=x, position=0 if model.n_lanes == 4 else 32, type='hold')])
    def forbidden(*a, **kw):
        pytest.fail('legacy tokenizer, hard onset selection or repairs entered anchored route')
    monkeypatch.setattr(anchored_ar, 'generate', predict)
    for name in ('generate_tokens', 'quantize_candidates', 'constrain_4k_notes', 'constrain_free_notes'):
        monkeypatch.setattr(pipeline, name, forbidden)
    report = pipeline.run_job(arguments(tmp_path, pair))
    assert seen == [(4, False), (64, True)]
    for mode, route in report['routes'].items():
        assert route['changed_notes'] == 0 and route['repairs'] == {'applied': False}
        assert route['termination'] == 'learned_stop'
        assert 'e0980' in route['model_label']
    assert 'AL-AR4K' in (tmp_path/'job/preview.html').read_text()


def test_runtime_preflight_accepts_portable_models(tmp_path, packed_inputs, monkeypatch):
    (_, root), _, _ = packed_inputs
    monkeypatch.setattr(runtime, 'check_resources', lambda *a, **kw: None)
    result = runtime.probe_models(root, tmp_path)
    assert result['4k']['decoder_positions'] == 4
    assert result['freex']['decoder_positions'] == 64


def test_unfinished_generation_without_any_notes_still_fails(tmp_path, packed_inputs, monkeypatch):
    pair, _, _ = packed_inputs
    from src import anchored_ar
    monkeypatch.setattr(anchored_ar, 'generate', lambda *a, **kw: dict(
        status='failed', termination='failed', notes=[], error='continued with no future audio interval'))
    with pytest.raises(ValueError, match='did not finish'):
        pipeline.run_job(arguments(tmp_path, pair))
    assert not (tmp_path/'job/manifest.json').exists()
    assert not list((tmp_path/'job').glob('*.phira.zip'))
    assert json.loads((tmp_path/'job/job.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('mode', ['4k', 'freex'])
@pytest.mark.parametrize('reason', list(PARTIAL_TERMINATIONS))
def test_unfinished_route_keeps_preview_download_and_persistent_warning(
        tmp_path, packed_inputs, monkeypatch, mode, reason):
    from src import anchored_ar
    from src.product.server import LocalJobs
    pair, _, _ = packed_inputs
    def predict(model, condition, bpm, **kw):
        current = '4k' if model.n_lanes == 4 else 'freex'
        note = dict(time_seconds=.12345678, end_seconds=.57345678, screen_x=.2890625,
                    position=0 if current == '4k' else 17, type='hold')
        result = dict(status='completed', termination='learned_stop', eos_clock_seconds=.6,
                      seconds=.01, notes=[note])
        if current == mode:
            result.update(status='failed', termination='failed', error=reason, eos_clock_seconds=None)
            result['notes'] += [dict(note, time_seconds=.7, end_seconds=1.3),
                               dict(note, time_seconds=1.1, end_seconds=1.1, type='tap'),
                               dict(note, time_seconds=.9, end_seconds=.9)]
        return result
    monkeypatch.setattr(anchored_ar, 'generate', predict)
    jobs = LocalJobs(tmp_path/'server', pair[1], 'cpu')
    identifier = 'a'*32
    try:
        report = pipeline.run_job(arguments(tmp_path, pair, output=str(jobs.root/'jobs'/identifier)))
        jobs.records[identifier] = dict(id=identifier, status='completed', name='song.wav')
        jobs.save(jobs.records[identifier])
        assert len(report['warnings']) == 1 and report['warnings'][0]['mode'] == mode
        route = report['routes'][mode]
        assert not route['model_completed'] and route['truncated']
        assert route['generation_status'] == 'failed' and route['generator_error'] == reason
        assert route['termination'] == PARTIAL_TERMINATIONS[reason][0]
        assert route['cutoff_changes'] == dict(notes_after_audio=1, shortened_holds=1, incomplete_notes=1)
        events = json.loads((jobs.root/'jobs'/identifier/f'{mode}.events.json').read_text())['events']
        assert len(events) == 2 and events[-1]['end_seconds'] == pytest.approx(1.)
        assert all(n['time_seconds'] < 1. and n['end_seconds'] <= 1. for n in events)
        other = 'freex' if mode == '4k' else '4k'
        assert report['routes'][other]['model_completed'] and not report['routes'][other]['warnings']
    finally:
        jobs.close()
    jobs = LocalJobs(tmp_path/'server', pair[1], 'cpu')
    try:
        status = jobs.status(identifier)
        assert status['status'] == 'completed' and status['warnings'] == report['warnings']
        assert set(status['packages']) == {'4k', 'freex'}
        assert jobs.artifact(identifier, 'preview.html').is_file()
        text = jobs.artifact(identifier, 'preview.html').read_text()
        assert report['warnings'][0]['message'] in text
        assert jobs.artifact(identifier, f'{mode}.phira.zip').is_file()
    finally:
        jobs.close()


def test_partial_export_preserves_input_and_keeps_last_audio_instant_inside_bounds(tmp_path):
    end = 135.55083900226757
    n = dict(time_seconds=math.nextafter(end, 0.), end_seconds=math.nextafter(end, 0.),
             screen_x=.2890625, position=0, type='tap')
    prediction = dict(status='failed', termination='failed', error='continued with no future audio interval', notes=[n])
    before = deepcopy(prediction)
    raw, info = export_prediction(prediction, '4k', 193., 0., end)
    chart = chart_document(raw, 193., 0., tmp_path/'end.json')
    events, _ = validate_chart(chart, '4k', end, None)
    assert events['events'][0]['time_seconds'] < end
    assert prediction == before and info['truncated']


def test_unknown_failures_are_not_published_as_partial_charts():
    predicted = dict(status='failed', termination='failed', error='unexpected model corruption', notes=[])
    with pytest.raises(ValueError, match='did not finish'):
        export_prediction(predicted, '4k', 120., 0., 1.)


def test_export_refuses_unmatched_training_cache(tmp_path, packed_inputs, inputs):
    _, sources, manifest = packed_inputs
    manifest.write_text(manifest.read_text() + ' ')
    _, base = inputs
    with pytest.raises(ValueError, match='differs from the training checkpoint'):
        portable_checkpoint(sources['4k'], base/'decoder/decoder_best.pt',
                            base/'transcriber/transcriber_best.pt', '4k', manifest)

import itertools
import json
import math
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audio_understanding import ANNOTATION_SCHEMA, FRAME_SECONDS, SCHEMA
from src.audio_understanding.annotations import event_targets, read_manifest, reviewed_regions
from src.audio_understanding.evaluation import match_events, evaluate_record, aggregate
from src.audio_understanding.inference import infer_mel, load_model, make_result, result_to_ticks, sha256
from src.data.timing import TIMING_SCHEMA
from src.models.transcriber import TranscriberNet
from src.train_audio_understanding import acoustic_loss, audit_splits


def record(**overrides):
    return dict(schema=ANNOTATION_SCHEMA, id='song', source_group='composition-1', audio='song.wav',
                label_source='reviewed_acoustic', provenance='synthetic test fixture',
                duration_seconds=1., coverage=[[0., 1.]], uncertain=[], events_seconds=[.1, .5], **overrides)


def small_model(rhythm_head=True):
    config = dict(n_mels=128, hidden_dim=32, conv_blocks=1, depth=1, num_heads=2,
                  dropout=0., rhythm_head=rhythm_head)
    return TranscriberNet(**config).eval(), config


def test_coverage_uncertainty_collisions_and_loss_gradient():
    r = record()
    r.update(coverage=[[0, .4], [.6, 1]], uncertain=[[.1, .2]], events_seconds=[.01, .02, .66, .95])
    target, valid, offsets, collisions = event_targets(r, 12, .1)
    assert reviewed_regions(r) == [[0, .1], [.2, .4], [.6, 1]]
    assert valid.tolist() == [True, False, True, True, False, False, True, True, True, True, False, False]
    assert collisions == 1 and not offsets[0] and target[1, 0] == pytest.approx(.1)
    logits = torch.zeros(1, 2, 12, requires_grad=True)
    loss, _ = acoustic_loss(logits, target[None], valid[None], offsets[None])
    loss.backward()
    assert torch.equal(logits.grad[:, :, ~valid], torch.zeros(1, 2, (~valid).sum()))
    assert logits.grad[0, 1, 0] == 0  # ambiguous timing must not train an invented average
    assert logits.grad[0, 1, 6] != 0


def test_manifest_rejects_unknown_labels_outside_events_and_empty_coverage(tmp_path):
    path = tmp_path/'labels.jsonl'
    for change in ({'label_source': 'midi'}, {'events_seconds': [1.]}, {'coverage': []},
                   {'uncertain': [[0, 1]]}, {'events_seconds': [float('nan')]},
                   {'source_group': ''}):
        r = record(); r.update(change)
        path.write_text(json.dumps(r)+'\n')
        with pytest.raises(ValueError):
            read_manifest(path)


@pytest.mark.parametrize('duration', [.1, .3, 1.1, 2.3])
def test_reviewed_partial_tail_has_float64_boundaries(duration):
    r = record()
    r.update(duration_seconds=duration, coverage=[[0, duration]], events_seconds=[duration-1e-5])
    frames = math.ceil(duration/FRAME_SECONDS)
    target, valid, _, _ = event_targets(r, frames)
    assert valid[-1] and target[0, -1] == 1
    r['coverage'] = [[0, duration-1e-4]]
    _, valid, _, _ = event_targets(r, frames)
    assert not valid[-1]


def brute_assignment(ref, pred, tolerance):
    best = (0, 0.)
    for count in range(1, min(len(ref), len(pred))+1):
        for a in itertools.combinations(ref, count):
            for b in itertools.combinations(pred, count):
                if all(abs(x-y) <= tolerance+1e-12 for x, y in zip(a, b)):
                    best = max(best, (count, -sum(abs(x-y) for x, y in zip(a, b))))
    return best


def test_matcher_matches_exhaustive_optimum_and_inclusive_boundary():
    rng = np.random.default_rng(9)
    for _ in range(100):
        ref = sorted(rng.choice(np.arange(10)/100, 4, replace=False).tolist())
        pred = sorted(rng.choice(np.arange(10)/100, 5, replace=False).tolist())
        pairs = match_events(ref, pred, .025)
        expected = brute_assignment(ref, pred, .025)
        assert len(pairs) == expected[0]
        assert sum(abs(a-b) for a, b in pairs) == pytest.approx(-expected[1])
    assert match_events([0], [.025], .025) == [(0, .025)]
    assert match_events([0, .04], [.03, .06], .035) == [(0, .03), (.04, .06)]
    assert match_events([.01], [0, .009], .025) == [(.01, .009)]


def test_evaluation_ignores_unreviewed_audio_and_reports_unmatched_time():
    r = record()
    r.update(coverage=[[0, .3]], events_seconds=[.1, .2])
    scores = evaluate_record(r, [.16, .8])
    assert scores['ignored_predictions'] == 1
    assert scores['at_25ms']['matched'] == 0
    assert scores['at_50ms']['matched'] == 1
    assert scores['timing_100ms']['median_absolute_ms'] == pytest.approx(40)
    assert scores['timing_100ms']['unmatched_reference_fraction'] == .5
    assert aggregate([scores])['macro_by_song']['recall_50ms']['ci95'] is None
    other = {**scores, 'label_source': 'chart_onset'}
    with pytest.raises(ValueError, match='separate'):
        aggregate([scores, other])


def test_song_metrics_are_invariant_to_slicing_and_overlap_is_rejected():
    r = record()
    r.update(duration_seconds=10., coverage=[[0, 10]], events_seconds=np.arange(.5, 10).tolist())
    full = aggregate([evaluate_record(r, [.5])])
    a = {**r, 'coverage': [[0, 1]]}
    b = {**r, 'coverage': [[1, 10]]}
    sliced = aggregate([evaluate_record(a, [.5]), evaluate_record(b, [.5])])
    assert sliced['macro_by_song'] == full['macro_by_song']
    assert sliced['per_song'][0]['at_50ms']['f1'] == pytest.approx(2/11)
    with pytest.raises(ValueError, match='overlap'):
        aggregate([evaluate_record(r, [.5]), evaluate_record(a, [.5])])
    # A reference/prediction pair on opposite sides of a slice boundary still matches.
    r['events_seconds'] = [.99]
    a = {**r, 'coverage': [[0, 1]]}; b = {**r, 'coverage': [[1, 10]]}
    assert aggregate([evaluate_record(a, [1.01]), evaluate_record(b, [1.01])])['macro_by_song'] \
        == aggregate([evaluate_record(r, [1.01])])['macro_by_song']


@pytest.mark.parametrize('frames', [1, 2, 15, 16, 17, 41])
@pytest.mark.parametrize('global_head', [False, True])
def test_chunked_inference_covers_short_odd_and_tail_frames(frames, global_head):
    model, _ = small_model(global_head)
    duration = (frames-.1)*FRAME_SECONDS
    mel = torch.randn(128, frames)
    output = infer_mel(model, mel, duration, chunk_frames=16, context_frames=4)
    assert len(output['features']) == math.ceil(frames/2)
    assert len(output['event_probability']) == math.ceil(frames/(1 if global_head else 2))
    assert np.all(np.diff(output['frame_times_seconds']) > 0)
    assert np.all(np.isfinite(output['features']))
    result = make_result(output, duration, {'label_source': 'chart_onset'}, threshold=0, min_interval_ms=0)
    assert all(0 <= e['time_seconds'] < duration for e in result['events'])
    if frames <= 16:
        padded = torch.nn.functional.pad(mel, (0, frames % 2))
        with torch.no_grad():
            expected = model.features(padded[None])[0].numpy()
        np.testing.assert_allclose(output['features'], expected, atol=1e-6)


def test_chunk_ownership_keeps_every_frame_exactly_once():
    model, _ = small_model(True)
    # Context-independent feature encoder exposes indexing errors at every seam.
    model.features = lambda mel: mel[:, :32, ::2].transpose(1, 2)
    mel = torch.arange(43).float()[None].expand(128, -1)
    output = infer_mel(model, mel, 42.9*FRAME_SECONDS, chunk_frames=16, context_frames=4)
    np.testing.assert_array_equal(output['features'][:, 0], np.arange(0, 43, 2))


def test_tick_adapter_preserves_audio_offset_and_rejects_invalid_events():
    r = dict(schema=SCHEMA, time_basis='audio_seconds', duration_seconds=1.,
             events=[dict(time_seconds=t, confidence=.9) for t in (.01, .25, .5, .99)])
    assert result_to_ticks(r, 120, .25) == [0, 16, 47]
    r['events'].append(dict(time_seconds=1., confidence=.9))
    with pytest.raises(ValueError, match='invalid event'):
        result_to_ticks(r, 120)


def test_split_audit_rejects_group_and_byte_leakage(tmp_path):
    a, b = tmp_path/'a.wav', tmp_path/'b.wav'
    a.write_bytes(b'audio'); b.write_bytes(b'audio')
    train = record(); train['audio'] = str(a)
    val = {**train, 'id': 'val', 'source_group': 'another', 'audio': str(b)}
    with pytest.raises(ValueError, match='duplicate audio'):
        audit_splits([train], [val])
    b.write_bytes(b'different')
    val['source_group'] = train['source_group']
    with pytest.raises(ValueError, match='source_group'):
        audit_splits([train], [val])
    val['source_group'] = 'another'; train['label_source'] = 'chart_onset'
    with pytest.raises(ValueError, match='acoustic label'):
        audit_splits([train], [val])


def test_song_sampler_covers_all_windows_without_redecoding():
    from src.train_audio_understanding import SongWindowSampler
    class Dataset:
        entries = [(song, start) for song in range(3) for start in range(5)]
        def __len__(self):
            return len(self.entries)
    dataset = Dataset()
    order = list(SongWindowSampler(dataset))
    assert sorted(order) == list(range(15))
    songs = [dataset.entries[i][0] for i in order]
    assert sum(a != b for a, b in zip(songs, songs[1:])) == 2


def test_audio_memory_guard_runs_before_decode(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.audio_understanding import inference
    audio = tmp_path/'audio.wav'; sf.write(audio, np.zeros(1000), 22050)
    calls = []
    budget = SimpleNamespace(directory=tmp_path, check=lambda **kw: calls.append(kw),
                             args=SimpleNamespace(min_available_ram_gib=1, max_rss_gib=2))
    monkeypatch.setattr(inference, 'resource_snapshot', lambda _: {'ram_available_gib': 1., 'rss_gib': 1.})
    monkeypatch.setattr(inference, 'AudioGPUprocessor', lambda **_: pytest.fail('must reject before decoding'))
    with pytest.raises(RuntimeError, match='decode estimate'):
        inference.load_audio(audio, budget=budget)
    assert calls == [{'force': True}]


def test_train_infer_evaluate_smoke_with_frozen_encoder(tmp_path, monkeypatch):
    from src import train_audio_understanding
    from script.understand_audio import analyze
    from script.evaluate_audio_understanding import evaluate

    torch.manual_seed(42)
    model, config = small_model(False)
    initial = tmp_path/'encoder.pt'
    torch.save(dict(model=model.state_dict(), config=config, timing_schema=TIMING_SCHEMA), initial)
    manifests = []
    for name, frequency in [('train', 200), ('val', 500)]:
        audio = tmp_path/f'{name}.wav'
        wave = .1*np.sin(2*np.pi*frequency*np.arange(4410)/22050)
        sf.write(audio, wave, 22050)
        r = record()
        r.update(id=name, source_group=name, audio=audio.name, duration_seconds=.2,
                 coverage=[[0, .2]], events_seconds=[.05, .1])
        path = tmp_path/f'{name}.jsonl'; path.write_text(json.dumps(r)+'\n')
        manifests.append(path)
    run = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--train-manifest', str(manifests[0]),
        '--val-manifest', str(manifests[1]), '--encoder-ckpt', str(initial), '--run-dir', str(run),
        '--epochs', '1', '--batch-size', '1', '--chunk-frames', '16', '--device', 'cpu',
        '--min-free-disk-gib', '0', '--min-available-ram-gib', '0'])
    train_audio_understanding.main()
    trained, metadata = load_model(run/'audio_best.pt')
    assert metadata['label_source'] == 'reviewed_acoustic'
    for key, value in model.state_dict().items():
        assert torch.equal(value, trained.state_dict()[key]), key
    prediction = tmp_path/'result.json'
    result = analyze(tmp_path/'val.wav', run/'audio_best.pt', prediction,
                     chunk_frames=16, context_frames=4)
    assert result['dense_sha256'] == sha256(prediction.with_suffix('.npz'))
    assert result['capabilities']['beat'] == 'unavailable'
    mapping = tmp_path/'predictions.json'; mapping.write_text(json.dumps({'val': prediction.name}))
    report = evaluate(manifests[1], mapping, tmp_path/'report.json')
    assert report['label_source'] == 'reviewed_acoustic'
    assert report['acceptance'] == 'not_assessed'
    with pytest.raises(ValueError, match='fresh'):
        analyze(tmp_path/'val.wav', run/'audio_best.pt', prediction)


@pytest.mark.parametrize('lanes', [4, 64])
def test_audio_evidence_enters_both_decoder_modes(tmp_path, lanes):
    from src.data.chart_tokenizer import ChartTokenizer, EOS
    from src.models.chart_decoder import ChartDecoder
    from src.test_decoder import generate_chart

    audio = tmp_path/'song.wav'; sf.write(audio, np.zeros(11025), 22050)
    evidence = dict(schema=SCHEMA, time_basis='audio_seconds', duration_seconds=.5,
                    audio_sha256=sha256(audio), events=[dict(time_seconds=.25, confidence=.9)])
    path = tmp_path/'events.json'; path.write_text(json.dumps(evidence))
    enc, _ = small_model(False)
    tk = ChartTokenizer(n_lanes=lanes)
    dec = ChartDecoder(vocab_size=tk.vocab_size, n_lanes=lanes, hidden_dim=32,
                       mem_dim=32, time_dim=32, depth=1, num_heads=2, offset_head=lanes == 64).eval()
    with torch.no_grad():
        dec.lm_head.weight.zero_(); dec.lm_head.bias.zero_()
        dec.lm_head.bias[EOS] = 100
        dec.lm_head.bias[tk.note(0, 1)] = 200
    output = tmp_path/'chart.json'
    count = generate_chart(str(audio), str(output), enc=enc, dec=dec, device=torch.device('cpu'),
                           bpm=120, offset=.125, audio_events=str(path), quiet=True)
    notes = json.loads(output.read_text())['judgeLineList'][0]['notesAbove']
    assert count > 0 and {n['time'] for n in notes} == {8}
    evidence['audio_sha256'] = 'wrong'; path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match='different audio'):
        generate_chart(str(audio), str(output), enc=enc, dec=dec, device=torch.device('cpu'),
                       bpm=120, audio_events=str(path), quiet=True)

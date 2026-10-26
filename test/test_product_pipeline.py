"""Engineering invariants, not evidence of learned or human-playable quality."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from script.generate_product import parser
from src.models.transcriber import TranscriberNet
from src.models.chart_decoder import ChartDecoder, generate_tokens
from src.data.chart_tokenizer import ChartTokenizer, EOS
from src.data.chart_constraints import constrain_4k_notes
from src.data.timing import TIMING_SCHEMA
from src.product.artifacts import chart_document, repair_log, validate_chart, write_preview
from src.product.pipeline import JobCancelled, run_job, quantize_candidates


@pytest.fixture(autouse=True)
def controlled_package_budget(monkeypatch):
    from src.product import phira
    monkeypatch.setattr(phira,'check_resources',lambda *a,**k:None)
    from src import resource_budget
    from src.product import pipeline
    from src.audio_understanding import inference
    state=lambda path:dict(disk_free_gib=20,ram_available_gib=20,rss_gib=.1)
    for module in (resource_budget,pipeline,inference):
        monkeypatch.setattr(module,'resource_snapshot',state)


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(2)
    config = dict(n_mels=128, hidden_dim=32, conv_blocks=1, depth=1, num_heads=2)
    encoder = TranscriberNet(**config).eval()
    with torch.no_grad():
        encoder.head.weight.zero_(); encoder.head.bias.zero_()
    root = tmp_path/'models'
    (root/'transcriber').mkdir(parents=True)
    torch.save(dict(config=config, model=encoder.state_dict(), timing_schema=TIMING_SCHEMA),
               root/'transcriber/transcriber_best.pt')
    for lanes, folder in [(4, 'decoder'), (64, 'decoder_freex')]:
        tk = ChartTokenizer(n_lanes=lanes)
        dec_config = dict(vocab_size=tk.vocab_size, n_lanes=lanes, hidden_dim=32,
                          mem_dim=32, time_dim=32, depth=1, num_heads=2, offset_head=lanes==64)
        decoder = ChartDecoder(**dec_config).eval()
        with torch.no_grad():
            decoder.lm_head.weight.zero_(); decoder.lm_head.bias.zero_()
            decoder.lm_head.bias[EOS] = 100
            decoder.lm_head.bias[tk.note(0, 3)] = 200
            decoder.lm_head.bias[tk.dtb(1)] = 250
            decoder.lm_head.bias[tk.durb(2)] = 300
        state = {'encoder.'+key: value for key, value in encoder.state_dict().items()}
        state.update({'decoder.'+key: value for key, value in decoder.state_dict().items()})
        (root/folder).mkdir()
        torch.save(dict(enc_config=config, dec_config=dec_config, model=state, timing_schema=TIMING_SCHEMA),
                   root/folder/'decoder_best.pt')
    audio = tmp_path/'song.wav'
    sf.write(audio, np.zeros(22050), 22050)
    return audio, root


def arguments(tmp_path, inputs, **overrides):
    audio, models = inputs
    args = parser().parse_args(['--audio', str(audio), '--model-root', str(models), '--bpm', '120',
                               '--output', str(tmp_path/'job'), '--min-free-disk-gib', '0',
                               '--min-available-ram-gib', '1', '--max-rss-gib', '64'])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_real_small_models_complete_both_routes_and_record_identity(tmp_path, inputs):
    args = arguments(tmp_path, inputs)
    report = run_job(args)
    root = Path(args.output)
    assert report['status'] == 'completed'
    assert set(report['routes']) == {'4k', 'freex'}
    assert report['audio']['label_source'] == 'chart_onset'
    assert report['config']['timing_source'] == 'provided'
    assert report['download_names']['4k.phira.zip'] == f"song_4k_{report['export_date']}.phira.zip"
    assert report['download_names']['freex.json'] == f"song_freex_{report['export_date']}.json"
    assert report['download_names']['4k.phira.zip'] in (root/'preview.html').read_text()
    for mode, checks in report['routes'].items():
        assert checks['note_count'] > 0 and checks['complete_candidate_coverage']
        assert checks['human_playability'] == 'unverified' and checks['chord_size_cap'] is None
        events = json.loads((root/f'{mode}.events.json').read_text())['events']
        assert len({event['id'] for event in events}) == checks['note_count']
        assert all(0 <= event['time_seconds'] <= event['end_seconds'] <= 1 for event in events)
        assert checks['repairs']['shortened_holds'] > 0
    assert json.loads((root/'job.json').read_text())['status'] == 'completed'
    assert (root/'audio.wav').read_bytes() == inputs[0].read_bytes()
    assert 'audio_events.npz' in report['artifacts']
    from src.product.phira import verify_package
    assert set(report['packages'])=={'4k','freex'}
    for mode,package in report['packages'].items():
        assert verify_package(root/package['file'])['status']=='passed'
        assert report['routes'][mode]['package_file']==package['file']
        assert report['artifacts'][package['file']]['sha256']==package['sha256']
    with pytest.raises(FileExistsError):
        run_job(args)
    assert json.loads((root/'manifest.json').read_text())['status'] == 'completed'


@pytest.mark.parametrize('failure', ['empty', 'token_limit', 'cancel', 'timeout', 'resource', 'second_route','package'])
def test_failures_never_mark_partial_job_complete(tmp_path, inputs, monkeypatch, failure):
    from src.product import pipeline
    args = arguments(tmp_path, inputs)
    if failure == 'empty':
        original = pipeline.make_result
        def no_events(*a, **k):
            result = original(*a, **k); result['events'] = []; result['status']='no_candidates'; return result
        monkeypatch.setattr(pipeline, 'make_result', no_events)
    elif failure == 'token_limit':
        args.max_decoder_steps = 1
    elif failure == 'cancel':
        cancel = tmp_path/'cancel'; cancel.touch(); args.cancel_file = str(cancel)
    elif failure == 'timeout':
        args.timeout_seconds = 1e-12
    elif failure == 'resource':
        def fail_budget(*a, **k):
            raise RuntimeError('Resource budget: insufficient disk reserve')
        monkeypatch.setattr(pipeline, 'ResourceBudget', fail_budget)
    elif failure == 'second_route':
        original = pipeline.decode_notes
        def fail_second(*a, **k):
            if a[4] == 'freex':
                raise RuntimeError('second route failed')
            return original(*a, **k)
        monkeypatch.setattr(pipeline, 'decode_notes', fail_second)
    elif failure=='package':
        original=pipeline.build_phira_package
        def fail_package(*a,**k):
            if k['mode']=='freex':raise RuntimeError('second Phira package failed')
            return original(*a,**k)
        monkeypatch.setattr(pipeline,'build_phira_package',fail_package)
    with pytest.raises((ValueError, RuntimeError, TimeoutError, JobCancelled)):
        run_job(args)
    root = Path(args.output)
    state = json.loads((root/'job.json').read_text())
    assert state['status'] == ('cancelled' if failure == 'cancel' else 'failed')
    assert not (root/'manifest.json').exists()
    assert not (root/'preview.html').exists()
    assert not list(root.glob('*.phira.zip'))


def test_offset_four_types_tail_and_repair_ids(tmp_path):
    raw = [dict(type=kind, time=i*16, positionX=lane, holdTime=64 if kind==3 else 0,
                speed=1., floorPosition=i*.25)
           for i, (kind, lane) in enumerate(zip([1, 2, 3, 4], [-3.75, -1.25, 1.25, 3.75]))]
    before = deepcopy(raw)
    final, repairs = constrain_4k_notes(raw, 64)
    chart = chart_document(final, 120, .25, tmp_path/'chart.json')
    events, audit = validate_chart(chart, '4k', 1.25, [0, 16, 32, 48])
    assert raw == before  # export writer must not mutate originals
    assert set(audit['note_types']) == {'tap','drag','hold','flick'}
    assert all(audit['note_types'].values())
    assert events['events'][0]['time_seconds'] == .25
    assert events['events'][2]['time_seconds'] == .75
    assert events['events'][2]['end_seconds'] == 1.25
    assert repair_log(raw, final)[0]['source_index'] == 2
    assert repairs['shortened_holds'] == 1
    for alteration in ['tail', 'position', 'floor', 'missing']:
        wrong = deepcopy(chart)
        notes = wrong['judgeLineList'][0]['notesAbove']
        if alteration == 'tail': notes[2]['holdTime']=1000
        if alteration == 'position': notes[0]['positionX']=0
        if alteration == 'floor': notes[1]['floorPosition']+=1
        if alteration == 'missing': notes.pop()
        with pytest.raises(ValueError):
            validate_chart(wrong, '4k', 1.25, [0, 16, 32, 48])


def test_preview_embeds_data_safely_and_uses_audio_clock(tmp_path):
    title = '</script><script>alert("not executable")</script>'
    target = tmp_path/'preview.html'
    write_preview(target, title=title, audio_name='audio.wav', duration=2,
                  routes={'4k':dict(events=[], chart_file='4k.json', events_file='4k.events.json')})
    html = target.read_text()
    assert title not in html
    assert 'audio.currentTime' in html and 'performance.now' not in html
    assert 'fetch(' not in html and '<script src=' not in html
    payload = html.split('<script id="data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload)['title'] == title


def test_decoder_callback_can_cancel_during_generation(tmp_path, inputs, monkeypatch):
    from src.product import pipeline
    args = arguments(tmp_path, inputs)
    def cancel_mid_decode(self, *progress):
        raise JobCancelled('cancel during decoder')
    monkeypatch.setattr(pipeline.JobControl, 'progress', cancel_mid_decode)
    with pytest.raises(JobCancelled):
        run_job(args)
    assert json.loads((Path(args.output)/'job.json').read_text())['status'] == 'cancelled'
    assert not (Path(args.output)/'manifest.json').exists()


def test_budget_exhaustion_fails_even_after_last_candidate_was_covered():
    tk = ChartTokenizer()
    dec = ChartDecoder(vocab_size=tk.vocab_size, n_lanes=4, hidden_dim=32,
                       mem_dim=32, time_dim=32, depth=1, num_heads=2).eval()
    with torch.no_grad():
        dec.lm_head.weight.zero_(); dec.lm_head.bias.zero_()
        dec.lm_head.bias[tk.note(0, 1)] = 100
    memory = dec.prepare_memory(torch.zeros(1, 2, 32))
    valid = torch.ones(1, 2, dtype=torch.bool)
    legacy = generate_tokens(dec, memory, valid, 120, 46.4, 2, skeleton_ticks=[0], max_tokens=1)
    assert len(tk.decode_tokens(legacy, strict=True)) == 1
    with pytest.raises(RuntimeError, match='before natural EOS'):
        generate_tokens(dec, memory, valid, 120, 46.4, 2, skeleton_ticks=[0], max_tokens=1,
                        require_complete=True)


def test_final_sub_tick_candidate_is_retained_and_reported():
    from src.audio_understanding import SCHEMA
    evidence=dict(schema=SCHEMA,time_basis='audio_seconds',duration_seconds=1,
                  events=[dict(time_seconds=.998,confidence=.9),dict(time_seconds=.999,confidence=.8)])
    ticks, report = quantize_candidates(evidence,120,0)
    assert ticks == [63]
    assert report['quantization_clamped_events'] == 2
    assert report['quantization_merged_events'] == 1
    assert report['max_quantization_error_ms'] < 15.625


def test_layout_memory_guard_runs_before_encoder_attention(tmp_path,inputs,monkeypatch):
    from types import SimpleNamespace
    from src.product import pipeline
    monkeypatch.setattr(pipeline,'resource_snapshot',lambda path:dict(rss_gib=1.,ram_available_gib=64.))
    def must_not_run(*a,**k):raise AssertionError('attention ran before budget guard')
    monkeypatch.setattr(TranscriberNet,'features',must_not_run)
    budget=SimpleNamespace(args=SimpleNamespace(max_rss_gib=2,min_available_ram_gib=8,gpu_memory_fraction=.65))
    control=SimpleNamespace(directory=tmp_path,budget=budget)
    with pytest.raises(RuntimeError,match='layout estimate exceeds RAM'):
        pipeline.decode_notes(torch.zeros(128,20000),400,[0],inputs[1]/'decoder/decoder_best.pt',
                              '4k',120,0,42,torch.device('cpu'),control)

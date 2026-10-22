"""Run the existing DiT on the same eight development cases as AR benchmark v1.

Keeps the uploaded historical chart separate from a newly sampled checkpoint.
No training, beat snapping, threshold tuning, or additional chart repairs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile

import torch

from script.benchmark_models import file_hash, read, write
from src.model_benchmark import SCHEMA, evaluate
from src.pattern_diagnostics import chart_events
from src.resource_budget import ResourceBudget, add_resource_args


def chart_prediction(chart):
    notes = chart_events(chart, '4k')
    return dict(status='completed', onsets_seconds=sorted({n['time_seconds'] for n in notes}), notes=notes)


def uploaded_prediction(path, expected_audio_sha256):
    """Read bounded members without extraction; score original chart seconds."""
    if Path(path).stat().st_size > 160*1024**2:
        raise ValueError('uploaded ZIP too large')
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        if len(members) > 32:
            raise ValueError('too many ZIP members')
        charts = [m for m in members if m.filename.endswith('.json') and Path(m.filename).name != 'audiolab.json']
        audio = [m for m in members if m.filename.lower().endswith(('.ogg', '.wav'))]
        if len(charts) != 1 or len(audio) != 1:
            raise ValueError('exactly one chart and one audio required')
        if charts[0].file_size > 8*1024**2 or audio[0].file_size > 128*1024**2:
            raise ValueError('ZIP member too large')
        digest = hashlib.sha256()
        with z.open(audio[0]) as f:
            for block in iter(lambda: f.read(1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != expected_audio_sha256:
            raise ValueError('uploaded audio differs from benchmark Eltaw')
        raw = z.read(charts[0])
        chart = json.loads(raw)
    return chart_prediction(chart), dict(zip_sha256=file_hash(path), chart_sha256=hashlib.sha256(raw).hexdigest(),
        audio_sha256=expected_audio_sha256, bpm=chart['judgeLineList'][0]['bpm'], offset=chart['offset'],
        checkpoint_provenance='user-reported diffusion; checkpoint and sampler not present in ZIP')


def input_audio(cases, cache, probes):
    index = {r['id']: Path(r['audio']) for r in read(probes/'fixtures/fixtures.json')['rows']}
    index.update({r['id']: Path(r['audio']['path']) for r in read(cache/'cache.json')['rows']})
    selected = {}
    for case in cases:
        path = index[case['id']]
        if file_hash(path) != case['audio_sha256']:
            raise ValueError(f'audio hash differs: {case["id"]}')
        selected[case['id']] = path
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-input', type=Path, default=Path('out/benchmark_v1/inputs.json'))
    p.add_argument('--protocol', type=Path, default=Path('benchmarks/v1/protocol.json'))
    p.add_argument('--cache', type=Path, default=Path('out/full_corpus_v1/cache'))
    p.add_argument('--probes', type=Path, default=Path('out/onset_attention_20261022/onsets'))
    p.add_argument('--dit', type=Path, default=Path('checkpoints/dit/dit_best.pt'))
    p.add_argument('--vae', type=Path, default=Path('checkpoints/vae/vae_final.pt'))
    p.add_argument('--wave', type=Path, default=Path('checkpoints/dit/wave_finetuned.pt'))
    p.add_argument('--upload', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ddim-steps', type=int, default=50)
    p.add_argument('--cfg-scale', type=float, default=3.)
    p.add_argument('--fixed-length', action='store_true')
    p.add_argument('--device', default='cuda:0')
    add_resource_args(p)
    p.set_defaults(max_rss_gib=4.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    budget = ResourceBudget(args, device, args.output)
    source = read(args.baseline_input)
    baseline = deepcopy(next(r for r in source['runs'] if r['id'] == '4k/global/seed42'))
    cases = deepcopy(baseline['cases'])
    audio = input_audio(cases, args.cache, args.probes)
    protocol = read(args.protocol)
    run = dict(id=f'diffusion/current-legacy-checkpoints/seed{args.seed}', cases=cases, predictions={},
        metadata=dict(stage='decoded_layout', mode='4k', seed=args.seed, ddim_steps=args.ddim_steps,
                      cfg_scale=args.cfg_scale, eta=0., fixed_length=args.fixed_length,
                      from_logits=True, repairs='none; existing converter threshold/hold interpretation retained',
                      checkpoint_sha256={k: file_hash(getattr(args, k)) for k in ('dit', 'vae', 'wave')},
                      upload_reproduction='not established', timing_schema='legacy_unknown',
                      caveats=['development/training songs only', 'single stochastic seed',
                               'benchmark unchanged; uploaded chart has unknown generation provenance']))
    bundle = dict(schema=SCHEMA, runs=[run, baseline])
    if args.upload:
        eltaw = next(c for c in cases if c['id'] == 'Eltaw_IN')
        prediction, provenance = uploaded_prediction(args.upload, eltaw['audio_sha256'])
        bundle['runs'].append(dict(id='diffusion/user-upload/Eltaw-v2', cases=deepcopy(cases),
            predictions={'Eltaw_IN': prediction}, metadata=dict(stage='uploaded_chart', mode='4k', **provenance)))
    sources = ('script/benchmark_diffusion.py', 'src/test.py', 'src/data/chart2array.py',
               'src/data/audio2mel.py', 'src/models/dit.py', 'src/diffusion/sampler.py',
               'src/encoder/encoder.py', 'src/condition/wave.py', 'src/model_benchmark.py')
    run['metadata']['source_sha256'] = {name: file_hash(name) for name in sources}

    from src.test import _build_dit, _load_ckpt_weights, generate_chart
    from src.condition.wave import AudioWaveEncoder, DEFAULT_WAVE_CONFIG
    from src.encoder.encoder import ChartVAE, DEFAULT_VAE_CONFIG
    from src.diffusion.schedule import NoiseSchedule, DEFAULT_SCHEDULE_CONFIG
    dit = _build_dit(str(args.dit), device)
    vae = ChartVAE(DEFAULT_VAE_CONFIG).to(device)
    wave = AudioWaveEncoder(**DEFAULT_WAVE_CONFIG).to(device)
    for model, path in ((dit, args.dit), (vae, args.vae), (wave, args.wave)):
        _load_ckpt_weights(str(path), model, device)
        model.eval()
    # Assert the condition encoder paired with this DiT, when stored together.
    ck = torch.load(args.dit, map_location='cpu', weights_only=True)
    if 'wave' in ck:
        actual = wave.state_dict()
        if any(not torch.equal(actual[k].cpu(), v) for k, v in ck['wave'].items()):
            raise ValueError('Wave weights differ from the DiT checkpoint training pair')
    run['metadata']['dit_epoch'] = ck.get('epoch')
    run['metadata']['vae_latent_scale'] = vae.scale
    del ck
    schedule = NoiseSchedule(**DEFAULT_SCHEDULE_CONFIG).to(device)
    def check_budget(_model, _inputs):
        budget.check()
    dit.register_forward_pre_hook(check_budget)
    run['metadata']['generation_seconds'] = {}
    for i, case in enumerate(cases):
        identity = case['id']
        print(json.dumps(dict(case=identity, index=i+1, total=len(cases)), ensure_ascii=False), flush=True)
        chart_path = args.output/f'case-{i:02d}.json'
        clock = case.get('clock')
        bpm, offset = (clock['bpm'], clock['beat_zero_seconds']) if clock else (120., 0.)
        # 120 here only encodes fixture seconds in PGR; fixtures have no rhythm score.
        started = time.monotonic()
        try:
            budget.check(force=True)
            with (args.output/f'case-{i:02d}.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                raw = generate_chart(str(audio[identity]), str(chart_path), dit=dit, vae=vae, wave=wave,
                    schedule=schedule, device=device, bpm=bpm, offset=offset, seed=args.seed,
                    ddim_steps=args.ddim_steps, cfg_scale=args.cfg_scale, fixed_length=args.fixed_length)
                del raw
            run['predictions'][identity] = chart_prediction(read(chart_path))
        except Exception as exc:
            run['predictions'][identity] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
            (args.output/f'case-{i:02d}.failure.log').write_text(traceback.format_exc())
        run['metadata']['generation_seconds'][identity] = time.monotonic()-started
        write(args.output/'inputs.json', bundle)
        report = evaluate(bundle, protocol)
        report['resources'] = budget.log_state()
        write(args.output/'report.json', report)
        row = report['runs'][0]['rows'][i]
        print(json.dumps(dict(case=identity, status=row['status'], gates=row.get('gates'),
                              f1_50ms=row.get('events', {}).get('at_50ms', {}).get('f1')), ensure_ascii=False), flush=True)
        if run['predictions'][identity]['status'] == 'failed' and 'Resource budget' in run['predictions'][identity]['error']:
            raise RuntimeError('resource guard stopped remaining cases; missing cases remain in report')
        gc.collect()
    print(json.dumps(dict(status='completed', output=str(args.output), resources=budget.log_state())), flush=True)


if __name__ == '__main__':
    main()

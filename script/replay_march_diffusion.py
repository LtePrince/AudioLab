"""Generate benchmark cases with the unmodified March 2026 source snapshot.

Run this file in a fresh Python process. The request contains hashed weights,
hashed audio, output paths and sampler settings; scoring remains in the current
benchmark process. No historical source or checkpoint is modified.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REVISION = '20edd4d35b584dccd6608dafa92d7028bf41d80f'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    output = Path(request['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if any(name == 'src' or name.startswith('src.') for name in sys.modules):
        raise RuntimeError('Use a fresh Python process; current src must not be imported')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/audiolab-mpl')
    import torch
    torch.set_num_threads(2)
    device = torch.device(args.device)

    # Resource checks have no src imports; load them outside the legacy package.
    spec = importlib.util.spec_from_file_location('audiolab_replay_budget', ROOT/'src/resource_budget.py')
    resources = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resources)
    limits = dict(min_free_disk_gib=8., min_available_ram_gib=8., max_rss_gib=4.,
                  gpu_memory_fraction=.65, keep_checkpoints=1)
    limits.update(request.get('resources', {}))
    budget = resources.ResourceBudget(SimpleNamespace(**limits), device, output)
    for item in [*request['weights'].values(), *request['cases']]:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f'Input hash differs: {item["path"]}')

    archive = subprocess.check_output(['git', 'archive', REVISION, 'src'], cwd=ROOT)
    if len(archive) > 10*1024**2:
        raise ValueError('Historical source archive exceeds budget')
    result = dict(code_commit=REVISION, request_sha256=digest(args.request),
                  runner_sha256=digest(__file__), weights=request['weights'],
                  sampler=request['sampler'], cases={}, resource_limits=limits,
                  fixed_length=True, from_logits=False, repairs='none',
                  timing='original March behavior, before the August 2x alignment fix')
    with tempfile.TemporaryDirectory(prefix='march-source-', dir=output) as directory:
        source = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter='data')
        del archive
        result['source_sha256'] = {
            str(f.relative_to(source)): digest(f) for f in sorted((source/'src').rglob('*.py'))}
        sys.path.insert(0, str(source))
        from src import test as historical
        if not Path(historical.__file__).resolve().is_relative_to(source):
            raise RuntimeError('Historical imports resolved outside the exported revision')
        vae = historical.ChartVAE(historical.DEFAULT_VAE_CONFIG).to(device)
        wave = historical.AudioWaveEncoder(**historical.DEFAULT_WAVE_CONFIG).to(device)
        dit = historical.RhythmDiT(**historical.DEFAULT_DIT_CONFIG).to(device)
        models = dict(dit=dit, vae=vae, wave=wave)
        for name, model in models.items():
            historical._load_ckpt_weights(request['weights'][name]['path'], model, device)
            model.eval()
            budget.check(force=True)
        ck = torch.load(request['weights']['dit']['path'], map_location='cpu', weights_only=True, mmap=True)
        result['checkpoint_metadata'] = {key: ck.get(key) for key in ('epoch', 'step', 'global_step')}
        del ck
        result['model_parameters'] = {name: sum(p.numel() for p in model.parameters())
                                      for name, model in models.items()}
        result['dit_config'] = historical.DEFAULT_DIT_CONFIG
        result['vae_scale'] = vae.scale
        schedule = historical.NoiseSchedule(**historical.DEFAULT_SCHEDULE_CONFIG).to(device)

        def check_budget(_model, _inputs):
            budget.check()
        dit.register_forward_pre_hook(check_budget)
        write(output/'generation.json', result)
        # The request keeps the frozen case IDs/indexes; Eltaw may run first.
        for case in request['cases']:
            started = time.monotonic()
            chart = output/f'case-{case["index"]:02d}.json'
            record = dict(status='running', chart=str(chart))
            result['cases'][case['id']] = record
            try:
                budget.check(force=True)
                with chart.with_suffix('.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                    raw = historical.generate_chart(
                        case['path'], str(chart), dit=dit, vae=vae, wave=wave,
                        schedule=schedule, device=device, bpm=case['bpm'], offset=case['offset'],
                        **request['sampler'])
                    del raw
                record.update(status='completed', chart_sha256=digest(chart))
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                chart.with_suffix('.failure.log').write_text(traceback.format_exc())
            record['seconds'] = time.monotonic() - started
            result['resources'] = budget.log_state()
            write(output/'generation.json', result)
            print(json.dumps(dict(case=case['id'], **record)), flush=True)
            if record['status'] == 'failed' and 'Resource budget' in record['error']:
                raise RuntimeError('Resource guard stopped remaining cases')
            gc.collect()
        sys.path.remove(str(source))
    result['temporary_source_cleaned'] = True
    result['resources'] = budget.log_state()
    write(output/'generation.json', result)
    print(json.dumps(dict(status='completed', output=str(output), resources=result['resources'])), flush=True)


if __name__ == '__main__':
    main()

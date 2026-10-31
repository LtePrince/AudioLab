"""Sequential, resumable 4K/free-x training with bounded resource use.

Run from the AudioLab root with the project's Python environment. Status,
commands, resource samples and logs are kept inside --run-dir. Only one
worker runs at once. Existing baseline checkpoints are read-only inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKERS = ['src.train_transcription', 'src.train_decoder', 'src.test_decoder',
           'src.test_transcription', 'script.eval_onset_f1', 'script.diagnose_decoder',
           'script.audit_audio_conditioning', 'script.calibrate_skeleton', 'src.train_rhythm',
           'src.train_audio_understanding', 'script.understand_audio', 'script.evaluate_audio_understanding',
           'src.train_chord_count', 'script.generate_product', 'script.audit_product_pipeline',
           'script.serve_product','script.audit_product_tempo','script.audit_audio_conditioning_v2',
           'script.ablate_layout_features', 'script.train_soft_onset', 'script.probe_isolated_onsets',
           'script.audit_decoder_alignment', 'script.train_alignment_scratch', 'script.benchmark_diffusion',
           'script.train_grounded_ar', 'script.train_anchored_ar', 'script.probe_audio_localization',
           'script.train_windowed_ar']


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def refuse_live_run(manifest: dict) -> None:
    """A resume must not start a second worker beside an existing supervisor."""
    pids = [manifest.get('pid')]
    active = manifest.get('active_stage')
    if active:
        pids.append(manifest.get('stages', {}).get(active, {}).get('pid'))
    for pid in pids:
        if pid and pid != os.getpid():
            cmdline = Path(f'/proc/{pid}/cmdline')
            if cmdline.exists() and b'script.train_models' in cmdline.read_bytes():
                raise RuntimeError(f'Run is already active (PID {pid}); refusing duplicate training')


def build_stages(args) -> list[tuple[str, str, list[str]]]:
    run = Path(args.run_dir)
    common = ['--data-list', 'data/train.txt', '--val-list', 'data/val.txt',
              '--cache-dir', 'data/cache_mel', '--num-workers', '0', '--device', 'cuda:0',
              '--ckpt-dir', str(run), '--save-every', '1', '--keep-checkpoints', '1',
              '--min-free-disk-gib', str(args.min_free_disk_gib),
              '--min-available-ram-gib', str(args.min_available_ram_gib),
              '--max-rss-gib', str(args.max_rss_gib),
              '--gpu-memory-fraction', str(args.gpu_memory_fraction), '--log-every', '10']
    stages = [('transcriber', 'src.train_transcription', common + [
        '--chart-cache-dir', 'data/cache_chart', '--init-ckpt', 'checkpoints/transcriber/transcriber_best.pt',
        '--batch-size', '2', '--grad-accum', '8', '--val-batch-size', '1',
        '--lr', '0.00005', '--epochs', str(args.transcriber_epochs), '--patience', '8'])]
    for name, free in [('decoder', False), ('decoder_freex', True)]:
        command = common + ['--encoder-ckpt', str(run / 'transcriber/transcriber_best.pt'),
            '--init-ckpt', f'checkpoints/{name}/decoder_best.pt', '--batch-size', '1', '--grad-accum', '6',
            '--lr', '0.00005', '--encoder-lr', '0.00001', '--epochs', str(args.decoder_epochs),
            '--patience', '8']
        if free:
            command += ['--gameplay-dir', 'data/gameplay']
        stages.append((name, 'src.train_decoder', command))
    for name in ['decoder', 'decoder_freex']:
        output = str(run / f'eval_{name}')
        stages.append((f'generate_{name}', 'src.test_decoder', [
            '--ckpt', str(run / name / 'decoder_best.pt'), '--list', 'data/val.txt',
            '--out-dir', output, '--skeleton-ckpt', str(run / 'transcriber/transcriber_best.pt'),
            '--onset-threshold', '0.7', '--seed', '42', '--device', 'cuda:0']))
        for tolerance in [25, 50, 92.9]:
            command = ['--gen-dir', output, '--list', 'data/val.txt', '--tol-ms', str(tolerance)]
            if name == 'decoder_freex':
                command += ['--ref-dir', 'data/gameplay_charts']
            stages.append((f'evaluate_{name}_{tolerance}ms', 'script.eval_onset_f1', command))
    return stages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=WORKERS)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--run-dir', default='checkpoints/timing_v1')
    p.add_argument('--resume-run', action='store_true')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--gpu-memory-fraction', type=float, default=0.65)
    p.add_argument('--min-free-disk-gib', type=float, default=8)
    p.add_argument('--min-available-ram-gib', type=float, default=8)
    p.add_argument('--max-rss-gib', type=float, default=10)
    p.add_argument('--transcriber-epochs', type=int, default=20)
    p.add_argument('--decoder-epochs', type=int, default=30)
    args, remaining = p.parse_known_args()
    if remaining and not args.worker:
        p.error(f'unrecognized arguments: {remaining}')
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/audiolab-mpl')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from src.resource_budget import ResourceBudget, GIB, resource_snapshot
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is unavailable; run outside the device-restricted sandbox.')
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    if args.worker:
        import runpy
        # The separator isolates worker flags from supervisor flags.
        sys.argv = [args.worker] + (remaining[1:] if remaining[:1] == ['--'] else remaining)
        runpy.run_module(args.worker, run_name='__main__')
        return
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    args.keep_checkpoints = 1
    budget = ResourceBudget(args, torch.device('cuda:0'), run)
    # Actual forward/backward checks runtime/driver compatibility, not just enumeration.
    probe = torch.randn(64, 64, device='cuda', requires_grad=True)
    (probe @ probe.T).square().mean().backward()
    torch.cuda.synchronize()
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                      'cuda': torch.version.cuda, **budget.log_state()}), flush=True)
    del probe
    torch.cuda.empty_cache()
    if args.preflight:
        return
    run_lock = (run / '.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another supervisor holds {run / ".lock"}') from None
    manifest_path = run / 'run.json'
    if manifest_path.exists() and not args.resume_run:
        raise ValueError(f'{manifest_path} exists; use --resume-run or a new --run-dir')
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                {'config': vars(args), 'stages': {}, 'created': time.time()})
    refuse_live_run(manifest)
    for key in ('transcriber_epochs', 'decoder_epochs'):
        if manifest['config'][key] != getattr(args, key):
            raise ValueError(f'Resume must keep the original {key}: {manifest["config"][key]}')
    manifest.update(status='running', pid=os.getpid())
    write_json(manifest_path, manifest)

    def stop_requested(signum, frame):
        raise KeyboardInterrupt(f'Supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, stop_requested)
    for name, module, command in build_stages(args):
        if manifest['stages'].get(name, {}).get('status') == 'complete':
            continue
        budget.check(force=True)
        if args.resume_run and name in ('transcriber', 'decoder', 'decoder_freex'):
            pattern = 'tsc_e*.pt' if name == 'transcriber' else 'dec_e*.pt'
            checkpoints = sorted((run / name).glob(pattern))
            if checkpoints:
                index = command.index('--init-ckpt')
                command[index:index + 2] = ['--resume', str(checkpoints[-1])]
        cmd = [sys.executable, '-u', '-m', 'script.train_models', '--gpu', str(args.gpu),
               '--gpu-memory-fraction', str(args.gpu_memory_fraction), '--worker', module, '--', *command]
        record = {'status': 'running', 'command': cmd, 'started': time.time()}
        manifest['stages'][name] = record
        manifest['active_stage'] = name
        print(f'[pipeline] starting {name}; log: {run / (name + ".log")}', flush=True)
        with (run / (name + '.log')).open('a') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            write_json(manifest_path, manifest)
            try:
                while child.poll() is None:
                    budget.check(force=True)
                    rss = 0.0
                    status = Path(f'/proc/{child.pid}/status')
                    if status.exists():
                        for line in status.read_text().splitlines():
                            if line.startswith('VmRSS:'):
                                rss = int(line.split()[1]) * 1024 / GIB
                    if rss > args.max_rss_gib:
                        raise RuntimeError(f'Worker RSS {rss:.2f} GiB exceeds limit')
                    sample = {'time': time.time(), 'stage': name, 'worker_rss_gib': rss,
                              **resource_snapshot(run)}
                    with (run / 'resources.jsonl').open('a') as fh:
                        fh.write(json.dumps(sample) + '\n')
                    time.sleep(10)
            except BaseException as exc:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                record.update(status='stopped', reason=str(exc), finished=time.time())
                manifest['status'] = 'stopped'
                write_json(manifest_path, manifest)
                raise
        record.update(status='complete' if child.returncode == 0 else 'failed',
                      returncode=child.returncode, finished=time.time())
        manifest['status'] = 'running' if child.returncode == 0 else 'failed'
        write_json(manifest_path, manifest)
        if child.returncode:
            raise SystemExit(f'{name} failed; inspect {run / (name + ".log")}')
    manifest.update(status='complete', active_stage=None, finished=time.time())
    write_json(manifest_path, manifest)
    print('[pipeline] complete', flush=True)


if __name__ == '__main__':
    main()

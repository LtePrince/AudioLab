"""Two bounded GPU queues for the authorized four independent 1000→1200 runs."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
RUN = ROOT/'out/ar_continue1200_20261027'
ASSETS = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(state):
    target = RUN/'supervisor.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    os.replace(temp, target)


def main():
    os.chdir(ROOT)
    plan = json.loads((ASSETS/'launch_plan.json').read_text())
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN/'supervisor.json').exists():
        raise RuntimeError('Existing launch record; recover explicitly, do not duplicate')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'train' in args and (b'script.train_anchored_ar' in args or
                                str(ASSETS/'worker.py').encode() in args):
            raise RuntimeError(f'Existing anchored training PID {proc.name}')
    state = dict(schema='audiolab-continue1200-supervisor-v1', status='running', pid=os.getpid(),
                 started_at_utc=now(), launch_plan_sha256=sha(ASSETS/'launch_plan.json'),
                 runs={r['id']:dict(r, status='queued') for r in plan['runs']})
    active = {}

    def launch(item):
        for path, digest in plan['source_sha256'].items():
            if sha(ROOT/path) != digest:
                raise RuntimeError(f'Queued stage source changed: {path}')
        if sha(ROOT/item['parent_checkpoint']) != item['parent_sha256']:
            raise RuntimeError('Parent checkpoint changed')
        output = ROOT/item['output']
        if output.exists():
            raise RuntimeError(f'Refuse existing output: {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(ROOT/'.venv/bin/python'), '-u', str(ASSETS/'worker.py'), 'train',
            '--mode', item['mode'], '--variant', item['variant'],
            '--continue-from', item['parent_checkpoint'], '--output', item['output'],
            '--pairs', item['pairs'], '--epochs', '1200', '--until-epoch', '1200',
            '--eval-every', '50', '--snapshot-every', '50', '--device', 'cuda:0',
            '--min-free-disk-gib', '8', '--min-available-ram-gib', '8',
            '--max-rss-gib', '4', '--gpu-memory-fraction', '0.65']
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(item['gpu']),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLCONFIGDIR='/tmp/audiolab-mpl')
        log = RUN/(item['id']+'.log')
        with log.open('xb', buffering=0) as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state['runs'][item['id']].update(status='running', pid=child.pid, started_at_utc=now(),
                                       command=command, log=str(log.relative_to(ROOT)))
        active[item['gpu']] = (child, item)
        save(state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        save(state)
        while True:
            for gpu, (child, item) in list(active.items()):
                record = state['runs'][item['id']]
                progress = ROOT/item['output']/'training.json'
                if progress.is_file():
                    curves = json.loads(progress.read_text())['curves']
                    record.update(epoch=curves[-1]['epoch'], steps=curves[-1]['steps'])
                code = child.poll()
                if code is None:
                    continue
                record.update(returncode=code, finished_at_utc=now())
                if code != 0:
                    record['status'] = 'failed'
                    raise RuntimeError(f"{item['id']} exited {code}; see {record['log']}")
                if record.get('epoch') != 1200 or record.get('steps') != 360000:
                    raise RuntimeError(f"{item['id']} did not finish all 200 added epochs")
                for epoch in (1050,1100,1150,1200):
                    folder = ROOT/item['output']
                    if not (folder/f'evaluation-e{epoch:04d}/fixed15/summary.json').is_file():
                        raise RuntimeError(f'Missing fixed15 evaluation: {item["id"]}/{epoch}')
                    snapshot = folder/f'model_e{epoch:04d}.pt'
                    if sha(snapshot) != json.loads(snapshot.with_suffix('.json').read_text())['sha256']:
                        raise RuntimeError('Snapshot checksum mismatch')
                record.update(status='completed', checkpoint_sha256=sha(ROOT/item['output']/'model_latest.pt'))
                del active[gpu]
            for gpu in (0,1):
                if gpu in active:
                    continue
                pending = next((r for r in plan['runs'] if r['gpu']==gpu and
                                state['runs'][r['id']]['status']=='queued'), None)
                if pending:
                    launch(pending)
            state['observed_at_utc'] = now()
            save(state)
            if not active:
                break
            time.sleep(30)
        state.update(status='completed', finished_at_utc=now())
        save(state)
    except BaseException as exc:
        for child, item in active.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            record = state['runs'][item['id']]
            if record['status'] != 'failed':
                record['status'] = 'stopped'
            record['returncode'] = child.returncode
        state.update(status='failed', error=str(exc), finished_at_utc=now())
        save(state)
        raise


if __name__ == '__main__':
    main()

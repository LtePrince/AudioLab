"""Launch the frozen historical replay independently of a CLI session."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/anchored_ar.py').exists())
OUT = ROOT / 'out/ar_early_replay_20261102'
OUT.mkdir(exist_ok=True)
if (OUT / 'launch.json').exists():
    raise RuntimeError('existing launch; inspect it before starting another process')
with (OUT / 'worker.log').open('xb', buffering=0) as log:
    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('replay.py')),
                              'evaluate', '--device', 'cuda:0'], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    (OUT / 'launch.json').write_text(json.dumps(dict(pid=child.pid, started=time.time(),
                                                     log=str(OUT / 'worker.log'))) + '\n')
time.sleep(3)
if child.poll() is not None:
    raise RuntimeError(f'replay exited during startup: {(OUT / "worker.log").read_text()[-2000:]}')
print(json.dumps(dict(pid=child.pid, status='launched', log=str(OUT / 'worker.log'))), flush=True)

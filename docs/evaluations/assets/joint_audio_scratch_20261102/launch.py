"""Launch E23-A: random acoustic encoder and decoder; fresh optimizer/RNG."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/joint_audio_ar.py').exists())
OUT=ROOT/'out/joint_audio_scratch_20261102'
OUT.mkdir(exist_ok=True)
if (OUT/'launch.json').exists():
    raise RuntimeError('existing launch: inspect process/checkpoint before resuming')
command=[sys.executable,'-u','-m','script.train_models','--gpu','0','--worker','script.train_joint_audio_a','--','run',
    '--initialization','scratch','--run-id','E23-A','--output','out/joint_audio_scratch_20261102',
    '--architecture-config','docs/evaluations/assets/joint_audio_scratch_20261102/architecture.json',
    '--mel-cache','out/joint_audio_a_20261102/mel','--epochs','300','--lr','0.0003','--encoder-lr','0.0003',
    '--warmup-steps','1500','--baseline-eval','focus','--eval-every','20','--full-eval-every','100']
with (OUT/'worker.log').open('xb',buffering=0) as log:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'launch.json').write_text(json.dumps(dict(pid=child.pid,started=time.time(),command=command,
        log=str(OUT/'worker.log'),scope='A only, random encoder+decoder, 300 songs, 300 epochs; shared fixed Mel'))+'\n')
time.sleep(3)
if child.poll() is not None:
    raise RuntimeError(f'worker exited during startup: {(OUT/"worker.log").read_text()[-2500:]}')
print(json.dumps(dict(pid=child.pid,status='launched',log=str(OUT/'worker.log'))),flush=True)

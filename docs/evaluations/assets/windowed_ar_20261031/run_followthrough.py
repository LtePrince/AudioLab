"""Finish stage-5 controls, then a bounded matched full300 adaptation pilot.

Run outside the sandbox for CUDA access. Source hashes are frozen before any
queued worker starts. Only this supervisor's own children are terminated on
failure; already-running independent experiments are not killed.
"""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

import hashlib
import json
import os
import subprocess
import time

from src.product.artifacts import write_json

BASE=ROOT/'out/windowed_ar_20261031'
FILES=('script/train_windowed_variants.py','script/train_pattern_continuation.py','script/train_windowed_corpus.py',
    'src/windowed_variants.py','src/windowed_ar.py','src/anchored_ar.py','src/models/chart_decoder.py',
    'src/models/chart_history.py','src/models/transcriber.py','src/resource_budget.py','script/train_anchored_ar.py',
    'script/train_grounded_ar.py','script/train_soft_onset.py','script/train_alignment_scratch.py',
    'docs/evaluations/assets/phira_generalization10_20261029/benchmark.py','benchmarks/v1/protocol.json',
    'docs/evaluations/assets/windowed_ar_20261031/summarize_stage5.py',
    'docs/evaluations/assets/windowed_ar_20261031/run_followthrough.py')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def running(folder):
    needle=f'out/windowed_ar_20261031/{folder}'.encode()
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if needle in cmd:return True
    return False


def main():
    status_path=BASE/'followthrough.json'
    if status_path.exists():raise RuntimeError('existing supervisor state: inspect, do not duplicate')
    sources={p:sha(ROOT/p) for p in FILES};children=[];logs=[]
    state=dict(schema='audiolab-e16-followthrough-v1',pid=os.getpid(),status='running',phase='stage5',
        source_sha256=sources,created=time.time(),started_runs={},latest={})

    def check_sources():
        if any(sha(ROOT/p)!=h for p,h in sources.items()):raise RuntimeError('queued source changed')

    def start(folder,module,extra):
        check_sources();cmd=[str(ROOT/'.venv/bin/python'),'-m',module,'train',*extra,
            '--output',f'out/windowed_ar_20261031/{folder}']
        log=(BASE/f'{folder}.log').open('ab',buffering=0);logs.append(log)
        child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children.append(child);state['started_runs'].setdefault(folder,[]).append(dict(pid=child.pid,command=cmd,time=time.time()))
        write_json(status_path,state);print(json.dumps(dict(started=folder,pid=child.pid)),flush=True)
        return child

    def tick(folders,acceptable=('completed',)):
        latest={};done=True
        for folder in folders:
            path=BASE/folder/'training.json'
            data=read(path) if path.exists() else {};curves=data.get('curves',[])
            record=curves[-1] if curves else {}
            latest[folder]=dict(status=data.get('status','starting'),epoch=record.get('epoch'),
                loss=record.get('objective',record.get('loss')))
            accepted=data.get('status') in acceptable
            alive=running(folder)
            if not accepted and not alive:
                raise RuntimeError(f'{folder} exited before completing; see its log')
            if not accepted or alive:done=False
        state['latest']=latest;state['updated']=time.time();write_json(status_path,state)
        print(json.dumps(dict(phase=state['phase'],progress=latest),ensure_ascii=False),flush=True)
        return done

    write_json(status_path,state)
    try:
        # The caller has already started E1 and P0 and checked their first log.
        while not tick(('encoder_tail',)):
            time.sleep(30)
        if (BASE/'pattern_kind_position').exists():raise RuntimeError('pattern P1 output already exists')
        start('pattern_kind_position','script.train_pattern_continuation',
              ['--variant','kind_position','--device','cuda:0'])
        while not tick(('pattern_base','pattern_kind_position')):
            time.sleep(30)
        check_sources()
        with (BASE/'stage5-summary.log').open('wb') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'docs/evaluations/assets/windowed_ar_20261031/summarize_stage5.py')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['stage5_report']='docs/evaluations/windowed_ar_stage5_20261031.md'
        state['phase']='natural_first_epoch';write_json(status_path,state)
        for gpu,architecture in enumerate(('whole','window')):
            folder=f'natural_{architecture}'
            if (BASE/folder).exists():raise RuntimeError('natural output already exists')
            start(folder,'script.train_windowed_corpus',['--architecture',architecture,'--device',f'cuda:{gpu}',
                '--epochs','20','--until-epoch','1','--eval-every','10'])
        while not tick(('natural_whole','natural_window'),('phase_completed',)):
            time.sleep(30)
        state['phase']='natural_adaptation';write_json(status_path,state)
        for gpu,architecture in enumerate(('whole','window')):
            start(f'natural_{architecture}','script.train_windowed_corpus',
                ['--architecture',architecture,'--device',f'cuda:{gpu}','--epochs','20','--eval-every','10','--resume'])
        while not tick(('natural_whole','natural_window')):
            time.sleep(30)
        state['status']='completed';state['phase']='reports_ready_for_analysis';state['finished']=time.time()
        write_json(status_path,state);print(json.dumps(dict(status='completed',reports=str(status_path))),flush=True)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished=time.time())
        write_json(status_path,state);raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()

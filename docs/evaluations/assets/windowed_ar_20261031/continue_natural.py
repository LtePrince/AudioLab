"""Audited NW1 e20 -> e100 continuation; preserve the matched e20 comparison.

Only the stopping epoch changes. Model, AdamW, sampler and Torch/CUDA RNG are
retained. The original training source and its constant LR remain unchanged.
This is a fitting study, not a matched e100/e20 architecture comparison.
"""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/windowed_ar.py').exists())
sys.path.insert(0, str(ROOT))

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from types import SimpleNamespace

import torch

from script.train_windowed_corpus import SCHEMA
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget
from docs.evaluations.assets.windowed_ar_20261031.summarize_natural import region_metrics

BASE = ROOT / 'out/windowed_ar_20261031'
ASSET = Path(__file__).resolve().parent
SOURCE = BASE / 'natural_window'
DEST = BASE / 'natural_window_fit100'


def read(path):
    return json.loads(Path(path).read_text())


def equal_tree(a, b):
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def extend_checkpoint(checkpoint, target_epoch):
    config = checkpoint['config']
    if checkpoint['schema'] != SCHEMA or config['architecture'] != 'window':
        raise ValueError('only the natural window protocol is supported')
    if checkpoint['epoch'] != config['epochs'] or target_epoch <= checkpoint['epoch']:
        raise ValueError('extend a completed run to a later stopping epoch')
    if len(checkpoint['curves']) != checkpoint['epoch']:
        raise ValueError('incomplete epoch history')
    if checkpoint['steps'] != checkpoint['curves'][-1]['steps']:
        raise ValueError('step/curve mismatch')
    if not config['schedule'].startswith('constant_lr;'):
        raise ValueError('extending a nonconstant schedule requires a separate experiment')
    state = checkpoint['training_state']
    if not all(k in state for k in ('optimizer', 'sampler', 'torch_rng', 'cuda_rng')):
        raise ValueError('complete optimizer and RNG state required')
    if any(group['lr'] != config['lr'] for group in state['optimizer']['param_groups']):
        raise ValueError('saved optimizer LR differs from the constant schedule')
    return dict(checkpoint, config=dict(config, epochs=target_epoch))


def budget(directory):
    args = SimpleNamespace(min_free_disk_gib=8, min_available_ram_gib=8,
                           max_rss_gib=4, gpu_memory_fraction=.65, keep_checkpoints=1)
    return ResourceBudget(args, torch.device('cpu'), directory)


def prepare():
    training = read(SOURCE / 'training.json')
    if training['status'] != 'completed' or training['curves'][-1]['epoch'] != 20:
        raise RuntimeError('matched e20 run must be complete')
    if not read(SOURCE / 'evaluation-e0020/summary.json')['full']:
        raise RuntimeError('finish matched full evaluation first')
    checkpoint_path = SOURCE / 'model_latest.pt'
    source_hash = sha256(checkpoint_path)
    if source_hash != training['checkpoint_sha256']:
        raise RuntimeError('source checkpoint changed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['steps'] != 6000 or checkpoint['config'] != training['config']:
        raise RuntimeError('unexpected e20 provenance')
    for path, digest in checkpoint['config']['source_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise RuntimeError(f'training source changed: {path}')
    extended = extend_checkpoint(checkpoint, 100)
    DEST.mkdir(exist_ok=False)
    budget(DEST).save(extended, DEST / 'model_latest.pt')
    reloaded = torch.load(DEST / 'model_latest.pt', map_location='cpu', weights_only=True)
    restored = dict(reloaded, config=dict(reloaded['config'], epochs=20))
    if not equal_tree(checkpoint, restored) or sha256(checkpoint_path) != source_hash:
        raise RuntimeError('continuation failed exact state preservation check')
    record = dict(schema='audiolab-nw1-continuation-v1', source=str(checkpoint_path.relative_to(ROOT)),
                  source_sha256=source_hash, source_epoch=20, source_steps=6000,
                  destination=str(DEST.relative_to(ROOT)), target_epoch=100,
                  changed_fields={'config.epochs': [20, 100]}, exact_state_preservation_verified=True,
                  initial_resume_sha256=sha256(DEST / 'model_latest.pt'),
                  training_source_sha256=training['config']['source_sha256'],
                  preparation_script_sha256=sha256(__file__), resources=budget(DEST).log_state(),
                  scope='W1-only fitting study. e20 W0/W1 remain immutable; unequal additional training is not an architecture ablation.')
    write_json(DEST / 'continuation.json', record)
    write_json(ASSET / 'natural_continuation.json', record)
    print(json.dumps(dict(prepared=str(DEST), retained=['weights', 'AdamW', 'sampler', 'Torch/CUDA RNG'],
                          epochs=[20, 100]), ensure_ascii=False), flush=True)


def report():
    training = read(DEST / 'training.json')
    if training['status'] != 'completed' or training['curves'][-1]['epoch'] != 100:
        raise RuntimeError('continuation not complete')
    final = read(DEST / 'evaluation-e0100/summary.json')
    if not final['full'] or len(final['rows']) != 25:
        raise RuntimeError('full 15+10 evaluation missing')
    prior = read(ASSET / 'natural_result.json')
    result = dict(schema='audiolab-nw1-fitting100-v1', continuation=read(DEST / 'continuation.json'),
                  checkpoint_sha256=sha256(DEST / 'model_latest.pt'), config=training['config'],
                  curves=training['curves'], resources=training['resources'], evaluation=final,
                  fixed_gap_regions=[], script_sha256=sha256(__file__))
    selection = read(ROOT / 'docs/evaluations/assets/phira_generalization10_20261029/selection.json')
    for old in prior['fixed_gap_regions']:
        song = next(s for s in selection['songs'] if s['id'] == old['id'])
        reference = read(ROOT / song['reference'])['notes']
        prediction = read(DEST / f'evaluation-e0100/{song["id"]}.json')['prediction']
        result['fixed_gap_regions'].append(dict(id=old['id'], name=old['name'], start=old['start'], end=old['end'],
            e20=old['window'], e100=region_metrics(reference, prediction, old['start'], old['end'])))
    result['intermediate_checks'] = [read(p) for p in sorted(DEST.glob('evaluation-e*/summary.json'))]
    write_json(ASSET / 'natural_fit100_result.json', result)
    lines = ['# 分窗自然歌曲续训：e20 → e100', '',
        '只续训 E16-NW1（AL-AR4K v0.6.0 / A），全 300 首、原数据顺序与配方。'
        '完整恢复 AdamW 和采样/随机状态，恒定 LR 3e-5；唯一元数据变化是计划停止轮次 20→100。'
        '新增 24,000 次更新，累计 30,000 次。原 e20 两组对照保持不动。', '',
        '这是训练充分性检查；没有新增等量训练的 W0，不能将结果解释为相同训练预算的架构胜负。', '',
        '| 数据 / 分窗轮数 | 起点 F1@50ms | 网格加分 | 时间＋类型 F1 | 正常完成 |',
        '| --- | ---: | ---: | ---: | --- |']
    for split in ('train15', 'E20_development'):
        for epoch, metrics in ((20, prior['runs']['window']['evaluation']['aggregates'][split]),
                               (100, final['aggregates'][split])):
            lines.append(f"| {split} / e{epoch} | {metrics['onset_f1_50ms']:.4f} | {metrics['grid_bonus']:.4f} | "
                         f"{metrics['time_type_f1']:.4f} | {metrics['completed']}/{metrics['examples']} |")
    first, last = training['curves'][19], training['curves'][-1]
    lines += ['', f"训练 objective：{first['objective']:.5f}→{last['objective']:.5f}；"
              f"真实历史下窗口结束召回：{first['end_stop_recall']:.4f}→{last['end_stop_recall']:.4f}。",
              '', '| 原缺音区间 | e20 最长空档 s | e100 最长空档 s | e100 匹配 P / R |',
              '| --- | ---: | ---: | --- |']
    for gap in result['fixed_gap_regions']:
        m = gap['e100']
        lines.append(f"| {gap['name']} {gap['start']:.2f}–{gap['end']:.2f}s | "
                     f"{gap['e20']['largest_onset_gap_seconds']:.2f} | {m['largest_onset_gap_seconds']:.2f} | "
                     f"{m['precision']:.3f}/{m['recall']:.3f} |")
    lines += ['', '以上只检查原八处中的两处 4K A 缺口；E20 为开发回归集，非盲测。'
              '学习结束状态仍按每窗 END_WINDOW 判断，调度器走完整曲不计作学会结束。'
              '这是自动生成的数值报告；解释与下一轮取舍须结合逐曲输出。', '',
              '[完整结果、训练曲线与来源](assets/windowed_ar_20261031/natural_fit100_result.json)。'
              '该续训不自动替换产品或 Release 权重。', '']
    (ROOT / 'docs/evaluations/windowed_ar_natural_fit100_20261101.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='reported', metrics=final['aggregates']), ensure_ascii=False), flush=True)


def run():
    record = read(DEST / 'continuation.json')
    lock = (DEST / '.supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = DEST / 'supervisor.json'
    if state_path.exists():
        raise RuntimeError('existing supervisor state; inspect before relaunch')
    sources = dict(record['training_source_sha256'])
    extras = ('script/train_alignment_scratch.py', 'src/resource_budget.py',
              'docs/evaluations/assets/phira_generalization10_20261029/benchmark.py',
              'docs/evaluations/assets/windowed_ar_20261031/continue_natural.py',
              'docs/evaluations/assets/windowed_ar_20261031/summarize_natural.py',
              'benchmarks/v1/protocol.json')
    sources.update({p: sha256(ROOT / p) for p in extras})
    for path, digest in sources.items():
        if sha256(ROOT / path) != digest:
            raise RuntimeError(f'source mismatch: {path}')
    if sha256(DEST / 'model_latest.pt') != record['initial_resume_sha256']:
        raise RuntimeError('prepared resume checkpoint changed before launch')
    # Preserve physical GPU index 1 and the original two-device RNG topology.
    if os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 2:
        raise RuntimeError('continuation requires the original visible two-GPU topology')
    cmd = [str(ROOT / '.venv/bin/python'), '-u', '-m', 'script.train_windowed_corpus', 'train',
           '--architecture', 'window', '--device', 'cuda:1', '--epochs', '100', '--eval-every', '20',
           '--resume', '--output', str(DEST.relative_to(ROOT))]
    state = dict(schema='audiolab-nw1-fitting-supervisor-v1', status='starting', pid=os.getpid(),
                 command=cmd, source_sha256=sources, created=time.time())
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        budget(DEST).check(force=True)
        with (DEST / 'worker.log').open('xb', buffering=0) as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', worker_pid=child.pid)
            write_json(state_path, state)
            while child.poll() is None:
                snapshot = budget(DEST).log_state()
                budget(DEST).check(force=True)
                if any(sha256(ROOT / p) != h for p, h in sources.items()):
                    raise RuntimeError('training source changed during continuation')
                training_path = DEST / 'training.json'
                training = read(training_path) if training_path.exists() else {}
                curve = training.get('curves', [{}])[-1]
                state.update(epoch=curve.get('epoch', 20), loss=curve.get('objective'),
                             stop_recall=curve.get('end_stop_recall'), resources=snapshot, updated=time.time())
                write_json(state_path, state)
                print(json.dumps({k: state[k] for k in ('status', 'epoch', 'loss', 'stop_recall', 'resources')}), flush=True)
                time.sleep(30)
        if child.returncode:
            raise RuntimeError(f'worker exit {child.returncode}; see worker.log')
        report()
        state.update(status='completed', finished=time.time())
        write_json(state_path, state)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished=time.time())
        write_json(state_path, state)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'report'))
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    {'prepare': prepare, 'run': run, 'report': report}[args.action]()

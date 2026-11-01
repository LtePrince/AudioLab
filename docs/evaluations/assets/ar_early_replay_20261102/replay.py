"""Matched inference of the previously selected early 4K A/S checkpoints.

All four models use original whole-song AR, the same frozen input tensors,
reference clocks and raw greedy decoding. No training or production promotion.
"""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/anchored_ar.py').exists())
sys.path.insert(0, str(ROOT))

import argparse
import gc
import json
import os
import statistics
from datetime import datetime, timezone
from types import SimpleNamespace

import torch

from docs.evaluations.assets.windowed_ar_20261031.summarize_natural import region_metrics
from script.train_anchored_ar import condition
from script.train_soft_onset import load_sample
from script.train_windowed_corpus import natural_reference, scoring_module
from src.anchored_ar import AnchoredAR, generate
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget

ASSET = Path(__file__).resolve().parent
OUT = ROOT / 'out/ar_early_replay_20261102'
CHECKS = ROOT / 'out/windowed_ar_20261031/natural_checks'
CACHE = ROOT / 'out/full_corpus_v1/cache'
SELECTION = ROOT / 'docs/evaluations/assets/phira_generalization10_20261029/selection.json'
PRIOR = ROOT / 'docs/evaluations/assets/runtime_selection15_20261026/selection.json'
ARCHIVE = ROOT / 'docs/evaluations/assets/ar_archive_20261030/manifest.json'
FILES = ['src/anchored_ar.py', 'src/models/chart_decoder.py', 'src/models/chart_history.py',
         'src/model_benchmark.py', 'src/audio_understanding/evaluation.py', 'src/soft_onset.py',
         'script/train_windowed_corpus.py', 'script/train_anchored_ar.py', 'script/train_soft_onset.py',
         'docs/evaluations/assets/phira_generalization10_20261029/benchmark.py',
         'docs/evaluations/assets/windowed_ar_20261031/summarize_natural.py',
         'benchmarks/v1/protocol.json', str(Path(__file__).relative_to(ROOT))]


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    destination = ASSET / 'selection.json'
    if destination.exists():
        raise RuntimeError('selection already frozen')
    original = read(PRIOR); archived = read(ARCHIVE); songs = read(SELECTION)
    checks = read(CHECKS / 'manifest.json'); cache = read(CACHE / 'cache.json')
    assert checks['selection_sha256'] == sha256(SELECTION)
    assert checks['frontend_sha256'] == songs['frozen_frontends']['4k']['sha256']
    for kind in ('4k', 'onset'):
        frontend = songs['frozen_frontends'][kind]
        assert sha256(frontend['path']) == frontend['sha256'] == cache['models'][kind]['sha256']
    candidates = []
    for variant, short, early_epoch in [('audio', 'A', 920), ('soft', 'S', 1000)]:
        selected = next(c for c in original['candidates']['4k'] if c['variant'] == variant and c['epoch'] == early_epoch)
        late = next(r for r in archived['runs'] if r['mode'] == '4k' and r['variant'] == variant)['resume']
        for epoch, path, expected in [(early_epoch, selected['checkpoint'], selected['sha256']),
                                      (1400, late['path'], late['sha256'])]:
            assert sha256(ROOT / path) == expected
            checkpoint = torch.load(ROOT / path, map_location='cpu', weights_only=True)
            assert checkpoint['epoch'] == epoch and checkpoint['config']['variant'] == variant
            cfg = checkpoint['config']['decoder_config']
            assert cfg.get('n_lanes', 4) == 4 and cfg.get('cross_window') is None
            candidates.append(dict(id=f'{short}{epoch:04d}', product='AL-AR4K v0.5.1', variant=variant,
                epoch=epoch, checkpoint=path, sha256=expected, config=checkpoint['config']))
            del checkpoint
        assert candidates[-1]['config']['decoder_config'] == candidates[-2]['config']['decoder_config']
    eltaw = next(row for row in cache['rows'] if row['id'] == 'Eltaw_IN')
    assert sha256(CACHE / eltaw['file']) == eltaw['sha256']
    selected_songs = [next(s for s in songs['songs'] if s['id'] == id) for id in (40717, 52417)]
    selected_songs += [s for s in songs['songs'] if s['id'] not in (40717, 52417)]
    for song in selected_songs:
        row = next(r for r in checks['rows'] if r['id'] == song['id'])
        assert sha256(CHECKS / row['file']) == row['sha256']
        assert sha256(ROOT / song['audio']) == song['audio_sha256']
        assert sha256(ROOT / song['reference']) == song['reference_sha256']
    plan = dict(schema='audiolab-early-checkpoint-replay-v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        candidates=candidates, songs=selected_songs, eltaw=eltaw,
        cache_manifest_sha256=sha256(CACHE / 'cache.json'), checks=checks,
        prior_selection_sha256=sha256(PRIOR), original_selection_sha256=sha256(SELECTION),
        source_sha256={p: sha256(ROOT / p) for p in FILES},
        controls=dict(training_updates=0, architecture='original whole-song AnchoredAR in all arms',
                      greedy=True, seed=42, max_tokens=12000, grid_repair=False,
                      bpm='frozen reference clock; no tempo estimation', positions_scored=False),
        selection_reason='Previously delivered A920 and S1000, chosen on 15 training songs before E20. '
                         'Compare each with its own A/S1400; do not select an early epoch based on new-song results.',
        input_scope='Eltaw uses original full-song training cache. All community inputs use the same '
                    'CPU-encoded frozen features/onsets from the window pilot. Not a bitwise replay of the original GPU-encoded E20.',
        scope='4K A/S only; E20 is development data. No free-x conclusion, new training or runtime replacement.')
    write_json(destination, plan)
    print(json.dumps(dict(candidates=[c['id'] for c in candidates], songs=11, inference_calls=44)), flush=True)


def gaps(reference, prediction, duration):
    truth = sorted({n['time_seconds'] for n in reference})
    edges = [0., *prediction['onsets_seconds'], duration]
    rows = []
    for start, end in zip(edges, edges[1:]):
        if end <= start:
            continue
        held = max((n['end_seconds'] for n in prediction['notes'] if n['time_seconds'] <= start), default=start)
        rows.append(dict(start=start, end=end, seconds=end-start,
                         reference_onsets=sum(start < t < end for t in truth),
                         no_active_note_seconds=max(0., end-max(start, held))))
    return sorted(rows, key=lambda r: r['seconds'], reverse=True)


@torch.inference_mode()
def evaluate(device_name):
    plan = read(ASSET / 'selection.json'); plan_hash = sha256(ASSET / 'selection.json')
    for p, digest in plan['source_sha256'].items():
        if sha256(ROOT / p) != digest:
            raise RuntimeError(f'frozen source changed: {p}')
    OUT.mkdir(exist_ok=True)
    device = torch.device(device_name)
    args = SimpleNamespace(min_free_disk_gib=8, min_available_ram_gib=8, max_rss_gib=4,
                           gpu_memory_fraction=.65, keep_checkpoints=1)
    budget = ResourceBudget(args, device, OUT); scorer = scoring_module()
    profile = read(ROOT / 'benchmarks/v1/protocol.json')['rhythm']
    complete = 0
    for candidate in plan['candidates']:
        path = ROOT / candidate['checkpoint']; assert sha256(path) == candidate['sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = AnchoredAR(checkpoint['config']['decoder_config']).to(device).eval()
        model.load_state_dict(checkpoint['model'], strict=True); del checkpoint
        destination = OUT / candidate['id']; destination.mkdir(exist_ok=True)
        for song in [dict(id='Eltaw_IN', name='Eltaw'), *plan['songs']]:
            budget.check(force=True)
            result_path = destination / f'{song["id"]}.json'
            if result_path.exists():
                assert read(result_path)['selection_sha256'] == plan_hash
                complete += 1
                continue
            if song['id'] == 'Eltaw_IN':
                assert sha256(CACHE / plan['eltaw']['file']) == plan['eltaw']['sha256']
                sample = load_sample(CACHE, plan['eltaw']); view = condition(sample)
                reference = natural_reference(sample); meta = sample['routes']['4k']['reference']
                clock = dict(bpm=meta['timebase_bpm'], beat_zero_seconds=meta['offset_seconds'],
                             source='training reference', verified=True)
                split = 'training_sanity'; del sample
            else:
                row = next(r for r in plan['checks']['rows'] if r['id'] == song['id'])
                assert sha256(CHECKS / row['file']) == row['sha256']
                view = torch.load(CHECKS / row['file'], map_location='cpu', weights_only=True)
                reference = read(ROOT / song['reference'])['notes']; clock = song['clock']; split = 'E20_development'
            prediction = generate(model, view, clock['bpm'], use_soft=candidate['variant']=='soft',
                                  greedy=True, seed=42, max_tokens=12000, check=budget.check)
            prediction.pop('trace', None)
            score = scorer.score(reference, prediction, clock, profile)
            intervals = gaps(reference, prediction, view['duration'])
            active = [g for g in intervals if g['reference_onsets'] > 0]
            record = dict(candidate=candidate['id'], song_id=song['id'], name=song['name'], split=split,
                          selection_sha256=plan_hash, checkpoint_sha256=candidate['sha256'],
                          status=prediction['status'], metrics=score, longest_gaps=intervals[:5],
                          longest_reference_active_gap=active[0] if active else None,
                          prediction=prediction, resources=budget.log_state())
            write_json(result_path, record); complete += 1
            progress = dict(status='running', done=complete, total=44, candidate=candidate['id'], song=song['name'],
                            result=prediction['status'], f1=score['onset_f1_50ms'], grid=score['grid_bonus'],
                            gap_seconds=active[0]['seconds'] if active else 0., resources=budget.log_state())
            write_json(OUT / 'progress.json', progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            del view, prediction, record
        del model; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(OUT / 'completed.json', dict(status='completed', done=complete, resources=budget.log_state()))


def summarize():
    plan = read(ASSET / 'selection.json'); rows = []; aggregates = []
    for candidate in plan['candidates']:
        part = [read(OUT / candidate['id'] / f'{id}.json') for id in ['Eltaw_IN', *[s['id'] for s in plan['songs']]]]
        for row in part:
            assert row['selection_sha256'] == sha256(ASSET / 'selection.json')
            row.pop('prediction'); rows.append(row)
        community = [r for r in part if r['split']=='E20_development']
        keys = ('onset_f1_25ms', 'onset_f1_50ms', 'onset_f1_100ms', 'grid_bonus', 'time_type_f1', 'tail_10s_recall')
        aggregates.append(dict(candidate=candidate['id'], songs=len(community),
            completed=sum(r['status']=='completed' for r in community),
            **{k: statistics.mean(r['metrics'][k] for r in community) for k in keys},
            median_largest_reference_active_gap_seconds=statistics.median(r['longest_reference_active_gap']['seconds'] for r in community)))
    fixed = []
    old = read(ROOT / 'docs/evaluations/assets/phira_gap_diagnosis_20261030/4k.json')
    for case in old['results']:
        short = case['run'].split('-')[-1]
        song = next(s for s in plan['songs'] if s['id']==case['song_id']); gap = case['largest_gaps'][0]
        item = dict(variant=short, song_id=song['id'], name=song['name'], start=gap['start'], end=gap['end'], models={})
        for candidate in plan['candidates']:
            if not candidate['id'].startswith(short):
                continue
            p = read(OUT / candidate['id'] / f'{song["id"]}.json')['prediction']
            item['models'][candidate['id']] = region_metrics(read(ROOT/song['reference'])['notes'], p, gap['start'], gap['end'])
        fixed.append(item)
    write_json(ASSET / 'summary.json', dict(schema='audiolab-early-checkpoint-replay-results-v1',
        selection_sha256=sha256(ASSET/'selection.json'), training_updates=0, aggregates=aggregates,
        rows=rows, fixed_gap_regions=fixed, scope=plan['scope']))
    lines = ['# 早期优选权重回测：A920 / S1000 与各自 1400', '',
        '用户提出回测 800–1000 轮表现较好的权重。本次使用当时 15 首训练曲已选出的 A920、S1000，'
        '并分别复跑 A1400、S1400。四组均为 AL-AR4K v0.5.1 原整曲 AR；没有套用分窗结构，也没有训练。', '',
        '原十首社区曲全部复测，另用 Eltaw 核对历史行为。所有组共享冻结音频特征、软起音、参考 BPM、'
        '12,000 token 预算与贪心解码；A 仍关闭额外起音输入，S 保持开启。社区特征来自同一 CPU 缓存，'
        '因此须比较本轮重跑的 1400，不能把旧 GPU 编码输出当成数值完全一致的对照。', '',
        '| 权重 | 十首起点 F1@50ms | 网格加分 | 时间＋类型 F1 | 自然结束 | 最长空档中位数 s |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for a in aggregates:
        lines.append(f"| {a['candidate']} | {a['onset_f1_50ms']:.4f} | {a['grid_bonus']:.4f} | {a['time_type_f1']:.4f} | "
                     f"{a['completed']}/10 | {a['median_largest_reference_active_gap_seconds']:.2f} |")
    lines += ['', '空档包含开头/结尾，只在区间内存在参考起点的空档中取每曲最大值；'
              'Hold 活跃覆盖、全部区间和固定旧缺口的 P/R 另列在 JSON，不用塞入一个键当作解决。', '',
              '| 曲目 / 权重 | 起点 F1@50ms | 网格 | 正常结束 | 最长参考有键空档 s |',
              '| --- | ---: | ---: | --- | ---: |']
    for id in ('Eltaw_IN', 40717, 52417):
        for r in [r for r in rows if r['song_id']==id]:
            lines.append(f"| {r['name']} / {r['candidate']} | {r['metrics']['onset_f1_50ms']:.4f} | "
                         f"{r['metrics']['grid_bonus']:.4f} | {r['status']=='completed'} | "
                         f"{r['longest_reference_active_gap']['seconds']:.2f} |")
    lines += ['', 'E20 已用于开发，不是盲测。参考谱是一个合理编排；时间＋类型指标不包含移动判定线坐标。'
              '此比较检验历史候选是否更好，不单独证明训练时长是唯一因果因素；未穷举该轮数区间所有权重，也不代表 free-x。', '',
              '[冻结选择、源码及权重 SHA](assets/ar_early_replay_20261102/selection.json)、'
              '[逐曲结果与固定缺口对照](assets/ar_early_replay_20261102/summary.json)。发布权重不替换。', '']
    (ROOT / 'docs/evaluations/ar_early_replay_20261102.md').write_text('\n'.join(lines))
    print(json.dumps(aggregates, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'evaluate', 'summarize'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); os.chdir(ROOT); torch.set_num_threads(2)
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'evaluate':
        evaluate(args.device); summarize()
    else:
        summarize()

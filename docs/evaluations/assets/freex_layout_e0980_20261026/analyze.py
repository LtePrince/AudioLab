"""Rank saved E18 Eltaw outputs and plot real continuous x; CPU replay only."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').is_file() and (p / 'src').is_dir())
sys.path.insert(0, str(ROOT))
import torch
from src.anchored_ar import residuals_by_note
from src.audio_understanding.evaluation import match_events
from src.data.chart_tokenizer import ChartTokenizer, bin_to_hit
from src.model_benchmark import rhythm_bonus
from src.pattern_diagnostics import KINDS

torch.set_num_threads(1)
BASE = ROOT / 'out/freex_anchored_20261025/continued1000'
DEST = ROOT / 'docs/evaluations/assets/freex_layout_e0980_20261026'
DEST.mkdir(parents=True, exist_ok=True)
CUTOFFS = {'audio': 1000, 'soft': 1000}
SONG = 'Eltaw_IN'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def groups(notes):
    result = defaultdict(list)
    for index, note in enumerate(notes):
        result[round(note['time_seconds'], 6)].append(index)
    return dict(sorted(result.items()))


def compare(reference, predicted):
    rg, pg = groups(reference), groups(predicted)
    rt, pt = list(rg), list(pg)
    pairs = match_events(rt, pt, .05)
    ri, pi = {t: i for i, t in enumerate(rt)}, {t: i for i, t in enumerate(pt)}
    matched_ref, matched_pred, mapped, exact = set(), set(), {}, {}
    dx, hold_errors = [], []
    for r, p in pairs:
        rr, pp = defaultdict(list), defaultdict(list)
        for i in rg[r]:
            n = reference[i]
            rr[n['position'], n['type']].append(i)
        for i in pg[p]:
            n = predicted[i]
            pp[n['position'], n['type']].append(i)
        mapped[ri[r]] = pi[p]
        exact[ri[r]] = Counter({k: len(v) for k, v in rr.items()}) == Counter({k: len(v) for k, v in pp.items()})
        for key in rr.keys() & pp.keys():
            left = sorted(rr[key], key=lambda i: reference[i]['screen_x'])
            right = sorted(pp[key], key=lambda i: predicted[i]['screen_x'])
            for a, b in zip(left, right):
                matched_ref.add(a)
                matched_pred.add(b)
                dx.append(abs(reference[a]['screen_x'] - predicted[b]['screen_x']))
                if key[1] == 'hold':
                    hold_errors.append(abs((reference[a]['end_seconds'] - reference[a]['time_seconds']) -
                                           (predicted[b]['end_seconds'] - predicted[b]['time_seconds'])))
    windows = max(0, len(rt) - 3)
    four_exact = sum(all(exact.get(i + k, False) and mapped.get(i + k) == mapped.get(i, -99) + k
                         for k in range(4)) for i in range(windows))
    tp = len(matched_ref)
    nr, np = len(reference), len(predicted)
    metrics = dict(reference_notes=nr, predicted_notes=np, matched_notes=tp,
                   onset_f1=2 * len(pairs) / (len(rt) + len(pt)),
                   joint_time_bin_type_f1=2 * tp / (nr + np),
                   exact_groups_given_matched=sum(exact.values()) / len(pairs) if pairs else 0,
                   four_group_reproduction=four_exact / windows if windows else None,
                   matched_coordinate_mae_screen=statistics.mean(dx) if dx else None,
                   matched_holds=len(hold_errors),
                   matched_hold_duration_mae_ms=1000 * statistics.mean(hold_errors) if hold_errors else None,
                   type_counts=dict(Counter(n['type'] for n in predicted)))
    return metrics, matched_ref, matched_pred


manifest_path = ROOT / 'out/full_corpus_v1/cache/cache.json'
manifest = json.loads(manifest_path.read_text())
entry = next(r for r in manifest['rows'] if r['id'] == SONG)
cache_path = manifest_path.parent / entry['file']
assert sha(cache_path) == entry['sha256']
sample = torch.load(cache_path, map_location='cpu', weights_only=True)
route = sample['routes']['freex']
metadata = route['reference']
residuals = residuals_by_note(route)
scale = 60 / (32 * metadata['timebase_bpm'])
reference = [dict(time_seconds=n.tick * scale + metadata['offset_seconds'],
                  end_seconds=(n.tick + n.dur) * scale + metadata['offset_seconds'],
                  position=n.lane, type=KINDS[n.type], screen_x=bin_to_hit(n.lane, residuals[n.tick, n.lane]))
             for n in ChartTokenizer(64).decode_tokens(route['tokens'].tolist(), strict=True)]
duration = float(sample['duration'])
del sample, route
self_check, _, _ = compare(reference, reference)
assert self_check['joint_time_bin_type_f1'] == self_check['four_group_reproduction'] == 1
assert self_check['matched_coordinate_mae_screen'] == self_check['matched_hold_duration_mae_ms'] == 0
protocol_path = ROOT / 'benchmarks/v1/protocol.json'
protocol = json.loads(protocol_path.read_text())
records, source_files = [], []
for variant, cutoff in CUTOFFS.items():
    for path in sorted((BASE / variant).glob('evaluation-e*/inputs.json')):
        epoch = int(path.parent.name.removeprefix('evaluation-e'))
        if epoch > cutoff or not path.with_name('diagnostics.json').is_file():
            continue
        digest = sha(path)
        source_files.append(dict(path=str(path.relative_to(ROOT)), sha256=digest))
        data = json.loads(path.read_text())
        for run in data['runs']:
            case = next(c for c in run['cases'] if c['id'] == SONG)
            assert set(groups(reference)) == {round(t, 6) for t in case['events_seconds']}
            prediction = run['predictions'][SONG]
            assert all(math.isfinite(n['screen_x']) and 0 <= n['screen_x'] <= 1 for n in prediction['notes'])
            metrics, _, _ = compare(reference, prediction['notes'])
            assert abs(metrics['onset_f1'] - prediction['f1_50ms']) < 1e-12
            metrics['grid_bonus'] = rhythm_bonus(prediction['onsets_seconds'], case['clock'], protocol['rhythm'])['bonus_per_onset']
            metrics['last_onset_delta_seconds'] = (prediction['last_onset_seconds'] - max(case['events_seconds'])
                                                   if prediction['last_onset_seconds'] is not None else None)
            records.append(dict(variant=variant, epoch=epoch, decoding=run['id'].rsplit('/', 1)[-1],
                                status=prediction['status'], metrics=metrics,
                                input_path=str(path.relative_to(ROOT)), input_sha256=digest,
                                source_checkpoint_sha256=run['metadata']['checkpoint_sha256']))

selected = []
for variant in CUTOFFS:
    eligible = [r for r in records if r['variant'] == variant and r['status'] == 'completed']
    best = max(eligible, key=lambda r: (r['metrics']['joint_time_bin_type_f1'],
                                      r['metrics']['four_group_reproduction'],
                                      r['metrics']['onset_f1'],
                                      -abs(r['metrics']['last_onset_delta_seconds']), r['epoch']))
    snapshot = BASE / variant / f"model_e{best['epoch']:04d}.pt"
    sidecar = json.loads(snapshot.with_suffix('.json').read_text())
    assert sha(snapshot) == sidecar['sha256']
    assert sidecar['source_checkpoint_sha256'] == best['source_checkpoint_sha256']
    best = dict(best, snapshot=str(snapshot.relative_to(ROOT)), snapshot_sha256=sidecar['sha256'])
    selected.append(best)

summary = dict(schema='audiolab-freex-layout-replay-v1', observed_at_utc=datetime.now(timezone.utc).isoformat(),
               product='AL-FreeX v0.4.0', experiment='E18', song=SONG, cutoffs=CUTOFFS,
               scope='One seen training song. Ranking stored greedy and seed42 outputs only; no inference or time snapping. Does not rank generalization or musical quality.',
               selection='Only completed outputs; maximize full-song time+64-bin+type note F1, then consecutive-four-group reproduction, onset F1, smaller tail error, later epoch.',
               definitions=dict(joint_f1='Existing 50ms one-to-one unique-onset matching, then same 64-bin position and type within each group. Denominator includes all notes. Excludes Hold duration.',
                                continuous_x='Plot actual bin_to_hit(bin,residual)=0.05+0.9*(bin+residual)/64. Matched-coordinate MAE is separate, no extra coordinate threshold.',
                                sequence='Four consecutive reference groups with identical bin+type multisets and consecutive prediction groups; denominator is all reference four-group windows.',
                                differences='Red rings: prediction note lacking a time+bin+type match. Gray crosses: unmatched reference note. Not a verdict on alternate valid compositions; excludes Hold-length differences.'),
               reference=dict(cache_path=str(cache_path.relative_to(ROOT)), cache_sha256=entry['sha256'],
                              manifest_sha256=sha(manifest_path), metadata=metadata, duration_seconds=duration,
                              notes=len(reference), type_counts=dict(Counter(n['type'] for n in reference))),
               analysis_script_sha256=sha(Path(__file__)), protocol_sha256=sha(protocol_path),
               control=self_check, compared_outputs=len(records), input_files=source_files,
               selected=selected, results=records)
(DEST / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
fields = ['variant', 'epoch', 'decoding', 'status', 'joint_time_bin_type_f1', 'onset_f1', 'grid_bonus',
          'four_group_reproduction', 'last_onset_delta_seconds', 'predicted_notes']
with (DEST / 'ranking.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in records:
        writer.writerow({key: row[key] if key in row else row['metrics'][key] for key in fields})
for row in selected:
    print('SELECTED', json.dumps({k: v for k, v in row.items() if k not in ('input_sha256', 'source_checkpoint_sha256', 'snapshot_sha256')}, ensure_ascii=False), flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
colors = dict(tap='#1685bc', drag='#cb9200', hold='#8756cf', flick='#da477b')
markers = dict(tap='o', drag='D', hold='|', flick='^')
rows = [('训练用 free-x 参考谱', reference, None)]
for item in selected:
    data = json.loads((ROOT / item['input_path']).read_text())
    run = next(r for r in data['runs'] if r['id'].endswith('/' + item['decoding']))
    title = f"{'A · 纯音频' if item['variant'] == 'audio' else 'S · 软起音'} / e{item['epoch']:04d} / {'贪心' if item['decoding'] == 'greedy' else 'seed42 采样'}"
    rows.append((title, run['predictions'][SONG]['notes'], item))

def plot(start, end, title, annotate=True):
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.subplots_adjust(top=.82, bottom=.10, left=.065, right=.98, hspace=.40)
    fig.suptitle(title, x=.065, ha='left', y=.98, fontsize=21, weight='bold')
    fig.text(.065, .94, 'AL-FreeX v0.4.0 / E18 · 已保存输出的整曲联合音符 F1 最优 · 比较范围：A/S均至1000轮', color='#475569', fontsize=11)
    fig.text(.065, .91, '横轴：音频时间；纵轴：真实连续横坐标（0=屏幕左，1=屏幕右）。紫色横线为 Hold 持续时间。', color='#475569', fontsize=11)
    handles = [Line2D([], [], color=colors[k], marker=markers[k], linestyle='-' if k == 'hold' else 'None', label=k.title(), markersize=7) for k in colors]
    if annotate:
        handles += [Line2D([], [], marker='o', color='#dc2626', markerfacecolor='none', linestyle='None', label='生成音符不匹配'),
                    Line2D([], [], marker='x', color='#64748b', linestyle='None', label='未匹配的参考音符')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.058, .892), ncol=6, frameon=False)
    for ax, (label, notes, record) in zip(axes, rows):
        ax.set_facecolor('#f8fafc')
        ax.set_ylim(-.02, 1.02)
        ax.set_xlim(start, end)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_ylabel('屏幕横坐标')
        step = 2 if end - start <= 31 else 10
        ax.set_xticks([x for x in range(math.ceil(start / step) * step, math.floor(end / step) * step + 1, step)])
        ax.grid(alpha=.20)
        ax.spines[['top', 'right']].set_visible(False)
        if record:
            metric = record['metrics']
            label += f"  |  整曲联合 F1 {metric['joint_time_bin_type_f1']:.3f} · 起点 F1 {metric['onset_f1']:.3f} · {len(notes)} 音符"
        else:
            label += f'  |  {len(notes)} 音符'
        ax.set_title(label, loc='left', fontsize=11, pad=8)
        for kind in colors:
            selected_notes = [n for n in notes if n['type'] == kind and start <= n['time_seconds'] <= end]
            if kind == 'hold':
                for note in notes:
                    if note['type'] == 'hold' and note['end_seconds'] >= start and note['time_seconds'] <= end:
                        ax.plot([max(start, note['time_seconds']), min(end, note['end_seconds'])], [note['screen_x']] * 2,
                                color=colors[kind], lw=4, alpha=.55, solid_capstyle='butt', zorder=2)
            ax.scatter([n['time_seconds'] for n in selected_notes], [n['screen_x'] for n in selected_notes],
                       color=colors[kind], marker=markers[kind], s=(22 if kind != 'hold' else 95), linewidths=.9, zorder=4)
        if record and annotate:
            _, mr, mp = compare(reference, notes)
            extra = [n for i, n in enumerate(notes) if i not in mp and start <= n['time_seconds'] <= end]
            missing = [n for i, n in enumerate(reference) if i not in mr and start <= n['time_seconds'] <= end]
            ax.scatter([n['time_seconds'] for n in missing], [n['screen_x'] for n in missing], marker='x', c='#64748b',
                       s=28, linewidths=.8, alpha=.55, zorder=3)
            ax.scatter([n['time_seconds'] for n in extra], [n['screen_x'] for n in extra], marker='o', facecolors='none',
                       edgecolors='#dc2626', s=62, linewidths=.8, alpha=.8, zorder=5)
    axes[-1].set_xlabel('音频时间（秒）')
    fig.text(.065, .040, '联合匹配：起点 ±50ms + 相同64位置bin + 键型；Hold长度另计。参考谱是训练目标，并非唯一正确编排。', color='#475569', fontsize=10)
    return fig

fig = plot(0, 30, 'Eltaw 前30秒：A/S 980轮贪心排布')
fig.savefig(DEST / 'eltaw_intro.png', dpi=160, facecolor='white')
fig.savefig(DEST / 'eltaw_intro.pdf', facecolor='white')
plt.close(fig)
fig = plot(0, duration, 'Eltaw 整曲：A/S 980轮贪心排布', annotate=False)
fig.savefig(DEST / 'eltaw_full.png', dpi=160, facecolor='white')
plt.close(fig)
with PdfPages(DEST / 'eltaw_full_detail.pdf') as pdf:
    for start in range(0, math.ceil(duration), 30):
        end = min(start + 30, duration)
        fig = plot(start, end, f'Eltaw {start}–{end:.1f}秒：A/S 980轮贪心排布')
        pdf.savefig(fig, facecolor='white')
        plt.close(fig)
assert sum(p.stat().st_size for p in DEST.iterdir() if p.is_file()) < 10 * 2**20
print('PASS: reference identity, saved-onset F1 agreement, immutable snapshot hashes, artifact budget <10 MiB.', flush=True)

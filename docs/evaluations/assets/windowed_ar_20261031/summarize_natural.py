"""Report the matched natural-song pilot, including the original gap regions."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

import json
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json

BASE=ROOT/'out/windowed_ar_20261031'
ASSET=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())


def region_metrics(reference,prediction,start,end):
    r=sorted({n['time_seconds'] for n in reference if start<n['time_seconds']<end})
    p=[t for t in prediction['onsets_seconds'] if start<t<end]
    matched=len(match_events(r,p,.05));edges=[start,*p,end]
    uncovered=[]
    for a,b in zip(edges,edges[1:]):
        held=max((n['end_seconds'] for n in prediction['notes'] if n['time_seconds']<=a),default=a)
        uncovered.append(max(0,b-max(a,held)))
    return dict(reference_onsets=len(r),predicted_onsets=len(p),matched_50ms=matched,
        precision=matched/len(p) if p else 0.,recall=matched/len(r) if r else 1.,
        largest_onset_gap_seconds=max(b-a for a,b in zip(edges,edges[1:])),
        longest_no_active_note_seconds=max(uncovered,default=0),predictions=p)


def main():
    result=dict(schema='audiolab-natural-window-pilot-comparison-v1',runs={},fixed_gap_regions=[])
    for arch in ('whole','window'):
        directory=BASE/f'natural_{arch}';training=read(directory/'training.json')
        if training['status']!='completed' or training['curves'][-1]['epoch']!=20:
            raise RuntimeError(f'{arch} not finished')
        final=read(directory/'evaluation-e0020/summary.json')
        if not final['full'] or len(final['rows'])!=25:raise RuntimeError('full 15+10 evaluation required')
        result['runs'][arch]=dict(config=training['config'],checkpoint_sha256=sha256(directory/'model_latest.pt'),
            final_training=training['curves'][-1],curves=training['curves'],resources=training['resources'],
            evaluation=final,first_epoch=training['curves'][0])
    selection=read(ROOT/'docs/evaluations/assets/phira_generalization10_20261029/selection.json')
    old=read(ROOT/'docs/evaluations/assets/phira_gap_diagnosis_20261030/4k.json')
    for case in [r for r in old['results'] if r['run']=='E14-A']:
        song=next(s for s in selection['songs'] if s['id']==case['song_id'])
        notes=read(ROOT/song['reference'])['notes'];gap=case['largest_gaps'][0]
        item=dict(id=song['id'],name=song['name'],start=gap['start'],end=gap['end'],original_gap=gap,
            original_source=case['source_result'],original_source_sha256=case['source_result_sha256'])
        for arch in ('whole','window'):
            prediction=read(BASE/f'natural_{arch}/evaluation-e0020/{song["id"]}.json')['prediction']
            item[arch]=region_metrics(notes,prediction,gap['start'],gap['end'])
        result['fixed_gap_regions'].append(item)
    result['scope']='Two matched 4K A warm-started runs; all300 training. Original two A gap regions only, '
    result['scope']+='not the four-model eight-region gate. E20 is development data; no blind-test claim or runtime replacement.'
    write_json(ASSET/'natural_result.json',result)
    lines=['# 全 300 首：整曲／分窗输出的同父权重适配','',
        '两组各完成 20 轮、6,000 次更新；每轮包含全部 300 首整曲，以及原有的两份物理片段和真实静音。'
        '两组从同一个只读 A1400 父权重开始，均重置 AdamW、恒定 LR 3e-5。W1 只增加 10 秒输出窗口协议，'
        '没有混入条件键型头、音频解冻、扩容或历史噪声。这是匹配的协议适配试验，不是从零训练的能力上限测量。','',
        '## 训练与资源','',
        '| 组别 | 首轮 / 末轮 objective | 末轮帧正确率 | 末轮停止召回 | 末轮误停率 | RSS / 显存峰值 GiB |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for arch,r in result['runs'].items():
        t=r['final_training'];s=r['resources']
        lines.append(f"| {arch} | {r['first_epoch']['objective']:.5f} / {t['objective']:.5f} | {t['frame_accuracy']:.4f} | {t['end_stop_recall']:.4f} | {t['false_stop_rate']:.5f} | {s['rss_gib']:.2f} / {s['gpu_peak_allocated_gib']:.2f} |")
    lines+=['','每轮两组均有 270,200 个时间目标与 633,694 个排布标签；停止标签为 W0 1,200、W1 6,712。'
        'W1 的停止是窗口边界监督，不能与 W0 的整曲 EOS 视为同一种语义。'
        '只保存 latest 和一个 best-fit，统一评价第 20 轮，不按开发集选择最优轮。','',
        '## 原始贪心生成','',
        '| 组别 / 数据 | 起点 F1@25ms | @50ms | @100ms | 网格加分 | 时间＋类型 F1 | 正常完成 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for arch,r in result['runs'].items():
        for split,m in sorted(r['evaluation']['aggregates'].items()):
            lines.append(f"| {arch} / {split} | {m['onset_f1_25ms']:.4f} | {m['onset_f1_50ms']:.4f} | {m['onset_f1_100ms']:.4f} | {m['grid_bonus']:.4f} | {m['time_type_f1']:.4f} | {m['completed']}/{m['examples']} |")
    lines+=['','15 首训练曲使用 Eltaw＋20261026 固定随机 14 首；E20 十首已经用于开发，是回归集。'
        '参考谱只是一个合理编排；时间＋类型不要求动态判定线的坐标一致。'
        '训练曲另外计算含轨道的联合 F1，失败记零：'+', '.join(f"{a} {r['evaluation']['aggregates']['train15']['completed_only_joint_f1_50ms']:.4f}" for a,r in result['runs'].items())+'。','',
        '两组使用相同原时钟与输入，无起音阈值、网格吸附或导出修复。十首开发曲由原冻结编码器在 CPU 预编码一次、两组共享；'
        '不声称逐位重现此前 GPU 编码的旧生成轨迹。分窗“完成”要求每个窗都学会 END_WINDOW；调度走完全曲不是学会全曲 EOS。','',
        '## 原先的两处 A 缺音区间','',
        '| 歌曲 / 固定区间 | 组别 | 输出 / 参考起点 | 匹配 P / R @50ms | 区间内最长起点空档 s | 最长无活动音符 s |',
        '| --- | --- | --- | --- | ---: | ---: |']
    for item in result['fixed_gap_regions']:
        for arch in ('whole','window'):
            m=item[arch]
            lines.append(f"| {item['name']} / {item['start']:.2f}–{item['end']:.2f}s | {arch} | {m['predicted_onsets']}/{m['reference_onsets']} | {m['precision']:.3f}/{m['recall']:.3f} | {m['largest_onset_gap_seconds']:.2f} | {m['longest_no_active_note_seconds']:.2f} |")
    lines+=['','这只覆盖原八处诊断中的两处 4K A 区间，不替 S/free-x 验收。每首另保存包括开头/结尾的最长空档和 Hold 活跃覆盖，'
        '避免只在旧缺口塞入一个键或把缺口移到其他段落就宣称成功。','',
        '[完整曲线、逐曲指标、固定区间事件、SHA256 与资源](assets/windowed_ar_20261031/natural_result.json)。'
        '20 轮是适配观察点；若末期仍明显学习或生成不稳定，不据此判定架构上限。发布权重与 main 未替换。','']
    (ROOT/'docs/evaluations/windowed_ar_natural_20261031.md').write_text('\n'.join(lines))
    print(json.dumps({k:v['evaluation']['aggregates'] for k,v in result['runs'].items()},ensure_ascii=False))


if __name__=='__main__':main()

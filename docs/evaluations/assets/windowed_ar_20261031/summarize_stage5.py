"""Reproduce stage-5 tables from fixed final checkpoints, not selected epochs."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

from collections import Counter
import json

from src.audio_understanding.inference import sha256
from src.model_benchmark import rhythm_bonus
from src.product.artifacts import write_json

BASE=ROOT/'out/windowed_ar_20261031'
ASSET=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())


def aggregate(rows,protocol):
    result={}
    clock=dict(bpm=120.,beat_zero_seconds=0.,source='constructed physical clock',verified=True)
    for split in sorted({r['split'] for r in rows}):
        subset=[r for r in rows if r['split']==split];positive=[r for r in subset if r['reference_onsets']]
        out=dict(examples=len(subset),positive_examples=len(positive),completed=sum(r['status']=='completed' for r in subset),
            silence_extra_onsets=sum(r['generated_onsets'] for r in subset if not r['reference_onsets']),
            forced_windows=sum(r['forced_windows'] for r in subset),reference_onsets=sum(len(r['reference_onsets']) for r in subset),
            predicted_onsets=sum(r['generated_onsets'] for r in subset),
            onset_f1={ms:sum(r['onset'][ms]['f1'] for r in positive)/len(positive) for ms in ('25','50','100')},
            grid_bonus=sum(rhythm_bonus(r['onsets_seconds'],clock,protocol['rhythm'])['bonus_per_onset'] for r in positive)/len(positive),
            reference_grid_bonus=sum(rhythm_bonus(r['reference_onsets'],clock,protocol['rhythm'])['bonus_per_onset'] for r in positive)/len(positive))
        if split!='fixed_probe':
            out['joint_f1_25ms']=sum(r['joint_f1_25ms'] if r['status']=='completed' else 0. for r in positive)/len(positive)
        teacher=Counter();zero=[]
        for r in subset:
            teacher.update(r.get('teacher',{}))
            if 'feature_zeroing_loss' in r:zero.append(r['feature_zeroing_loss']['delta'])
        if teacher:
            out['teacher_counts']=dict(teacher)
            out['teacher_continue_and_time_25ms']=teacher['continue_and_time_correct_25ms']/max(1,teacher['events'])
            out['teacher_layout_accuracy']=teacher['layout_correct']/max(1,teacher['layout_total'])
        if zero:out['mean_loss_increase_zeroed_features']=sum(zero)/len(zero)
        # A visited window can be empty despite reference activity. This is not
        # erased by scheduler traversal or declared successful END_WINDOW.
        ref_active=empty=0
        for r in subset:
            for w in r['windows']:
                if any(w['start']<=t<w['end'] for t in r['reference_onsets']):
                    ref_active+=1
                    empty+=not any(w['start']<=t<w['end'] for t in r['onsets_seconds'])
        out.update(reference_active_windows=ref_active,empty_reference_active_windows=empty)
        result[split]=out
    return result


def main():
    protocol=read(ROOT/'benchmarks/v1/protocol.json');runs={}
    for name,folder in [('W1','window'),('K1','kind_position'),('E1','encoder_tail'),('C1','capacity512')]:
        directory=BASE/folder;training=read(directory/'training.json')
        if training['status']!='completed' or training['curves'][-1]['epoch']!=400:
            raise RuntimeError(f'{name} final training not complete')
        evaluation=read(directory/'evaluation-e0400.json')
        if not evaluation['full'] or len(evaluation['rows'])!=69:raise RuntimeError('full fixed evaluation required')
        runs[name]=dict(checkpoint=str((directory/'model_latest.pt').relative_to(ROOT)),
            checkpoint_sha256=sha256(directory/'model_latest.pt'),evaluation_sha256=sha256(directory/'evaluation-e0400.json'),
            config=training['config'],final_training=training['curves'][-1],resources=training['resources'],
            metrics=aggregate(evaluation['rows'],protocol))
    patterns={}
    for name,folder in [('P0','pattern_base'),('P1','pattern_kind_position')]:
        directory=BASE/folder;training=read(directory/'training.json');evaluation=read(directory/'evaluation-e0400.json')
        if training['status']!='completed' or not evaluation['full'] or len(evaluation['rows'])!=80:
            raise RuntimeError('pattern final comparison not complete')
        patterns[name]=dict(config=training['config'],checkpoint=str((directory/'model_latest.pt').relative_to(ROOT)),
            checkpoint_sha256=sha256(directory/'model_latest.pt'),evaluation_sha256=sha256(directory/'evaluation-e0400.json'),
            final_training=training['curves'][-1],resources=training['resources'],metrics=evaluation['aggregates'])
    result=dict(schema='audiolab-stage5-isolated-comparisons-v1',runs=runs,patterns=patterns,
        protocol_sha256=sha256(ROOT/'benchmarks/v1/protocol.json'),
        scope='Controlled tasks; no natural-song quality claim. Baseline reused at same epoch/data/schedule. '
            'Pattern times/count/Tap type given; only suffix positions generated.')
    write_json(ASSET/'stage5_result.json',result)
    lines=['# 第 5 阶段：键型、音频适配、容量与模式续写','',
        '三项独立对照及两组模式训练均完成 400 轮。生成任务每组 12,800 次更新；模式任务每组 16,000 次更新。'
        '这是构造数据上的架构归因，没有把 32 条诊断训练写成全 300 首自然歌曲训练。','',
        '## 模型修改与控制条件','',
        '| 运行 | 改动 | 参数 |','| --- | --- | ---: |',
        f"| W1 | 原联合 NOTE 输出，冻结音频编码器，复用第 2 阶段完整对照 | {runs['W1']['config']['parameters']:,} |",
        f"| K1 | 条件输出 p(action) p(type\\|NOTE) p(position\\|type,NOTE) | {runs['K1']['config']['parameters']:,} |",
        f"| E1 | 仅解冻最后一个全局双向音频 block 与末端 LayerNorm | {runs['E1']['config']['parameters']:,} |",
        f"| C1 | 解码器宽度 256→512，深度 6、头数 8 不变 | {runs['C1']['config']['parameters']:,} |",'',
        '各组全曲原分辨率音频、10 秒输出责任窗、完整因果历史、四键型、多押及 Hold 表示不变。'
        '没有起音阈值、硬节拍吸附或键型比例配额。K1 分解输出概率，输入 token 流保持不变；'
        '不是原先 factorized 输入嵌入开关。E1 缓存的是末层之前的全曲激活，末层每步实际重算并接收谱面梯度。'
        '69 条样本的末层重建及原半精度缓存误差均为零。编码器 LR 3e-5，其余 3e-4，使用相同 5% warmup/cosine。','',
        '小模型共享部分使用相同随机初始化；C1 张量尺寸改变，只保证 seed、数据、曝光与调度相同。'
        '统一评第 400 轮，未按随机检查集挑轮数。W1 在物理区间浮点修复前训练，但固定数据的目标帧索引全部不变。','',
        '## 自由成谱结果','',
        '起点为非静音样本的宏平均 F1@25ms；联合要求时间、位置与类型一致，失败整谱记零。'
        '新组合为 32 条中的 28 条有音样本；固定探针的虚构类型和位置从不评分。','',
        '| 运行 | 训练起点 / 联合 | 新组合起点 | 新组合联合 | 新组合结束 | 固定起点 | 新组合网格加分 |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for name,r in runs.items():
        tr=r['metrics']['train'];v=r['metrics']['random_check'];p=r['metrics']['fixed_probe']
        lines.append(f"| {name} | {tr['onset_f1']['25']:.4f} / {tr['joint_f1_25ms']:.4f} | {v['onset_f1']['25']:.4f} | {v['joint_f1_25ms']:.4f} | {v['completed']}/32 | {p['onset_f1']['25']:.4f} | {v['grid_bonus']:.4f} |")
    lines+=['','网格只加分，不决定输出排布。调度器走完整曲不代表学会整曲 EOS；结束要求每窗均由模型 END_WINDOW 结束。','',
        '| 运行 | 新组合有参考却为空的窗口 | 静音额外起点（训练/新组合/固定） | 训练秒数 | 显存峰值 GiB |',
        '| --- | --- | --- | ---: | ---: |']
    for name,r in runs.items():
        v=r['metrics']['random_check'];quiet='/'.join(str(r['metrics'][s]['silence_extra_onsets']) for s in ('train','random_check','fixed_probe'))
        lines.append(f"| {name} | {v['empty_reference_active_windows']}/{v['reference_active_windows']} | {quiet} | {r['final_training']['elapsed_seconds']:.1f} | {r['resources']['gpu_peak_allocated_gib']:.3f} |")
    lines+=['','## 明确前缀的模式续写','',
        '输入 24 个给定起点，每个一个 Tap；前 8 个谱面位置可见，自回归续写后 16 个。'
        '同一音频有五种不同模式；模型没有模式 ID 或未来位置提示。训练/新组合各 40 条、各 8 段共享音频；'
        '新组合使用未入训的轨道排列。没有输出周期约束，跨窗仍保留完整历史。'
        '该任务检验位置续写，不能据此声称时间、密度、类型选择已经解决。','',
        '| 任务 | P0 联合头：位置 / 整段正确 | P1 条件头：位置 / 整段正确 |',
        '| --- | --- | --- |']
    for split in ('train','new_combinations'):
        for family in ('','/jack','/short_jacks','/long_jacks','/alternation','/staircase'):
            key=split+family;a=patterns['P0']['metrics'][key];b=patterns['P1']['metrics'][key]
            lines.append(f"| {key} | {a['position_accuracy']:.4f} / {a['exact_suffix']}/{a['examples']} | {b['position_accuracy']:.4f} / {b['exact_suffix']}/{b['examples']} |")
    lines+=['','## 归因边界','',
        '清零音频特征的损失变化及真实历史下的时间/停止/排布计数见原始汇总。'
        '清零特征测依赖性，真正静音另外使用静音波形；两者不能互换。'
        '若训练集拟合但新组合仍差，不能把“结构能记住”当“已学会泛化编谱”；'
        '若给定时间与前缀的模式能续写，只说明当前 self-attention 可以利用该模式历史，'
        '不能自动推导无前缀自然生成也有同样能力。','',
        '[完整数据、参数、checkpoint/source SHA256、资源与诊断](assets/windowed_ar_20261031/stage5_result.json)。',
        '以上均为单 seed、有限构造分布。自然歌曲全量训练与缺音回归仍须单独执行；产品权重不自动替换。','']
    (ROOT/'docs/evaluations/windowed_ar_stage5_20261031.md').write_text('\n'.join(lines))
    print(json.dumps({k:v['metrics'] for k,v in runs.items()},ensure_ascii=False))


if __name__=='__main__':main()

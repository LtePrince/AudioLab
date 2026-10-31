"""Run the prepared stage-4 diagnostic controls, then write an evidence report.

Two GPU workers at a time; A completes before separate S weights begin. The
fixed recipe is W0/W1 without history noise: H1 has not earned promotion merely
by reducing its training loss. This does not launch natural full300 training.
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
FILES=('script/train_windowed_freex.py','script/train_windowed_ar.py','script/train_models.py',
       'src/windowed_ar.py','src/history_time_noise.py','src/anchored_ar.py',
       'src/models/chart_decoder.py','src/models/chart_history.py','script/train_grounded_ar.py',
       'docs/evaluations/assets/windowed_ar_20261031/run_freex_controls.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def report(results):
    asset=ROOT/'docs/evaluations/assets/windowed_ar_20261031/freex_controlled_result.json'
    write_json(asset,dict(schema='audiolab-controlled-freex-comparison-v1',runs=results,
        scope='Constructed mapping only; full-resolution whole audio. No natural full300 training or product replacement.'))
    lines=['# 第 4 阶段：free-x 分窗与独立 A/S 对照','',
        '四组各 400 轮、12,800 次更新完成。下表是 32 条构造训练样本及 32 条未入训随机组合的诊断；'
        '不代表自然歌曲的可玩性。原 5 个固定起音探针另外评测，静音不混入起点宏平均。','',
        '## 修改方案与公平条件','',
        'W0 维持整曲输出，W1 加入窗口条件、窗口内 pointer 和 END_WINDOW 监督。'
        '音频仍使用原 free-x 编码器的整曲特征，每步均能看到全曲。64 bin 与连续 residual、'
        '跨窗 Hold 和因果坐标反馈均保留。A 仅音频，S 另外输入连续软起音，各自独立权重，'
        '没有硬起音门限。每对主干同随机初始化、相同数据曝光和学习率；本轮都不加历史噪声，'
        '从而单独检查输出协议的迁移。','',
        '构造音频与 4K 对照相同；位置由已知音色及可听增益确定，'
        '用于检查连续坐标的可学习性，不是自然音乐的编排规则。'
        '每轮同样 140 个时间组，W0/W1 的停止标签分别为 32/80。','',
        '训练前审计原全 300 首 free-x 标签：217,391 个起点组、264,872 个音符全部保留，'
        '64-bin residual 逐项不变，时间往返最大误差 0。修正了 Drop It_IN 的浮点帧边界：'
        '统一使用同一组物理边界查找帧，避免除法舍入把边界前一个浮点数分到下一帧。'
        '没有移动参考起点或吸附网格；原 69 条 4K 受控样本的 292 个时间目标帧完全不变。'
        '[全量 codec 审计](assets/windowed_ar_20261031/freex_corpus_codec_audit.json)、'
        '[数值回归证据](assets/windowed_ar_20261031/physical_bin_boundary_fix.json)。','',
        '| 组别 | 参数 | 训练起点 F1@25ms | 随机起点 F1@25ms | 随机联合 F1 | 随机正常结束 | 固定探针起点 F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        s=r['evaluation'];a=s['train'];b=s['random_check'];c=s['fixed_probe']
        lines.append(f"| {name} | {r['config']['parameters']:,} | {a['positive_macro_raw_onset_f1_25ms']:.4f} | "
            f"{b['positive_macro_raw_onset_f1_25ms']:.4f} | {b['positive_macro_joint_f1_25ms']:.4f} | "
            f"{b['completed']}/{b['examples']} | {c['positive_macro_raw_onset_f1_25ms']:.4f} |")
    lines+=['','联合 F1 要求时间、位置 bin、类型一致，未正常结束样本记零。'
        'W1 正常结束要求每个窗口都由模型 END_WINDOW 结束；调度器走完整曲不能冒充模型学会全曲 EOS。','',
        '## 坐标、静音与资源','',
        '| 组别 | 随机匹配音符数 | 匹配音符 x MAE（屏幕宽度比例） | 训练联合 F1 | 静音额外起点（训练/随机/探针） |',
        '| --- | ---: | ---: | ---: | --- |']
    for name,r in results.items():
        s=r['evaluation'];b=s['random_check'];error=b['matched_screen_x_mae']
        lines.append(f"| {name} | {b['matched_notes']} | {error:.6f} | "
            f"{s['train']['positive_macro_joint_f1_25ms']:.4f} | "
            +'/'.join(str(s[k]['silence_extra_onsets']) for k in ('train','random_check','fixed_probe'))+' |'
            if error is not None else f'| {name} | 0 | 无匹配，不计算 | — | — |')
    lines+=['','坐标 MAE 只覆盖已匹配时间/bin/类型的音符，必须与匹配数共同看，不能掩盖漏音或错位。'
        '缓存特征训练的资源峰值不等于整曲服务峰值；各组完整计数、checkpoint SHA256、数据/源码 SHA256 '
        '和峰值见 [原始汇总](assets/windowed_ar_20261031/freex_controlled_result.json)。','',
        '## 归因与下一步','']
    for condition in ('A','S'):
        a=results[f'{condition}-W0']['evaluation'];b=results[f'{condition}-W1']['evaluation']
        delta=b['random_check']['positive_macro_raw_onset_f1_25ms']-a['random_check']['positive_macro_raw_onset_f1_25ms']
        probe=b['fixed_probe']['positive_macro_raw_onset_f1_25ms']-a['fixed_probe']['positive_macro_raw_onset_f1_25ms']
        lines.append(f'{condition} 的窗口协议相对整曲输出，随机组合起点 F1 变化 {delta:+.4f}，'
            f'原固定探针变化 {probe:+.4f}。这是整套输出协议的对照，不能再拆成 mask 或某个头的单独因果结论。')
        lines.append('')
    lines+=['本轮样本量小、单 seed，只能证明或反驳当前构造任务上的行为。'
        '如果训练拟合和新组合之间仍有明显落差，继续相同样本的长训不足以解释泛化；'
        '如果时间仍不稳定，键型属性、编码器适配与扩容也只能分别作为诊断，不能直接推出产品升级。'
        '自然歌曲全 300 首训练、E20 缺音回归和 Phira 试玩仍未执行。'
        '第 5 阶段的独立表示/编码器/容量实验尚未执行，不以本报告冒充完成。','']
    (ROOT/'docs/evaluations/windowed_ar_stage4_20261031.md').write_text('\n'.join(lines))


def main():
    status_path=BASE/'freex_supervisor.json'
    if status_path.exists():raise RuntimeError('supervisor state exists; inspect before launching another queue')
    if read(BASE/'history20/training.json')['status']!='completed':raise RuntimeError('finish stage 3 before training stage 4')
    if not (ROOT/'docs/evaluations/assets/windowed_ar_20261031/h0_h1_controlled_result.json').exists():
        raise RuntimeError('stage 3 final diagnostics required before stage 4')
    sources={p:digest(ROOT/p) for p in FILES};results={};workers=[]
    state=dict(schema='audiolab-freex-controlled-supervisor-v1',pid=os.getpid(),status='running',source_sha256=sources,
        scope='controlled only; natural training and stage 5 not launched',runs={},created=time.time())
    write_json(status_path,state)
    try:
        for condition in ('A','S'):
            if any(digest(ROOT/p)!=value for p,value in sources.items()):raise RuntimeError('source changed while queue active')
            workers=[]
            for gpu,architecture in enumerate(('whole','window')):
                name=f'{condition}-W{gpu}';directory=BASE/f'freex_{condition.lower()}_{architecture}'
                if directory.exists():raise RuntimeError(f'refuse output overwrite: {directory}')
                command=[str(ROOT/'.venv/bin/python'),'-m','script.train_models','--gpu',str(gpu),
                    '--worker','script.train_windowed_freex','--','train','--architecture',architecture,
                    '--condition',condition,'--output',str(directory),'--epochs','400','--eval-every','100']
                log=BASE/f'freex-{name}.log'
                with log.open('xb',buffering=0) as stream:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                workers.append((name,directory,child));state['runs'][name]=dict(pid=child.pid,status='running',command=command,log=str(log))
                write_json(status_path,state)
            while any(p.poll() is None for _,_,p in workers):
                for name,directory,child in workers:
                    code=child.poll()
                    if code is not None and code!=0:raise RuntimeError(f'{name} exited {code}; see log')
                    progress=directory/'training.json'
                    if progress.exists():
                        data=read(progress);curves=data.get('curves',[])
                        state['runs'][name].update(epoch=curves[-1]['epoch'] if curves else 0,
                            objective=curves[-1]['objective'] if curves else None,resources=data.get('resources'))
                state['updated']=time.time();write_json(status_path,state);time.sleep(10)
            for name,directory,child in workers:
                if child.returncode!=0:raise RuntimeError(f'{name} exited {child.returncode}')
                training=read(directory/'training.json');assert training['status']=='completed' and training['curves'][-1]['epoch']==400
                evaluation=read(directory/'evaluation-e0400.json');checkpoint=directory/'model_latest.pt'
                results[name]=dict(config=training['config'],final_training=training['curves'][-1],resources=training['resources'],
                    checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),
                    evaluation_sha256=digest(directory/'evaluation-e0400.json'),evaluation=evaluation['summaries'])
                state['runs'][name].update(status='completed',checkpoint_sha256=results[name]['checkpoint_sha256'])
                write_json(status_path,state)
        report(results);state.update(status='completed',updated=time.time(),report='docs/evaluations/windowed_ar_stage4_20261031.md')
        write_json(status_path,state);print(json.dumps(state),flush=True)
    except BaseException as exc:
        for _,_,child in workers:
            if child.poll() is None:child.terminate()
        for _,_,child in workers:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='failed',error=str(exc),updated=time.time());write_json(status_path,state)
        raise


if __name__=='__main__':main()

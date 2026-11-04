"""Summarize E24 read-only probes; plot frozen per-window diagnostic results."""
from pathlib import Path
import argparse
import bisect
import gzip
import hashlib
import json
import math
import os
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/joint_audio_ar.py').exists())
sys.path.insert(0,str(ROOT))
from src.audio_understanding.inference import sha256
from src.product.artifacts import write_json
from src.soft_onset import MEMORY_SECONDS

ASSET=Path(__file__).resolve().parent


def compact():
    data=((ASSET/'results.json').read_bytes() if (ASSET/'results.json').exists()
          else gzip.decompress((ASSET/'results.json.gz').read_bytes()))
    raw=json.loads(data);plan=json.loads((ASSET/'plan.json').read_text())
    assert raw['status']=='completed' and raw['training_updates']==0 and len(raw['results'])==4
    assert raw['plan_sha256']==sha256(ASSET/'plan.json')
    assert raw['checkpoint_sha256']==sha256(ROOT/plan['checkpoint'])
    for path,digest in plan['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    rows=[]
    for song in raw['results']:
        assert song['exact_replay'] and song['exact_native_prefix_control']
        accepted={raw['plan_sha256']}
        amendment=plan.get('validation_amendment')
        if amendment and song['id'] in amendment['preserved_completed_songs']:
            accepted.add(amendment['initial_plan_sha256'])
        assert song['plan_sha256'] in accepted
        selected={w['window'] for w in song['native_windows']}
        def ends(decisions):
            return [{key:d[key] for key in ('window','clock','native_stop_probability','native_end')}
                    for d in decisions if d['window_start'] and d['window'] in selected]
        row={k:song[k] for k in ('id','split','exact_replay','exact_native_prefix_control','native_windows',
                                 'audio','audio_evidence','resources')}
        row['source_plan_sha256']=song['plan_sha256']
        row['forced_prefix_metadata_differences']=song.get('forced_prefix_metadata_differences',[])
        row['native_window_starts']=ends(song['native_decisions'])
        row['native_final_decisions']=[next(d for d in reversed(song['native_decisions']) if d['window']==w)
                                      for w in sorted(selected)]
        row['arms']=[]
        for arm in song['arms']:
            assert all(not d['forced'] for d in arm['decisions'])
            row['arms'].append({**{k:v for k,v in arm.items() if k!='decisions'},
                               'window_starts':ends(arm['decisions'])})
        teacher=song['teacher']
        row['teacher']=None if teacher is None else {
            **{k:v for k,v in teacher.items() if k!='decisions'},
            'selected_decisions':[d for d in teacher['decisions']
                                  if d['window'] in selected and (d['window_start'] or d['truth_end'])]}
        if 'oracle_timing_decisions' in song:
            row['oracle_timing_window_starts']=ends(song['oracle_timing_decisions'])
            positive=[d for d in song['oracle_timing_decisions'] if not d.get('forced_end')]
            edges=[i*MEMORY_SECONDS for i in range(math.ceil(max(d['selected_next_seconds'] for d in positive)/MEMORY_SECONDS)+2)]
            row['oracle_timing_pointer_diagnostic']=[]
            for w in sorted(selected):
                part=[d for d in positive if d['window']==w]
                hits=[d['native_pointer_frame']==bisect.bisect_right(edges,d['selected_next_seconds'])-1 for d in part]
                row['oracle_timing_pointer_diagnostic'].append(dict(window=w,reference_next_onsets=len(part),
                    native_frame_hits=sum(hits),native_false_stops=sum(d['native_end'] for d in part),
                    continue_and_frame_hits=sum(hit and not d['native_end'] for hit,d in zip(hits,part))))
        rows.append(row)
    summary=dict(experiment='E24',status='completed',training_updates=0,
        checkpoint_sha256=raw['checkpoint_sha256'],raw_results_sha256=hashlib.sha256(data).hexdigest(),
        summary_source_sha256=sha256(Path(__file__)),
        plan_sha256=raw['plan_sha256'],results=rows,resources=raw['resources'],scope=raw['scope'],
        validation='All four original outputs reproduced exactly; native-prefix replay preserved all notes '
                   'and all free-continuation window records exactly; '
                   'all scored continuation decisions free; checkpoint and frozen sources unchanged.')
    write_json(ASSET/'summary.json',summary)
    return summary


def plot(summary):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/audiolab-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'#f5f7fb','axes.facecolor':'white',
        'pdf.fonttype':42,'axes.unicode_minus':False})
    songs={r['id']:r for r in summary['results']};gap=songs['52417'];inc=songs['Incyde_IN'];elt=songs['Eltaw_IN']
    fig,axes=plt.subplots(2,2,figsize=(13,9));fig.subplots_adjust(top=.86,bottom=.16,hspace=.42,wspace=.25)
    fig.suptitle('E24 · 固定300轮权重的归因实验',x=.06,y=.965,ha='left',fontsize=21,weight='bold')
    fig.text(.06,.915,'AL-AR4K v0.7.0 / A / E23-A e0300   |   训练更新：0   |   原始输出与原历史重放均核验通过',fontsize=10)
    ax=axes[0,0]
    for rows,label,color in [(gap['native_window_starts'],'自产历史','#d76a23'),
                             (gap['oracle_timing_window_starts'],'参考时间历史（键型仍由模型生成）','#2463c7')]:
        ax.plot([r['window']*10 for r in rows],[r['native_stop_probability'] for r in rows],
                'o-',color=color,label=label)
    ax.axhline(.5,color='#777777',ls='--',lw=1,label='原贪心决策边界');ax.set_ylim(-.03,1.03)
    ax.set_title('匙ノ咒：窗口入口的结束概率',loc='left');ax.set_xlabel('窗口起点 / 秒');ax.set_ylabel('P(END_WINDOW)')
    ax.legend(fontsize=8,loc='best');ax.grid(alpha=.15)
    ax=axes[0,1]
    native=gap['native_windows'];ax.plot([r['window']*10 for r in native],[r['reference_onsets'] for r in native],
        '--',color='#727c8c',label='参考起点数')
    ax.plot([r['window']*10 for r in native],[r['generated_onsets'] for r in native],'o-',color='#d76a23',label='原始自由生成')
    for arm,color in zip([r for r in gap['arms'] if r['kind']=='reference_time_model_layout_prefix'],['#2463c7','#15917e','#9655b4']):
        rows=arm['windows'];ax.plot([r['window']*10 for r in rows],[r['generated_onsets'] for r in rows],
            'o-',color=color,label=f"修正此前时间历史；{arm['release_window']*10}秒起自由生成")
    ax.set_title('匙ノ咒：释放后的自由生成覆盖',loc='left');ax.set_xlabel('窗口起点 / 秒');ax.set_ylabel('唯一起点数量')
    ax.set_ylim(-3,120)
    ax.legend(fontsize=7,loc='best');ax.grid(alpha=.15)
    ax=axes[1,0];native=inc['native_windows'];repair=next(a for a in inc['arms'] if a['kind']=='true_4k_prefix')['windows']
    x=list(range(len(native)))
    ax.bar([v-.24 for v in x],[r['reference_onsets'] for r in native],width=.24,color='#a1a9b5',label='参考')
    ax.bar(x,[r['generated_onsets'] for r in native],width=.24,color='#d76a23',label='自产历史')
    ax.bar([v+.24 for v in x],[r['generated_onsets'] for r in repair],width=.24,color='#2463c7',label='180秒前真实历史，之后自由生成')
    ax.set_xticks(x,[str(r['window']*10) for r in native]);ax.set_xlabel('窗口起点 / 秒');ax.set_ylabel('唯一起点数量')
    ax.set_title('Incyde：尾段结束失败的历史对照',loc='left');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    ax=axes[1,1];native=next(r for r in elt['native_windows'] if r['window']==6)
    labels=['原音频','当前窗口静音','整曲物理静音','编码特征清零']
    audio={r['kind']:r for r in elt['audio']}
    values=[native['notes']]+[audio[k]['window']['notes'] for k in ['local_mute','physical_silence','zero_features']]
    bars=ax.bar(labels,values,color=['#2463c7','#15917e','#9655b4','#d76a23']);ax.bar_label(bars,padding=3)
    ax.set_ylim(0,max(values)*1.2+1);ax.set_ylabel('音符数');ax.set_title('Eltaw 60–70秒：固定原历史的音频干预',loc='left')
    ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
    fig.text(.06,.09,'参考历史仅注入释放时刻之前；社区谱只提供时间，不能当作完整真实4K历史。数量恢复不等于节奏和键型正确。',fontsize=10)
    fig.text(.06,.05,'音频干预保留非空原历史；整曲静音结果不代表空历史静音测试。特征清零是分布外输入，不等于无声音频。',fontsize=10,color='#526074')
    fig.savefig(ASSET/'attribution.png',dpi=150);fig.savefig(ASSET/'attribution.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
    data=json.loads((ASSET/'summary.json').read_text()) if args.plot_only else compact()
    plot(data);print(json.dumps(dict(status='completed',summary=str(ASSET/'summary.json')),ensure_ascii=False))

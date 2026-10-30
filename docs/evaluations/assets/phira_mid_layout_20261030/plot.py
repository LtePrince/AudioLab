"""Replay saved E20 outputs for the two middle-ranked songs; no inference."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
DEST = Path(__file__).resolve().parent
SOURCE = ROOT/'docs/evaluations/assets/phira_generalization10_20261029'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,
                     'pdf.fonttype':42,'savefig.facecolor':'white'})
COLORS = dict(tap='#1685bc',drag='#c58c00',hold='#8756cf',flick='#d44878')
MARKERS = dict(tap='o',drag='D',hold='|',flick='^')
KINDS = list(COLORS)
LANES = [.2890625,.4296875,.5703125,.7109375]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def configure(ax, start, end, *, mode=None, clock=None):
    ax.set_xlim(start,end)
    ax.set_facecolor('#f8fafc')
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis='x',alpha=.18)
    if mode is None:
        ax.set_ylim(-.5,3.5)
        ax.set_yticks(range(4),[k.title() for k in KINDS])
        for i in range(4):ax.axhline(i,color='#dbe2eb',lw=.6,zorder=0)
        ax.set_ylabel('键型')
    else:
        ax.set_ylim(0,1)
        if mode=='4k':
            ax.set_yticks(LANES,['1','2','3','4']);ax.set_ylabel('轨道')
            for y in LANES:ax.axhline(y,color='#d5dce5',lw=.8,zorder=0)
        else:
            ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('屏幕横坐标')
            ax.grid(axis='y',alpha=.16)
        if clock:
            period=60/clock['bpm'];origin=clock['beat_zero_seconds']
            for beat in range(math.ceil((start-origin)/period),math.floor((end-origin)/period)+1):
                ax.axvline(origin+beat*period,color='#94a3b8',lw=.6 if beat%4 else 1.,
                           alpha=.18 if beat%4 else .32,zorder=0)


def draw(ax, notes, start, end, *, mode=None):
    def y(note):return KINDS.index(note['type']) if mode is None else note['screen_x']
    holds=[[(max(start,n['time_seconds']),y(n)),(min(end,n['end_seconds']),y(n))]
           for n in notes if n['type']=='hold' and n['end_seconds']>=start and n['time_seconds']<=end]
    if holds:ax.add_collection(LineCollection(holds,colors=COLORS['hold'],linewidths=2.7,alpha=.45,zorder=2))
    for kind in KINDS:
        visible=[n for n in notes if n['type']==kind and start<=n['time_seconds']<=end]
        counts=Counter((n['time_seconds'],y(n)) for n in visible)
        if not counts:continue
        points=list(counts)
        size=[(32 if kind=='hold' else 13)*(1+.65*math.log2(counts[p])) for p in points]
        ax.scatter([p[0] for p in points],[p[1] for p in points],s=size,c=COLORS[kind],
                   marker=MARKERS[kind],linewidths=.9,alpha=.9,zorder=3)


def legend(fig, y):
    handles=[Line2D([],[],color=COLORS[k],marker=MARKERS[k],linestyle='-' if k=='hold' else 'None',
                    markersize=6,label=k.title()) for k in KINDS]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,y),ncol=4,frameon=False)


def label(record):
    m=record['metrics'];mode='4K' if record['mode']=='4k' else 'free-x'
    variant=record['run_id'][-1]
    status='自然结束' if record['status']=='completed' else '结束失败'
    return (f'{mode} {variant} / e{record["epoch"]}  ·  起点 F1 {m["onset_f1_50ms"]:.3f}'
            f'  ·  网格 {m["grid_bonus"]:.3f}  ·  {len(record["prediction"]["notes"])} 键  ·  {status}')


def full_plot(song, reference, records):
    end=song['duration_seconds']
    fig,axes=plt.subplots(5,1,figsize=(18,11),sharex=True)
    fig.subplots_adjust(top=.83,bottom=.095,left=.065,right=.985,hspace=.52)
    fig.suptitle(f'{song["name"]} · 整曲节奏与键型对照',x=.065,ha='left',y=.975,fontsize=22,weight='bold')
    fig.text(.065,.935,f'Phira #{song["id"]} · 参考难度 {song["difficulty"]:.1f} · {song["clock"]["bpm"]:g} BPM · {end:.2f} 秒 · 原始贪心输出',color='#475569')
    fig.text(.065,.903,'五行采用相同键型纵轴；这张图比较时间与键型，不表示音符的屏幕位置。',color='#475569')
    legend(fig,.89)
    rows=[(f'社区参考谱 · {len(reference)} 键',reference,None)]+[(label(r),r['prediction']['notes'],r) for r in records]
    for ax,(title,notes,record) in zip(axes,rows):
        configure(ax,0,end);draw(ax,notes,0,end)
        ax.set_title(title,loc='left',fontsize=10.5,pad=7)
        if record:
            ax.axvline(end,color='#b45309',lw=1.2,ls='--')
    axes[-1].set_xlabel('音频时间（秒）')
    fig.text(.065,.042,'标记颜色/形状表示键型，紫色横线表示 Hold 时长；重合标记大小随同刻同型数量增加。未做吸附、补音或截断修正。',color='#475569',fontsize=9)
    fig.text(.065,.020,'起点 F1：唯一事件 ±50 ms 匹配。图中指标均为整曲成绩；结束失败的前缀保留。参考谱是作者的一种编排，不是唯一正确答案。',color='#475569',fontsize=9)
    return fig


def layout_plot(song, reference, records, windows, title):
    columns=len(windows)
    fig,axes=plt.subplots(5,columns,figsize=(20 if columns==2 else 18,12),squeeze=False,sharex='col')
    fig.subplots_adjust(top=.82,bottom=.09,left=.065,right=.98,hspace=.53,wspace=.16)
    fig.suptitle(title,x=.065,ha='left',y=.975,fontsize=22,weight='bold')
    fig.text(.065,.94,f'Phira #{song["id"]} · {song["clock"]["bpm"]:g} BPM · 4K A/S e1400；free-x A e1300 / S e1400 · A=纯音频，S=额外软起音',color='#475569')
    fig.text(.065,.909,'首行只画参考时间与键型；下四行画生成的真实横坐标。4K 轨道与 free-x 坐标均从屏幕左到右递增。',color='#475569')
    legend(fig,.894)
    for column,(start,end,window_name) in enumerate(windows):
        ax=axes[0,column]
        configure(ax,start,end);draw(ax,reference,start,end)
        ax.set_title(f'{window_name} · {start:.2f}–{end:.2f} 秒\n社区参考：时间＋键型',loc='left',fontsize=11,pad=8)
        for index,record in enumerate(records,1):
            ax=axes[index,column];mode=record['mode']
            configure(ax,start,end,mode=mode,clock=song['clock'])
            draw(ax,record['prediction']['notes'],start,end,mode=mode)
            ax.set_title(label(record),loc='left',fontsize=9.5,pad=7)
        axes[-1,column].set_xlabel('音频时间（秒）')
    fig.text(.065,.044,'竖线是参考节拍，仅作阅读标尺；音符时间保持原样。Hold 跨窗口时显示实际延续部分，不人为添加起点。',color='#475569',fontsize=9)
    fig.text(.065,.022,'参考谱含动态判定线，因此不以未经还原的局部 positionX 冒充屏幕坐标；只对生成的四套静态谱比较排布。',color='#475569',fontsize=9)
    return fig


def main():
    assert shutil.disk_usage(DEST).free>8*2**30+20*2**20,'preserve 8 GiB disk reserve'
    selection=read(SOURCE/'selection.json');summary=read(SOURCE/'summary.json')
    assert sha(SOURCE/'selection.json')==summary['selection_sha256']
    ranking=[]
    for song in selection['songs']:
        rows=[r for r in summary['rows'] if r['song_id']==song['id']]
        assert len(rows)==4
        ranking.append(dict(id=song['id'],name=song['name'],mean_onset_f1_50ms=statistics.mean(r['metrics']['onset_f1_50ms'] for r in rows),
                            mean_grid_bonus=statistics.mean(r['metrics']['grid_bonus'] for r in rows)))
    ranking.sort(key=lambda r:(-r['mean_onset_f1_50ms'],r['id']))
    for rank,row in enumerate(ranking,1):row['rank']=rank
    picked=ranking[4:6]
    manifest=dict(schema='audiolab-phira-mid-layout-v1',experiment='E20',kind='saved_prediction_visualization',
        selection_rule='rank ten songs by mean unique onset F1 at50ms over all four fixed candidates; choose ranks5 and6; failed prefixes included',
        new_training=False,new_inference=False,ranking=ranking,selected=[],
        source_summary_sha256=sha(SOURCE/'summary.json'),plot_script_sha256=sha(__file__))
    for choice in picked:
        song=next(s for s in selection['songs'] if s['id']==choice['id'])
        reference_path=ROOT/song['reference'];assert sha(reference_path)==song['reference_sha256']
        reference=read(reference_path)['notes'];records=[];inputs=[]
        for candidate in selection['candidates']:
            path=ROOT/'out/phira_generalization10_20261029'/str(song['id'])/candidate['id']/'result.json'
            record=read(path)
            assert record['checkpoint_sha256']==candidate['sha256'] and record['selection_sha256']==summary['selection_sha256']
            assert record['prediction']['note_count']==len(record['prediction']['notes'])
            records.append(record);inputs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        duration=song['duration_seconds'];clock=song['clock'];period=60/clock['bpm'];origin=clock['beat_zero_seconds']
        first=min(n['time_seconds'] for n in reference)
        start=max(0,origin+math.floor((first-origin)/period+1e-8)*period)
        middle=max(0,origin+math.floor((duration/2-16*period-origin)/period/8)*8*period)
        windows=[(start,min(start+32*period,duration),'起始 32 拍'),(middle,min(middle+32*period,duration),'中段 32 拍')]
        stem=str(song['id']);fig=full_plot(song,reference,records)
        full=DEST/f'{stem}_full.png';fig.savefig(full,dpi=140);plt.close(fig)
        fig=layout_plot(song,reference,records,windows,f'{song["name"]} · A/S 局部排布对照')
        detail=DEST/f'{stem}_layout.png';fig.savefig(detail,dpi=145);plt.close(fig)
        pdf_path=DEST/f'{stem}_full_detail.pdf';pages=0
        with PdfPages(pdf_path) as pdf:
            for start in range(0,math.ceil(duration),30):
                end=min(start+30,duration)
                fig=layout_plot(song,reference,records,[(start,end,'连续整曲分页')],f'{song["name"]} · {start}–{end:.1f} 秒')
                pdf.savefig(fig);plt.close(fig);pages+=1
        manifest['selected'].append(dict(**choice,reference= song['reference'],reference_sha256=song['reference_sha256'],
            inputs=inputs,windows=[dict(start_seconds=a,end_seconds=b,label=c) for a,b,c in windows],
            pdf_pages=pages,artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in (full,detail,pdf_path)]))
        print(json.dumps(dict(song=song['name'],rank=choice['rank'],mean_f1=choice['mean_onset_f1_50ms'],pdf_pages=pages),ensure_ascii=False),flush=True)
    assert sum(p.stat().st_size for p in DEST.iterdir() if p.is_file())<20*2**20
    manifest['disk_free_gib']=shutil.disk_usage(DEST).free/2**30
    save_json(DEST/'selection.json',manifest)


if __name__=='__main__':main()

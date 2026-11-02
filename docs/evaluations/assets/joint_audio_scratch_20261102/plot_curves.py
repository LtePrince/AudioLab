"""Freeze one E23 progress snapshot and plot training versus fixed-song rollout."""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/audiolab-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/joint_audio_ar.py').exists())
RUN=ROOT/'out/joint_audio_scratch_20261102'
ASSET=Path(__file__).resolve().parent


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-snapshot',type=Path,help='Re-render an existing frozen snapshot')
    args=parser.parse_args()
    raw=(args.training_snapshot or RUN/'training.json').read_bytes()
    progress=json.loads(raw);curves=progress['curves']
    if not curves or progress['config']['run']!='E23-A':raise ValueError('E23 training curves required')
    last=curves[-1]['epoch'];dest=ASSET/f'curves_e{last:04d}'
    if args.training_snapshot and args.training_snapshot.resolve().parent!=dest.resolve():
        raise ValueError('snapshot must reside in its epoch-specific artifact directory')
    dest.mkdir(exist_ok=args.training_snapshot is not None)
    songs={'Eltaw_IN':'Eltaw（训练曲）','40717':'天使にふれたよ!（开发曲）','52417':'匙ノ咒（开发曲）'}
    rollouts=[];sources=[];full=[]
    for path in sorted(RUN.glob('evaluation-e*/summary.json')):
        data=path.read_bytes();evaluation=json.loads(data)
        if evaluation['epoch']>last:continue
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(data)))
        if evaluation['full']:full.append(dict(epoch=evaluation['epoch'],aggregates=evaluation['aggregates']))
        for row in evaluation['rows']:
            if str(row['id']) not in songs:continue
            gap=next((g for g in row['longest_gaps'] if g['reference_onsets']>0),None)
            rollouts.append(dict(epoch=evaluation['epoch'],song=str(row['id']),status=row['status'],
                onset_f1_50ms=row['metrics']['onset_f1_50ms'],grid_bonus=row['metrics']['grid_bonus'],
                time_type_f1=row['metrics']['time_type_f1'],forced_windows=row['forced_windows'],
                longest_reference_active_gap_seconds=gap['seconds'] if gap else 0.))
    latest_eval=max(r['epoch'] for r in rollouts)
    (dest/'training_snapshot.json').write_bytes(raw)
    snapshot=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),run='E23-A',product='AL-AR4K v0.7.0 / A',
        completed_epoch=last,latest_rollout_epoch=latest_eval,training_source_sha256=digest(raw),
        plot_source_sha256=digest(Path(__file__).read_bytes()),evaluation_sources=sources,
        evaluation=rollouts,full_evaluation_aggregates=full,
        scope='Raw epoch values, no smoothing. Training uses true history. Rollout curves always use the same '
              'three songs; never join the 15/10-song means to these individual curves. Failed outputs included; '
              'F1/grid are not completion or playability scores.')
    (dest/'provenance.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    columns=['epoch','steps','objective','frame_ce','layout_ce','subframe_mse','stop_bce','continue_bce',
             'frame_accuracy','end_stop_recall','false_stop_rate','conditional_offset_mae_seconds']
    with (dest/'training.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');writer.writeheader()
        writer.writerows({key:row[key] for key in columns} for row in curves)
    with (dest/'evaluation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rollouts[0]),lineterminator='\n');writer.writeheader();writer.writerows(rollouts)

    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f5f7fb','axes.facecolor':'white',
        'axes.edgecolor':'#c4ccd8','text.color':'#17283d','axes.labelcolor':'#34465b',
        'xtick.color':'#526074','ytick.color':'#526074','pdf.fonttype':42,'axes.unicode_minus':False})
    fig,axes=plt.subplots(3,2,figsize=(13.4,12.4))
    fig.subplots_adjust(top=.88,bottom=.13,hspace=.48,wspace=.25)
    fig.suptitle(f'E23-A · 全参数从零训练曲线',fontsize=21,x=.065,ha='left',y=.97,weight='bold')
    fig.text(.065,.932,f'AL-AR4K v0.7.0 / A   |   已完成 {last}/300 轮   |   全 300 首   |   13.37M 参数',fontsize=11)
    fig.text(.065,.906,'上两排：真实谱面历史下的训练指标。下排：固定三首的自由生成评测。每轮原始值，无平滑。',fontsize=10,color='#526074')
    x=[c['epoch'] for c in curves]
    colors=['#2463c7','#d76a23','#15917e','#9655b4']
    for ax in axes.flat:
        ax.set_xlim(0,max(last,latest_eval)+3);ax.set_xlabel('训练轮数');ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
    ax=axes[0,0];ax.plot(x,[c['objective'] for c in curves],color=colors[0],lw=1.9)
    ax.set_yscale('log');ax.set_title('总训练损失（对数纵轴）',loc='left');ax.set_ylabel('损失')
    ax.text(.97,.92,f"{curves[0]['objective']:.3f} → {curves[-1]['objective']:.4f}",transform=ax.transAxes,ha='right',weight='bold')
    ax=axes[0,1]
    for key,label,color in zip(['frame_ce','layout_ce','stop_bce','subframe_mse'],
                              ['起点时间 CE','键型／布局 CE','窗口结束 BCE','帧内偏移 MSE'],colors):
        ax.plot(x,[c[key] for c in curves],label=label,color=color,lw=1.5)
    ax.set_yscale('log');ax.set_title('各项损失（未加权，对数纵轴）',loc='left');ax.set_ylabel('损失')
    ax.legend(frameon=False,fontsize=9,ncol=2,loc='upper right')
    ax=axes[1,0]
    for key,label,color in [('frame_accuracy','起点时间帧准确率',colors[0]),('end_stop_recall','窗口结束召回',colors[2])]:
        ax.plot(x,[c[key] for c in curves],label=label,color=color,lw=1.7)
    ax.set_ylim(0,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,loc='lower right')
    ax.set_title('真实历史下的训练准确率／召回',loc='left');ax.set_ylabel('比例')
    ax=axes[1,1];ax.plot(x,[1000*c['conditional_offset_mae_seconds'] for c in curves],color=colors[3],lw=1.7)
    ax.set_ylim(bottom=0);ax.set_title('已给定正确时间帧时的偏移误差',loc='left');ax.set_ylabel('平均绝对误差（ms）')
    ax.text(.97,.92,f"最新 {1000*curves[-1]['conditional_offset_mae_seconds']:.2f} ms",transform=ax.transAxes,ha='right')
    for ax,metric,title in [(axes[2,0],'onset_f1_50ms','自由生成：起点 F1@50ms'),
                            (axes[2,1],'grid_bonus','自由生成：网格匹配加分')]:
        for (song,label),color in zip(songs.items(),colors):
            rows=sorted((r for r in rollouts if r['song']==song),key=lambda r:r['epoch'])
            ax.plot([r['epoch'] for r in rows],[r[metric] for r in rows],marker='o',ms=4,color=color,lw=1.7,label=label)
        ax.set_ylim(0,1.03);ax.set_title(title,loc='left');ax.set_ylabel('分数')
        ax.legend(frameon=False,fontsize=8.5,loc='upper left' if metric=='grid_bonus' else 'lower right')
    fig.text(.065,.074,f'自由生成最新检查：e{latest_eval:04d}。第 100 轮的 15 首／10 首均值未混入下排曲线。',fontsize=10)
    fig.text(.065,.049,'高训练准确率不等于自主生成成功；帧内误差不包含选错时间帧。F1／网格保留结束失败的输出，不能代替结束率。',fontsize=9,color='#526074')
    fig.savefig(dest/'training_curves.png',dpi=150,facecolor=fig.get_facecolor())
    fig.savefig(dest/'training_curves.pdf',facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps(dict(directory=str(dest),completed_epoch=last,latest_evaluation=latest_eval,
        first_loss=curves[0]['objective'],latest_loss=curves[-1]['objective'],latest_focus=[r for r in rollouts if r['epoch']==latest_eval]),ensure_ascii=False))


if __name__=='__main__':main()

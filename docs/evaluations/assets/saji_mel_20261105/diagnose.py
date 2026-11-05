"""Read-only Mel audit and same-checkpoint crop controls; no training changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from script.train_joint_audio_a import mel_from_wave
from script.train_windowed_corpus import natural_reference
from src.audio_understanding.evaluation import match_events
from src.data.audio2mel import AudioGPUprocessor
from src.event4k import Event4K,generate
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget,add_resource_args
from src.soft_onset import MEMORY_SECONDS,state_hash

MEL=ROOT/'out/joint_audio_a_20261102/mel'
SELECTION=ROOT/'docs/evaluations/assets/phira_generalization10_20261029/selection.json'
RUN=ROOT/'out/event4k_fullsong_bpm_20261105'
OUT=ROOT/'out/saji_mel_20261105'
ASSETS=Path(__file__).resolve().parent
HOP=512/22050
GAP=(124.7700008470185,207.69255383549904)

def read(p):return json.loads(Path(p).read_text())
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def full_view(row):
    x=load(MEL/row['file']);return x['views'][0] if 'views' in x else x
def quantiles(x):return {str(p):float(np.quantile(x,p/100)) for p in (0,5,50,95,100)}

def mel_stats(view):
    x=view['mel'][:,:math.ceil(view['duration']/HOP)].float().numpy()
    power=np.power(10.,x/10.)
    energy=10*np.log10(np.maximum(power.sum(0),1e-20))
    flux=np.maximum(np.diff(x,axis=1),0).mean(0)
    return dict(duration=view['duration'],mel_frames=x.shape[1],memory_frames=view['mel'].shape[1]//2,
        finite=bool(np.isfinite(x).all()),floor_frame_fraction=float((x.max(0)<=-99.99).mean()),
        logmel_mean=float(x.mean()),logmel_std=float(x.std()),mel_energy_db=quantiles(energy),
        positive_logmel_difference=quantiles(flux),mean_spectrum_db=x.mean(1).tolist())

def rms_curve(wave):
    mono=wave.mean(1).flatten().numpy();hop=11025;n=math.ceil(len(mono)/hop)
    values=np.pad(mono,(0,n*hop-len(mono))).reshape(n,hop)
    rms=np.sqrt((values**2).mean(1));return (np.arange(n)+.5)*.5,20*np.log10(np.maximum(rms,1e-10))

def songs():
    selected={str(x['id']):x for x in read(SELECTION)['songs']}
    cache=read(ROOT/'out/full_corpus_v1/cache/cache.json');row=next(x for x in cache['rows'] if x['id']=='Eltaw_IN')
    sample=load(ROOT/'out/full_corpus_v1/cache'/row['file'])
    eltaw=dict(id='Eltaw_IN',name='Eltaw',audio=row['audio']['path'],reference_notes=natural_reference(sample))
    return [eltaw,dict(selected['40717'],reference_notes=read(ROOT/selected['40717']['reference'])['notes']),
            dict(selected['52417'],reference_notes=read(ROOT/selected['52417']['reference'])['notes'])]

def audit(budget):
    processor=AudioGPUprocessor(device='cpu');manifest=read(MEL/'manifest.json')
    report=dict(kind='audio_and_cache_audit',training_changed=False,mel_manifest_sha256=digest(MEL/'manifest.json'),
        source_sha256=digest(__file__),training=[],development=[],detailed=[])
    for split in ('rows','development'):
        destination='training' if split=='rows' else split
        for i,row in enumerate(manifest[split]):
            budget.check();view=full_view(row);stats=mel_stats(view)
            report[destination].append(dict(id=str(row['id']),**stats))
            if (i+1)%100==0:print(json.dumps(dict(stage='mel_stats',split=split,songs=i+1)),flush=True)
            del view
    rows={str(x['id']):x for x in manifest['rows']+manifest['development']};plots=[]
    for song in songs():
        budget.check(force=True);id=str(song['id']);row=rows[id];view=full_view(row)
        assert digest(MEL/row['file'])==row['sha256']
        wave=processor.load_from_path(ROOT/song['audio']);fresh,duration=mel_from_wave(processor,wave)
        assert digest(ROOT/song['audio'])==row['audio_sha256']
        times,rms=rms_curve(wave)
        detail=dict(id=id,name=song['name'],duration=duration,channels=wave.shape[1],
            mel_exactly_reproduced=bool(torch.equal(fresh,view['mel'])),max_recomputed_mel_error=float((fresh-view['mel']).abs().max()),
            waveform_rms_dbfs=quantiles(rms),silent_half_seconds_fraction=float((rms<-60).mean()),
            cache_sha256=row['sha256'],audio_sha256=row['audio_sha256'],segments=[])
        if id=='52417':
            for name,a,b in [('before',0,GAP[0]),('gap',*GAP),('after',GAP[1],duration)]:
                part=(times>=a)&(times<b);start=int(a/HOP);end=min(view['mel'].shape[1],math.ceil(b/HOP))
                part_mel=view['mel'][:,start:end]
                detail['segments'].append(dict(name=name,start=a,end=b,rms_dbfs=quantiles(rms[part]),
                    silent_half_seconds_fraction=float((rms[part]<-60).mean()),finite=bool(torch.isfinite(part_mel).all()),
                    floor_frame_fraction=float((part_mel.max(0).values<=-99.99).float().mean()),
                    reference_onsets=sum(a<t<b for t in set(n['time_seconds'] for n in song['reference_notes']))))
        prediction=read(RUN/f'evaluation-e0020/{id}.json')['prediction']
        plots.append((song,view['mel'].numpy(),duration,times,rms,prediction))
        report['detailed'].append(detail)
        print(json.dumps(dict(stage='waveform_check',**detail),ensure_ascii=False),flush=True)
        del wave,fresh,view
    train=report['training'];target=next(x for x in report['development'] if x['id']=='52417')
    report['training_duration_seconds']=quantiles([x['duration'] for x in train])
    report['saji_percentile_among_training']={key:100*sum(x[key]<=target[key] for x in train)/len(train)
        for key in ('duration','memory_frames','logmel_mean','logmel_std','floor_frame_fraction')}
    report['resources']=budget.log_state();write_json(ASSETS/'mel_audit.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(14,12),sharex=True,gridspec_kw={'height_ratios':[1.5,1.5,1.5,1,1]},layout='constrained')
    titles=['Eltaw (training song)','Community #40717','Community #52417: Saji no Noroi']
    for ax,(_,mel,duration,_,_,_),title in zip(axes[:3],plots,titles):
        img=ax.imshow(mel,aspect='auto',origin='lower',extent=(0,mel.shape[1]*HOP,0,128),cmap='magma',vmin=-80,vmax=20)
        ax.set_title(f'{title} | {duration:.2f} s',loc='left');ax.set_ylabel('Mel band');ax.set_facecolor('#eeeeee')
    song,_,duration,times,rms,pred=plots[-1]
    axes[3].plot(times,rms,color='#146a9b',lw=.8);axes[3].set_ylim(-65,0);axes[3].set_ylabel('Mono RMS\n(dBFS)')
    bins=np.arange(0,275,5);ref=np.array(sorted(set(n['time_seconds'] for n in song['reference_notes'])))
    axes[4].stairs(np.histogram(ref,bins)[0],bins,label='Reference',color='#c06b10',lw=1.5)
    axes[4].stairs(np.histogram(pred['onsets_seconds'],bins)[0],bins,label='Generated at epoch 20',color='#146a9b',lw=1.5)
    axes[4].set_ylabel('Onsets / 5 s');axes[4].legend(loc='upper right');axes[4].set_xlabel('Physical audio time (seconds)')
    for ax in axes[2:]:ax.axvspan(*GAP,color='#ff5577',alpha=.17)
    axes[3].text(126,-60,'83 s generation gap; audio remains present',color='#9d2348')
    axes[0].set_xlim(0,270)
    fig.colorbar(img,ax=axes[:3],label='Log-Mel power (dB; shared scale)',fraction=.018,pad=.01)
    fig.savefig(ASSETS/'mel_comparison.png',dpi=160);plt.close(fig)

def compare_segment(predicted,reference,start,end):
    pt=sorted({t for t in predicted if start<=t<end});rt=sorted({t for t in reference if start<=t<end})
    edges=[start,*pt,end]
    gaps=[dict(start=a,end=b,seconds=b-a,reference_onsets=sum(a<t<b for t in rt)) for a,b in zip(edges,edges[1:]) if b>a]
    return dict(predicted_onsets=len(pt),reference_onsets=len(rt),onset_f1_50ms=2*len(match_events(rt,pt,.05))/max(1,len(rt)+len(pt)),
        longest_gaps=sorted(gaps,key=lambda x:x['seconds'],reverse=True)[:3])

def traced_generate(model,mel,duration,bpm,budget):
    c=model.chart;real_step=c.step;real_logits=c.time_logits;clock=0.;trace=[]
    def step(event,time_seconds,position,caches,bpm):
        nonlocal clock
        clock=time_seconds;return real_step(event,time_seconds,position,caches,bpm)
    def logits(h,memory,allowed):
        raw=real_logits(h,memory,allowed);p=raw[0].softmax(-1);n=memory.shape[1]
        values,indices=p.topk(min(5,len(p)));starts=torch.arange(n,device=p.device)*MEMORY_SECONDS
        trace.append(dict(previous_time=clock,eos_probability=float(p[-1]),
            probability_next_5s=float(p[:-1][(starts>=clock-MEMORY_SECONDS)&(starts<clock+5)].sum()),
            top=[dict(frame=int(i),seconds=float(i)*MEMORY_SECONDS if int(i)<n else None,probability=float(v)) for i,v in zip(indices,values)]))
        return raw
    c.step=step;c.time_logits=logits
    try:result=generate(model,mel,duration,bpm,check=budget.check)
    finally:c.step=real_step;c.time_logits=real_logits
    for i,row in enumerate(trace):row['chosen_event_time']=result['event_times_seconds'][i] if i<len(result['event_times_seconds']) else None
    return result,trace

def controls(args,budget,device):
    path=RUN/'model_best_fit.pt'
    # A single open inode remains stable even when training atomically replaces latest/best.
    with path.open('rb') as f:
        checkpoint_sha=hashlib.file_digest(f,'sha256').hexdigest();f.seek(0);saved=torch.load(f,map_location='cpu',weights_only=True)
    config=saved['config'];model=Event4K(**dict(encoder_config=config['architecture']['encoder_config'],decoder_config=config['architecture']['decoder_config']))
    model.load_state_dict(saved['model'],strict=True);assert state_hash(model.state_dict())==saved['parameter_sha256']
    model.to(device).eval();processor=AudioGPUprocessor(device='cpu');song=songs()[-1]
    wave=processor.load_from_path(ROOT/song['audio']);duration=wave.shape[-1]/22050
    report=dict(kind='read_only_same_checkpoint_controls',product='AL-Event4K@0.3.0',epoch=saved['epoch'],steps=saved['steps'],
        checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=checkpoint_sha,parameter_sha256=saved['parameter_sha256'],
        checkpoint_copy_saved=False,weights_updated=False,source_sha256=digest(__file__),
        limitations=['These controls use the sampled current checkpoint, not the unavailable epoch-20 weights.',
            'Cropping changes total duration, available future audio, and for the tail crop the time origin/history; it does not isolate length alone.'],rows=[])
    write_json(OUT/'control_config.json',{k:v for k,v in report.items() if k!='rows'});del saved
    reference=sorted(set(n['time_seconds'] for n in song['reference_notes']))
    with torch.inference_mode():
        for name,start,end in [('full',0.,duration),('first_200s',0.,200.),('tail_from_120s',120.,duration)]:
            budget.check(force=True)
            segment=wave[...,round(start*22050):round(end*22050)];mel,seconds=mel_from_wave(processor,segment)
            pred,trace=traced_generate(model,mel,seconds,song['clock']['bpm'],budget)
            shifted=[t+start for t in pred['onsets_seconds']]
            row=dict(name=name,start=start,end=end,duration=seconds,status=pred['status'],event_count=pred['event_count'],
                unclosed_holds=pred['unclosed_holds'],full_interval=compare_segment(shifted,reference,start,end),
                common_125_200=compare_segment(shifted,reference,125.,200.),
                largest_time_jumps=sorted([r for r in trace if r['chosen_event_time'] is not None],key=lambda r:r['chosen_event_time']-r['previous_time'],reverse=True)[:3],
                resources=budget.log_state())
            write_json(OUT/f'{name}.json',dict(row=row,prediction=pred,trace=trace))
            report['rows'].append(row);write_json(ASSETS/'controls.json',report)
            print(json.dumps(dict(stage='crop_control',epoch=report['epoch'],**row),ensure_ascii=False),flush=True)
            del pred,trace,mel

def main():
    import os
    os.chdir(ROOT);p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','controls'])
    p.add_argument('--device',default='cpu');add_resource_args(p);p.set_defaults(max_rss_gib=4.,gpu_memory_fraction=.25)
    args=p.parse_args();torch.set_num_threads(2);device=torch.device(args.device)
    budget=ResourceBudget(args,device,OUT)
    audit(budget) if args.stage=='audit' else controls(args,budget,device)

if __name__=='__main__':main()

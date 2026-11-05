"""Full-song Event4K: one EOS, full history, bounded storage and exact resume."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

from script.train_alignment_scratch import training_state, restore_training
from script.train_grounded_ar import read
from script.train_joint_audio_a import set_learning_rates
from script.train_soft_onset import cache_inputs, load_sample
from script.train_windowed_corpus import E20, natural_reference, scoring_module
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.event4k import SCHEMA, COLUMNS, COUNTS, Event4K, encode_events, events_to_notes, source_notes, normalized_loss, generate
from src.joint_audio_ar import gradient_audit
from src.pattern_diagnostics import KINDS
from src.product.artifacts import write_json
from src.resource_budget import GIB, ResourceBudget, add_resource_args
from src.soft_onset import state_hash

SOURCES=('src/event4k.py','script/train_event4k.py','script/train_models.py','src/joint_audio_ar.py',
    'src/models/transcriber.py','src/models/chart_decoder.py','src/models/attention.py',
    'src/condition/wave.py','src/windowed_ar.py','src/chart_windows.py','src/resource_budget.py',
    'script/train_alignment_scratch.py','script/train_joint_audio_a.py','script/train_windowed_corpus.py',
    'src/audio_understanding/evaluation.py','benchmarks/v1/protocol.json',
    'docs/evaluations/assets/phira_generalization10_20261029/benchmark.py')


def roundtrip_error(reference,decoded):
    """Compare physical times, not rounded decimal strings on a rounding boundary."""
    key=lambda n:(n['position'],n['type'],n['time_seconds'],n['end_seconds'])
    if len(reference)!=len(decoded):raise ValueError('event codec changed note count')
    maximum=0.
    for a,b in zip(sorted(reference,key=key),sorted(decoded,key=key),strict=True):
        if (a['position'],a['type'])!=(b['position'],b['type']):raise ValueError('event codec changed position/type')
        error=max(abs(a[k]-b[k]) for k in ('time_seconds','end_seconds'))
        if error>1e-9:raise ValueError('event codec changed physical time beyond 1 ns')
        maximum=max(maximum,error)
    return maximum


def prepare(args,budget):
    manifest=read(args.mel_cache/'manifest.json')
    if manifest['status']!='completed' or len(manifest['rows'])!=300 or len(manifest['development'])!=10:
        raise ValueError('complete existing full300 Mel cache required')
    directory=args.output/'targets';directory.mkdir(exist_ok=True)
    path=directory/'manifest.json'
    contract=dict(schema=SCHEMA,mel_manifest_sha256=sha256(args.mel_cache/'manifest.json'),
        codec_source_sha256=sha256('src/event4k.py'),output_protocol='full_song_single_eos',precision_seconds=1e-9)
    if path.exists():
        result=read(path)
        if result['status']!='completed' or result['contract']!=contract:raise ValueError('target cache contract changed')
        for row in result['rows']:
            if sha256(directory/row['file'])!=row['sha256']:raise ValueError('target cache changed')
        for row in manifest['rows']+manifest['development']:
            if sha256(args.mel_cache/row['file'])!=row['sha256']:raise ValueError('shared Mel cache changed')
        return manifest,result
    result=dict(schema=SCHEMA,status='preparing',contract=contract,rows=[],counts={},errors=[])
    counts=Counter();total_bytes=0;max_roundtrip_error=0.
    for index,row in enumerate(manifest['rows']):
        budget.check()
        if sha256(args.mel_cache/row['file'])!=row['sha256']:raise ValueError('shared Mel row changed')
        sample=torch.load(args.mel_cache/row['file'],map_location='cpu',weights_only=True);targets=[]
        for vi,view in enumerate(sample['views']):
            notes=source_notes(view['target'])
            try:
                target=encode_events(notes,view['duration'],view['target']['bpm'])
                decoded=events_to_notes(target['times'],target['events'])
                max_roundtrip_error=max(max_roundtrip_error,roundtrip_error(notes,decoded))
                if target['eos_count']!=1 or target['event_indices'][-1]!=-1 or (target['event_indices'][:-1]<0).any():
                    raise ValueError('exactly one terminal EOS required')
                if target['prefix'].tolist()!=list(range(target['event_count']+1)):
                    raise ValueError('full chronological prefixes required')
            except Exception as error:
                result['errors'].append(dict(id=row['id'],view=vi,error=str(error)));continue
            counts['views']+=1;counts['notes']+=len(notes);counts['events']+=target['event_count']
            counts['hold_ends']+=int(target['events'][:,:,4].sum())
            counts['chord_events']+=int((target['events'][:,:,:4].sum((1,2))>1).sum())
            counts['eos_labels']+=target['eos_count']
            counts['legacy_intermediate_window_ends_removed']+=view['target']['stop_count']-1
            targets.append(target)
        if len(targets)!=4:continue
        saved=directory/f'train-{index:03d}.pt'
        budget.save(dict(id=row['id'],targets=targets),saved);total_bytes+=saved.stat().st_size
        if total_bytes>512*1024**2:raise RuntimeError('event target cache exceeds 512 MiB')
        result['rows'].append(dict(id=row['id'],file=saved.name,sha256=sha256(saved),
            events=targets[0]['event_count'],audio_frames=row['frames'],mel_file=row['file']))
        if (index+1)%30==0:print(json.dumps(dict(stage='event_codec_audit',songs=index+1,errors=len(result['errors']))),flush=True)
    for row in manifest['development']:
        if sha256(args.mel_cache/row['file'])!=row['sha256']:raise ValueError('development Mel changed')
    result.update(status='failed' if result['errors'] else 'completed',counts=dict(counts),bytes=total_bytes,
                  max_roundtrip_error_seconds=max_roundtrip_error)
    write_json(path,result)
    if result['errors'] or len(result['rows'])!=300:raise ValueError('lossless event codec audit failed; no songs silently excluded')
    return manifest,result


def load_initial(args,targets):
    architecture=read(args.architecture_config);torch.manual_seed(args.seed)
    if architecture.get('output_protocol')!=SCHEMA:raise ValueError('full-song architecture contract required')
    model=Event4K(architecture['encoder_config'],architecture['decoder_config'])
    if model.chart.bpm_condition != args.bpm_condition:raise ValueError('requested BPM configuration mismatch')
    config=dict(schema=SCHEMA,run='E27-BPM' if args.bpm_condition else 'E27-NOBPM',
        requested_version='AL-Event4K@0.3.0' if args.bpm_condition else 'AL-Event4K@0.4.0',
        architecture=architecture,initialization='scratch',inherited_parameter_tensors=0,parent_checkpoint=None,
        seed=args.seed,epochs=args.epochs,lr=args.lr,encoder_lr=args.encoder_lr,warmup_steps=args.warmup_steps,
        output_protocol='full_song_single_eos',eval_every=args.eval_every,full_eval_every=args.full_eval_every,
        mel_cache=str(args.mel_cache),mel_manifest_sha256=sha256(args.mel_cache/'manifest.json'),
        targets_manifest_sha256=sha256(args.output/'targets/manifest.json'),
        selection_sha256=sha256(E20),initial_state_sha256=state_hash(model.state_dict()),
        initial_encoder_sha256=state_hash(model.acoustic.state_dict()),
        initial_decoder_sha256=state_hash(model.chart.state_dict()),
        parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=sum(p.numel() for p in model.acoustic.parameters()),
        decoder_parameters=sum(p.numel() for p in model.chart.parameters()),
        source_sha256={p:sha256(p) for p in SOURCES},architecture_sha256=sha256(args.architecture_config),
        columns=list(COLUMNS),time_encoding='physical memory frames; sinusoidal projection',
        bpm_encoding='log2(BPM/120), bias-free learned projection' if args.bpm_condition else 'absent',
        history_summary=False,boundary_embedding=False,window_policy='absent; monotonic time and physical audio limits only',
        event_loss='mean-of-20 BCE per event',time_loss='joint remaining-time/song-EOS categorical CE; event/EOS separately normalized',
        coordinate_loss='frame-conditional fractional squared error',feature_precision='float32 end to end',
        training_views='all 300 full songs + two frozen physical crops + true waveform silence per song',
        retention='latest exact-resume + best-fit inference; first and 100/200/final inference snapshots; no cache duplication',
        schedule='AdamW; 1500-step linear warmup then constant; fresh optimizer/RNG; all parameters trainable')
    return model,config


def batch(args,row):
    sample=torch.load(args.mel_cache/row['mel_file'],map_location='cpu',weights_only=True)
    labels=torch.load(args.output/'targets'/row['file'],map_location='cpu',weights_only=True)
    for view,target in zip(sample['views'],labels['targets'],strict=True):
        view['target']=target
    return sample['views']


def probe(args,budget,targets,device):
    model,config=load_initial(args,targets);model.to(device)
    rows={r['id']:r for r in targets['rows']}
    selected=list(dict.fromkeys(['Eltaw_IN',max(rows,key=lambda k:rows[k]['audio_frames']),max(rows,key=lambda k:rows[k]['events'])]))
    report=dict(config=config,rows=[])
    for id in selected:
        views=batch(args,rows[id]);totals={k:sum(v['target'][k] for v in views) for k in COUNTS}
        model.train();optimizer=torch.optim.AdamW(model.optimizer_groups(decoder_lr=args.lr,encoder_lr=args.encoder_lr))
        set_learning_rates(optimizer,args,0);optimizer.zero_grad(set_to_none=True)
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for view in views:
            budget.check(force=True)
            loss=normalized_loss(model.components(view['target'],view['mel'],view['duration']),totals)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite gradient probe')
            loss.backward();del loss
        audit=gradient_audit(model);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),before[k])]
        for prefix in audit['group_gradient_norms']:
            if not any(k.startswith(prefix) for k in changed):raise RuntimeError('disconnected block '+prefix)
        report['rows'].append(dict(id=id,gradient=audit,changed_tensors=len(changed),resources=budget.log_state()))
        print(json.dumps(dict(stage='event_gradient_probe',id=id,all_connected=True,parameters=config['parameters'],resources=budget.log_state())),flush=True)
        del views,optimizer,before
    smoke=batch(args,rows['Eltaw_IN'])[1]
    prediction=generate(model,smoke['mel'],smoke['duration'],smoke['target']['bpm'],max_events=500,check=budget.check)
    if any(not 0<=t<smoke['duration'] for t in prediction['event_times_seconds']):
        raise RuntimeError('smoke generated an out-of-audio event')
    report['generation_smoke']=dict(status=prediction['status'],events=prediction['event_count'],
        error=prediction['error'],termination=prediction['termination'],learned_song_eos=prediction['learned_song_eos'],unclosed_holds=prediction['unclosed_holds'],
        scope='bounded rollout on an existing physical 24-second crop; no quality claim')
    report.update(status='passed',probe_weights_discarded=True);write_json(args.output/'gradient_probe.json',report)
    del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return report


def launch(args):
    """Detach only after a validated probe; never leave an unobserved duplicate worker."""
    if not (args.output/'gradient_probe.json').exists():raise ValueError('run GPU probe before detached launch')
    if read(args.output/'gradient_probe.json')['config'].get('schema')!=SCHEMA:
        raise ValueError('legacy window probe cannot authorize a full-song launch')
    lock=(args.output/'.launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'launch.json'
    if path.exists() and not args.resume:raise ValueError('existing launch; inspect state before explicit resume')
    if path.exists():
        pid=read(path)['pid'];cmdline=Path(f'/proc/{pid}/cmdline')
        if cmdline.exists() and b'script.train_event4k' in cmdline.read_bytes():raise ValueError('training worker is still alive')
    log_path=args.output/('resume.log' if args.resume else 'worker.log')
    command=[sys.executable,'-u','-m','script.train_models','--gpu',os.environ.get('CUDA_VISIBLE_DEVICES','0'),
             '--worker','script.train_event4k','--','run',*sys.argv[2:]]
    with log_path.open('ab' if args.resume else 'xb',buffering=0) as log:
        child=subprocess.Popen(command,cwd=Path.cwd(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(path,dict(pid=child.pid,started=time.time(),command=command,log=str(log_path),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        requested_version='AL-Event4K@0.3.0' if args.bpm_condition else 'AL-Event4K@0.4.0'))
    time.sleep(3)
    if child.poll() is not None:raise RuntimeError('worker exited during startup: '+log_path.read_text()[-2000:])
    print(json.dumps(dict(status='launched',pid=child.pid,log=str(log_path))),flush=True)


def reference_clock(meta):
    return dict(bpm=meta['timebase_bpm'],beat_zero_seconds=meta.get('offset_seconds',0.),
                source='training_chart_metadata',verified=False)


@torch.no_grad()
def evaluate(model,args,budget,manifest,epoch,version,*,full):
    model.eval();scorer=scoring_module();profile=read('benchmarks/v1/protocol.json')['rhythm']
    cache=cache_inputs(args.cache);pool=[r for r in cache['rows'] if r['id']!='Eltaw_IN']
    chosen=[next(r for r in cache['rows'] if r['id']=='Eltaw_IN')]+random.Random(20261026).sample(pool,14)
    if not full:chosen=chosen[:1]
    mel_rows={r['id']:r for r in manifest['rows']};dev_rows={r['id']:r for r in manifest['development']}
    result=[];output=args.output/f'evaluation-e{epoch:04d}';output.mkdir(exist_ok=True)
    def one(id,split,view,reference,clock):
        prediction=generate(model,view['mel'],view['duration'],clock['bpm'],check=budget.check)
        metrics=scorer.score(reference,prediction,clock,profile)
        if split=='train15':
            matched=sum(len(match_events([n['time_seconds'] for n in reference if n['position']==lane and n['type']==kind],
                [n['time_seconds'] for n in prediction['notes'] if n['position']==lane and n['type']==kind],.05)) for lane in range(4) for kind in KINDS.values())
            metrics['joint_time_position_type_f1_50ms']=2*matched/max(1,len(reference)+len(prediction['notes']))
            ref_end=[n['end_seconds'] for n in reference if n['type']=='hold']
            pred_end=[n['end_seconds'] for n in prediction['notes'] if n['type']=='hold' and not n.get('unclosed_hold')]
            metrics['hold_end_f1_50ms']=2*len(match_events(ref_end,pred_end,.05))/max(1,len(ref_end)+len(pred_end))
        times=[0.,*prediction['onsets_seconds'],view['duration']]
        gaps=sorted([dict(start=a,end=b,seconds=b-a,reference_onsets=sum(a<t<b for t in {n['time_seconds'] for n in reference}))
            for a,b in zip(times,times[1:]) if b>a],key=lambda g:g['seconds'],reverse=True)[:5]
        row=dict(id=id,split=split,status=prediction['status'],metrics=metrics,longest_gaps=gaps,
            termination=prediction['termination'],learned_song_eos=prediction['learned_song_eos'],
            unclosed_holds=prediction['unclosed_holds'])
        write_json(output/f'{id}.json',dict(**row,version=version,epoch=epoch,prediction=prediction));result.append(row)
        print(json.dumps(dict(stage='event_evaluation',epoch=epoch,id=id,f1=metrics['onset_f1_50ms'],grid=metrics['grid_bonus'],status=prediction['status'])),flush=True)
    for row in chosen:
        sample=load_sample(args.cache,row);meta=sample['routes']['4k']['reference']
        view=torch.load(args.mel_cache/mel_rows[row['id']]['file'],map_location='cpu',weights_only=True)['views'][0]
        one(row['id'],'train15',view,natural_reference(sample),reference_clock(meta))
        del sample,view
    for song in read(E20)['songs']:
        if not full and song['id'] not in (40717,52417):continue
        if sha256(song['reference'])!=song['reference_sha256']:raise ValueError('development reference changed')
        view=torch.load(args.mel_cache/dev_rows[str(song['id'])]['file'],map_location='cpu',weights_only=True)
        one(str(song['id']),'E20_development',view,read(song['reference'])['notes'],song['clock']);del view
    aggregates={}
    for split in {r['split'] for r in result}:
        rows=[r for r in result if r['split']==split]
        aggregates[split]=dict(examples=len(rows),completed=sum(r['status']=='completed' for r in rows),
            **{k:sum(r['metrics'][k] for r in rows)/len(rows) for k in ['onset_f1_50ms','grid_bonus','time_type_f1']})
    write_json(output/'summary.json',dict(epoch=epoch,version=version,full=full,rows=result,aggregates=aggregates,
        scope='Same fixed train15 and reused development10 benchmark; HoldEnd excluded from onset F1. Not a blind test.'))
    return aggregates


def checkpoint_manifest(manifest,version):
    """One product version per run; preserve old labels as checkpoint aliases."""
    result=dict(manifest,product=version,version_policy='one_version_per_training_plan')
    result['versions']=[];aliases={}
    for entry in manifest['versions']:
        row=dict(entry);identity=f'{version}/e{row["epoch"]:04d}'
        aliases[row['version']]=identity
        if row['version']!=version:row.setdefault('legacy_version',row['version'])
        row.update(version=version,checkpoint_id=identity);result['versions'].append(row)
    result['aliases']={k:aliases.get(v,v) for k,v in manifest.get('aliases',{}).items()}
    return result


def check_resume_config(saved,current,output,checkpoint):
    if saved==current:return
    # Only an explicitly recorded metadata/evaluation patch may change this file.
    # Model, data, optimizer and schedule changes still require a separate experiment.
    if {k:v for k,v in saved.items() if k!='source_sha256'}!={k:v for k,v in current.items() if k!='source_sha256'}:
        raise ValueError('resume training config changed')
    old,new=saved['source_sha256'],current['source_sha256']
    if set(old)!=set(new) or {k for k in old if old[k]!=new[k]}!={'script/train_event4k.py'}:
        raise ValueError('resume model/data source changed')
    path=output/'resume_compatibility.json'
    expected=dict(scope='version_metadata_and_evaluation_only',checkpoint_sha256=sha256(checkpoint),
                  old_trainer_sha256=old['script/train_event4k.py'],new_trainer_sha256=new['script/train_event4k.py'])
    if not path.exists() or any(read(path).get(k)!=v for k,v in expected.items()):
        raise ValueError('resume source patch requires exact recorded compatibility')


def save_version(args,budget,model,optimizer,sampler,config,epoch,steps,curves,best):
    version=config.get('requested_version',f'AL-Event4K@0.{3 if args.bpm_condition else 4}.0')
    identity=f'{version}/e{epoch:04d}'
    versions_path=args.output/'versions.json';versions=read(versions_path) if versions_path.exists() else dict(versions=[],aliases={})
    versions=checkpoint_manifest(versions,version)
    if any(r['epoch']==epoch for r in versions['versions']):raise RuntimeError('immutable checkpoint already registered')
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    payload=dict(schema=SCHEMA,product=version,checkpoint_id=identity,config=config,epoch=epoch,steps=steps,model=state,
        curves=curves,parameter_sha256=state_hash(state),training_state=training_state(optimizer,sampler))
    latest=args.output/'model_latest.pt';budget.save(payload,latest)
    if curves[-1]['objective']<best:
        best=curves[-1]['objective'];budget.save({k:v for k,v in payload.items() if k!='training_state'},args.output/'model_best_fit.pt')
        versions['aliases']['best_fit']=identity
    retained=[]
    if epoch==1 or epoch%100==0 or epoch==args.epochs:
        path=args.output/f'model_e{epoch:04d}.pt';budget.save({k:v for k,v in payload.items() if k!='training_state'},path);retained.append(path.name)
    versions['versions'].append(dict(version=version,checkpoint_id=identity,epoch=epoch,steps=steps,parameter_sha256=payload['parameter_sha256'],
        checkpoint_sha256=sha256(latest),retained_inference_files=retained))
    versions['aliases']['latest']=identity;write_json(versions_path,versions)
    return version,best


def train(args,budget,manifest,targets,device):
    model,config=load_initial(args,targets);model.to(device)
    optimizer=torch.optim.AdamW(model.optimizer_groups(decoder_lr=args.lr,encoder_lr=args.encoder_lr))
    sampler=random.Random(args.seed);curves=[];steps=0;started=time.monotonic();best=math.inf
    latest=args.output/'model_latest.pt'
    if latest.exists():
        if not args.resume:raise ValueError('checkpoint exists; resume must be explicit')
        old=torch.load(latest,map_location='cpu',weights_only=True)
        if old.get('schema')!=SCHEMA:raise ValueError('legacy window checkpoint cannot resume the full-song protocol')
        check_resume_config(old['config'],config,args.output,latest)
        model.load_state_dict(old['model'],strict=True);restore_training(old['training_state'],optimizer,sampler)
        curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds'];best=min(r['objective'] for r in curves);del old
    else:torch.manual_seed(args.seed+1000)
    write_json(args.output/'config.json',config)
    def status(state,**extra):
        write_json(args.output/'training.json',dict(status=state,pid=os.getpid(),updated=time.time(),config=config,
            steps=steps,curves=curves,resources=budget.log_state(),**extra))
    version=config['requested_version']
    status('running')
    # A crash after checkpoint saving must not silently skip its scheduled evaluation.
    if curves:
        previous=len(curves);full=previous%args.full_eval_every==0 or previous==args.epochs
        summary=args.output/f'evaluation-e{previous:04d}'/'summary.json'
        if (previous%args.eval_every==0 or full) and (not summary.exists() or full and not read(summary).get('full')):
            status('evaluating',latest_version=version)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                evaluate(model,args,budget,manifest,previous,version,full=full)
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    for epoch in range(len(curves),args.epochs):
        order=list(targets['rows']);sampler.shuffle(order);model.train();stats=Counter()
        for index,row in enumerate(order):
            views=batch(args,row);counts={k:sum(v['target'][k] for v in views) for k in COUNTS}
            optimizer.zero_grad(set_to_none=True);set_learning_rates(optimizer,args,steps)
            for view in views:
                budget.check()
                parts=model.components(view['target'],view['mel'],view['duration'])
                loss=normalized_loss(parts,counts)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite Event4K loss')
                loss.backward()
                for k,v in parts.items():stats[k]+=float(v.detach()) if torch.is_tensor(v) else v
                for k in COUNTS:stats[k]+=view['target'][k]
                del parts,loss
            if index==0:write_json(args.output/'last_gradient.json',dict(epoch=epoch+1,step=steps+1,**gradient_audit(model)))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1;del views
            if (index+1)%10==0:
                status('running',active_epoch=epoch+1,songs=index+1)
                print(json.dumps(dict(stage='event_train',epoch=epoch+1,songs=index+1,steps=steps,resources=budget.log_state())),flush=True)
        record=dict(epoch=epoch+1,steps=steps,elapsed_seconds=time.monotonic()-started,
            time_ce=stats['time_ce']/stats['event_count'],fraction_mse=stats['fraction_mse']/stats['event_count'],
            event_bce=stats['event_bce']/stats['event_count'],eos_ce=stats['eos_ce']/stats['eos_count'],
            event_exact=stats['event_exact']/stats['event_count'],frame_accuracy=stats['frame_correct']/stats['event_count'],
            false_eos_rate=stats['false_eos']/stats['continue_count'],eos_recall=stats['eos_correct']/stats['eos_count'],
            conditional_time_mae_seconds=stats['offset_error']/stats['event_count'],counts={k:int(stats[k]) for k in COUNTS})
        record['objective']=sum(record[k] for k in ['time_ce','fraction_mse','event_bce','eos_ce']);curves.append(record)
        version,best=save_version(args,budget,model,optimizer,sampler,config,epoch+1,steps,curves,best)
        print(json.dumps(dict(version=version,**record)),flush=True);status('running',latest_version=version)
        full=(epoch+1)%args.full_eval_every==0 or epoch+1==args.epochs
        if (epoch+1)%args.eval_every==0 or full:
            status('evaluating',latest_version=version)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                evaluate(model,args,budget,manifest,epoch+1,version,full=full)
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    final=args.output/f'evaluation-e{args.epochs:04d}'/'summary.json'
    if not final.exists() or not read(final).get('full'):
        status('evaluating',latest_version=version)
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            evaluate(model,args,budget,manifest,args.epochs,version,full=True)
    status('completed',latest_version=version,checkpoint_sha256=sha256(latest))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','probe','run','launch'])
    p.add_argument('--output',type=Path,default=Path('out/event4k_fullsong_bpm_20261105'))
    p.add_argument('--architecture-config',type=Path,default=Path('docs/evaluations/assets/event4k_fullsong_20261105/architecture.json'))
    p.add_argument('--mel-cache',type=Path,default=Path('out/joint_audio_a_20261102/mel'))
    p.add_argument('--cache',type=Path,default=Path('out/full_corpus_v1/cache'))
    p.add_argument('--bpm-condition',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--eval-every',type=int,default=20);p.add_argument('--full-eval-every',type=int,default=100)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--encoder-lr',type=float,default=3e-4)
    p.add_argument('--warmup-steps',type=int,default=1500);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true')
    add_resource_args(p);p.set_defaults(max_rss_gib=6.)
    args=p.parse_args()
    if min(args.epochs,args.eval_every,args.full_eval_every)<1 or args.warmup_steps<0:p.error('invalid schedule')
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='launch':launch(args);return
    lock=(args.output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=torch.device('cpu' if args.stage=='prepare' else args.device);budget=ResourceBudget(args,device,args.output)
    try:
        manifest,targets=prepare(args,budget)
        if args.stage=='prepare':return
        path=args.output/'gradient_probe.json'
        if args.stage=='probe' or not path.exists():probe(args,budget,targets,device)
        else:
            model,config=load_initial(args,targets);del model
            if read(path)['config']!=config:raise ValueError('probe source/config changed; rerun probe')
        if args.stage=='run':train(args,budget,manifest,targets,device)
    except BaseException as error:
        write_json(args.output/'failure.json',dict(type=type(error).__name__,message=str(error),time=time.time(),pid=os.getpid()))
        path=args.output/'training.json'
        if path.exists():
            state=read(path);state.update(status='failed',error=str(error),updated=time.time());write_json(path,state)
        raise


if __name__=='__main__':main()

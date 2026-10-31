"""Matched full300 W0/W1 adaptation from one frozen natural-music parent.

Both arms reset AdamW, use the same parent weights, data, order and constant
learning rate. W1 changes the output-window protocol only. This is adaptation,
not the controlled from-scratch study and not automatic product promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.util
import json
import math
from pathlib import Path
import random
import time

import torch

from script.train_alignment_scratch import training_state, restore_training
from script.train_anchored_ar import condition, epoch_objective, COUNTS
from script.train_grounded_ar import checked_pairs, encode_waveform, load_inputs, read
from script.train_soft_onset import cache_inputs, load_sample
from src.anchored_ar import AnchoredAR, encode_target, normalized_loss, generate as whole_generate
from src.audio_understanding.inference import sha256
from src.audio_understanding.evaluation import match_events
from src.conditional_training import empty_targets
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart_tokenizer import ChartTokenizer
from src.pattern_diagnostics import KINDS, LANE_X
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import state_hash, training_preflight
from src.windowed_ar import WindowedAR, encode_window_target, generate as window_generate

SCHEMA='audiolab-full300-window-adaptation-v1'
SOURCES=('script/train_windowed_corpus.py','script/train_anchored_ar.py','src/anchored_ar.py',
    'src/windowed_ar.py','src/models/chart_decoder.py','src/models/chart_history.py',
    'script/train_grounded_ar.py','script/train_soft_onset.py')
E20=Path('docs/evaluations/assets/phira_generalization10_20261029/selection.json')


def load_parent_model(parent,architecture,seed):
    if parent['config']['variant']!='audio' or parent['config']['decoder_config'].get('n_lanes',4)!=4:
        raise ValueError('this matched pilot requires the independent 4K A parent')
    torch.manual_seed(seed)
    cfg=parent['config']['decoder_config']
    model=WindowedAR(cfg) if architecture=='window' else AnchoredAR(cfg)
    keys=model.load_state_dict(parent['model'],strict=architecture=='whole')
    if architecture=='window':
        if keys.missing_keys!=['bounds_proj.weight'] or keys.unexpected_keys:
            raise ValueError('unexpected migration differences')
        assert not model.bounds_proj.weight.any()
    common={k:v for k,v in model.state_dict().items() if k!='bounds_proj.weight'}
    if state_hash(common)!=state_hash(parent['model']):raise ValueError('parent weights changed on migration')
    return model


@torch.no_grad()
def prepare_checks(args,budget,device):
    dest=args.checks;manifest=dest/'manifest.json'
    if manifest.exists() or list(dest.glob('*.pt')):raise ValueError('existing/partial evaluation cache must be inspected')
    selection=read(E20);encoder,onset=load_inputs(args,device);processor=AudioGPUprocessor(device='cpu')
    result=dict(schema=SCHEMA,selection_sha256=sha256(E20),
        frontend_sha256=sha256(args.model_root/'decoder/decoder_best.pt'),rows=[])
    if result['frontend_sha256']!=selection['frozen_frontends']['4k']['sha256']:
        raise ValueError('E20 frontend mismatch')
    for song in selection['songs']:
        if sha256(song['audio'])!=song['audio_sha256'] or sha256(song['reference'])!=song['reference_sha256']:
            raise ValueError('frozen development song changed')
        view=encode_waveform(processor.load_from_path(song['audio']),encoder,onset,budget,device)
        path=dest/f'{song["id"]}.pt';budget.save(view,path)
        result['rows'].append(dict(id=song['id'],file=path.name,sha256=sha256(path)))
        print(json.dumps(dict(stage='prepare_checks',song=song['id'],duration=view['duration'])),flush=True)
    write_json(manifest,result)


def natural_reference(sample):
    route=sample['routes']['4k'];scale=60/(32*route['reference']['timebase_bpm']);offset=route['reference']['offset_seconds']
    return [dict(time_seconds=n.tick*scale+offset,end_seconds=(n.tick+n.dur)*scale+offset,
        position=n.lane,type=KINDS[n.type],screen_x=LANE_X[n.lane]) for n in ChartTokenizer(4).decode_tokens(route['tokens'].tolist(),strict=True)]


def scoring_module():
    path=Path('docs/evaluations/assets/phira_generalization10_20261029/benchmark.py')
    spec=importlib.util.spec_from_file_location('frozen_e20_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@torch.no_grad()
def evaluate(model,args,budget,epoch,full=True):
    model.eval();generator=window_generate if args.architecture=='window' else whole_generate
    profile=read('benchmarks/v1/protocol.json')['rhythm'];scorer=scoring_module();result=[]
    cache=cache_inputs(args.cache);pool=[r for r in cache['rows'] if r['id']!='Eltaw_IN']
    chosen=[next(r for r in cache['rows'] if r['id']=='Eltaw_IN')]+random.Random(20261026).sample(pool,14)
    if not full:chosen=chosen[:1]
    selected=read(E20);check_manifest=read(args.checks/'manifest.json')
    if check_manifest['selection_sha256']!=sha256(E20):raise ValueError('evaluation selection changed')
    output=args.output/f'evaluation-e{epoch:04d}';output.mkdir(exist_ok=True)

    def one(id,split,view,reference,clock):
        p=generator(model,view,clock['bpm'],use_soft=False,greedy=True,max_tokens=12000,check=budget.check)
        p.pop('trace',None);score=scorer.score(reference,p,clock,profile)
        if split=='train15':
            matched=0
            for lane in range(4):
                for kind in KINDS.values():
                    a=[n['time_seconds'] for n in reference if n['position']==lane and n['type']==kind]
                    b=[n['time_seconds'] for n in p['notes'] if n['position']==lane and n['type']==kind]
                    matched+=len(match_events(a,b,.05))
            score['joint_time_position_type_f1_50ms']=2*matched/(len(reference)+len(p['notes']))
            score['completed_only_joint_f1_50ms']=score['joint_time_position_type_f1_50ms'] if p['status']=='completed' else 0.
        # Include leading/trailing inactivity, so empty-tail collapse cannot
        # appear as a good longest-gap score merely because it emitted once.
        times=[0.,*p['onsets_seconds'],view['duration']]
        gaps=[dict(start=a,end=b,seconds=b-a,reference_onsets=sum(a<t<b for t in {n['time_seconds'] for n in reference}))
              for a,b in zip(times,times[1:]) if b>a]
        gaps=sorted(gaps,key=lambda g:g['seconds'],reverse=True)
        for gap in gaps:
            held_until=max((n['end_seconds'] for n in p['notes'] if n['time_seconds']<=gap['start']),default=gap['start'])
            gap['no_active_note_seconds']=max(0,gap['end']-max(gap['start'],held_until))
        record=dict(id=id,split=split,status=p['status'],metrics=score,longest_gaps=gaps[:5],
            forced_windows=p.get('forced_windows'),learned_window_end_fraction=p.get('learned_window_end_fraction'))
        write_json(output/f'{id}.json',dict(**record,prediction=p))
        result.append(record);print(json.dumps(dict(stage='evaluation',epoch=epoch,id=id,split=split,
            status=p['status'],f1=score['onset_f1_50ms'],grid=score['grid_bonus'],largest_gap_seconds=gaps[0]['seconds'] if gaps else 0)),flush=True)

    for row in chosen:
        sample=load_sample(args.cache,row);meta=sample['routes']['4k']['reference']
        # Canonical training notes already use physical audio time; clock offset
        # comes only from the same frozen reference metadata used in old checks.
        clock=dict(bpm=meta['timebase_bpm'],beat_zero_seconds=meta.get('offset_seconds',0.),source='training reference',verified=True)
        one(row['id'],'train15',condition(sample),natural_reference(sample),clock)
        del sample
    songs=selected['songs'] if full else [s for s in selected['songs'] if s['id'] in (40717,52417)]
    for song in songs:
        row=next(r for r in check_manifest['rows'] if r['id']==song['id']);path=args.checks/row['file']
        if sha256(path)!=row['sha256']:raise ValueError('evaluation features changed')
        view=torch.load(path,map_location='cpu',weights_only=True)
        one(str(song['id']),'E20_development',view,read(song['reference'])['notes'],song['clock']);del view
    aggregates={}
    for split in {r['split'] for r in result}:
        part=[r for r in result if r['split']==split]
        aggregates[split]=dict(examples=len(part),completed=sum(r['status']=='completed' for r in part),
            **{k:sum(r['metrics'][k] for r in part)/len(part) for k in ('onset_f1_25ms','onset_f1_50ms','onset_f1_100ms','grid_bonus','time_type_f1')})
        if split=='train15':
            aggregates[split]['completed_only_joint_f1_50ms']=sum(r['metrics']['completed_only_joint_f1_50ms'] for r in part)/len(part)
    write_json(output/'summary.json',dict(epoch=epoch,full=full,rows=result,aggregates=aggregates,
        scope='Raw generation; E20 is a development regression set, not a new blind test. '
            'Time+type ignores moving-line coordinates; not time+position+type joint F1.'))
    return aggregates


def train(args,budget,device):
    if args.mode!='4k':raise ValueError('first natural pilot isolates 4K A')
    cache=cache_inputs(args.cache);pairs=checked_pairs(args);assert len(cache['rows'])==300
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    model=load_parent_model(parent,args.architecture,args.seed).to(device);model.decoder.gradient_checkpointing=True
    config=dict(architecture=args.architecture,mode='4k',variant='audio',seed=args.seed,epochs=args.epochs,lr=args.lr,
        parent=str(args.parent),parent_sha256=sha256(args.parent),parent_epoch=parent['epoch'],
        initial_shared_sha256=state_hash(parent['model']),decoder_config=parent['config']['decoder_config'],
        parameters=sum(p.numel() for p in model.parameters()),cache_sha256=sha256(args.cache/'cache.json'),
        pairs_sha256=sha256(args.pairs/'pairs.json'),source_sha256={p:sha256(p) for p in SOURCES},
        schedule='constant_lr; new AdamW in BOTH arms; matched parent weights and sample/dropout seeds',
        policy='All 300 full songs + unchanged two physical crops and real silence per song. '
            'Frozen whole audio; no type/encoder/capacity/history-noise changes; raw greedy evaluation.')
    del parent
    target_fn=encode_window_target if args.architecture=='window' else encode_target
    targets={};by_id={r['id']:r for r in pairs['rows']}
    for row in cache['rows']:
        sample=load_sample(args.cache,row);targets[row['id']]=target_fn(sample['routes']['4k'],sample['duration']);budget.check()
    del sample
    quiet={float(k):torch.load(args.pairs/v['file'],map_location='cpu',weights_only=True) for k,v in pairs['silence'].items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr);sampler=random.Random(args.seed)
    curves=[];steps=0;started=time.monotonic();torch.manual_seed(args.seed+1000)
    if (args.output/'model_latest.pt').exists():
        if not args.resume:raise ValueError('checkpoint exists; --resume required')
        old=torch.load(args.output/'model_latest.pt',map_location='cpu',weights_only=True)
        if old['config']!=config:raise ValueError('resume config/source changed')
        model.load_state_dict(old['model']);restore_training(old['training_state'],optimizer,sampler)
        curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds'];del old
    best=min((r['objective'] for r in curves),default=math.inf)
    write_json(args.output/'training.json',dict(status='running',config=config,curves=curves))
    print(json.dumps(dict(stage='validated',songs=len(targets),config=config,resources=budget.log_state())),flush=True)
    for epoch in range(len(curves),min(args.epochs,args.until_epoch or args.epochs)):
        order=list(cache['rows']);sampler.shuffle(order);model.train();stats=Counter()
        for i,row in enumerate(order):
            sample=load_sample(args.cache,row);pair=torch.load(args.pairs/by_id[row['id']]['file'],map_location='cpu',weights_only=True)
            crops=[target_fn(t,pair['duration']) for t in pair['targets']]
            silent=target_fn(empty_targets(crops[0]['bpm']),pair['duration'])
            batch=[(targets[row['id']],condition(sample)),*zip(crops,pair['views']),(silent,quiet[pair['duration']])]
            totals={k:sum(t[k] for t,_ in batch) for k in COUNTS};optimizer.zero_grad(set_to_none=True)
            for target,view in batch:
                estimate=training_preflight(model.decoder,dict(routes={'4k':dict(tokens=target['tokens'],features=view['features'])}),'4k',budget,device)
                if device.type=='cuda':
                    free,total=torch.cuda.mem_get_info(device)
                    if estimate['bytes']+target['time_count']*view['features'].shape[1]*32>min(free,total*args.gpu_memory_fraction-torch.cuda.memory_allocated(device)):
                        raise RuntimeError('pointer working set exceeds resource budget')
                parts=model.components(target,view,use_soft=False);loss=normalized_loss(parts,totals)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite natural adaptation loss')
                loss.backward()
                for k,v in parts.items():stats[k]+=float(v.detach()) if torch.is_tensor(v) else v
                for k in COUNTS:stats[k]+=target[k]
                del parts,loss
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
            del batch,sample,pair,crops
            if (i+1)%100==0:print(json.dumps(dict(stage='train',epoch=epoch+1,songs=i+1,steps=steps)),flush=True)
        record=dict(epoch=epoch+1,steps=steps,elapsed_seconds=time.monotonic()-started,counts=dict((k,int(stats[k])) for k in COUNTS),
            frame_ce=stats['frame_ce']/stats['time_count'],frame_accuracy=stats['frame_correct']/stats['time_count'],
            subframe_mse=stats['subframe_mse']/stats['time_count'],layout_ce=stats['layout_ce']/stats['layout_count'],
            stop_bce=stats['stop_bce']/stats['stop_count'],continue_bce=stats['continue_bce']/stats['continue_count'],
            false_stop_rate=stats['false_stops']/stats['continue_count'],end_stop_recall=stats['correct_stops']/stats['stop_count'],
            conditional_offset_mae_seconds=stats['conditional_offset_absolute_seconds']/stats['time_count'])
        record['objective']=epoch_objective(record);curves.append(record)
        saved=dict(schema=SCHEMA,config=config,curves=curves,epoch=epoch+1,steps=steps,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},training_state=training_state(optimizer,sampler))
        budget.save(saved,args.output/'model_latest.pt')
        if record['objective']<best:
            best=record['objective'];budget.save({k:v for k,v in saved.items() if k!='training_state'},args.output/'model_best_fit.pt')
        del saved
        write_json(args.output/'training.json',dict(status='running',config=config,curves=curves,resources=budget.log_state()))
        print(json.dumps(record),flush=True)
        last=epoch+1==min(args.epochs,args.until_epoch or args.epochs)
        if (args.eval_every and (epoch+1)%args.eval_every==0) or (last and epoch+1==args.epochs):
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                summary=evaluate(model,args,budget,epoch+1,full=epoch+1==args.epochs)
            print(json.dumps(dict(stage='evaluated',epoch=epoch+1,aggregates=summary)),flush=True)
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    write_json(args.output/'training.json',dict(status='completed' if epoch+1==args.epochs else 'phase_completed',
        config=config,curves=curves,resources=budget.log_state(),checkpoint_sha256=sha256(args.output/'model_latest.pt')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare-checks','train'))
    p.add_argument('--architecture',choices=('whole','window'),default='window')
    p.add_argument('--parent',type=Path,default=Path('out/ar_saturation_20261023/continued1400/audio/model_latest.pt'))
    p.add_argument('--output',type=Path);p.add_argument('--epochs',type=int,default=20);p.add_argument('--until-epoch',type=int)
    p.add_argument('--eval-every',type=int,default=10);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true');p.add_argument('--cache',type=Path,default=Path('out/full_corpus_v1/cache'))
    p.add_argument('--pairs',type=Path,default=Path('out/grounded_ar_20261023/pairs'))
    p.add_argument('--checks',type=Path,default=Path('out/windowed_ar_20261031/natural_checks'))
    p.add_argument('--model-root',type=Path,default=Path('checkpoints/timing_v1'));p.add_argument('--mode',default='4k',choices=('4k',))
    p.add_argument('--device',default='cuda:0');add_resource_args(p);p.set_defaults(max_rss_gib=4.)
    args=p.parse_args();torch.set_num_threads(2)
    if args.stage=='train' and args.output is None:p.error('--output required')
    directory=args.output or args.checks;directory.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device);budget=ResourceBudget(args,device,directory)
    (prepare_checks if args.stage=='prepare-checks' else train)(args,budget,device)


if __name__=='__main__':main()

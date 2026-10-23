"""Bounded time-pointer experiments, including state-preserving longer training."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import random
import time

import torch

from script.train_alignment_scratch import training_state, restore_training
from script.train_grounded_ar import (checked_pairs, load_inputs, encode_waveform, read, SONGS)
from script.train_soft_onset import cache_inputs, load_sample
from src.anchored_ar import SCHEMA, AnchoredAR, encode_target, normalized_loss, generate
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.conditional_training import empty_targets
from src.data.audio2mel import AudioGPUprocessor
from src.model_benchmark import evaluate as benchmark
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import state_hash, training_preflight

SOURCES=('src/anchored_ar.py','script/train_anchored_ar.py','src/models/chart_decoder.py',
         'src/models/chart_history.py','src/conditional_training.py','script/train_grounded_ar.py')
COUNTS=('time_count','layout_count','stop_count','continue_count')


def continuation_config(parent, config, *, parent_path, parent_sha256):
    """Extend a completed pilot without changing its training task or state.

    A continuation is a new run with its own provenance, not a rewrite of the
    pilot's cosine horizon. Keep the actual terminal AdamW learning rates.
    """
    if parent.get('schema') != SCHEMA:
        raise ValueError('continuation requires an absolute-time AR checkpoint')
    previous=parent['config']
    for key in ('schema','variant','decoder_config','seed','parameters',
                'full_cache_sha256','pair_cache_sha256','loss','policy'):
        if previous.get(key) != config.get(key):
            raise ValueError(f'continuation changes training task: {key}')
    curves=parent.get('curves',[])
    if not curves or parent['epoch']!=len(curves) or curves[-1]['epoch']!=parent['epoch'] \
            or curves[-1]['steps']!=parent['steps'] or parent['steps']!=300*parent['epoch']:
        raise ValueError('continuation requires a complete 300-song epoch boundary')
    if config['epochs'] <= parent['epoch']:
        raise ValueError('continuation target must exceed parent epoch')
    state=parent.get('training_state',{})
    if not all(key in state for key in ('optimizer','sampler','torch_rng','cuda_rng')):
        raise ValueError('continuation requires optimizer and RNG state')
    rates=[float(g['lr']) for g in state['optimizer']['param_groups']]
    if not rates or any(not math.isfinite(rate) or rate<=0 for rate in rates):
        raise ValueError('invalid parent terminal learning rate')
    result=deepcopy(config)
    result.update(initialization='continue parent weights, AdamW moments and sampler/torch RNG; frozen original acoustic frontends',
        initial_state_sha256=state_hash(parent['model']),
        continuation=dict(parent_path=str(parent_path),parent_sha256=parent_sha256,
            parent_epoch=parent['epoch'],parent_steps=parent['steps'],
            parent_source_sha256=previous['source_sha256'],
            schedule='constant_parent_terminal_lr',learning_rates=rates))
    return result


def epoch_objective(record):
    counts=record['counts']
    stop=(record['stop_bce']*counts['stop_count']+
          record['continue_bce']*counts['continue_count'])/max(1,counts['stop_count']+counts['continue_count'])
    return record['frame_ce']+record['subframe_mse']+.25*record['layout_ce']+.5*stop


def condition(sample):
    return dict(features=sample['routes']['4k']['features'],soft_onsets=sample['soft_onsets'],
                valid=sample['valid'],duration=sample['duration'])


def train(args,budget,device):
    cache=cache_inputs(args.cache);pairs=checked_pairs(args)
    decoder_config=read('docs/evaluations/diffusion_ar_attribution_20261023.json')['ar']['config']
    torch.manual_seed(args.seed)
    model=AnchoredAR(decoder_config).to(device)
    model.decoder.gradient_checkpointing=True
    config=dict(schema=SCHEMA,variant=args.variant,decoder_config=decoder_config,
        initialization='all decoder and pointer weights from scratch; frozen original acoustic frontends',
        initial_state_sha256=state_hash(model.state_dict()),seed=args.seed,epochs=args.epochs,lr=args.lr,
        parameters=sum(p.numel() for p in model.parameters()),
        full_cache_sha256=sha256(args.cache/'cache.json'),pair_cache_sha256=sha256(args.pairs/'pairs.json'),
        source_sha256={p:sha256(p) for p in SOURCES},
        loss='frame CE + conditional normalized subframe MSE + .25 layout CE + .5 event-normalized stop/continue BCE; joint update denominators',
        policy='global audio; pointer output chronological intervals; no grid snapping/onset threshold/type quota; continuous seconds export')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    rng=random.Random(args.seed);curves=[];steps=0;started=time.monotonic()
    path=args.output/'model_latest.pt'
    if args.continue_from:
        if args.continue_from.resolve()==path.resolve():
            raise ValueError('continuation must preserve the parent in a separate output')
        parent=torch.load(args.continue_from,map_location='cpu',weights_only=True)
        config=continuation_config(parent,config,parent_path=args.continue_from,
                                   parent_sha256=sha256(args.continue_from))
        model.load_state_dict(parent['model'],strict=True)
        restore_training(parent['training_state'],optimizer,rng)
        curves=deepcopy(parent['curves']);steps=parent['steps']
        started-=curves[-1]['elapsed_seconds'];del parent
    if path.exists():
        if not args.resume:raise ValueError('checkpoint exists; use --resume')
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['config']!=config:raise ValueError('resume source/config differs')
        model.load_state_dict(old['model'],strict=True);restore_training(old['training_state'],optimizer,rng)
        curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds'];del old
    # Validate every reference before the first update. Only compact CPU labels
    # are retained; acoustic feature arrays continue to load one song at a time.
    targets={};by_id={r['id']:r for r in pairs['rows']}
    for row in cache['rows']:
        budget.check()
        sample=load_sample(args.cache,row)
        try:targets[row['id']]=encode_target(sample['routes']['4k'],sample['duration'])
        except ValueError as exc:raise ValueError(f"{row['id']}: {exc}") from exc
        del sample
    quiet={float(k):torch.load(args.pairs/v['file'],map_location='cpu',weights_only=True)
           for k,v in pairs['silence'].items()}
    print(json.dumps(dict(stage='validated',songs=len(targets),config=config,resources=budget.log_state())),flush=True)
    last_epoch=min(args.epochs,args.until_epoch or args.epochs)
    if last_epoch<len(curves):raise ValueError('requested stop precedes the resumed epoch')
    best_objective=min((epoch_objective(r) for r in curves),default=math.inf)
    write_json(args.output/'training.json',dict(config=config,curves=curves))
    for epoch in range(len(curves),last_epoch):
        order=list(cache['rows']);rng.shuffle(order);model.train();stats=Counter()
        for i,row in enumerate(order):
            sample=load_sample(args.cache,row)
            pair=torch.load(args.pairs/by_id[row['id']]['file'],map_location='cpu',weights_only=True)
            crops=[encode_target(t,pair['duration']) for t in pair['targets']]
            silent=encode_target(empty_targets(crops[0]['bpm']),pair['duration'])
            batch=[(targets[row['id']],condition(sample)),*zip(crops,pair['views']),
                   (silent,quiet[pair['duration']])]
            totals={k:sum(t[k] for t,_ in batch) for k in COUNTS}
            if 'continuation' in config:
                for group,rate in zip(optimizer.param_groups,config['continuation']['learning_rates'],strict=True):
                    group['lr']=rate
            else:
                factor=min(1.,(steps+1)/max(1,.05*args.epochs*300))*(.1+.9*(1+math.cos(math.pi*steps/max(1,args.epochs*300-1)))/2)
                for group in optimizer.param_groups:group['lr']=args.lr*factor
            optimizer.zero_grad(set_to_none=True)
            for target,view in batch:
                fake=dict(routes={'4k':dict(tokens=target['tokens'],features=view['features'])})
                estimate=training_preflight(model.decoder,fake,'4k',budget,device)
                extra=target['time_count']*view['features'].shape[1]*32
                if device.type=='cuda':
                    free,total=torch.cuda.mem_get_info(device)
                    if estimate['bytes']+extra>min(free,total*args.gpu_memory_fraction-torch.cuda.memory_allocated(device)):
                        raise RuntimeError('Resource budget: pointer training estimate exceeds allowance')
                components=model.components(target,view,use_soft=args.variant=='soft')
                loss=normalized_loss(components,totals)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite anchored loss')
                loss.backward()
                for key,value in components.items():stats[key]+=float(value.detach()) if torch.is_tensor(value) else value
                for key in COUNTS:stats[key]+=target[key]
                del components,loss
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1
            del batch,sample,pair,crops
            if (i+1)%100==0:print(json.dumps(dict(stage='train',variant=args.variant,epoch=epoch+1,songs=i+1,steps=steps)),flush=True)
        record=dict(epoch=epoch+1,steps=steps,elapsed_seconds=time.monotonic()-started,
            counts={k:stats[k] for k in COUNTS},
            frame_ce=stats['frame_ce']/stats['time_count'],
            frame_accuracy=stats['frame_correct']/stats['time_count'],
            conditional_offset_mae_seconds=stats['conditional_offset_absolute_seconds']/stats['time_count'],
            subframe_mse=stats['subframe_mse']/stats['time_count'],
            layout_ce=stats['layout_ce']/stats['layout_count'],
            stop_bce=stats['stop_bce']/stats['stop_count'],continue_bce=stats['continue_bce']/stats['continue_count'],
            false_stop_rate=stats['false_stops']/stats['continue_count'],
            end_stop_recall=stats['correct_stops']/stats['stop_count'],
            learning_rates=[g['lr'] for g in optimizer.param_groups])
        record['objective']=epoch_objective(record)
        curves.append(record)
        saved=dict(schema=SCHEMA,config=config,curves=curves,epoch=epoch+1,steps=steps,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},training_state=training_state(optimizer,rng))
        budget.save(saved,path)
        if record['objective']<best_objective:
            best_objective=record['objective']
            # A compact inference snapshot, never a substitute for the latest
            # optimizer/RNG checkpoint. Preserve only one best-fit snapshot.
            compact={k:v for k,v in saved.items() if k!='training_state'}
            budget.save(compact,args.output/'model_best_fit.pt');del compact
        del saved
        write_json(args.output/'training.json',dict(config=config,curves=curves))
        print(json.dumps(record),flush=True)
        if args.eval_every and ((epoch+1)%args.eval_every==0 or epoch+1==last_epoch):
            evaluation=deepcopy(args)
            evaluation.checkpoint=path
            evaluation.output=args.output/f'evaluation-e{epoch+1:04d}'
            evaluation.output.mkdir(parents=True,exist_ok=True)
            # Encoder construction and diagnostics must not consume the
            # training dropout RNG. Sampler is a separate random.Random.
            devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                evaluate(evaluation,budget,device)
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    resources=budget.log_state();write_json(args.output/'training_resources.json',resources)
    print(json.dumps(dict(status='trained',resources=resources)),flush=True)


def score_rollout(model,view,bpm,reference,args,budget,greedy):
    result=generate(model,view,bpm,use_soft=args.variant=='soft',greedy=greedy,check=budget.check)
    result.pop('trace',None)
    if 'onsets_seconds' in result:
        times=result['onsets_seconds']
        for ms in (25,50):
            count=len(match_events(reference,times,ms/1000))
            result[f'f1_{ms}ms']=2*count/(len(reference)+len(times)) if reference or times else 1.
        result['reference_onsets']=len(reference)
        result['metric_scope']='completed_generation' if result['status']=='completed' else 'partial_failed_generation_diagnostic_only'
    return result


@torch.no_grad()
def evaluate(args,budget,device):
    checkpoint=args.checkpoint or args.output/'model_latest.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['schema']!=SCHEMA or ck['config']['variant']!=args.variant:raise ValueError('wrong anchored schema/variant')
    checkpoint_epoch,checkpoint_steps=ck['epoch'],ck['steps']
    model=AnchoredAR(ck['config']['decoder_config']).to(device).eval();model.load_state_dict(ck['model'],strict=True);del ck
    cache=cache_inputs(args.cache);pairs=checked_pairs(args)
    source=read(args.baseline_input);baseline=next(r for r in source['runs'] if r['id']=='4k/global/seed42')
    runs=[dict(id=f'4k/anchored-{args.variant}/{suffix}',cases=deepcopy(baseline['cases']),predictions={},
        metadata=dict(stage='raw_layout',mode='4k',variant=args.variant,greedy=greedy,seed=42,
            checkpoint_sha256=sha256(checkpoint),checkpoint_epoch=checkpoint_epoch,checkpoint_steps=checkpoint_steps,
            sources={p:sha256(p) for p in SOURCES},
            repairs='none',time_representation='predicted continuous seconds',schema=SCHEMA))
          for greedy,suffix in ((True,'greedy'),(False,'seed42'))]
    cases={c['id']:c for c in baseline['cases']};teacher=[]
    for row in cache['rows']:
        if row['id'] not in SONGS:continue
        sample=load_sample(args.cache,row);view=condition(sample);target=encode_target(sample['routes']['4k'],sample['duration'])
        diagnostic={}
        for name,acoustic,soft in (('matched',True,True),('zero_audio',False,True),('zero_soft',True,False)):
            v=dict(view)
            if not acoustic:v['features']=torch.zeros_like(v['features'])
            if not soft:v['soft_onsets']=torch.zeros_like(v['soft_onsets'])
            component=model.components(target,v,use_soft=args.variant=='soft')
            diagnostic[name]={k:float(v) for k,v in component.items()}
        teacher.append(dict(id=row['id'],counts={k:target[k] for k in COUNTS},components=diagnostic))
        for run in runs:
            result=score_rollout(model,view,target['bpm'],cases[row['id']]['events_seconds'],args,budget,run['metadata']['greedy'])
            run['predictions'][row['id']]=result
            print(json.dumps(dict(stage='evaluate',case=row['id'],greedy=run['metadata']['greedy'],status=result['status'],f1=result.get('f1_50ms'),notes=result.get('note_count'))),flush=True)
        del sample,view,target
    encoder,onset=load_inputs(args,device)
    for row in read(args.probes/'fixtures/fixtures.json')['rows']:
        if sha256(row['audio'])!=cases[row['id']]['audio_sha256']:raise ValueError('frozen probe changed')
        waveform=AudioGPUprocessor(device='cpu').load_from_path(row['audio'])
        view=encode_waveform(waveform,encoder,onset,budget,device)
        for run in runs:
            result=score_rollout(model,view,120.,cases[row['id']]['events_seconds'],args,budget,run['metadata']['greedy'])
            run['predictions'][row['id']]=result
            print(json.dumps(dict(stage='evaluate',case=row['id'],greedy=run['metadata']['greedy'],status=result['status'],f1=result.get('f1_50ms'),notes=result.get('note_count'))),flush=True)
    del encoder,onset;gc.collect();torch.cuda.empty_cache()
    shifts=[]
    for row in pairs['diagnostics']:
        pair=torch.load(args.pairs/row['file'],map_location='cpu',weights_only=True)
        for index,(route,view) in enumerate(zip(pair['targets'],pair['views'])):
            target=encode_target(route,pair['duration'])
            result=score_rollout(model,view,target['bpm'],target['reference_times'],args,budget,False)
            shifts.append(dict(id=row['id'],view=index,reference_first_seconds=target['reference_times'][0],
                generated_first_seconds=(result.get('onsets_seconds') or [None])[0],
                generation={k:v for k,v in result.items() if k not in ('notes','onsets_seconds')}))
    bundle=dict(schema=source['schema'],runs=runs);write_json(args.output/'inputs.json',bundle)
    report=benchmark(bundle,read(args.protocol));report['resources']=budget.log_state()
    write_json(args.output/'report.json',report);write_json(args.output/'diagnostics.json',dict(teacher=teacher,shift_pairs=shifts))
    print(json.dumps(dict(status='evaluated',summaries={r['id']:r['summary'] for r in report['runs']},resources=budget.log_state())),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('train','evaluate'));p.add_argument('--variant',choices=('audio','soft'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    p.add_argument('--continue-from',type=Path,help='continue a completed pilot, retaining AdamW/RNG and terminal LR')
    p.add_argument('--until-epoch',type=int,help='stop at this cumulative epoch without changing the configured schedule')
    p.add_argument('--eval-every',type=int,default=0,help='evaluate free generation at cumulative epoch intervals; no RNG change')
    p.add_argument('--checkpoint',type=Path,help='evaluation checkpoint, allowing reports outside its training directory')
    p.add_argument('--cache',type=Path,default=Path('out/full_corpus_v1/cache'))
    p.add_argument('--pairs',type=Path,default=Path('out/grounded_ar_20261023/pairs'))
    p.add_argument('--model-root',type=Path,default=Path('checkpoints/timing_v1'))
    p.add_argument('--probes',type=Path,default=Path('out/onset_attention_20261022/onsets'))
    p.add_argument('--baseline-input',type=Path,default=Path('out/benchmark_v1/inputs.json'))
    p.add_argument('--protocol',type=Path,default=Path('benchmarks/v1/protocol.json'))
    p.add_argument('--device',default='cuda:0');add_resource_args(p);p.set_defaults(max_rss_gib=4.)
    args=p.parse_args()
    if args.epochs<1 or not math.isfinite(args.lr) or args.lr<=0:raise ValueError('positive epochs/lr required')
    if args.eval_every<0 or (args.until_epoch is not None and args.until_epoch<1):raise ValueError('invalid evaluation/stop interval')
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    budget=ResourceBudget(args,torch.device(args.device),args.output)
    {'train':train,'evaluate':evaluate}[args.stage](args,budget,budget.device)


if __name__=='__main__':main()

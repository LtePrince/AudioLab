"""E16 K pattern diagnostic: observed chart prefix, unknown future positions.

Timing and one-Tap-per-onset are GIVEN in this isolation task. No time, type or
note-count quality claims may be made from it. Identical audio has five chart
patterns; only the observed prefix identifies the requested continuation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from script.probe_audio_localization import sources
from script.train_alignment_scratch import training_state, restore_training
from script.train_grounded_ar import load_inputs, encode_waveform
from src.audio_understanding import SAMPLE_RATE
from src.audio_understanding.inference import sha256
from src.chart_windows import chart_windows
from src.data.chart_tokenizer import BOS, ChartTokenizer, TokenNote
from src.models.chart_history import CausalChartHistory
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import MEMORY_SECONDS, state_hash
from src.windowed_ar import encode_window_target
from src.windowed_variants import VariantAR

SCHEMA='audiolab-explicit-prefix-pattern-v1'
FAMILIES=('jack','short_jacks','long_jacks','alternation','staircase')
PREFIX=8
SOURCE_FILES=('script/train_pattern_continuation.py','src/windowed_variants.py',
    'src/windowed_ar.py','src/models/chart_decoder.py','src/models/chart_history.py')


def lanes_for(family,order,count=24):
    a,b,*_=order
    period={'jack':[a],'short_jacks':[a,a,b,b],'long_jacks':[a]*4+[b]*4,
            'alternation':[a,b],'staircase':list(order)}[family]
    return [period[i%len(period)] for i in range(count)]


def prefix_target(row):
    target=encode_window_target(row['route'],row['duration'])
    labels=target['layout_targets'];note=(labels>=50)&(labels<66)
    indices=note.nonzero().flatten()[PREFIX:]
    # Only suffix position loss. The prefix and future times are observed inputs;
    # future positions remain causal teacher-forcing labels, never input hints.
    return target,indices,((labels[indices]-50)//4)


@torch.no_grad()
def continuation(model,condition,times,prefix,*,check=lambda:None):
    """Only the observed prefix is passed; no future chart argument exists."""
    if len(prefix)!=PREFIX or len(times)<=PREFIX:raise ValueError('eight-note prefix required')
    device=next(model.parameters()).device;decoder=model.decoder;tk=ChartTokenizer(4)
    memory=decoder.prepare_memory(condition['features'].float().to(device),torch.zeros_like(condition['soft_onsets']).to(device))
    valid=condition['valid'].to(device);caches=decoder.init_cache(memory)
    history=CausalChartHistory(4,60/120/MEMORY_SECONDS)
    windows=chart_windows(condition['duration'],10.);wi=0;position=0
    selected=[];candidates=torch.tensor([tk.note(lane,1) for lane in range(4)],device=device)

    def step(token,clock):
        nonlocal position
        frame=clock/MEMORY_SECONDS;bounds=torch.tensor([[windows[wi].start,windows[wi].end]],device=device,dtype=torch.float64)
        logits,_=decoder.step(torch.tensor([token],device=device),torch.tensor([frame],device=device),position,
            caches,memory.shape[1],valid,history=torch.tensor([history.features(frame)],device=device),
            input_condition=model.window_condition(bounds))
        history.observe(token,frame);position+=1
        return logits[0]

    step(BOS,0.)
    for i,clock in enumerate(times):
        check()
        while clock>=windows[wi].end:
            wi+=1;step(tk.cond(2),windows[wi].start)
        logits=step(tk.cond(0),clock)
        lane=int(prefix[i]) if i<PREFIX else int(logits[candidates].argmax())
        selected.append(lane);step(tk.note(lane,1),clock);step(tk.cond(1),clock)
    return selected


@torch.no_grad()
def prepare(args,budget,device):
    if args.data.exists():raise ValueError('pattern dataset exists; refusing overwrite')
    encoder,onset=load_inputs(args,device);clips,metadata=sources(args.samples)
    permutations=list(itertools.permutations(range(4)));random.Random(817).shuffle(permutations)
    rows=[]
    for split,orders,seed in [('train',permutations[:8],817),('new_combinations',permutations[8:16],100817)]:
        for index,order in enumerate(orders):
            rng=random.Random(seed+index);spacing=(.25,.375,.5,.625)[index%4]
            start=(.5,.75,1.,1.25)[index//2%4]
            ticks=[round(64*(start+j*spacing)) for j in range(24)];times=[t/64 for t in ticks]
            duration=float(math.ceil(times[-1]+1.));wave=np.zeros(round(duration*SAMPLE_RATE),dtype='float32')
            # Waveform generation does not receive the pattern or lane order.
            for onset_time in times:
                clip=clips[3];p=round(onset_time*SAMPLE_RATE)
                wave[p:p+len(clip)]+=rng.uniform(.2,.9)*clip
            condition=encode_waveform(torch.from_numpy(wave).view(1,1,-1),encoder,onset,budget,device)
            for family in FAMILIES:
                lanes=lanes_for(family,order)
                notes=[TokenNote(t,lane,1) for t,lane in zip(ticks,lanes)]
                rows.append(dict(id=f'{split}-{index}-{family}',split=split,family=family,audio_group=f'{split}-{index}',
                    duration=duration,times=times,lanes=lanes,prefix=lanes[:PREFIX],condition=condition,
                    route=dict(tokens=torch.tensor(ChartTokenizer(4).encode_notes(notes)),reference=dict(timebase_bpm=120.,offset_seconds=0.))))
    # Equal audio plus different, unambiguous prefixes is the intended contract.
    for group in {r['audio_group'] for r in rows}:
        part=[r for r in rows if r['audio_group']==group]
        assert len({tuple(r['prefix']) for r in part})==len(FAMILIES)
        assert all(torch.equal(r['condition']['features'],part[0]['condition']['features']) for r in part)
    payload=dict(schema=SCHEMA,rows=rows,prefix_notes=PREFIX,sources=metadata,
        frontend_sha256=sha256(args.model_root/'decoder/decoder_best.pt'),
        policy='24 known onset slots, each one Tap; eight observed chart notes; predict next 16 positions. '
            'No pattern-id input, no future-chart hints, no output-period constraint. All song audio visible.')
    budget.save(payload,args.data)
    write_json(args.data.with_suffix('.json'),dict(schema=SCHEMA,sha256=sha256(args.data),rows=len(rows),
        train=40,new_combinations=40,prefix_notes=PREFIX,policy=payload['policy'],resources=budget.log_state()))
    print(json.dumps(dict(status='prepared',rows=len(rows),sha256=sha256(args.data))),flush=True)


def loss_for(model,row,target,indices,labels):
    d=model.decoder;device=next(model.parameters()).device;condition=row['condition']
    memory=d.prepare_memory(condition['features'].float().to(device),torch.zeros_like(condition['soft_onsets']).to(device))
    logits=d(target['tokens'][None].to(device),target['frames'][None].to(device),memory,condition['valid'].to(device),
        history=target['history'][None].to(device),input_condition=model.window_condition(target['bounds'].to(device))[None])
    note_ids=torch.tensor([ChartTokenizer(4).note(lane,1) for lane in range(4)],device=device)
    return F.cross_entropy(logits[0,indices.to(device)][:,note_ids],labels.to(device))


@torch.no_grad()
def evaluate(model,rows,args,budget,epoch,full):
    model.eval();results=[]
    for row in rows if full else [r for r in rows if r['split']=='train' and r['audio_group']=='train-0']:
        predicted=continuation(model,row['condition'],row['times'],row['prefix'],check=budget.check)
        expected=row['lanes'];right=sum(a==b for a,b in zip(predicted[PREFIX:],expected[PREFIX:]))
        transitions=sum(predicted[i]-predicted[i-1]==expected[i]-expected[i-1] for i in range(PREFIX,len(expected)))
        results.append(dict(id=row['id'],split=row['split'],family=row['family'],audio_group=row['audio_group'],
            prefix=row['prefix'],reference_suffix=expected[PREFIX:],predicted_suffix=predicted[PREFIX:],
            correct=right,count=len(expected)-PREFIX,transition_correct=transitions,exact=predicted==expected))
    groups=defaultdict(list)
    for row in results:
        groups[row['split']].append(row);groups[row['split']+'/'+row['family']].append(row)
    aggregates={k:dict(examples=len(v),position_accuracy=sum(r['correct'] for r in v)/sum(r['count'] for r in v),
        transition_accuracy=sum(r['transition_correct'] for r in v)/sum(r['count'] for r in v),
        exact_suffix=sum(r['exact'] for r in v),total_positions=sum(r['count'] for r in v)) for k,v in groups.items()}
    write_json(args.output/f'evaluation-e{epoch:04d}.json',dict(epoch=epoch,full=full,aggregates=aggregates,rows=results,
        scope='Ground-truth times and one-Tap contract GIVEN; free-running suffix positions after an eight-note observed prefix.'))
    return aggregates


def train(args,budget,device):
    data=torch.load(args.data,map_location='cpu',weights_only=True);assert data['schema']==SCHEMA
    rows=data['rows'];training=[r for r in rows if r['split']=='train'];targets={r['id']:prefix_target(r) for r in training}
    config=json.loads(Path('docs/evaluations/diffusion_ar_attribution_20261023.json').read_text())['ar']['config']
    torch.manual_seed(args.seed);model=VariantAR(config,variant=args.variant).to(device)
    model.decoder.gradient_checkpointing=True
    cfg=dict(variant=args.variant,seed=args.seed,epochs=args.epochs,lr=args.lr,data_sha256=sha256(args.data),
        decoder_config=model.decoder_config,parameters=sum(p.numel() for p in model.parameters()),
        initial_backbone_sha256=state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('decoder.lm_head.')}),
        source_sha256={p:sha256(p) for p in SOURCE_FILES},policy=data['policy'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr);sampler=random.Random(args.seed);torch.manual_seed(args.seed+1000)
    curves=[];steps=0;started=time.monotonic()
    if (args.output/'model_latest.pt').exists():
        if not args.resume:raise ValueError('checkpoint exists; --resume required')
        old=torch.load(args.output/'model_latest.pt',map_location='cpu',weights_only=True);assert old['config']==cfg
        model.load_state_dict(old['model']);restore_training(old['training_state'],optimizer,sampler)
        curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds']
    best=min((r['loss'] for r in curves),default=math.inf)
    write_json(args.output/'training.json',dict(status='running',config=cfg,curves=curves))
    print(json.dumps(dict(status='validated',config=cfg,resources=budget.log_state())),flush=True)
    for epoch in range(len(curves),min(args.epochs,args.until_epoch or args.epochs)):
        order=list(training);sampler.shuffle(order);model.train();total=0.
        for row in order:
            factor=min(1.,(steps+1)/max(1,.05*args.epochs*len(training)))*(.1+.9*(1+math.cos(math.pi*steps/max(1,args.epochs*len(training)-1)))/2)
            for group in optimizer.param_groups:group['lr']=args.lr*factor
            optimizer.zero_grad(set_to_none=True);loss=loss_for(model,row,*targets[row['id']])
            if not torch.isfinite(loss):raise RuntimeError('nonfinite pattern loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1;total+=float(loss.detach());budget.check()
        record=dict(epoch=epoch+1,steps=steps,loss=total/len(order),elapsed_seconds=time.monotonic()-started)
        curves.append(record);last=epoch+1==min(args.epochs,args.until_epoch or args.epochs)
        if (epoch+1)%25==0 or last:
            saved=dict(schema=SCHEMA,config=cfg,epoch=epoch+1,steps=steps,curves=curves,
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},training_state=training_state(optimizer,sampler))
            budget.save(saved,args.output/'model_latest.pt')
            if record['loss']<best:
                best=record['loss'];budget.save({k:v for k,v in saved.items() if k!='training_state'},args.output/'model_best_fit.pt')
            del saved
            write_json(args.output/'training.json',dict(status='running',config=cfg,curves=curves,resources=budget.log_state()))
            print(json.dumps(record),flush=True)
        if (args.eval_every and (epoch+1)%args.eval_every==0) or last:
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                result=evaluate(model,rows,args,budget,epoch+1,full=epoch+1==args.epochs)
            print(json.dumps(dict(stage='evaluation',epoch=epoch+1,aggregates=result)),flush=True)
    write_json(args.output/'training.json',dict(status='completed' if epoch+1==args.epochs else 'phase_completed',
        config=cfg,curves=curves,resources=budget.log_state(),checkpoint_sha256=sha256(args.output/'model_latest.pt')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','train'))
    p.add_argument('--variant',choices=('base','kind_position'),default='base')
    p.add_argument('--data',type=Path,default=Path('out/windowed_ar_20261031/patterns/conditions.pt'))
    p.add_argument('--output',type=Path);p.add_argument('--model-root',type=Path,default=Path('checkpoints/timing_v1'))
    p.add_argument('--samples',type=Path,default=Path('out/onset_attention_20261022/samples'))
    p.add_argument('--epochs',type=int,default=400);p.add_argument('--until-epoch',type=int);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda:0');add_resource_args(p);p.set_defaults(max_rss_gib=4.)
    args=p.parse_args();torch.set_num_threads(2)
    if args.stage=='train' and args.output is None:p.error('--output required')
    directory=args.output or args.data.parent;directory.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device);budget=ResourceBudget(args,device,directory)
    (prepare if args.stage=='prepare' else train)(args,budget,device)


if __name__=='__main__':main()

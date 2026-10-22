"""Bounded alignment probes and paired attention adaptation using frozen inputs.

All music-cache songs were seen by full300. Results are training diagnostics,
never an independent test. Oracle soft onsets are a diagnostic distribution
shift, not a guaranteed upper bound and never a product input.
"""
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

import numpy as np
import torch

from script.train_soft_onset import cache_inputs, load_sample
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import code_version, load_audio, sha256
from src.data.chart_tokenizer import ChartTokenizer
from src.models.chart_decoder import generate_tokens
from src.models.chart_history import history_features
from src.product.artifacts import write_json
from src.product.experimental_models import load_models
from src.soft_onset import MEMORY_SECONDS, dense_condition, forward_loss, training_preflight
from src.conditioning_audit import token_scores
from src.resource_budget import ResourceBudget, add_resource_args

ROUTES = {'4k':'decoder', 'freex':'decoder_freex'}
VARIANTS = ('local', 'global', 'time_bias')


def configure(decoder, variant):
    decoder.cross_window = 64 if variant == 'local' else None
    decoder.cross_time_scale = 64. if variant == 'time_bias' else None


def oracle_condition(times, length):
    result = torch.zeros(1,length,2)
    for time in times:
        frame = time/MEMORY_SECONDS
        index = math.floor(frame)
        if not 0 <= index < length:
            raise ValueError('oracle onset outside audio')
        result[0,index] = torch.tensor([1., frame-index])
    return result


@torch.no_grad()
def rollout(decoder, features, soft, valid, bpm, offset, duration, reference, greedy, budget):
    tk = ChartTokenizer(decoder.n_lanes)
    memory = decoder.prepare_memory(features,soft)
    started = time.monotonic()
    try:
        output = generate_tokens(decoder,memory,valid,bpm,MEMORY_SECONDS*1000,duration/MEMORY_SECONDS,
                    offset=offset,tokenizer=tk,greedy=greedy,top_p=.95,max_tokens=12000,
                    generator=torch.Generator(device=memory.device).manual_seed(42),
                    require_complete=True,progress_callback=lambda *_:budget.check())
        tokens, residuals = output if decoder.offset_head is not None else (output,{})
        notes = tk.decode_tokens(tokens,strict=True)
        scale = 60/(32*bpm)
        times = sorted({n.tick*scale+offset for n in notes})
        score = {}
        for ms in (25,50):
            pairs = match_events(reference,times,ms/1000)
            score[f'f1_{ms}ms'] = 2*len(pairs)/(len(reference)+len(times)) if reference or times else 1.
            score[f'matched_{ms}ms'] = len(pairs)
        return dict(status='completed',greedy=greedy,seed=42,seconds=time.monotonic()-started,
                    reference_onsets=len(reference),generated_onsets=len(times),note_count=len(notes),
                    last_onset_seconds=times[-1] if times else None,
                    eos_clock_seconds=tk.ticks_per_position(tokens)[-1]*scale+offset,
                    types=dict(Counter(n.type for n in notes)),onsets_seconds=times,
                    tokens=tokens,residuals=residuals,**score)
    except RuntimeError as exc:
        if 'exhausted step budget' not in str(exc):
            raise
        return dict(status='failed',greedy=greedy,error=str(exc))


@torch.no_grad()
def teacher(decoder, sample, mode, device, zero_audio=False, zero_soft=False):
    row=sample['routes'][mode]
    tokens,frames=row['tokens'].to(device),row['frames'].to(device)
    features=row['features'].to(device); soft=sample['soft_onsets'].to(device)
    if zero_audio: features=torch.zeros_like(features)
    if zero_soft: soft=torch.zeros_like(soft)
    memory=decoder.prepare_memory(features,soft); valid=sample['valid'].to(device)
    history=history_features(tokens[:-1].tolist(),frames[:-1].tolist(),n_lanes=decoder.n_lanes,
             frames_per_beat=60/row['reference']['timebase_bpm']/MEMORY_SECONDS,
             offsets=row['offsets'][:-1] if decoder.n_lanes==64 else None)[None].to(device)
    # Recover actual last-layer attention for a bounded, uniform set of queries.
    block=decoder.blocks[-1]; captured=[]
    indices=torch.linspace(0,len(tokens)-2,min(32,len(tokens)-1),device=device).long().unique()
    handle=block.q_x.register_forward_hook(lambda _m,_a,out:captured.append(out[:,indices].detach()))
    try:
        logits=decoder(tokens[None,:-1],frames[None,:-1],memory,valid,history=history)
    finally:
        handle.remove()
    q=captured[0].reshape(1,-1,block.h,block.dh).transpose(1,2)
    k=block.kv_x(memory).reshape(1,memory.shape[1],2,block.h,block.dh).permute(2,0,3,1,4)[0]
    weights=q@k.transpose(-1,-2)/math.sqrt(block.dh)
    mask=decoder.cross_mask(frames[None,indices],memory.shape[1],valid)
    if mask is not None:
        weights=weights.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else weights+mask
    weights=weights.softmax(-1)
    distance=(frames[indices,None]-torch.arange(memory.shape[1],device=device)).abs()*MEMORY_SECONDS
    return dict(scores=token_scores(logits[0],tokens[1:],ChartTokenizer(decoder.n_lanes)),
                attention=dict(layer='last',queries=len(indices),clock='true reference history',
                  mean_mass_within_250ms=float((weights*(distance<=.25)).sum(-1).mean()),
                  mean_mass_within_3s=float((weights*(distance<=3)).sum(-1).mean()),
                  mean_mass_beyond_10s=float((weights*(distance>10)).sum(-1).mean())))


def fixtures(args, decoder, encoder, budget, device, report):
    manifest=json.loads((args.probes/'fixtures/fixtures.json').read_text())
    for row in manifest['rows']:
        if row['id'] not in ('piano_three','piano_weak_middle','silence'): continue
        mel,duration=load_audio(row['audio'],budget=budget)
        with torch.no_grad():
            mel=torch.nn.functional.pad(mel,(0,mel.shape[-1]%2))
            features=encoder.features(mel[None].to(device)).half().float()
        arrays=dict(np.load(args.probes/f'{row["id"]}.npz'))
        soft=dense_condition(arrays,features.shape[1]).to(device)
        valid=torch.ones(features.shape[:2],device=device,dtype=torch.bool)
        conditions={'predicted':soft,'oracle':oracle_condition(row['events_seconds'],features.shape[1]).to(device),
                    'zero':torch.zeros_like(soft)}
        for name,condition in conditions.items():
            for greedy in (True,False):
                result=rollout(decoder,features,condition,valid,120,0,duration,row['events_seconds'],greedy,budget)
                result.update(id=row['id'],condition=name,reference_kind='constructed acoustic onset')
                result['notes_in_silence']=sum(not any(a-.1<=t<=b+.1 for a,b in row['active_intervals'])
                                               for t in result.get('onsets_seconds',[]))
                report['rows'].append(result)
                write_json(args.output/'report.json',report)
                print(json.dumps({k:v for k,v in result.items() if k not in ('tokens','residuals','onsets_seconds')}),flush=True)


def music(args, decoder, cache, budget, device, report, variants):
    chosen={'Eltaw_IN','-SURREALISM-_IN','ABYSS MOTION 192.333_IN'}
    rows=[r for r in cache['rows'] if r['id'] in chosen]
    if len(rows)!=3: raise ValueError('missing fixed diagnostic songs')
    for variant in variants:
        configure(decoder,variant)
        for row in rows:
            sample=load_sample(args.cache,row); route=sample['routes'][args.mode]
            meta=route['reference']; bpm=meta['timebase_bpm']; offset=meta['offset_seconds']
            tk=ChartTokenizer(decoder.n_lanes)
            reference=sorted({n.tick*60/(32*bpm)+offset for n in tk.decode_tokens(route['tokens'].tolist(),strict=True)})
            result=dict(id=row['id'],variant=variant,cache_sha256=row['sha256'],
                        reference_kind='seen training chart onset',teacher={})
            for name,za,zs in [('matched',False,False),('zero_audio',True,False),('zero_soft',False,True)]:
                budget.check(force=True)
                result['teacher'][name]=teacher(decoder,sample,args.mode,device,za,zs)
            result['generation']=[]
            for greedy in (True,False):
                result['generation'].append(rollout(decoder,route['features'].to(device),sample['soft_onsets'].to(device),
                    sample['valid'].to(device),bpm,offset,sample['duration'],reference,greedy,budget))
            report['rows'].append(result);write_json(args.output/'report.json',report)
            print(json.dumps(dict(id=row['id'],variant=variant,bits=result['teacher']['matched']['scores']['all']['bits'],
                  generations=[{k:g.get(k) for k in ('greedy','f1_50ms','note_count','last_onset_seconds','status')}
                               for g in result['generation']])),flush=True)


def adapt(args, decoder, ck, cache, budget, device, report):
    configure(decoder,args.variant);decoder.gradient_checkpointing=True
    torch.manual_seed(42); order=list(cache['rows']); random.Random(42).shuffle(order)
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=5e-5)
    decoder.train(); losses=[]
    for i,row in enumerate(order):
        sample=load_sample(args.cache,row)
        training_preflight(decoder,sample,args.mode,budget,device)
        optimizer.zero_grad(set_to_none=True)
        loss,_=forward_loss(decoder,sample,args.mode,'soft_history',device)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite adaptation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.);optimizer.step()
        losses.append(float(loss.detach()))
        if (i+1)%25==0:
            print(f'[adapt {args.mode}/{args.variant}] {i+1}/{len(order)} loss={sum(losses[-25:])/25:.5f}',flush=True)
    decoder.eval(); del optimizer
    saved=deepcopy(ck)
    saved['model']={k:v.detach().cpu() for k,v in decoder.state_dict().items()}
    saved['dec_config'].update(cross_window=decoder.cross_window,cross_time_scale=decoder.cross_time_scale)
    # Deliberately not a product bundle. Export needs an explicit quality decision.
    saved.pop('runtime_contract',None);saved.pop('encoder_model',None)
    saved['adaptation']=dict(variant=args.variant,steps=len(order),source=report['checkpoint'],
                            seed=42,lr=5e-5,training_ids=[r['id'] for r in order],losses=losses,
                            evaluation_scope='training_diagnostic',optimizer_resume=False)
    budget.save(saved,args.output/'decoder_adapted.pt')
    report['adaptation']=saved['adaptation']; del saved
    music(args,decoder,cache,budget,device,report,[args.variant])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('fixtures','music','adapt'))
    p.add_argument('--mode',choices=ROUTES,required=True)
    p.add_argument('--variant',choices=VARIANTS,default='time_bias')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('out/full_corpus_v1/cache'))
    p.add_argument('--probes',type=Path,default=Path('out/onset_attention_20261022/onsets'))
    p.add_argument('--model-root',type=Path,default=Path('checkpoints/full300_history'))
    p.add_argument('--device',default='cuda:0');add_resource_args(p);args=p.parse_args()
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    budget=ResourceBudget(args,torch.device(args.device),args.output);device=budget.device
    path=args.model_root/ROUTES[args.mode]/'decoder_best.pt'
    encoder,decoder=load_models(path,device);ck=torch.load(path,map_location='cpu',weights_only=True)
    report=dict(schema='audiolab-alignment-audit-v1',mode=args.mode,action=args.action,
                checkpoint=dict(path=str(path.resolve()),sha256=sha256(path)),rows=[],
                implementation=code_version(),status='running',no_postprocessing=True,
                notes=['Cache music was seen in full300; no independent quality claims.',
                       'Fixtures/ideal soft conditions are distribution shifts, not playable chart targets.',
                       'Natural EOS does not certify whole-song coverage.'])
    if args.action=='fixtures': fixtures(args,decoder,encoder,budget,device,report)
    else:
        del encoder; gc.collect();cache=cache_inputs(args.cache)
        report['cache_sha256']=sha256(args.cache/'cache.json')
        if args.action=='music': music(args,decoder,cache,budget,device,report,VARIANTS)
        else: adapt(args,decoder,ck,cache,budget,device,report)
    report.update(status='completed',resources=budget.log_state());write_json(args.output/'report.json',report)


if __name__=='__main__':main()

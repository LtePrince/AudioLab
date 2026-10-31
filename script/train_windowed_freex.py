"""Controlled free-x port: separate A/S weights, full-waveform audio memory.

This is a diagnostic synthetic mapping, not a new natural-music product model.
Coordinates use 64 bins plus continuous residuals, including in causal history.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch

from script.probe_audio_localization import sources
from script.train_alignment_scratch import training_state, restore_training
from script.train_grounded_ar import load_inputs, encode_waveform
from script.train_windowed_ar import constructed_chart, score as score_onsets
from src.anchored_ar import AnchoredAR, encode_target, normalized_loss, generate as whole_generate
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart_tokenizer import ChartTokenizer, TokenNote, hit_to_bin, bin_to_hit
from src.history_time_noise import perturb_history_times
from src.pattern_diagnostics import KINDS
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import state_hash
from src.windowed_ar import WindowedAR, encode_window_target, generate as window_generate

SCHEMA='audiolab-controlled-freex-windows-v1'
SOURCES=('script/train_windowed_freex.py','script/train_windowed_ar.py','src/windowed_ar.py',
         'src/history_time_noise.py','src/anchored_ar.py','src/models/chart_decoder.py',
         'src/models/chart_history.py','script/train_grounded_ar.py')
COUNTS=('time_count','layout_count','stop_count','continue_count','coordinate_count')


def read(path):return json.loads(Path(path).read_text())


def free_route(events):
    """Map known construction factors to a continuous target, without rounding x.

    Timbre sets coarse region/type; audible gain varies x inside that region.
    This known mapping only establishes learnability; it is not a charting rule.
    """
    tk=ChartTokenizer(64);notes=[];residuals={};reference=[]
    for tick,voice,gain in sorted(events):
        x=(.20,.38,.58,.80,.80)[voice]+.07*(gain-.5)
        lane,residual=hit_to_bin(x);kind=(1,2,4,1,3)[voice];duration=64 if kind==3 else 0
        if (tick,lane) in residuals:raise ValueError('synthetic simultaneous coordinates collide in one bin')
        notes.append(TokenNote(tick,lane,kind,duration));residuals[tick,lane]=residual
        reference.append(dict(time_seconds=tick/64,end_seconds=(tick+duration)/64,
            position=lane,residual=residual,screen_x=x,type=KINDS[kind]))
    tokens=tk.encode_notes(notes);offsets=[];mask=[]
    for token,tick in zip(tokens,tk.ticks_per_position(tokens)):
        note=tk.kind(token)=='NOTE';mask.append(note)
        offsets.append(residuals[tick,tk.value(token)//4] if note else 0.)
    route=dict(tokens=torch.tensor(tokens),offsets=torch.tensor(offsets,dtype=torch.float32),
        offset_mask=torch.tensor(mask,dtype=torch.bool),reference=dict(timebase_bpm=120.,offset_seconds=0.))
    return route,reference


def prepare(args,budget,device):
    path=args.data_root/'conditions.pt';manifest_path=args.data_root/'manifest.json'
    if path.exists() or manifest_path.exists():raise ValueError('free-x data exists; do not overwrite')
    encoder,onset=load_inputs(args,device);clips,metadata=sources(args.samples);rows=[]
    for split,seed in [('train',args.seed),('random_check',args.seed+10000019)]:
        for index in range(32):
            wave,row=constructed_chart(index,seed,clips,split)
            row['route'],row['reference']=free_route(row['source_events'])
            row['condition']=encode_waveform(wave,encoder,onset,budget,device);rows.append(row)
    processor=AudioGPUprocessor(device='cpu');fixture=read(args.fixed_probes)
    for f in fixture['rows']:
        assert sha256(f['audio'])==f['audio_sha256']
        wave=processor.load_from_path(f['audio']);route,reference=free_route([(round(t*64),0,.5) for t in f['events_seconds']])
        rows.append(dict(id=f['id'],split='fixed_probe',duration=f['duration_seconds'],route=route,reference=reference,
            condition=encode_waveform(wave,encoder,onset,budget,device),source_audio_sha256=f['audio_sha256'],
            note='Only onset scores are meaningful; fictitious position/type labels are never trained or scored.'))
    for row in rows:
        a=encode_target(row['route'],row['duration'],n_lanes=64)
        b=encode_window_target(row['route'],row['duration'],n_lanes=64,width=args.window_seconds)
        assert all(a[k]==b[k] for k in ('time_count','layout_count','coordinate_count'))
        assert torch.equal(a['offsets'][a['offset_mask']],b['offsets'][b['offset_mask']])
    budget.save(rows,path)
    manifest=dict(schema=SCHEMA,mode='freex',seed=args.seed,window_seconds=args.window_seconds,
        data_sha256=sha256(path),sources=metadata,fixed_probes_sha256=sha256(args.fixed_probes),
        frontends={k:dict(path=str(p),sha256=sha256(p)) for k,p in {
            'freex':args.model_root/'decoder_freex/decoder_best.pt','onset':args.model_root/'transcriber/transcriber_best.pt'}.items()},
        rows=[{k:v for k,v in row.items() if k not in ('condition','route')} for row in rows],
        policy='Same physical waves/event times as controlled 4K; original free-x acoustic encoder; '
            'continuous x from audible voice/gain, 64 bins plus residual; no coordinate clipping or grid snapping.',
        source_sha256={p:sha256(p) for p in SOURCES},resources=budget.log_state())
    write_json(manifest_path,manifest)
    print(json.dumps(dict(status='prepared',data_sha256=manifest['data_sha256'],rows=len(rows),resources=manifest['resources'])),flush=True)


def score(result,row):
    # Reuse onset matching without ever treating 4-lane labels as free-x labels.
    score_onsets(result,{**row,'split':'fixed_probe'})
    if row['split']=='fixed_probe':return result
    matches=0;errors=[];hold_errors=[]
    for lane in range(64):
        for kind in KINDS.values():
            refs=[n for n in row['reference'] if n['position']==lane and n['type']==kind]
            preds=[n for n in result['notes'] if n['position']==lane and n['type']==kind]
            pairs=match_events([n['time_seconds'] for n in refs],[n['time_seconds'] for n in preds],.025)
            matches+=len(pairs)
            for a,b in pairs:
                r=min(refs,key=lambda n:abs(n['time_seconds']-a));p=min(preds,key=lambda n:abs(n['time_seconds']-b))
                errors.append(abs(r['screen_x']-p['screen_x']))
                if kind=='hold':hold_errors.append(abs((r['end_seconds']-r['time_seconds'])-(p['end_seconds']-p['time_seconds'])))
    total=len(row['reference'])+len(result['notes'])
    result['joint_f1_25ms']=2*matches/total if total else 1.
    result['matched_position_bin_type_count']=matches
    result['matched_screen_x_errors']=errors;result['matched_hold_duration_errors']=hold_errors
    result['coordinate_out_of_bounds']=sum(not .05<=n['screen_x']<=.95 for n in result['notes'])
    return result


@torch.no_grad()
def evaluate(model,rows,args,budget,epoch,*,full):
    model.eval();generator=window_generate if args.architecture=='window' else whole_generate
    selected=rows if full else [r for r in rows if r['split']=='train' and int(r['id'].split('-')[-1]) in (0,1,2,7,12,15,23,31)]
    outputs=[]
    for row in selected:
        result=generator(model,row['condition'],120.,use_soft=args.condition=='S',greedy=True,
            max_tokens=512,check=budget.check);result.pop('trace',None);score(result,row)
        outputs.append(dict(id=row['id'],split=row['split'],**result))
    summaries={}
    for split in sorted({r['split'] for r in outputs}):
        subset=[r for r in outputs if r['split']==split];positive=[r for r in subset if r['reference_onsets']]
        s=dict(examples=len(subset),positive_examples=len(positive),completed=sum(r['status']=='completed' for r in subset),
            positive_macro_raw_onset_f1_25ms=sum(r['onset']['25']['f1'] for r in positive)/max(1,len(positive)),
            silence_extra_onsets=sum(len(r['onsets_seconds']) for r in subset if not r['reference_onsets']))
        if split!='fixed_probe':
            errors=[e for r in subset for e in r['matched_screen_x_errors']]
            s.update(positive_macro_joint_f1_25ms=sum(r['joint_f1_25ms'] if r['status']=='completed' else 0. for r in positive)/max(1,len(positive)),
                matched_notes=len(errors),matched_screen_x_mae=sum(errors)/len(errors) if errors else None,
                coordinate_out_of_bounds=sum(r['coordinate_out_of_bounds'] for r in subset))
        if args.architecture=='window':s['forced_windows']=sum(r['forced_windows'] for r in subset)
        summaries[split]=s
    write_json(args.output/f'evaluation-e{epoch:04d}.json',dict(epoch=epoch,architecture=args.architecture,condition=args.condition,
        full=full,summaries=summaries,rows=outputs,scope='Controlled diagnostic; no claim about natural chart quality or learned song EOS.'))
    return summaries


def train(args,budget,device):
    manifest=read(args.data_root/'manifest.json')
    assert manifest['schema']==SCHEMA and manifest['data_sha256']==sha256(args.data_root/'conditions.pt')
    assert manifest['window_seconds']==args.window_seconds
    rows=torch.load(args.data_root/'conditions.pt',map_location='cpu',weights_only=True)
    train_rows=[r for r in rows if r['split']=='train'];assert len(train_rows)==32
    decoder_config=read('docs/evaluations/diffusion_ar_attribution_20261023.json')['ar']['config']
    decoder_config.update(n_lanes=64,vocab_size=369,offset_head=True)
    torch.manual_seed(args.seed)
    model=(WindowedAR(decoder_config,width=args.window_seconds) if args.architecture=='window' else AnchoredAR(decoder_config)).to(device)
    model.decoder.gradient_checkpointing=True
    config=dict(architecture=args.architecture,condition=args.condition,mode='freex',seed=args.seed,epochs=args.epochs,lr=args.lr,
        parameters=sum(p.numel() for p in model.parameters()),decoder_config=decoder_config,window_seconds=args.window_seconds,
        data_sha256=manifest['data_sha256'],history_noise_ms=args.history_noise_ms,history_noise_seed=args.seed+2000,
        initial_backbone_sha256=state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('bounds_proj.')}),
        source_sha256={p:sha256(p) for p in SOURCES},loss='Original anchored per-example normalized losses, including coordinate MSE.',
        scope='32 constructed examples; independent random combinations; not natural music or product weights.')
    target_fn=(lambda r:encode_window_target(r['route'],r['duration'],n_lanes=64,width=args.window_seconds)) if args.architecture=='window' \
        else (lambda r:encode_target(r['route'],r['duration'],n_lanes=64))
    targets={r['id']:target_fn(r) for r in train_rows};optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    sampler=random.Random(args.seed);noise_rng=random.Random(args.seed+2000);curves=[];steps=0;started=time.monotonic()
    torch.manual_seed(args.seed+1000)
    if (args.output/'model_latest.pt').exists():
        if not args.resume:raise ValueError('checkpoint exists; explicit --resume required')
        old=torch.load(args.output/'model_latest.pt',map_location='cpu',weights_only=True)
        assert old['config']==config,'resume source/config changed'
        model.load_state_dict(old['model']);restore_training(old['training_state'],optimizer,sampler)
        noise_rng.setstate(old['history_noise_rng']);curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds']
    best=min((r['objective'] for r in curves),default=math.inf)
    write_json(args.output/'training.json',dict(status='running',config=config,curves=curves))
    print(json.dumps(dict(status='validated',config=config,resources=budget.log_state())),flush=True)
    for epoch in range(len(curves),min(args.epochs,args.until_epoch or args.epochs)):
        order=list(train_rows);sampler.shuffle(order);model.train();stats=Counter();noise_stats=Counter();objective=0.
        for row in order:
            target,audit=perturb_history_times(targets[row['id']],n_lanes=64,max_seconds=args.history_noise_ms/1000,rng=noise_rng)
            noise_stats.update({k:v for k,v in audit.items() if k!='max_absolute_noise_seconds'})
            noise_stats['max_absolute_noise_seconds']=max(noise_stats['max_absolute_noise_seconds'],audit['max_absolute_noise_seconds'])
            factor=min(1.,(steps+1)/max(1,.05*args.epochs*len(train_rows)))*(.1+.9*(1+math.cos(math.pi*steps/max(1,args.epochs*len(train_rows)-1)))/2)
            for group in optimizer.param_groups:group['lr']=args.lr*factor
            optimizer.zero_grad(set_to_none=True);parts=model.components(target,row['condition'],use_soft=args.condition=='S')
            loss=normalized_loss(parts,target)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite controlled free-x loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1;objective+=float(loss.detach())
            for k,v in parts.items():stats[k]+=float(v.detach()) if torch.is_tensor(v) else v
            for k in COUNTS:stats[k]+=target[k]
            budget.check()
        record=dict(epoch=epoch+1,steps=steps,objective=objective/len(order),elapsed_seconds=time.monotonic()-started,
            frame_accuracy=stats['frame_correct']/max(1,stats['time_count']),stop_recall=stats['correct_stops']/max(1,stats['stop_count']),
            false_stop_rate=stats['false_stops']/max(1,stats['continue_count']),counts={k:int(stats[k]) for k in COUNTS},
            conditional_time_mae_seconds=stats['conditional_offset_absolute_seconds']/max(1,stats['time_count']),
            teacher_screen_x_mae=stats['coordinate_absolute_screen']/max(1,stats['coordinate_count']),
            learning_rate=optimizer.param_groups[0]['lr'],history_noise=dict(noise_stats))
        curves.append(record);last=epoch+1==min(args.epochs,args.until_epoch or args.epochs)
        if (epoch+1)%25==0 or last:
            saved=dict(schema=SCHEMA,config=config,epoch=epoch+1,steps=steps,curves=curves,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                training_state=training_state(optimizer,sampler),history_noise_rng=noise_rng.getstate())
            budget.save(saved,args.output/'model_latest.pt')
            if record['objective']<best:
                best=record['objective'];budget.save({k:v for k,v in saved.items() if k!='training_state'},args.output/'model_best_fit.pt')
            del saved
            write_json(args.output/'training.json',dict(status='running',config=config,curves=curves,resources=budget.log_state()))
            print(json.dumps(record),flush=True)
        if (args.eval_every and (epoch+1)%args.eval_every==0) or last:
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                summary=evaluate(model,rows,args,budget,epoch+1,full=last)
            print(json.dumps(dict(stage='evaluation',epoch=epoch+1,summaries=summary)),flush=True)
    write_json(args.output/'training.json',dict(status='completed' if epoch+1==args.epochs else 'phase_completed',config=config,
        curves=curves,resources=budget.log_state(),checkpoint_sha256=sha256(args.output/'model_latest.pt')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','train'))
    p.add_argument('--architecture',choices=('whole','window'),default='window');p.add_argument('--condition',choices=('A','S'),default='A')
    p.add_argument('--data-root',type=Path,default=Path('out/windowed_ar_20261031/freex_controlled'));p.add_argument('--output',type=Path)
    p.add_argument('--model-root',type=Path,default=Path('checkpoints/timing_v1'));p.add_argument('--samples',type=Path,default=Path('out/onset_attention_20261022/samples'))
    p.add_argument('--fixed-probes',type=Path,default=Path('out/onset_attention_20261022/onsets/fixtures/fixtures.json'))
    p.add_argument('--window-seconds',type=float,default=10.);p.add_argument('--history-noise-ms',type=float,default=0.)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=400);p.add_argument('--until-epoch',type=int)
    p.add_argument('--eval-every',type=int,default=100);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda:0');add_resource_args(p);p.set_defaults(max_rss_gib=4.)
    args=p.parse_args();args.mode='freex';torch.set_num_threads(2)
    if not math.isfinite(args.history_noise_ms) or args.history_noise_ms<0:p.error('history noise must be finite and nonnegative')
    if args.stage=='train' and args.output is None:p.error('--output is required')
    directory=args.data_root if args.stage=='prepare' else args.output;directory.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device);budget=ResourceBudget(args,device,directory)
    (prepare if args.stage=='prepare' else train)(args,budget,device)


if __name__=='__main__':main()

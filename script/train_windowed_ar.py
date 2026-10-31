"""E16 W0/W1 controlled generation: same initial backbone, global audio.

Small constructed charts establish protocol/learning ability before full-corpus
experiments. Their deterministic type mapping is not a music-mapping objective.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from script.probe_audio_localization import sources
from script.train_alignment_scratch import training_state, restore_training
from script.train_grounded_ar import load_inputs, encode_waveform
from src.anchored_ar import AnchoredAR, encode_target, normalized_loss, generate as whole_generate
from src.audio_understanding import SAMPLE_RATE
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart_tokenizer import ChartTokenizer, TokenNote
from src.pattern_diagnostics import KINDS, LANE_X
from src.history_time_noise import perturb_history_times
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import state_hash
from src.windowed_ar import WindowedAR, encode_window_target, generate as window_generate

DATA_SCHEMA='audiolab-controlled-chart-windows-v1'
SOURCES=('script/train_windowed_ar.py','src/windowed_ar.py','src/chart_windows.py',
         'src/anchored_ar.py','src/models/chart_decoder.py','src/models/chart_history.py',
         'src/history_time_noise.py')
COUNTS=('time_count','layout_count','stop_count','continue_count')


def read(path):return json.loads(Path(path).read_text())


def constructed_chart(index,seed,clips,split):
    rng=random.Random(seed+1009*index)
    duration=(10.,20.,30.,35.)[(index//8+index)%4]
    count=(0,1,3,5,8,12,4,2)[index%8]
    events=[];scale=60/(32*120.)
    # A complete Hold crosses an internal output boundary in several examples.
    if index%8==6 and duration>10.:events.append((round(9.5/scale),4,rng.uniform(.1,1.)))
    while len(events)<count:
        tick=round(rng.uniform(.15,duration-1.1)/scale)
        if all(abs(tick-old)*scale>.18 for old,_,_ in events):
            events.append((tick,rng.randrange(len(clips)),rng.uniform(.06,1.)))
    if count>=3 and index%3==0:
        tick,voice,_=events[0]
        voice2=1 if voice in (0,3,4) else 0
        events.append((tick,voice2,rng.uniform(.1,.6)))
    wave=np.zeros(round(duration*SAMPLE_RATE),dtype='float32');notes=[]
    for tick,voice,gain in sorted(events):
        start=round(tick*scale*SAMPLE_RATE);clip=clips[voice]
        wave[start:start+len(clip)]+=gain*clip
        lane=(0,1,2,3,3)[voice];kind=(1,2,4,1,3)[voice]
        notes.append(TokenNote(tick,lane,kind,64 if kind==3 else 0))
    route=dict(tokens=torch.tensor(ChartTokenizer(4).encode_notes(notes)),
               reference=dict(timebase_bpm=120.,offset_seconds=0.))
    reference=[dict(time_seconds=n.tick*scale,end_seconds=(n.tick+n.dur)*scale,
                    position=n.lane,type=KINDS[n.type],screen_x=LANE_X[n.lane]) for n in notes]
    return torch.from_numpy(wave).view(1,1,-1),dict(id=f'{split}-{index:03d}',split=split,route=route,
        reference=reference,source_events=[list(e) for e in events],duration=duration)


def prepare(args,budget,device):
    manifest_path=args.data_root/'manifest.json';data_path=args.data_root/'conditions.pt'
    if manifest_path.exists() or data_path.exists():
        raise ValueError('controlled data already exists or is partial; inspect instead of overwriting')
    encoder,onset=load_inputs(args,device)
    clips,metadata=sources(args.samples);rows=[]
    for split,seed in [('train',args.seed),('random_check',args.seed+10000019)]:
        for i in range(32):
            wave,row=constructed_chart(i,seed,clips,split)
            row['condition']=encode_waveform(wave,encoder,onset,budget,device);rows.append(row)
    fixture=read(args.fixed_probes);processor=AudioGPUprocessor(device='cpu')
    for f in fixture['rows']:
        assert sha256(f['audio'])==f['audio_sha256']
        wave=processor.load_from_path(f['audio'])
        notes=[TokenNote(round(t*64),0,1) for t in f['events_seconds']]
        row=dict(id=f['id'],split='fixed_probe',duration=f['duration_seconds'],
            route=dict(tokens=torch.tensor(ChartTokenizer(4).encode_notes(notes)),reference=dict(timebase_bpm=120.,offset_seconds=0.)),
            reference=[dict(time_seconds=t,end_seconds=t,position=0,type='tap',screen_x=LANE_X[0]) for t in f['events_seconds']],
            source_audio_sha256=f['audio_sha256'],
            note='Fixed acoustic probes score onsets only; the artificial Tap/column labels are never trained or scored.')
        row['condition']=encode_waveform(wave,encoder,onset,budget,device);rows.append(row)
    # Freeze labels, all physical audio conditions, and evaluation before training.
    for row in rows:
        a=encode_target(row['route'],row['duration'])
        b=encode_window_target(row['route'],row['duration'],width=args.window_seconds)
        assert a['time_count']==b['time_count'] and a['layout_count']==b['layout_count']
    budget.save(rows,data_path)
    manifest=dict(schema=DATA_SCHEMA,seed=args.seed,window_seconds=args.window_seconds,mode='4k',
        data_sha256=sha256(data_path),sources=metadata,fixed_probes_sha256=sha256(args.fixed_probes),
        frontends={name:dict(path=str(p),sha256=sha256(p)) for name,p in {
            '4k':args.model_root/'decoder/decoder_best.pt','onset':args.model_root/'transcriber/transcriber_best.pt'}.items()},
        rows=[{k:v for k,v in r.items() if k not in ('route','condition')} for r in rows],
        policy='Whole waveform encoded once per example. No local audio masking; no feature crops. '
               'Synthetic type/column mapping tests learnability only, not musical chart quality.',
        source_sha256={p:sha256(p) for p in SOURCES},resources=budget.log_state())
    write_json(manifest_path,manifest)
    print(json.dumps(dict(status='prepared',rows=len(rows),train=32,random_check=32,fixed_probes=5,
                         data_sha256=manifest['data_sha256'],resources=manifest['resources'])),flush=True)


def score(result,row):
    reference=sorted({n['time_seconds'] for n in row['reference']});prediction=result['onsets_seconds']
    scores={}
    for ms in (25,50,100):
        matched=len(match_events(reference,prediction,ms/1000))
        scores[str(ms)]=dict(reference=len(reference),predicted=len(prediction),matched=matched,
            precision=matched/len(prediction) if prediction else float(not reference),
            recall=matched/len(reference) if reference else 1.,
            f1=2*matched/(len(reference)+len(prediction)) if reference or prediction else 1.)
    result.update(onset=scores,reference_notes=len(row['reference']))
    if row['split']!='fixed_probe':
        matched=0
        for lane in range(4):
            for kind in KINDS.values():
                a=[n['time_seconds'] for n in row['reference'] if n['position']==lane and n['type']==kind]
                b=[n['time_seconds'] for n in result['notes'] if n['position']==lane and n['type']==kind]
                matched+=len(match_events(a,b,.025))
        denominator=len(row['reference'])+len(result['notes'])
        result['joint_f1_25ms']=2*matched/denominator if denominator else 1.
    result['reference_onsets']=reference
    return result


@torch.no_grad()
def evaluate(model,rows,args,budget,epoch,*,full):
    model.eval();generator=window_generate if args.architecture=='window' else whole_generate
    selected=rows if full else [r for r in rows if r['split']=='train' and int(r['id'].split('-')[-1]) in (0,1,2,7,12,15,23,31)]
    outputs=[]
    for row in selected:
        result=generator(model,row['condition'],120.,use_soft=False,greedy=True,
                         max_tokens=512,check=budget.check)
        result.pop('trace',None);score(result,row)
        outputs.append(dict(id=row['id'],split=row['split'],**result))
    aggregates={}
    for split in sorted({r['split'] for r in outputs}):
        subset=[r for r in outputs if r['split']==split]
        summary=dict(examples=len(subset),completed=sum(r['status']=='completed' for r in subset),
            mean_raw_onset_f1_25ms=sum(r['onset']['25']['f1'] for r in subset)/len(subset),
            mean_raw_onset_f1_50ms=sum(r['onset']['50']['f1'] for r in subset)/len(subset),
            silence_false_positives=sum(r['note_count'] for r in subset if not r['reference_onsets']))
        if split!='fixed_probe':summary['mean_joint_f1_25ms']=sum(r['joint_f1_25ms'] if r['status']=='completed' else 0. for r in subset)/len(subset)
        if args.architecture=='window':
            summary.update(learned_window_end_fraction=sum(r['learned_window_end_fraction'] for r in subset)/len(subset),
                scheduler_traversal_fraction=sum(r['scheduler_traversal_fraction'] for r in subset)/len(subset),
                forced_windows=sum(r['forced_windows'] for r in subset))
        aggregates[split]=summary
    report=dict(epoch=epoch,architecture=args.architecture,full=full,aggregates=aggregates,rows=outputs,
        scope='Constructed diagnostic chart mapping, not natural-music chart quality. '
              'W1 completion requires learned END_WINDOW in every window; scheduler is not learned song EOS.')
    write_json(args.output/f'evaluation-e{epoch:04d}.json',report)
    return aggregates


def train(args,budget,device):
    manifest=read(args.data_root/'manifest.json')
    assert manifest['schema']==DATA_SCHEMA and manifest['data_sha256']==sha256(args.data_root/'conditions.pt')
    assert manifest['window_seconds']==args.window_seconds
    rows=torch.load(args.data_root/'conditions.pt',map_location='cpu',weights_only=True)
    train_rows=[r for r in rows if r['split']=='train'];assert len(train_rows)==32
    decoder_config=read('docs/evaluations/diffusion_ar_attribution_20261023.json')['ar']['config']
    torch.manual_seed(args.seed)
    model=(WindowedAR(decoder_config,width=args.window_seconds) if args.architecture=='window'
           else AnchoredAR(decoder_config)).to(device)
    model.decoder.gradient_checkpointing=True
    config=dict(architecture=args.architecture,condition='A',seed=args.seed,epochs=args.epochs,lr=args.lr,
        parameters=sum(p.numel() for p in model.parameters()),decoder_config=decoder_config,
        window_seconds=args.window_seconds,data_sha256=manifest['data_sha256'],
        initial_backbone_sha256=state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('bounds_proj.')}),
        source_sha256={p:sha256(p) for p in SOURCES},
        loss='Original anchored per-example normalized losses; window-boundary labels and bounds conditioning only in W1.',
        scope='32 controlled training examples; separate 32 combinations and original fixed probes; not natural music.')
    config['history_noise_ms']=args.history_noise_ms
    config['history_noise_seed']=args.seed+2000
    target_fn=(lambda r:encode_window_target(r['route'],r['duration'],width=args.window_seconds)) if args.architecture=='window' \
        else (lambda r:encode_target(r['route'],r['duration']))
    targets={r['id']:target_fn(r) for r in train_rows}
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    sampler=random.Random(args.seed);noise_rng=random.Random(args.seed+2000)
    curves=[];steps=0;started=time.monotonic()
    # Match initial dropout RNG too; the extra zero-init boundary layer consumes initialization draws.
    torch.manual_seed(args.seed+1000)
    if (args.output/'model_latest.pt').exists():
        if not args.resume:raise ValueError('checkpoint exists; explicit --resume required')
        old=torch.load(args.output/'model_latest.pt',map_location='cpu',weights_only=True)
        assert old['config']==config,'resume source/config changed'
        model.load_state_dict(old['model']);restore_training(old['training_state'],optimizer,sampler)
        noise_rng.setstate(old['history_noise_rng'])
        curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds']
    best=min((r['objective'] for r in curves),default=math.inf)
    write_json(args.output/'training.json',dict(status='running',config=config,curves=curves))
    print(json.dumps(dict(status='validated',config=config,resources=budget.log_state())),flush=True)
    for epoch in range(len(curves),min(args.epochs,args.until_epoch or args.epochs)):
        order=list(train_rows);sampler.shuffle(order);model.train();stats=Counter();noise_stats=Counter();objective=0.
        for row in order:
            target,audit=perturb_history_times(targets[row['id']],n_lanes=model.decoder.n_lanes,
                max_seconds=args.history_noise_ms/1000,rng=noise_rng)
            noise_stats.update({k:v for k,v in audit.items() if k!='max_absolute_noise_seconds'})
            noise_stats['max_absolute_noise_seconds']=max(noise_stats['max_absolute_noise_seconds'],audit['max_absolute_noise_seconds'])
            factor=min(1.,(steps+1)/max(1,.05*args.epochs*len(train_rows))) \
                   *(.1+.9*(1+math.cos(math.pi*steps/max(1,args.epochs*len(train_rows)-1)))/2)
            for group in optimizer.param_groups:group['lr']=args.lr*factor
            optimizer.zero_grad(set_to_none=True)
            parts=model.components(target,row['condition'],use_soft=False)
            loss=normalized_loss(parts,target)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite controlled AR loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1;objective+=float(loss.detach())
            for k,v in parts.items():stats[k]+=float(v.detach()) if torch.is_tensor(v) else v
            for k in COUNTS:stats[k]+=target[k]
            budget.check()
        record=dict(epoch=epoch+1,steps=steps,objective=objective/len(order),
            elapsed_seconds=time.monotonic()-started,frame_accuracy=stats['frame_correct']/max(1,stats['time_count']),
            conditional_time_mae_seconds=stats['conditional_offset_absolute_seconds']/max(1,stats['time_count']),
            stop_recall=stats['correct_stops']/max(1,stats['stop_count']),
            false_stop_rate=stats['false_stops']/max(1,stats['continue_count']),counts={k:int(stats[k]) for k in COUNTS},
            learning_rate=optimizer.param_groups[0]['lr'])
        record['history_noise']=dict(noise_stats)
        curves.append(record)
        last=epoch+1==min(args.epochs,args.until_epoch or args.epochs)
        if (epoch+1)%25==0 or last:
            saved=dict(schema='audiolab-controlled-window-training-v1',config=config,epoch=epoch+1,steps=steps,curves=curves,
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},training_state=training_state(optimizer,sampler),
                history_noise_rng=noise_rng.getstate())
            budget.save(saved,args.output/'model_latest.pt')
            if record['objective']<best:
                best=record['objective'];budget.save({k:v for k,v in saved.items() if k!='training_state'},args.output/'model_best_fit.pt')
            del saved
            write_json(args.output/'training.json',dict(status='running',config=config,curves=curves,resources=budget.log_state()))
            print(json.dumps(record),flush=True)
        if (args.eval_every and (epoch+1)%args.eval_every==0) or last:
            devices=[device.index or 0] if device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                summary=evaluate(model,rows,args,budget,epoch+1,full=last)
            print(json.dumps(dict(stage='evaluation',epoch=epoch+1,aggregates=summary)),flush=True)
    write_json(args.output/'training.json',dict(status='completed' if epoch+1==args.epochs else 'phase_completed',
        config=config,curves=curves,resources=budget.log_state(),checkpoint_sha256=sha256(args.output/'model_latest.pt')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','train'))
    parser.add_argument('--architecture',choices=('whole','window'),default='window')
    parser.add_argument('--data-root',type=Path,default=Path('out/windowed_ar_20261031/controlled'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--model-root',type=Path,default=Path('checkpoints/timing_v1'))
    parser.add_argument('--samples',type=Path,default=Path('out/onset_attention_20261022/samples'))
    parser.add_argument('--fixed-probes',type=Path,default=Path('out/onset_attention_20261022/onsets/fixtures/fixtures.json'))
    parser.add_argument('--window-seconds',type=float,default=10.)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--until-epoch',type=int);parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--lr',type=float,default=3e-4);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--history-noise-ms',type=float,default=0.)
    parser.add_argument('--device',default='cuda:0');add_resource_args(parser);parser.set_defaults(max_rss_gib=4.)
    args=parser.parse_args();torch.set_num_threads(2)
    if not math.isfinite(args.history_noise_ms) or args.history_noise_ms<0:parser.error('--history-noise-ms must be finite and nonnegative')
    if args.stage=='train' and args.output is None:parser.error('--output is required for training')
    directory=args.output if args.stage=='train' else args.data_root;directory.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device);budget=ResourceBudget(args,device,directory)
    (prepare if args.stage=='prepare' else train)(args,budget,device)


if __name__=='__main__':main()

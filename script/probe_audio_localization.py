"""E16-D0: can a small head read physical onsets from frozen AR audio features?

This is a controlled acoustic diagnostic, not a chart generator. Independent
random arrangements and the original fixed probes never enter its training set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F

from src.audio_understanding import FRAME_SECONDS, SAMPLE_RATE
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.data.audio2mel import AudioGPUprocessor
from src.product.artifacts import write_json
from src.product.experimental_models import load_models
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import MEMORY_SECONDS

SCHEMA='audiolab-frozen-feature-localizer-v1'


class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(256)
        self.head=nn.Sequential(nn.Conv1d(256,64,5,padding=2),nn.GELU(),
                                nn.Conv1d(64,64,3,padding=1),nn.GELU(),nn.Conv1d(64,2,1))

    def forward(self,features):
        return self.head(self.norm(features).transpose(1,2)).transpose(1,2)


def sources(directory):
    clips=[];metadata=[]
    for name in ('C4','Fs4','A4'):
        path=directory/f'{name}.mp3'
        wave,rate=sf.read(path,dtype='float32',always_2d=True)
        wave=wave.mean(axis=1);divisor=math.gcd(rate,SAMPLE_RATE)
        wave=resample_poly(wave,SAMPLE_RATE//divisor,rate//divisor)
        start=int(np.flatnonzero(abs(wave)>abs(wave).max()*1e-3)[0])
        wave=wave[start:start+SAMPLE_RATE].copy()
        wave*=.6/max(abs(wave).max(),1e-8)
        wave[:44]*=np.linspace(0,1,44);wave[-2205:]*=np.linspace(1,0,2205)
        clips.append(wave)
        metadata.append(dict(name=name,path=str(path),sha256=sha256(path),trim_samples=start))
    # Additional explicitly synthetic timbres; no claims of natural-instrument generalization.
    t=np.arange(SAMPLE_RATE,dtype=np.float64)/SAMPLE_RATE
    for name,freq,partials,decay in [('additive_pluck',330.,8,6.),('additive_bell',440.,3,3.)]:
        wave=sum(np.sin(2*np.pi*freq*k*t)*np.exp(-decay*k*t)/k for k in range(1,partials+1))
        wave=(wave*.6/max(abs(wave).max(),1e-8)).astype('float32')
        wave[:44]*=np.linspace(0,1,44);wave[-2205:]*=np.linspace(1,0,2205)
        clips.append(wave);metadata.append(dict(name=name,frequency=freq,partials=partials,decay=decay,
                                               sha256=hashlib.sha256(wave.tobytes()).hexdigest()))
    return clips,metadata


def constructed(index,seed,split,clips,*,close_pair=False,extended=False):
    rng=np.random.default_rng(seed+1009*index)
    count=(0,1,1,1,2,3,4,5)[index%8]
    duration=(10.,16.,32.)[index%3] if extended else 10.
    starts=[]
    minimum_interval=.06 if close_pair else .8
    if close_pair and count>=2:
        first=float(rng.uniform(.15,duration-1.3))
        starts=[first,first+float(rng.uniform(.06,.25))]
    while len(starts)<count:
        candidate=float(rng.uniform(.15,duration-1.))
        if all(abs(candidate-t)>minimum_interval for t in starts):starts.append(candidate)
    starts=sorted(round(t*SAMPLE_RATE)/SAMPLE_RATE for t in starts)
    gains=np.exp(rng.uniform(np.log(.02),np.log(1.),count)).tolist()
    voices=rng.integers(0,len(clips),count).tolist()
    wave=np.zeros(round(duration*SAMPLE_RATE),dtype='float32')
    for onset,gain,voice in zip(starts,gains,voices):
        offset=round(onset*SAMPLE_RATE);clip=clips[voice]
        wave[offset:offset+len(clip)]+=gain*clip
    return wave,dict(id=f'{split}-{index:03d}',split=split,duration=duration,times=starts,
        gains=gains,voices=voices,close_pair=close_pair,
        waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest())


@torch.no_grad()
def encode(wave,encoder,processor,device,budget):
    duration=len(wave)/SAMPLE_RATE
    mel=processor(torch.from_numpy(wave).view(1,1,-1))[0]
    mel=mel[:,:math.ceil(duration/FRAME_SECONDS)]
    mel=F.pad(mel,(0,mel.shape[-1]%2))
    features=encoder.features(mel[None].to(device))[0].half().cpu()
    assert features.shape[1]==256 and torch.isfinite(features).all()
    budget.check()
    return features


def prepare(args,budget,device):
    cache_path=args.output/'conditions.pt';manifest_path=args.output/'conditions.json'
    # Keep head initialization identical whether the frozen-feature cache exists.
    encoder,unused=load_models(args.frontend,device);del unused
    if cache_path.exists() or manifest_path.exists():
        assert cache_path.exists() and manifest_path.exists(),'partial cache; inspect before resuming'
        manifest=json.loads(manifest_path.read_text())
        assert manifest['seed']==args.seed and manifest['cache_sha256']==sha256(cache_path)
        assert manifest['frontend_sha256']==sha256(args.frontend)
        assert manifest.get('training_examples',32)==args.training_examples
        del encoder
        return torch.load(cache_path,map_location='cpu',weights_only=True),manifest
    encoder.requires_grad_(False).eval();processor=AudioGPUprocessor(device='cpu')
    clips,source_metadata=sources(args.samples);rows=[]
    specifications=[('train',args.training_examples,args.seed),('random_check',64,args.seed+10000019)]
    if args.training_examples>32:specifications.append(('dense_check',64,args.seed+20000033))
    for split,n,seed in specifications:
        for i in range(n):
            # Original 32 training examples and random checks remain byte-identical.
            close_pair=(split=='train' and i>=32) or split=='dense_check'
            wave,row=constructed(i,seed,split,clips,close_pair=close_pair,
                                 extended=split=='train' and i>=128)
            row['features']=encode(wave,encoder,processor,device,budget);rows.append(row)
    fixture=json.loads(args.fixed_probes.read_text())
    for original in fixture['rows']:
        assert sha256(original['audio'])==original['audio_sha256']
        wave,rate=sf.read(original['audio'],dtype='float32');assert rate==SAMPLE_RATE and wave.ndim==1
        rows.append(dict(id=original['id'],split='fixed_probe',duration=len(wave)/rate,
                         times=original['events_seconds'],audio_sha256=original['audio_sha256'],
                         features=encode(wave,encoder,processor,device,budget)))
    # Check time origins and true silence around output-window boundaries.
    for name,starts,duration in [('boundary_sides',[9.97,10.03,19.97,20.03],32.),
                                 ('long_silence',[.25,29.25],32.)]:
        wave=np.zeros(round(duration*SAMPLE_RATE),dtype='float32')
        for i,onset in enumerate(starts):
            offset=round(onset*SAMPLE_RATE);clip=clips[i%3]
            wave[offset:offset+len(clip)]+=clip
        rows.append(dict(id=name,split='boundary_check',duration=duration,times=starts,
                         waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                         features=encode(wave,encoder,processor,device,budget)))
    # Separate boundary ownership from the additional difficulty of overlapping notes.
    for i,onset in enumerate((9.97,10.03,19.97,20.03)):
        wave=np.zeros(32*SAMPLE_RATE,dtype='float32');clip=clips[i%3];start=round(onset*SAMPLE_RATE)
        wave[start:start+len(clip)]+=clip
        rows.append(dict(id=f'boundary_single-{i}',split='boundary_single',duration=32.,times=[onset],
                         waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                         features=encode(wave,encoder,processor,device,budget)))
    budget.save(rows,cache_path)
    manifest=dict(schema=SCHEMA,seed=args.seed,training_examples=args.training_examples,
        additional_training='Original 32; clips 32..127 add close pairs; clips >=128 also cover 10/16/32-second durations.' if args.training_examples>32 else None,
        frontend=str(args.frontend),
        frontend_sha256=sha256(args.frontend),cache_sha256=sha256(cache_path),sources=source_metadata,
        source_attribution='Piano: Alexander Holm, Salamander Grand Piano / Tonejs/audio, CC BY 3.0; '
                           'two additive timbres constructed in this script.',
        code_sha256=sha256(__file__),fixed_probes_sha256=sha256(args.fixed_probes),
        rows=[{k:v for k,v in r.items() if k!='features'} for r in rows],
        label_contract='Onsets are physical sample construction times, not chart targets. '
                       'Audio memory remains whole and frozen; no onset head/skeleton is input.',
        limitations='Random arrangements of shared short samples; not unseen natural-music transcription.')
    write_json(manifest_path,manifest)
    del encoder
    torch.cuda.empty_cache() if device.type=='cuda' else None
    return rows,manifest


def batch(rows,device):
    length=max(len(r['features']) for r in rows)
    features=torch.zeros(len(rows),length,256,device=device)
    mask=torch.zeros(len(rows),length,dtype=torch.bool,device=device)
    labels=torch.zeros(len(rows),length,device=device);offsets=torch.zeros_like(labels)
    for b,row in enumerate(rows):
        size=len(row['features']);features[b,:size]=row['features'].to(device).float();mask[b,:size]=True
        for t in row['times']:
            frame=int(t/MEMORY_SECONDS)
            assert frame<size and labels[b,frame]==0,'two events collide in a diagnostic frame'
            lo=frame*MEMORY_SECONDS;hi=min(lo+MEMORY_SECONDS,row['duration'])
            labels[b,frame]=1.;offsets[b,frame]=(t-lo)/(hi-lo)
    return features,mask,labels,offsets


def objective(prediction,mask,labels,offsets):
    positive=mask & (labels>0);negative=mask & ~positive
    positive_loss=-F.logsigmoid(prediction[...,0][positive]).mean()
    negative_loss=-F.logsigmoid(-prediction[...,0][negative]).mean()
    offset_loss=F.mse_loss(prediction[...,1][positive].sigmoid(),offsets[positive])
    return positive_loss+negative_loss+4*offset_loss


@torch.no_grad()
def evaluate(model,rows,device):
    model.eval();out=[];errors=[]
    totals={ms:dict(reference=0,predicted=0,matched=0) for ms in (25,50,100)}
    for row in rows:
        pred=model(row['features'].float().to(device)[None])[0].sigmoid().cpu()
        # One decision per frame; fixed 0.5 threshold, no NMS/count cap/grid snap.
        times=[]
        for frame in (pred[:,0]>=.5).nonzero().flatten().tolist():
            lo=frame*MEMORY_SECONDS;hi=min(lo+MEMORY_SECONDS,row['duration'])
            if hi>lo:times.append(lo+(hi-lo)*min(float(pred[frame,1]),1-1e-7))
        scores={}
        for ms in totals:
            pairs=match_events(row['times'],times,ms/1000)
            total=totals[ms];total['reference']+=len(row['times']);total['predicted']+=len(times);total['matched']+=len(pairs)
            scores[str(ms)]=len(pairs)
            if ms==25:errors.extend(abs(a-b)*1000 for a,b in pairs)
        out.append(dict(id=row['id'],reference=row['times'],predictions=times,matched=scores))
    for t in totals.values():
        t.update(precision=t['matched']/t['predicted'] if t['predicted'] else float(not t['reference']),
                 recall=t['matched']/t['reference'] if t['reference'] else 1.,
                 f1=2*t['matched']/(t['reference']+t['predicted']) if t['reference']+t['predicted'] else 1.)
    return dict(at_ms=totals,mean_error_25ms=float(np.mean(errors)) if errors else None,
                silence_false_positives=sum(len(r['predictions']) for r in out if not r['reference']),rows=out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('out/windowed_ar_20261031/d0'))
    parser.add_argument('--frontend',type=Path,default=Path('checkpoints/timing_v1/decoder/decoder_best.pt'))
    parser.add_argument('--samples',type=Path,default=Path('out/onset_attention_20261022/samples'))
    parser.add_argument('--fixed-probes',type=Path,default=Path('out/onset_attention_20261022/onsets/fixtures/fixtures.json'))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=4000);parser.add_argument('--eval-every',type=int,default=250)
    parser.add_argument('--training-examples',type=int,default=32)
    add_resource_args(parser);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('completed run exists; choose a new output for a new experiment')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    device=torch.device(args.device);budget=ResourceBudget(args,device,args.output)
    rows,manifest=prepare(args,budget,device)
    train_rows=[r for r in rows if r['split']=='train']
    tensors=batch(train_rows,device);model=Localizer().to(device)
    initialization_sha256=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes()
        for t in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best_loss=math.inf;history=[];fit_checks=0;started=time.monotonic()
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=objective(model(tensors[0]),*tensors[1:]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();budget.check()
        if step%args.eval_every and step!=args.steps:continue
        score=evaluate(model,train_rows,device)
        with torch.no_grad():measured=float(objective(model(tensors[0]),*tensors[1:]))
        entry=dict(step=step,loss=measured,at_25ms=score['at_ms'][25],mae_ms=score['mean_error_25ms'],elapsed=time.monotonic()-started)
        history.append(entry)
        state=dict(schema=SCHEMA,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                   seed=args.seed,torch_rng=torch.get_rng_state(),source_sha256=sha256(__file__),
                   frontend_sha256=manifest['frontend_sha256'],conditions_sha256=manifest['cache_sha256'])
        budget.save(state,args.output/'model_latest.pt')
        if measured<best_loss:
            best_loss=measured;budget.save(state,args.output/'model_best_train.pt')
        write_json(args.output/'progress.json',dict(status='running',history=history,resources=budget.log_state()))
        print(json.dumps(entry),flush=True)
        fit=score['at_ms'][25]['precision']>=.99 and score['at_ms'][25]['recall']>=.99 \
            and score['silence_false_positives']==0 and score['mean_error_25ms']<5.
        fit_checks=fit_checks+1 if fit else 0
        if step>=1000 and fit_checks>=3:break
    selected=torch.load(args.output/'model_best_train.pt',map_location=device,weights_only=True)
    model.load_state_dict(selected['model'])
    results={split:evaluate(model,[r for r in rows if r['split']==split],device)
             for split in sorted({r['split'] for r in rows})}
    report=dict(schema=SCHEMA,status='completed',kind='diagnostic_training',parameters=sum(p.numel() for p in model.parameters()),
        selected_step=selected['step'],initialization_sha256=initialization_sha256,
        selection='lowest training loss; no held-out threshold/weight selection',
        checkpoint_sha256=sha256(args.output/'model_best_train.pt'),conditions=manifest,history=history,results=results,
        resources=budget.log_state(),elapsed_training_seconds=time.monotonic()-started,
        decoding='fixed probability >= 0.5 per frame + continuous offset; no NMS, cap, snapping or known count',
        quality_scope='Acoustic localization from frozen 4K features only; no chart types/positions or music quality claim.')
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(status='completed',parameters=report['parameters'],selected_step=report['selected_step'],
        results={s:{'at_25ms':r['at_ms'][25],'silence_false_positives':r['silence_false_positives']} for s,r in results.items()},
        resources=report['resources'])),flush=True)


if __name__=='__main__':main()

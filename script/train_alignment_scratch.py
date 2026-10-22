"""Matched random-initialization decoder pilots, with known-silent negatives.

The acoustic encoders stay frozen. All 300 cached songs train every epoch;
music diagnostics are therefore in-sample, not an architecture benchmark.
Pure digital silence is encoded through the real frontend, never represented
by zeroing learned features. Held-out 10-second silence is not a training row.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import random
import time

import torch

from script.audit_decoder_alignment import configure, fixtures, music, teacher, ROUTES, VARIANTS
from script.train_soft_onset import cache_inputs, load_sample
from src.audio_understanding.inference import code_version, infer_mel, load_model, sha256
from src.audio_understanding import FRAME_SECONDS, SAMPLE_RATE
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart_tokenizer import BOS, EOS
from src.models.chart_decoder import ChartDecoder
from src.product.artifacts import write_json
from src.product.experimental_models import load_models
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import MEMORY_SECONDS, dense_condition, forward_loss, state_hash, training_preflight


@torch.no_grad()
def silent_samples(encoder, onset, mode, device):
    rows=[];processor=AudioGPUprocessor(device='cpu')
    for duration in (5.,13.,26.,41.):
        mel=processor(torch.zeros(1,1,round(duration*SAMPLE_RATE)))[0]
        # Slice the time axis (mel has [frequency,time]).
        mel=mel[:,:math.ceil(duration/FRAME_SECONDS)]
        mel=torch.nn.functional.pad(mel,(0,mel.shape[-1]%2))
        features=encoder.features(mel[None].to(device)).half().float().cpu()
        arrays=infer_mel(onset,mel,duration)
        row=dict(features=features,tokens=torch.tensor([BOS,EOS]),frames=torch.tensor([0.,0.]),
                 offsets=torch.zeros(2),offset_mask=torch.zeros(2,dtype=torch.bool),
                 reference=dict(timebase_bpm=120.,offset_seconds=0.))
        rows.append(dict(id=f'digital-silence-{duration:g}s',duration=duration,routes={mode:row},
                         valid=torch.ones(features.shape[:2],dtype=torch.bool),
                         soft_onsets=dense_condition(arrays,features.shape[1])))
    return rows


def train_variant(args, variant, source, encoder, silence, cache, budget, device):
    destination=args.output/variant;destination.mkdir(parents=True,exist_ok=True)
    # There are no attention-policy-dependent parameters. This seed therefore
    # gives byte-identical initial tensors for all three attention variants.
    torch.manual_seed(args.seed)
    config=deepcopy(source['dec_config'])
    config.update(cross_window=64 if variant=='local' else None,
                  cross_time_scale=64. if variant=='time_bias' else None)
    decoder=ChartDecoder(**config).to(device)
    decoder.gradient_checkpointing=True
    initial_hash=state_hash(decoder.state_dict())
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=args.lr)
    sampler=random.Random(args.seed)
    epochs=[];steps=0; started=time.monotonic()
    diagnostic=[r for r in cache['rows'] if r['id'] in ('Eltaw_IN','-SURREALISM-_IN','ABYSS MOTION 192.333_IN')]
    rows=cache['rows']
    if len(rows)!=300 or len(diagnostic)!=3:raise ValueError('expected the frozen 300-song corpus and three diagnostic songs')
    total=args.epochs*(len(rows)+len(rows)//10)
    provenance=dict(schema='audiolab-scratch-alignment-v1',mode=args.mode,variant=variant,
        initialization='random decoder, frozen original acoustic encoders',seed=args.seed,
        initial_state_sha256=initial_hash,cache_sha256=sha256(args.cache/'cache.json'),
        lr=args.lr,epochs=args.epochs,silence_every_music_updates=10,
        silent_training_seconds=[s['duration'] for s in silence],
        silence_target='BOS EOS; actual zero waveform through original frontend',
        training_ids=[r['id'] for r in rows],evaluation_scope='training_diagnostic',
        checkpoint_sources=cache['models'],implementation=code_version(),curves=epochs)
    write_json(destination/'training.json',provenance)
    for epoch in range(args.epochs):
        order=list(rows);sampler.shuffle(order);decoder.train()
        music_losses=[];silence_losses=[]
        for i,row in enumerate(order):
            batch=[(load_sample(args.cache,row),music_losses)]
            if (i+1)%10==0:batch.append((silence[((epoch*len(rows)+i)//10)%len(silence)],silence_losses))
            for sample,losses in batch:
                training_preflight(decoder,sample,args.mode,budget,device)
                progress=steps/max(1,total-1)
                factor=min(1.,(steps+1)/max(1,.05*total))*(.1+.9*(1+math.cos(math.pi*progress))/2)
                for group in optimizer.param_groups:group['lr']=args.lr*factor
                optimizer.zero_grad(set_to_none=True)
                loss,_=forward_loss(decoder,sample,args.mode,'soft_history',device)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite scratch loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()));steps+=1
            if (i+1)%100==0:print(f'[scratch {args.mode}/{variant}] epoch {epoch+1}/{args.epochs} songs {i+1}/300',flush=True)
        decoder.eval();scores=[]
        for row in diagnostic:
            sample=load_sample(args.cache,row)
            with torch.no_grad():
                _,score=forward_loss(decoder,sample,args.mode,'soft_history',device,score=True)
            scores.append(dict(id=row['id'],**score['scores']['all']))
        quiet=[]
        for sample in silence:
            with torch.no_grad():
                loss,_=forward_loss(decoder,sample,args.mode,'soft_history',device)
            quiet.append(float(loss))
        epochs.append(dict(epoch=epoch+1,steps=steps,music_loss=sum(music_losses)/len(music_losses),
                           silence_loss=sum(silence_losses)/len(silence_losses),
                           teacher_scores=scores,silence_eos_nll=quiet,elapsed_seconds=time.monotonic()-started))
        write_json(destination/'training.json',provenance)
        saved={key:deepcopy(source[key]) for key in ('soft_onset_schema','timing_schema','condition','history_schema','mode') if key in source}
        saved.update(model={k:v.detach().cpu() for k,v in decoder.state_dict().items()},
                     dec_config=config,training_scope='all',evaluation_scope='training_diagnostic',
                     scratch_training=provenance,epoch=epoch+1,steps=steps,
                     recovery='model-only; optimizer/RNG resume not supported')
        budget.save(saved,destination/'decoder_latest.pt')
        print(json.dumps(dict(mode=args.mode,variant=variant,**epochs[-1])),flush=True)
    del optimizer,saved;gc.collect()
    report=dict(schema='audiolab-alignment-audit-v1',mode=args.mode,rows=[],status='running',
                checkpoint=dict(path=str((destination/'decoder_latest.pt').resolve()),sha256=sha256(destination/'decoder_latest.pt')),
                scratch_training=provenance,no_postprocessing=True)
    local_args=deepcopy(args);local_args.output=destination
    music(local_args,decoder,cache,budget,device,report,[variant])
    # Same held-out fixture waveforms as the old decoder; never gradient inputs.
    fixtures(local_args,decoder,encoder,budget,device,report)
    report.update(status='completed',resources=budget.log_state())
    write_json(destination/'report.json',report)
    del decoder;gc.collect();torch.cuda.empty_cache()
    return dict(variant=variant,initial_state_sha256=initial_hash,report=str(destination/'report.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=ROUTES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('out/full_corpus_v1/cache'))
    p.add_argument('--probes',type=Path,default=Path('out/onset_attention_20261022/onsets'))
    p.add_argument('--model-root',type=Path,default=Path('checkpoints/full300_history'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=3e-4)
    add_resource_args(p);args=p.parse_args()
    if args.epochs<1 or not math.isfinite(args.lr) or args.lr<=0:raise ValueError('positive epochs and learning rate required')
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    budget=ResourceBudget(args,torch.device(args.device),args.output);device=budget.device
    cache=cache_inputs(args.cache)
    path=args.model_root/ROUTES[args.mode]/'decoder_best.pt'
    encoder,old=load_models(path,device);source=torch.load(path,map_location='cpu',weights_only=True);del old
    encoder.requires_grad_(False)
    onset,_=load_model(args.model_root/'transcriber/transcriber_best.pt',device)
    silence=silent_samples(encoder,onset,args.mode,device);del onset;gc.collect()
    result=[]
    for variant in VARIANTS:
        result.append(train_variant(args,variant,source,encoder,silence,cache,budget,device))
        if len({r['initial_state_sha256'] for r in result})!=1:raise RuntimeError('initialization was not paired')
        write_json(args.output/'index.json',dict(status='completed' if len(result)==3 else 'running',models=result))


if __name__=='__main__':main()

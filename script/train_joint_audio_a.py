"""Bounded full-corpus A adaptation: Mel -> entire acoustic encoder -> window AR.

Only deterministic Mel/labels are cached. Every neural forward is recomputed
from the current checkpoint. S, the original encoder and the parent stay intact.
"""
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
import time

import torch
import torch.nn.functional as F

from script.train_alignment_scratch import restore_training, training_state
from script.train_anchored_ar import COUNTS, epoch_objective
from script.train_grounded_ar import checked_pairs, read
from script.train_soft_onset import cache_inputs, load_sample
from script.train_windowed_corpus import E20, evaluate
from src.anchored_ar import normalized_loss
from src.audio_understanding import FRAME_SECONDS, SAMPLE_RATE
from src.audio_understanding.inference import sha256
from src.conditional_training import empty_targets, shifted_waveform
from src.data.audio2mel import AudioGPUprocessor
from src.joint_audio_ar import SCHEMA, JointAudioWindowAR, gradient_audit
from src.product.artifacts import write_json
from src.resource_budget import GIB, ResourceBudget, add_resource_args
from src.soft_onset import state_hash, training_preflight
from src.windowed_ar import encode_window_target

SOURCES = ('script/train_joint_audio_a.py', 'script/train_models.py', 'src/joint_audio_ar.py', 'src/models/transcriber.py',
           'src/models/attention.py', 'src/condition/wave.py', 'src/data/audio2mel.py',
           'src/windowed_ar.py', 'src/anchored_ar.py', 'src/models/chart_decoder.py',
           'src/models/chart_history.py', 'script/train_windowed_corpus.py',
           'src/conditional_training.py', 'script/train_alignment_scratch.py',
           'docs/evaluations/assets/phira_generalization10_20261029/benchmark.py',
           'benchmarks/v1/protocol.json')


def mel_from_wave(processor, waveform):
    duration = waveform.shape[-1] / SAMPLE_RATE
    waveform = F.pad(waveform, (0, max(0, 1025-waveform.shape[-1])))
    mel = processor(waveform)[0, :, :math.ceil(duration / FRAME_SECONDS)]
    return F.pad(mel, (0, mel.shape[-1] % 2)).contiguous(), duration


def prepare(args, budget):
    cache = cache_inputs(args.cache)
    pairs = checked_pairs(args)
    pair_rows = {r['id']:r for r in pairs['rows']}
    directory = args.output / 'mel'
    directory.mkdir(exist_ok=True)
    path = directory / 'manifest.json'
    contract = dict(schema=SCHEMA, cache_sha256=sha256(args.cache/'cache.json'),
                    pairs_sha256=sha256(args.pairs/'pairs.json'), selection_sha256=sha256(E20),
                    processing='AudioGPUprocessor on CPU; original full waveform and frozen physical crops; '
                               '128 bins, 22050 Hz, hop512; ceil physical duration then even zero padding; float32 Mel')
    manifest = read(path) if path.exists() else dict(contract=contract, status='preparing', rows=[], development=[])
    if manifest['contract'] != contract:
        raise ValueError('Mel cache inputs changed')
    for row in manifest['rows'] + manifest['development']:
        if sha256(directory/row['file']) != row['sha256']:
            raise ValueError('Mel cache is incomplete or changed')
    if manifest['status'] == 'completed':
        return manifest
    processor = AudioGPUprocessor(device='cpu')
    existing = {r['id'] for r in manifest['rows']}
    total_bytes = sum((directory/r['file']).stat().st_size for r in manifest['rows']+manifest['development'])

    def save(payload, name):
        nonlocal total_bytes
        estimated = sum(v['mel'].numel()*4 for v in payload.get('views', [payload]))
        if total_bytes+estimated > 2*GIB:
            raise RuntimeError('Mel cache would exceed the explicit 2 GiB budget')
        budget.save(payload, directory/name)
        total_bytes += (directory/name).stat().st_size
        return dict(file=name, sha256=sha256(directory/name))

    for index, row in enumerate(cache['rows']):
        if row['id'] in existing:
            continue
        budget.check(force=True)
        if sha256(row['audio']['path']) != row['audio']['sha256']:
            raise ValueError('source audio changed')
        sample = load_sample(args.cache, row)
        pair = torch.load(args.pairs/pair_rows[row['id']]['file'], map_location='cpu', weights_only=True)
        waveform = processor.load_from_path(row['audio']['path'])
        mel, duration = mel_from_wave(processor, waveform)
        if abs(duration-sample['duration']) > 1/SAMPLE_RATE or mel.shape[-1]//2 != sample['valid'].shape[1]:
            raise ValueError('full-song Mel no longer matches the original clock')
        views = [dict(mel=mel, duration=duration, target=encode_window_target(sample['routes']['4k'], duration))]
        for shift, target in zip(pair['plan']['shift_samples'], pair['targets'], strict=True):
            crop, seconds = mel_from_wave(processor, shifted_waveform(waveform, pair['plan'], shift))
            views.append(dict(mel=crop, duration=seconds, target=encode_window_target(target, seconds)))
        quiet, seconds = mel_from_wave(processor, torch.zeros(1,1,round(pair['duration']*SAMPLE_RATE)))
        views.append(dict(mel=quiet, duration=seconds,
                          target=encode_window_target(empty_targets(views[1]['target']['bpm']), seconds)))
        stored = save(dict(id=row['id'], views=views), f'train-{index:03d}.pt')
        manifest['rows'].append(dict(id=row['id'], **stored, audio_sha256=row['audio']['sha256'],
            duration=duration, frames=mel.shape[-1]//2, tokens=len(views[0]['target']['tokens'])))
        manifest['bytes'] = total_bytes
        write_json(path, manifest)
        if (index+1)%30 == 0:
            print(json.dumps(dict(stage='prepare_mel', songs=index+1, total=len(cache['rows']), bytes=total_bytes)), flush=True)
        del sample, pair, waveform, views, mel, crop, quiet
    existing = {str(r['id']) for r in manifest['development']}
    for song in read(E20)['songs']:
        if str(song['id']) in existing:
            continue
        if sha256(song['audio']) != song['audio_sha256'] or sha256(song['reference']) != song['reference_sha256']:
            raise ValueError('development selection changed')
        mel, duration = mel_from_wave(processor, processor.load_from_path(song['audio']))
        stored = save(dict(mel=mel, duration=duration), f'dev-{song["id"]}.pt')
        manifest['development'].append(dict(id=str(song['id']), **stored, audio_sha256=song['audio_sha256']))
        write_json(path, manifest)
    if len(manifest['rows']) != 300 or len(manifest['development']) != 10:
        raise ValueError('all 300 songs and ten development songs required')
    manifest.update(status='completed', bytes=total_bytes)
    write_json(path, manifest)
    return manifest


def load_initial(args, device):
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    frontend = torch.load(args.frontend, map_location='cpu', weights_only=True)
    cache = read(args.cache/'cache.json')
    if sha256(args.frontend) != cache['models']['4k']['sha256']:
        raise ValueError('initial encoder must match the original acoustic cache')
    torch.manual_seed(args.seed)
    model, decoder_config = JointAudioWindowAR.from_parents(parent, frontend)
    config = dict(schema=SCHEMA, product='AL-AR4K@0.7.0', run='E22-A', variant='audio',
                  architecture='joint_audio_window', encoder_config=frontend['enc_config'],
                  decoder_config=decoder_config, parent=str(args.parent), parent_sha256=sha256(args.parent),
                  parent_epoch=parent['epoch'], frontend=str(args.frontend), frontend_sha256=sha256(args.frontend),
                  epochs=args.epochs, seed=args.seed, decoder_lr=args.lr, encoder_lr=args.encoder_lr,
                  schedule='new AdamW; constant two-group LR; no frozen A parameters',
                  feature_precision='float32 network; differentiable float16->float32 memory matching parent',
                  encoder_dropout='original pretrained dropout enabled during joint training',
                  policy='all 300 full songs + original two crops + real silence; full bidirectional audio; '
                         'unchanged 10-second output windows and loss; no S, no grid snap or type quota',
                  removed_unused_parameters=['encoder transcription output head', 'zero-input soft-onset projection'],
                  mel_manifest_sha256=sha256(args.output/'mel/manifest.json'),
                  source_sha256={p:sha256(p) for p in SOURCES},
                  parameters=sum(p.numel() for p in model.parameters()),
                  encoder_parameters=sum(p.numel() for p in model.acoustic.parameters()),
                  decoder_parameters=sum(p.numel() for p in model.chart.parameters()))
    return model.to(device), config


def load_row(args, row):
    return torch.load(args.output/'mel'/row['file'], map_location='cpu', weights_only=True)


def preflight(model, view, budget, device):
    length = view['mel'].shape[-1]//2
    fake = dict(routes={'4k':dict(tokens=view['target']['tokens'],features=torch.empty(1,length,0))})
    estimate = training_preflight(model.chart.decoder, fake, '4k', budget, device)
    # Conservative math-attention estimate with block recomputation. Real peak
    # is recorded by the full-song probe before the long job is started.
    encoder = model.acoustic
    extra = (encoder.num_heads*length*length*16 + length*encoder.conv_in.out_channels*4*128
             + sum(p.numel()*p.element_size()*4 for p in encoder.parameters())
             + view['target']['time_count']*length*32)
    if device.type == 'cuda':
        free,total = torch.cuda.mem_get_info(device)
        allowance = min(free,total*budget.args.gpu_memory_fraction-torch.cuda.memory_allocated(device))
        if estimate['bytes']+extra > allowance:
            raise RuntimeError('joint encoder/decoder estimate exceeds the GPU budget')
    budget.check(force=True)


def probe(args, budget, manifest, device):
    model, config = load_initial(args, device)
    rows = {r['id']:r for r in manifest['rows']}
    selected = list(dict.fromkeys(['Eltaw_IN', max(rows,key=lambda k:rows[k]['frames']),
                                  max(rows,key=lambda k:rows[k]['tokens'])]))
    report = dict(config=config, rows=[])
    for id in selected:
        sample = load_row(args,rows[id]); model.train()
        optimizer = torch.optim.AdamW(model.optimizer_groups(decoder_lr=args.lr,encoder_lr=args.encoder_lr))
        before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        totals = {k:sum(v['target'][k] for v in sample['views']) for k in COUNTS}
        optimizer.zero_grad(set_to_none=True)
        for view in sample['views']:
            preflight(model,view,budget,device)
            loss = normalized_loss(model.components(view['target'],view['mel'],view['duration']),totals)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite joint probe loss')
            loss.backward()
            del loss
        audit = gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        changed = [k for k,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),before[k])]
        for prefix in audit['group_gradient_norms']:
            if not any(k.startswith(prefix) for k in changed):raise RuntimeError(f'no update in {prefix}')
        report['rows'].append(dict(id=id, gradient=audit, changed_parameter_tensors=len(changed),
                                   resources=budget.log_state()))
        print(json.dumps(dict(stage='joint_probe',id=id,all_connected=True,parameters=audit['parameters'],
                              changed_tensors=len(changed),resources=budget.log_state())),flush=True)
        del sample,optimizer,before
    report.update(status='passed', probe_weights_discarded=True)
    write_json(args.output/'gradient_probe.json',report)
    del model
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return report


def evaluate_joint(model,args,budget,manifest,epoch,full):
    model.eval()
    rows = {r['id']:r for r in manifest['rows']}
    development = {r['id']:r for r in manifest['development']}
    def provider(id,split):
        sample=load_row(args,(rows if split=='train15' else development)[id])
        view=sample['views'][0] if split=='train15' else sample
        return model.condition(view['mel'],view['duration'])
    # Shared scorer/selection; the neural cache is replaced by this checkpoint's
    # full online encoder. The optional provider is inside evaluate's no_grad.
    return evaluate(model.chart,args,budget,epoch,full=full,condition_provider=provider)


def train(args,budget,manifest,device):
    model,config=load_initial(args,device)
    optimizer=torch.optim.AdamW(model.optimizer_groups(decoder_lr=args.lr,encoder_lr=args.encoder_lr))
    sampler=random.Random(args.seed);curves=[];steps=0;started=time.monotonic()
    path=args.output/'model_latest.pt'
    if path.exists():
        if not args.resume:raise ValueError('joint checkpoint exists; explicit resume required')
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['config']!=config:raise ValueError('joint resume config/source changed')
        model.load_state_dict(old['model'],strict=True)
        restore_training(old['training_state'],optimizer,sampler)
        curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds'];del old
    else:
        torch.manual_seed(args.seed+1000)
    write_json(args.output/'config.json',config)
    def status(state,**extra):
        write_json(args.output/'training.json',dict(status=state,pid=os.getpid(),updated=time.time(),
            config=config,curves=curves,steps=steps,resources=budget.log_state(),**extra))
    if not curves:
        status('baseline_evaluation')
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            evaluate_joint(model,args,budget,manifest,0,full=True)
    best=min((r['objective'] for r in curves),default=math.inf)
    status('running')
    for epoch in range(len(curves),args.epochs):
        order=list(manifest['rows']);sampler.shuffle(order);stats=Counter();model.train()
        for index,row in enumerate(order):
            sample=load_row(args,row);views=sample['views']
            totals={k:sum(v['target'][k] for v in views) for k in COUNTS}
            optimizer.zero_grad(set_to_none=True)
            for view in views:
                preflight(model,view,budget,device)
                parts=model.components(view['target'],view['mel'],view['duration'])
                loss=normalized_loss(parts,totals)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite joint loss')
                loss.backward()
                for k,v in parts.items():stats[k]+=float(v.detach()) if torch.is_tensor(v) else v
                for k in COUNTS:stats[k]+=view['target'][k]
                del loss,parts
            if index==0:
                audit=gradient_audit(model)
                write_json(args.output/'last_gradient.json',dict(epoch=epoch+1,step=steps+1,**audit))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1
            del sample,views
            if (index+1)%10==0:
                status('running',active_epoch=epoch+1,songs=index+1)
                print(json.dumps(dict(stage='train',epoch=epoch+1,songs=index+1,steps=steps,
                                      resources=budget.log_state())),flush=True)
        record=dict(epoch=epoch+1,steps=steps,elapsed_seconds=time.monotonic()-started,
            counts={k:int(stats[k]) for k in COUNTS},frame_ce=stats['frame_ce']/stats['time_count'],
            frame_accuracy=stats['frame_correct']/stats['time_count'],subframe_mse=stats['subframe_mse']/stats['time_count'],
            layout_ce=stats['layout_ce']/stats['layout_count'],stop_bce=stats['stop_bce']/stats['stop_count'],
            continue_bce=stats['continue_bce']/stats['continue_count'],false_stop_rate=stats['false_stops']/stats['continue_count'],
            end_stop_recall=stats['correct_stops']/stats['stop_count'],
            conditional_offset_mae_seconds=stats['conditional_offset_absolute_seconds']/stats['time_count'])
        record['objective']=epoch_objective(record);curves.append(record)
        saved=dict(schema=SCHEMA,config=config,epoch=epoch+1,steps=steps,curves=curves,
                   model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   training_state=training_state(optimizer,sampler))
        budget.save(saved,path)
        if record['objective']<best:
            best=record['objective'];budget.save({k:v for k,v in saved.items() if k!='training_state'},args.output/'model_best_fit.pt')
        del saved
        print(json.dumps(record),flush=True);status('running')
        if (epoch+1)%args.eval_every==0 or epoch+1==args.epochs:
            status('evaluating',active_epoch=epoch+1)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                evaluate_joint(model,args,budget,manifest,epoch+1,full=epoch+1==args.epochs)
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    final_summary=args.output/f'evaluation-e{args.epochs:04d}'/'summary.json'
    if not final_summary.exists() or not read(final_summary).get('full'):
        status('evaluating',active_epoch=args.epochs)
        evaluate_joint(model,args,budget,manifest,args.epochs,full=True)
    status('completed',checkpoint_sha256=sha256(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','probe','run'))
    parser.add_argument('--output',type=Path,default=Path('out/joint_audio_a_20261102'))
    parser.add_argument('--parent',type=Path,default=Path('out/windowed_ar_20261031/natural_window_fit100/model_latest.pt'))
    parser.add_argument('--frontend',type=Path,default=Path('checkpoints/timing_v1/decoder/decoder_best.pt'))
    parser.add_argument('--cache',type=Path,default=Path('out/full_corpus_v1/cache'))
    parser.add_argument('--pairs',type=Path,default=Path('out/grounded_ar_20261023/pairs'))
    parser.add_argument('--checks',type=Path,default=Path('out/windowed_ar_20261031/natural_checks'))
    parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--eval-every',type=int,default=5)
    parser.add_argument('--lr',type=float,default=3e-5);parser.add_argument('--encoder-lr',type=float,default=3e-6)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',default='cuda:0');add_resource_args(parser);parser.set_defaults(max_rss_gib=6.)
    args=parser.parse_args();args.mode='4k';args.architecture='window'
    if args.epochs<1 or args.eval_every<1:parser.error('positive epoch/evaluation intervals required')
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu' if args.stage=='prepare' else args.device)
    lock=(args.output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=ResourceBudget(args,device,args.output)
    try:
        manifest=prepare(args,budget)
        if args.stage=='prepare':return
        existing=args.output/'gradient_probe.json'
        if args.stage=='probe' or not existing.exists():probe(args,budget,manifest,device)
        else:
            initial,config=load_initial(args,torch.device('cpu'))
            del initial
            if read(existing)['config']!=config:raise ValueError('probe source/config changed; repeat probe')
        if args.stage=='run':train(args,budget,manifest,device)
    except BaseException as error:
        write_json(args.output/'failure.json',dict(type=type(error).__name__,message=str(error),time=time.time(),pid=os.getpid()))
        progress=args.output/'training.json'
        if progress.exists():
            state=read(progress);state.update(status='failed',updated=time.time(),error=str(error))
            write_json(progress,state)
        raise


if __name__=='__main__':main()

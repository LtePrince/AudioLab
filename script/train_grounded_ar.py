"""Bounded 4K conditional-generation experiment inspired by VLM training.

Stages: prepare physical waveform pairs, train a continued/data/rank control,
then evaluate unchanged benchmark v1 plus separate shift-response diagnostics.
Only the AR decoder is trained; original full-song features are reused and
short-waveform inputs are re-encoded. No test probes enter training.
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

import torch
import torch.nn.functional as F

from script.audit_decoder_alignment import rollout, teacher
from script.benchmark_models import decode_notes
from script.train_alignment_scratch import silent_samples, training_state, restore_training
from script.train_soft_onset import cache_inputs, load_sample
from src.audio_understanding import FRAME_SECONDS, SAMPLE_RATE
from src.audio_understanding.inference import infer_mel, load_audio, load_model, sha256
from src.conditional_training import (PAIR_SCHEMA, LENGTHS, choose_pair, shifted_waveform,
    shifted_targets, empty_targets, paired_loss, training_logits, time_log_probability)
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart_tokenizer import ChartTokenizer
from src.model_benchmark import evaluate as benchmark
from src.product.artifacts import write_json
from src.product.experimental_models import load_models
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import MEMORY_SECONDS, dense_condition, forward_loss, load_decoder, training_preflight

SONGS = ('-SURREALISM-_IN','ABYSS MOTION 192.333_IN','Eltaw_IN')
SOURCES = ('script/train_grounded_ar.py','src/conditional_training.py',
           'src/models/chart_decoder.py','src/models/chart_history.py','src/soft_onset.py')


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(args, device):
    encoder, unused = load_models(args.model_root/'decoder/decoder_best.pt',device)
    del unused
    onset, _ = load_model(args.model_root/'transcriber/transcriber_best.pt',device)
    return encoder.requires_grad_(False).eval(), onset.requires_grad_(False).eval()


@torch.no_grad()
def encode_waveform(waveform, encoder, onset, budget, device):
    duration = waveform.shape[-1]/SAMPLE_RATE
    mel = AudioGPUprocessor(device='cpu')(waveform)[0]
    mel = mel[:,:math.ceil(duration/FRAME_SECONDS)]
    mel = F.pad(mel,(0,mel.shape[-1]%2))
    budget.check(force=True)
    features = encoder.features(mel[None].to(device)).half().cpu()
    arrays = infer_mel(onset,mel,duration,check=budget.check)
    return dict(features=features,soft_onsets=dense_condition(arrays,features.shape[1]),
                valid=torch.ones(features.shape[:2],dtype=torch.bool),duration=duration)


def prepare(args, budget, device):
    dest = args.pairs
    manifest_path = dest/'pairs.json'
    if manifest_path.exists():
        completed = checked_pairs(args)
        print(json.dumps(dict(status='cached',pairs=len(completed['rows']))),flush=True)
        return
    dest.mkdir(parents=True,exist_ok=True)
    # Refuse partial overwrite; preparation can be restarted in a new directory.
    if list(dest.glob('pair-*.pt')):
        raise ValueError('partial pair cache exists; use a new --pairs directory')
    cache = cache_inputs(args.cache)
    encoder, onset = load_inputs(args,device)
    expected = cache['models']
    for key, path in (('4k',args.model_root/'decoder/decoder_best.pt'),
                      ('onset',args.model_root/'transcriber/transcriber_best.pt')):
        if sha256(path) != expected[key]['sha256']:
            raise ValueError('pair frontend must be identical to full-song cache frontend')
    manifest = dict(schema=PAIR_SCHEMA,status='running',seed=args.seed,
        full_cache_sha256=sha256(args.cache/'cache.json'),models={k:expected[k] for k in ('4k','onset')},
        source_sha256={p:sha256(p) for p in SOURCES},lengths=list(LENGTHS),rows=[],diagnostics=[],silence={},
        contract=dict(input='raw cropped waveform, both translated views re-encoded',
          label='all four types; complete Holds; physical sample clock, then original tick quantization',
          target_used_in_features=False,weak_onsets='all dense predictions retained',
          negative='same content, same duration/BPM/mask, different physical time shift',
          shift_overlap='less than 50 percent of unique onsets matched at 25 ms',
          diagnostic_scope='new transformations of three trained songs; not unseen songs'))
    for duration in LENGTHS:
        condition = encode_waveform(torch.zeros(1,1,round(duration*SAMPLE_RATE)),encoder,onset,budget,device)
        path = dest/f'silence-{duration:g}.pt';budget.save(condition,path)
        manifest['silence'][str(duration)] = dict(file=path.name,sha256=sha256(path))
    proc = AudioGPUprocessor(device='cpu');tk = ChartTokenizer(4)
    for i,row in enumerate(cache['rows']):
        budget.check(force=True)
        sample = load_sample(args.cache,row)
        route = sample['routes']['4k'];ref=route['reference']
        if sha256(row['audio']['path']) != row['audio']['sha256']:
            raise ValueError('source waveform fingerprint changed')
        waveform = proc.load_from_path(row['audio']['path'])
        if not torch.isfinite(waveform).all():raise ValueError('nonfinite source waveform')
        notes = tk.decode_tokens(route['tokens'].tolist(),strict=True)
        for diagnostic in (False,True) if row['id'] in SONGS else (False,):
            seed = args.seed+1009*i+(1000003 if diagnostic else 0)
            duration = 12. if diagnostic else LENGTHS[i%len(LENGTHS)]
            plan = choose_pair(notes,ref['timebase_bpm'],ref['offset_seconds'],
                               waveform.shape[-1]/SAMPLE_RATE,duration,random.Random(seed))
            pair = dict(id=row['id'],plan=plan,duration=duration,views=[],targets=[])
            for shift in plan['shift_samples']:
                wave = shifted_waveform(waveform,plan,shift)
                pair['views'].append(encode_waveform(wave,encoder,onset,budget,device))
                pair['targets'].append(shifted_targets(notes,ref['timebase_bpm'],ref['offset_seconds'],plan,shift))
            path = dest/f'pair-{i:03d}{"-diagnostic" if diagnostic else ""}.pt'
            budget.save(pair,path)
            manifest['diagnostics' if diagnostic else 'rows'].append(dict(id=row['id'],file=path.name,
                sha256=sha256(path),duration=duration,plan=plan,
                tokens=[len(t['tokens'])-1 for t in pair['targets']],source_audio_sha256=row['audio']['sha256']))
        if (i+1)%30==0:
            print(json.dumps(dict(stage='prepare',songs=i+1,total=len(cache['rows']))),flush=True)
        del sample,waveform,pair
    manifest.update(status='completed',resources=budget.log_state())
    write_json(manifest_path,manifest)
    print(json.dumps(dict(status='prepared',rows=len(manifest['rows']),resources=budget.log_state())),flush=True)


def checked_pairs(args):
    index = read(args.pairs/'pairs.json')
    if index['schema'] != PAIR_SCHEMA or index['status'] != 'completed' or index['seed'] != args.seed \
            or index['full_cache_sha256'] != sha256(args.cache/'cache.json') or len(index['rows']) != 300:
        raise ValueError('complete matching 300-song pair cache required')
    for row in index['rows']+index['diagnostics']+list(index['silence'].values()):
        path=(args.pairs/row['file']).resolve()
        if path.parent != args.pairs.resolve() or sha256(path)!=row['sha256']:
            raise ValueError('pair-cache path/hash differs')
    return index


def pair_sample(target, condition):
    route = dict(target,features=condition['features'].float())
    return dict(routes={'4k':route},valid=condition['valid'],soft_onsets=condition['soft_onsets'])


def train(args, budget, device):
    cache = cache_inputs(args.cache);pairs=checked_pairs(args)
    decoder, source = load_decoder(args.init,device)
    if decoder.n_lanes!=4 or decoder.cross_window is not None or decoder.cross_time_scale is not None:
        raise ValueError('this experiment requires the global 4K history checkpoint')
    decoder.gradient_checkpointing=True
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(decoder.parameters(),lr=args.lr)
    rng=random.Random(args.seed)
    by_id={r['id']:r for r in pairs['rows']}
    quiet={float(k):torch.load(args.pairs/v['file'],map_location='cpu',weights_only=True)
           for k,v in pairs['silence'].items()}
    old_silence=[]
    if args.variant=='continued':
        encoder,onset=load_inputs(args,device)
        old_silence=silent_samples(encoder,onset,'4k',device)
        del encoder,onset;gc.collect();torch.cuda.empty_cache()
    config=dict(variant=args.variant,epochs=args.epochs,lr=args.lr,seed=args.seed,
        rank_weight=args.rank_weight if args.variant=='rank' else 0.,rank_beta=args.rank_beta,
        init_sha256=sha256(args.init),init_epoch=source['epoch'],init_steps=source['steps'],
        full_cache_sha256=sha256(args.cache/'cache.json'),
        pair_cache_sha256=sha256(args.pairs/'pairs.json'),source_sha256={p:sha256(p) for p in SOURCES},
        formula=('original song-mean CE; separate silence EOS optimizer step every ten songs'
                 if args.variant=='continued' else
                 'joint token CE + weight * mean_bidirectional softplus(-beta * (mean_DT_logp_correct - mean_DT_logp_wrong))'),
        attribution='reference-free time-token adaptation inspired by CoPO; not mDPO reproduction',
        inference='unchanged global attention, temperature=1, top_p=.95, no repair or threshold changes')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'decoder_latest.pt';curves=[];steps=0;started=time.monotonic()
    if path.exists():
        if not args.resume:raise ValueError('checkpoint exists; use --resume')
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['grounding_training']['config']!=config:raise ValueError('resume config/source differs')
        decoder.load_state_dict(old['model'],strict=True)
        restore_training(old['training_state'],optimizer,rng)
        curves=old['grounding_training']['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds']
        del old
    total=args.epochs*(330 if args.variant=='continued' else 300)
    for epoch in range(len(curves),args.epochs):
        order=list(cache['rows']);rng.shuffle(order);decoder.train();stats=Counter();rank_rows=[]
        for i,row in enumerate(order):
            sample=load_sample(args.cache,row)
            training_preflight(decoder,sample,'4k',budget,device)
            factor=min(1.,(steps+1)/max(1,.05*total))*(.1+.9*(1+math.cos(math.pi*steps/max(1,total-1)))/2)
            for group in optimizer.param_groups:group['lr']=args.lr*factor
            optimizer.zero_grad(set_to_none=True)
            music_tokens=len(sample['routes']['4k']['tokens'])-1
            if args.variant=='continued':
                loss,details=forward_loss(decoder,sample,'4k','soft_history',device)
                loss.backward()
                stats['music_ce_sum']+=float(loss.detach())*music_tokens;stats['music_tokens']+=music_tokens
            else:
                pair=torch.load(args.pairs/by_id[row['id']]['file'],map_location='cpu',weights_only=True)
                silence=quiet[pair['duration']];empty=empty_targets(pair['targets'][0]['reference']['timebase_bpm'])
                crop_tokens=sum(len(t['tokens'])-1 for t in pair['targets'])
                denominator=music_tokens+crop_tokens+1
                loss,_=forward_loss(decoder,sample,'4k','soft_history',device)
                (loss*music_tokens/denominator).backward()
                stats['music_ce_sum']+=float(loss.detach())*music_tokens;stats['music_tokens']+=music_tokens
                for a,b in ((0,1),(1,0)):
                    estimate=training_preflight(decoder,pair_sample(pair['targets'][a],pair['views'][a]),'4k',budget,device)
                    if config['rank_weight'] and device.type=='cuda':
                        free,total_memory=torch.cuda.mem_get_info(device)
                        available=min(free,total_memory*args.gpu_memory_fraction-torch.cuda.memory_allocated(device))
                        if 2*estimate['bytes']>available:
                            raise RuntimeError('Resource budget: paired backward estimate exceeds GPU allowance')
                    ce,rank,detail=paired_loss(decoder,pair['targets'][a],pair['views'][a],pair['views'][b],device,
                        rank_weight=config['rank_weight'],beta=args.rank_beta)
                    (ce/denominator+rank/2).backward()
                    stats['crop_ce_sum']+=detail['ce_sum'];stats['crop_tokens']+=detail['supervised_tokens']
                    if 'gap' in detail:rank_rows.append(detail)
                silent_logits=training_logits(decoder,empty,silence,device)
                eos=F.cross_entropy(silent_logits,empty['tokens'][1:].to(device),reduction='sum')
                (eos/denominator).backward()
                stats['silence_eos_nll_sum']+=float(eos.detach());stats['silence_samples']+=1
                stats['silence_effective_weight_sum']+=1/denominator
                del pair,ce,rank,silent_logits,eos
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1
            if args.variant=='continued' and (i+1)%10==0:
                optimizer.zero_grad(set_to_none=True)
                silence=old_silence[((epoch*300+i)//10)%len(old_silence)]
                loss,_=forward_loss(decoder,silence,'4k','soft_history',device)
                loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();steps+=1
                stats['silence_eos_nll_sum']+=float(loss.detach());stats['silence_samples']+=1
                stats['silence_effective_weight_sum']+=1.
            if (i+1)%100==0:
                print(json.dumps(dict(stage='train',variant=args.variant,epoch=epoch+1,songs=i+1,steps=steps)),flush=True)
            del sample,loss
        record=dict(epoch=epoch+1,steps=steps,elapsed_seconds=time.monotonic()-started,**stats,
            music_ce=stats['music_ce_sum']/stats['music_tokens'],
            crop_ce=stats['crop_ce_sum']/stats['crop_tokens'] if stats['crop_tokens'] else None,
            silence_eos_nll=stats['silence_eos_nll_sum']/stats['silence_samples'],
            silence_mean_effective_weight=stats['silence_effective_weight_sum']/stats['silence_samples'],
            ranking={k:sum(r[k] for r in rank_rows)/len(rank_rows) for k in
                     ('matched_time_nll','mismatched_time_nll','gap','rank_loss')} if rank_rows else None)
        curves.append(record)
        saved={k:deepcopy(source[k]) for k in ('soft_onset_schema','timing_schema','condition','history_schema','mode','dec_config')}
        saved.update(model={k:v.detach().cpu() for k,v in decoder.state_dict().items()},
            epoch=epoch+1,steps=steps,training_scope='all',evaluation_scope='training_diagnostic',
            grounding_training=dict(config=config,curves=curves),training_state=training_state(optimizer,rng))
        budget.save(saved,path);write_json(args.output/'training.json',saved['grounding_training'])
        del saved
        print(json.dumps(record),flush=True)
    print(json.dumps(dict(status='trained',checkpoint=str(path),resources=budget.log_state())),flush=True)


@torch.no_grad()
def evaluate(args,budget,device):
    cache=cache_inputs(args.cache);pairs=checked_pairs(args)
    decoder,_=load_decoder(args.output/'decoder_latest.pt',device)
    source=read(args.baseline_input)
    baseline=next(r for r in source['runs'] if r['id']=='4k/global/seed42')
    runs=[dict(id=f'4k/grounded-{args.variant}/{suffix}',cases=deepcopy(baseline['cases']),predictions={},
        metadata=dict(stage='raw_layout',mode='4k',variant=args.variant,greedy=greedy,seed=42,
            checkpoint_sha256=sha256(args.output/'decoder_latest.pt'),temperature=1.,top_p=.95,
            repairs='none',sources={p:sha256(p) for p in SOURCES}))
          for greedy,suffix in ((True,'greedy'),(False,'seed42'))]
    cases={c['id']:c for c in baseline['cases']};audit=[]
    for row in cache['rows']:
        if row['id'] not in SONGS:continue
        sample=load_sample(args.cache,row);r=sample['routes']['4k'];case=cases[row['id']]
        audit.append(dict(id=row['id'],teacher={name:teacher(decoder,sample,'4k',device,**options)
            for name,options in (('matched',{}),('zero_audio',dict(zero_audio=True)),('zero_soft',dict(zero_soft=True)))}))
        for run in runs:
            result=rollout(decoder,r['features'].to(device),sample['soft_onsets'].to(device),sample['valid'].to(device),
                r['reference']['timebase_bpm'],r['reference']['offset_seconds'],sample['duration'],
                case['events_seconds'],run['metadata']['greedy'],budget)
            prediction={k:v for k,v in result.items() if k not in ('tokens','residuals')}
            if result['status']=='completed':
                prediction['notes']=decode_notes(result,'4k',r['reference']['timebase_bpm'],r['reference']['offset_seconds'])
            run['predictions'][row['id']]=prediction
            print(json.dumps(dict(stage='evaluate',case=row['id'],greedy=run['metadata']['greedy'],f1=result.get('f1_50ms'),types=result.get('types'))),flush=True)
        del sample
    encoder,onset=load_inputs(args,device)
    fixtures=read(args.probes/'fixtures/fixtures.json')
    for row in fixtures['rows']:
        case=cases[row['id']]
        if sha256(row['audio'])!=case['audio_sha256']:raise ValueError('frozen probe audio changed')
        waveform=AudioGPUprocessor(device='cpu').load_from_path(row['audio'])
        condition=encode_waveform(waveform,encoder,onset,budget,device)
        for run in runs:
            result=rollout(decoder,condition['features'].float().to(device),condition['soft_onsets'].to(device),condition['valid'].to(device),
                120.,0.,condition['duration'],case['events_seconds'],run['metadata']['greedy'],budget)
            prediction={k:v for k,v in result.items() if k not in ('tokens','residuals')}
            if result['status']=='completed':prediction['notes']=decode_notes(result,'4k',120.,0.)
            run['predictions'][row['id']]=prediction
            print(json.dumps(dict(stage='evaluate',case=row['id'],greedy=run['metadata']['greedy'],f1=result.get('f1_50ms'),notes=result.get('note_count'))),flush=True)
    del encoder,onset;gc.collect();torch.cuda.empty_cache()
    shifts=[]
    for row in pairs['diagnostics']:
        pair=torch.load(args.pairs/row['file'],map_location='cpu',weights_only=True)
        for a,b in ((0,1),(1,0)):
            target=pair['targets'][a];positive=pair['views'][a];negative=pair['views'][b]
            p,n=time_log_probability(training_logits(decoder,target,positive,device),target['tokens'])
            q,_=time_log_probability(training_logits(decoder,target,negative,device),target['tokens'])
            bpm=target['reference']['timebase_bpm'];tk=ChartTokenizer(4)
            times=sorted({n.tick*60/(32*bpm) for n in tk.decode_tokens(target['tokens'].tolist())})
            result=rollout(decoder,positive['features'].float().to(device),positive['soft_onsets'].to(device),positive['valid'].to(device),
                bpm,0.,pair['duration'],times,False,budget)
            shifts.append(dict(id=row['id'],view=a,plan=pair['plan'],matched_time_nll=float(-p),mismatched_time_nll=float(-q),
                gap=float(p-q),time_tokens=n,reference_first_seconds=times[0],
                generated_first_seconds=result.get('onsets_seconds',[None])[0] if result.get('onsets_seconds') else None,
                generation={k:v for k,v in result.items() if k not in ('tokens','residuals')}))
    bundle=dict(schema=source['schema'],runs=runs)
    write_json(args.output/'inputs.json',bundle)
    report=benchmark(bundle,read(args.protocol));report['resources']=budget.log_state()
    write_json(args.output/'report.json',report)
    write_json(args.output/'diagnostics.json',dict(teacher=audit,shift_pairs=shifts,
        scope='new transformations of seen songs; shift changes chiefly the initial DT; not full-rhythm certification'))
    print(json.dumps(dict(status='evaluated',summaries={r['id']:r['summary'] for r in report['runs']},resources=budget.log_state())),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','train','evaluate'))
    p.add_argument('--variant',choices=('continued','data','rank'),default='data')
    p.add_argument('--cache',type=Path,default=Path('out/full_corpus_v1/cache'))
    p.add_argument('--pairs',type=Path,default=Path('out/grounded_ar_20261023/pairs'))
    p.add_argument('--output',type=Path,default=Path('out/grounded_ar_20261023/data'))
    p.add_argument('--model-root',type=Path,default=Path('checkpoints/timing_v1'))
    p.add_argument('--init',type=Path,default=Path('out/onset_attention_20261022/4k-scratch/global/decoder_latest.pt'))
    p.add_argument('--probes',type=Path,default=Path('out/onset_attention_20261022/onsets'))
    p.add_argument('--baseline-input',type=Path,default=Path('out/benchmark_v1/inputs.json'))
    p.add_argument('--protocol',type=Path,default=Path('benchmarks/v1/protocol.json'))
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--rank-weight',type=float,default=.2)
    p.add_argument('--rank-beta',type=float,default=2.);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cuda:0');add_resource_args(p);p.set_defaults(max_rss_gib=4.)
    args=p.parse_args()
    if args.epochs<1 or not math.isfinite(args.lr) or args.lr<=0:raise ValueError('positive epochs/lr required')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    budget=ResourceBudget(args,torch.device(args.device),args.output)
    {'prepare':prepare,'train':train,'evaluate':evaluate}[args.stage](args,budget,budget.device)


if __name__=='__main__':main()

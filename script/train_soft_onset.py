"""Frozen audio features, fair free-decoding A/B, bounded decoder-only training.

prepare computes inputs from audio only, then attaches independent chart targets.
train never loads a target-derived skeleton; evaluate uses the same saved inputs.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import time

import torch

from script.diagnose_patterns import compact
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import code_version, infer_mel, load_audio, load_model, sha256
from src.conditioning_audit import (encode_audio, enforce_preflight, estimate_working_set,
                                    file_fingerprint, load_reference, tensor_fingerprint)
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.data.chart_tokenizer import ChartTokenizer, free_notes_to_phigros
from src.data.timing import TIMING_SCHEMA, tick_seconds
from src.models.chart_decoder import generate_tokens
from src.models.chart_history import HISTORY_SCHEMA, RELATION_SCHEMA, history_features
from src.pattern_diagnostics import chart_events, summarize
from src.product.artifacts import chart_document, repair_log, validate_chart, write_json, write_preview
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import (CACHE_SCHEMA, MEMORY_SECONDS, SCHEMA, condition_input, dense_condition,
                           forward_loss, initialize_decoder, load_decoder, state_hash,
                           training_preflight, weighted_scores)
from src.test_decoder import load_models

ROUTES = {'4k':'decoder','freex':'decoder_freex'}


def check_size(path, budget, maximum_mib=700):
    state = budget.check(force=True)
    size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
    if size > maximum_mib*1024**2:
        raise RuntimeError('experiment output exceeds explicit artifact budget')
    return dict(bytes=size, snapshot=state)


def prepare(args, output, budget, device):
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest['rows']
    if len({r['id'] for r in rows}) != len(rows) or any(r['split'] not in ('train','val') for r in rows):
        raise ValueError('unique sample IDs and explicit train/val split required')
    hashes = {split:{sha256(r['audio']) for r in rows if r['split']==split} for split in ('train','val')}
    if not hashes['train'] or not hashes['val'] or hashes['train'] & hashes['val']:
        raise ValueError('train/val audio overlap or empty split')
    model_root = Path(args.model_root)
    onset_path = model_root/'transcriber/transcriber_best.pt'
    onset, onset_meta = load_model(onset_path, device)
    if onset.rhythm_head is not None:
        raise ValueError('this experiment requires the same-rate lane onset head; global head unsupported')
    onset.requires_grad_(False)
    models, sources = {}, {'onset':file_fingerprint(onset_path)}
    for mode, subdir in ROUTES.items():
        path = model_root/subdir/'decoder_best.pt'
        enc, dec = load_models(path, device)
        enc.requires_grad_(False); dec.requires_grad_(False)
        models[mode] = (enc,dec)
        sources[mode] = file_fingerprint(path)
    report = dict(schema=CACHE_SCHEMA,status='running',source_manifest=file_fingerprint(args.manifest),
                  source_root=manifest['source_root'],models=sources,rows=[],implementation=code_version(),
                  contract=dict(onset='frozen predicted probability + subframe offset, no threshold/NMS',
                    audio_features='each route frozen legacy encoder, full attention',
                    onset_encoder='1024 mel core + 128 context, same train and inference',
                    memory_frame_seconds=MEMORY_SECONDS,condition_offset_unit='memory frames',
                    target_used_for_acoustic_or_soft_inputs=False,freeze_frontend=True,selection='fixed before training'))
    for index,row in enumerate(rows):
        budget.check(force=True)
        mel,duration = load_audio(row['audio'],max_seconds=args.max_seconds,budget=budget)
        if row.get('audio_sha256') and sha256(row['audio']) != row['audio_sha256']:
            raise ValueError('input audio differs from the frozen selection')
        references = {mode:load_reference(row['chart'],duration,
                      gameplay_path=row['gameplay'] if mode=='freex' else None,
                      source_root=manifest['source_root']) for mode in ROUTES}
        if any(len(ref.tokens)>args.max_reference_tokens for ref in references.values()):
            raise ValueError(f'{row["id"]}: complete reference exceeds token limit; no truncation')
        # All input encoders are frozen, and the input API receives no reference.
        arrays = infer_mel(onset,mel,duration,chunk_frames=1024,context_frames=128,check=budget.check)
        sample = dict(id=row['id'],duration=duration,audio=file_fingerprint(row['audio']),routes={})
        for mode,(enc,dec) in models.items():
            ref = references[mode]
            enforce_preflight(estimate_working_set(enc,dec,mel.shape[-1],len(ref.tokens)),budget,device)
            feats,valid,geometry = encode_audio(enc,mel,duration,device)
            sample['routes'][mode] = dict(features=feats.cpu(),tokens=ref.tokens,frames=ref.frames,
                offsets=ref.offsets,offset_mask=ref.offset_mask,reference=ref.metadata,geometry=geometry)
            if 'valid' in sample and not torch.equal(sample['valid'],valid.cpu()):
                raise ValueError('two route encoders have different clocks')
            sample['valid'] = valid.cpu()
        sample['soft_onsets'] = dense_condition(arrays,sample['valid'].shape[1])
        path = output/f'sample-{index:03d}.pt'
        budget.save(sample,path)
        report['rows'].append(dict(id=row['id'],split=row['split'],file=path.name,sha256=sha256(path),
            audio=sample['audio'],duration=duration,
            token_counts={m:len(r['tokens']) for m,r in sample['routes'].items()},
            condition=tensor_fingerprint(sample['soft_onsets']),valid=tensor_fingerprint(sample['valid'])))
        print(f'[prepare] {index+1}/{len(rows)} {row["id"]}',flush=True)
        del sample,arrays,references,mel,feats,valid
        check_size(output,budget)
    report.update(status='completed',resources=check_size(output,budget))
    write_json(output/'cache.json',report)
    return report


def cache_inputs(path):
    path = Path(path)
    manifest = json.loads((path/'cache.json').read_text())
    if manifest.get('schema') != CACHE_SCHEMA or manifest.get('status') != 'completed':
        raise ValueError('completed versioned cache required')
    for row in manifest['rows']:
        target = (path/row['file']).resolve()
        if target.parent != path.resolve() or sha256(target) != row['sha256']:
            raise ValueError('frozen sample cache changed or path escaped')
    return manifest


def load_sample(cache,row):
    return torch.load(Path(cache)/row['file'],map_location='cpu',weights_only=True)


def sample_order(rows, steps, seed):
    if not rows or steps<1:
        raise ValueError('nonempty training rows and positive steps required')
    rng = random.Random(seed); order = []
    while len(order)<steps:
        epoch = list(range(len(rows))); rng.shuffle(epoch); order.extend(epoch)
    return order[:steps]


def train(args,output,budget,device):
    cache = cache_inputs(args.cache)
    source = cache['models'][args.mode]
    if sha256(source['path']) != source['sha256']:
        raise ValueError('initial decoder changed')
    decoder,config,initial = initialize_decoder(source['path'],args.seed,
        with_history=args.condition.startswith('soft_history'),with_relation_aux=args.condition=='soft_history_aux')
    decoder.to(device).train()
    rows = [r for r in cache['rows'] if r['split']=='train']
    if args.train_limit:
        rows = rows[:args.train_limit]
    order = sample_order(rows,args.steps,args.seed)
    optimizer = torch.optim.AdamW(decoder.parameters(),lr=args.lr,weight_decay=.01)
    # Model constructors may consume RNG even for zero-initialized adapters.
    # Reset after construction so every arm starts the same dropout stream.
    torch.manual_seed(args.seed)
    losses = []; estimates = {}; first_gradient = None; first_history_gradient=None; started=time.monotonic()
    for step,index in enumerate(order,1):
        row = rows[index]; sample = load_sample(args.cache,row)
        if row['id'] not in estimates:
            estimates[row['id']] = training_preflight(decoder,sample,args.mode,budget,device)
        budget.check()
        optimizer.zero_grad(set_to_none=True)
        loss,details = forward_loss(decoder,sample,args.mode,args.condition,device)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite training loss')
        loss.backward()
        gradient = decoder.soft_onset_proj.weight.grad
        if first_gradient is None:
            first_gradient = float(gradient.norm())
            if args.condition!='none' and first_gradient<=0:
                raise RuntimeError('soft onset projection received no gradient')
            if decoder.history_dim:
                first_history_gradient=float(decoder.history_proj.weight.grad.norm())
                if first_history_gradient<=0:
                    raise RuntimeError('causal history projection received no gradient')
        torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)
        optimizer.step()
        losses.append(dict(step=step,sample=row['id'],loss=float(loss.detach()),**details))
        if step==1 or step%25==0 or step==args.steps:
            print(f'[train {args.mode}/{args.condition}] {step}/{args.steps} loss={float(loss.detach()):.4f} '
                  f'seconds={time.monotonic()-started:.1f}',flush=True)
        del sample,loss
    checkpoint = dict(soft_onset_schema=SCHEMA,timing_schema=TIMING_SCHEMA,condition=args.condition,
                      history_schema=HISTORY_SCHEMA if decoder.history_dim else None,
                      relation_schema=RELATION_SCHEMA if decoder.relation_head is not None else None,
                      relation_weight=.1 if decoder.relation_head is not None else None,
                      model={k:v.detach().cpu() for k,v in decoder.state_dict().items()},dec_config=config,
                      mode=args.mode,steps=args.steps,seed=args.seed,lr=args.lr,initialization=initial,
                      cache_manifest=file_fingerprint(Path(args.cache)/'cache.json'),
                      frozen_sources=cache['models'],training_order=[rows[i]['id'] for i in order],
                      inference_contract=cache['contract'])
    budget.save(checkpoint,output/'decoder_final.pt')
    report = dict(schema=SCHEMA,status='trained',mode=args.mode,condition=args.condition,steps=args.steps,
        seed=args.seed,lr=args.lr,optimizer='AdamW, weight_decay=.01, clip_grad_norm=1',
        dropout_rng_reset='after model/optimizer construction',
        initialization=initial,first_projection_gradient_norm=first_gradient,
        final_projection_norm=float(decoder.soft_onset_proj.weight.detach().norm()),
        first_history_projection_gradient_norm=first_history_gradient,
        final_history_projection_norm=float(decoder.history_proj.weight.detach().norm()) if decoder.history_dim else None,
        training_order=checkpoint['training_order'],train_samples=len(rows),metrics=losses,
        checkpoint=file_fingerprint(output/'decoder_final.pt'),cache=checkpoint['cache_manifest'],
        estimates=estimates,training_seconds=time.monotonic()-started,implementation=code_version())
    del optimizer,checkpoint,decoder
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    report['evaluation'] = evaluate(args,output,budget,device,output/'decoder_final.pt')
    report.update(status='completed',resources=check_size(output,budget))
    write_json(output/'report.json',report)
    return report


def onset_metrics(raw,reference,bpm,offset,duration):
    generated = sorted({n['time']*tick_seconds(1,bpm)+offset for n in raw})
    truth = sorted({n.tick*tick_seconds(1,bpm)+offset for n in reference})
    errors=[abs(t-r) for r,t in match_events(truth,generated,.05)]
    p=len(errors)/len(generated) if generated else 0.;r=len(errors)/len(truth) if truth else 0.
    return dict(reference='retained chart onsets, not acoustic ground truth',tolerance_seconds=.05,
        generated_onsets=len(generated),reference_onsets=len(truth),matched=len(errors),
        precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.,
        matched_mean_abs_error_seconds=sum(errors)/len(errors) if errors else None,
        audio_duration_seconds=duration,last_onset_seconds=generated[-1] if generated else None,
        last_note_end_seconds=max(((n['time']+n['holdTime'])*tick_seconds(1,bpm)+offset for n in raw),default=None),
        last_onset_fraction_of_audio=generated[-1]/duration if generated else 0.)


@torch.no_grad()
def evaluate(args,output,budget,device,checkpoint=None):
    checkpoint = checkpoint or args.checkpoint
    cache=cache_inputs(args.cache);decoder,ck=load_decoder(checkpoint,device)
    if ck['cache_manifest']['sha256'] != sha256(Path(args.cache)/'cache.json'):
        raise ValueError('evaluation cache differs from the frozen training inputs')
    mode,condition=ck['mode'],ck['condition'];tk=ChartTokenizer(decoder.n_lanes)
    rows=[r for r in cache['rows'] if r['split']=='val']
    if args.val_limit: rows=rows[:args.val_limit]
    scores=[];generated=[]
    for i,row in enumerate(rows):
        budget.check(force=True);sample=load_sample(args.cache,row)
        _,score=forward_loss(decoder,sample,mode,condition,device,score=True)
        scores.append(dict(id=row['id'],**score))
        if i>=args.generate_limit: continue
        route=sample['routes'][mode];meta=route['reference']
        bpm,offset=meta['timebase_bpm'],meta['offset_seconds'];duration=sample['duration']
        folder=output/f'val-{i:02d}';folder.mkdir()
        result=dict(id=row['id'],mode=mode,condition=condition,forced_skeleton=None,
                    status='running',seed=args.seed,max_decoder_steps=args.max_decoder_steps)
        try:
            memory=decoder.prepare_memory(route['features'].to(device),
                      condition_input(sample['soft_onsets'].to(device),condition))
            gen=torch.Generator(device=device).manual_seed(args.seed)
            sequence=generate_tokens(decoder,memory,sample['valid'].to(device),bpm,MEMORY_SECONDS*1000,
                duration/MEMORY_SECONDS,top_p=.95,temperature=1.,max_tokens=args.max_decoder_steps,
                generator=gen,tokenizer=tk,skeleton_ticks=None,offset=offset,require_complete=True,
                progress_callback=lambda *_:budget.check())
            tokens,offsets=sequence if decoder.offset_head is not None else (sequence,{})
            notes=tk.decode_tokens(tokens,strict=True)
            ticks=tk.ticks_per_position(tokens)
            positions={(ticks[j],tk.value(tokens[j])//4):x for j,x in offsets.items()}
            raw=(free_notes_to_phigros(tk,notes,positions,bpm) if decoder.n_lanes==64
                 else tk.to_phigros_notes(notes,bpm))
            rawchart=chart_document(raw,bpm,offset,folder/'raw.json')
            end_tick=(duration-offset)/tick_seconds(1,bpm)
            final,repairs=(constrain_4k_notes(raw,end_tick) if mode=='4k' else constrain_free_notes(raw,end_tick))
            chart=chart_document(final,bpm,offset,folder/'chart.json')
            events,checks=validate_chart(chart,mode,duration,None)
            write_json(folder/'events.json',events)
            audio=Path(sample['audio']['path']);(folder/('audio'+audio.suffix)).symlink_to(audio)
            write_preview(folder/'preview.html',title=f'{row["id"]} · {mode} / {condition} · 训练实验候选',
                audio_name='audio'+audio.suffix,duration=duration,
                routes={mode:dict(events=events['events'],chart_file='chart.json',events_file='events.json')})
            result.update(status='completed',termination='natural_eos',tokens=len(tokens),checks=checks,
                raw=compact(summarize(chart_events(rawchart,mode))),
                final=compact(summarize(chart_events(chart,mode))),repairs=repairs,
                changes=repair_log(raw,final),
                onset=onset_metrics(raw,tk.decode_tokens(route['tokens'].tolist(),strict=True),bpm,offset,duration),
                preview=str((folder/'preview.html').resolve()))
            write_json(folder/'tokens.json',dict(tokens=tokens,continuous_offsets=offsets))
        except Exception as error:
            result.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
            (folder/'preview.html').unlink(missing_ok=True)
        generated.append(result)
        print(f'[generate {mode}/{condition}] {row["id"]}: {result["status"]}',flush=True)
    # Same trained B, true predictions vs zero is a finite sensitivity probe.
    sensitivity=None
    if condition!='none' and rows:
        sample=load_sample(args.cache,rows[0]);route=sample['routes'][mode]
        tokens=route['tokens'][:65].to(device);frames=route['frames'][:65].to(device)
        outputs=[]
        history=None
        if decoder.history_dim:
            history=history_features(route['tokens'][:len(tokens)-1],route['frames'][:len(tokens)-1],
                n_lanes=decoder.n_lanes,frames_per_beat=60/route['reference']['timebase_bpm']/MEMORY_SECONDS,
                offsets=route['offsets'][:len(tokens)-1] if decoder.n_lanes==64 else None)[None].to(device)
        for selected in ('none','soft'):
            mem=decoder.prepare_memory(route['features'].to(device),
                        condition_input(sample['soft_onsets'].to(device),selected))
            outputs.append(decoder(tokens[None,:-1],frames[None,:-1],mem,sample['valid'].to(device),history=history))
        sensitivity=dict(sample=rows[0]['id'],first_input_tokens=len(tokens)-1,
                         max_abs_logit_difference=float((outputs[0]-outputs[1]).abs().max()))
    return dict(val_samples=len(rows),teacher_forcing=scores,weighted=weighted_scores(scores),
                free_generation=generated,soft_condition_sensitivity=sensitivity,
                interpretation='development holdout; chart-reference similarity and unreviewed samples, not playable certification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','evaluate'))
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--manifest');parser.add_argument('--model-root');parser.add_argument('--cache')
    parser.add_argument('--mode',choices=tuple(ROUTES));parser.add_argument('--condition',choices=('none','soft','soft_history','soft_history_aux'))
    parser.add_argument('--checkpoint');parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--train-limit',type=int,default=0);parser.add_argument('--val-limit',type=int,default=0)
    parser.add_argument('--generate-limit',type=int,default=2);parser.add_argument('--max-decoder-steps',type=int,default=6000)
    parser.add_argument('--max-seconds',type=float,default=240);parser.add_argument('--max-reference-tokens',type=int,default=4096)
    add_resource_args(parser);parser.set_defaults(max_rss_gib=4)
    args=parser.parse_args()
    if args.steps<1 or not 0<args.lr<1 or args.max_decoder_steps<1 or args.max_reference_tokens<2 \
            or min(args.train_limit,args.val_limit,args.generate_limit)<0:
        parser.error('positive steps/lr and nonnegative sample limits required')
    required={'prepare':('manifest','model_root'),'train':('cache','mode','condition'),
              'evaluate':('cache','checkpoint')}[args.action]
    if any(getattr(args,key) is None for key in required):
        parser.error(f'{args.action} requires {required}')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);device=torch.device(args.device)
    started=time.monotonic()
    try:
        budget=ResourceBudget(args,device,output)
        if args.action=='prepare': report=prepare(args,output,budget,device)
        elif args.action=='train': report=train(args,output,budget,device)
        else:
            report=dict(schema=SCHEMA,status='completed',evaluation=evaluate(args,output,budget,device))
            write_json(output/'report.json',report)
        write_json(output/'job.json',dict(status='completed',seconds=time.monotonic()-started,
                                         resources=check_size(output,budget)))
    except BaseException as error:
        (output/'cache.json').unlink(missing_ok=True);(output/'report.json').unlink(missing_ok=True)
        write_json(output/'job.json',dict(status='failed',error=dict(type=type(error).__name__,message=str(error))))
        raise


if __name__=='__main__':main()

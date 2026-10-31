"""Stage 5: separate output factorization, encoder adaptation and capacity.

Uses the frozen stage-2 diagnostic split, independent random starts and the
same exposure/schedule. Not a natural-song production training run.
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
import torch.nn.functional as F

from script.train_alignment_scratch import training_state, restore_training
from script.train_grounded_ar import load_inputs
from script.train_windowed_ar import constructed_chart, read, score, COUNTS
from script.probe_audio_localization import sources
from src.anchored_ar import normalized_loss
from src.audio_understanding import SAMPLE_RATE, FRAME_SECONDS
from src.audio_understanding.inference import sha256
from src.data.audio2mel import AudioGPUprocessor
from src.product.artifacts import write_json
from src.resource_budget import ResourceBudget, add_resource_args
from src.soft_onset import state_hash
from src.windowed_ar import WindowedAR, encode_window_target, generate, window_intervals
from src.windowed_variants import AcousticTail, VariantAR

SCHEMA = 'audiolab-window-representation-ablation-v1'
SOURCES = ('script/train_windowed_variants.py', 'src/windowed_variants.py',
    'script/train_windowed_ar.py', 'src/windowed_ar.py', 'src/anchored_ar.py',
    'src/models/chart_decoder.py', 'src/models/transcriber.py', 'src/models/chart_history.py')


@torch.no_grad()
def prepare_tail(args, budget, device):
    dest = args.tail_data
    if dest.exists():
        raise ValueError('refuse to overwrite prepared audio tail cache')
    rows = torch.load(args.data_root/'conditions.pt', map_location='cpu', weights_only=True)
    manifest = read(args.data_root/'manifest.json')
    assert sha256(args.data_root/'conditions.pt') == manifest['data_sha256']
    encoder, unused = load_inputs(args, device)
    del unused
    assert sha256(args.model_root/'decoder/decoder_best.pt') == manifest['frontends']['4k']['sha256']
    clips, metadata = sources(args.samples)
    assert metadata == manifest['sources']
    processor = AudioGPUprocessor(device='cpu')
    fixed = {r['id']:r for r in read(args.fixed_probes)['rows']}
    captured = []
    handle = encoder.blocks[-1].register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach().float().cpu()))
    tail = AcousticTail.from_encoder(encoder).to(device).eval()
    comparisons = []
    for row in rows:
        captured.clear()
        if row['split'] == 'fixed_probe':
            item = fixed[row['id']]
            assert sha256(item['audio']) == row['source_audio_sha256']
            wave = processor.load_from_path(item['audio'])
        else:
            seed = manifest['seed']+(10000019 if row['split']=='random_check' else 0)
            wave, reproduced = constructed_chart(int(row['id'].split('-')[-1]), seed, clips, row['split'])
            assert reproduced['source_events'] == row['source_events']
        mel = processor(wave)[0][:, :math.ceil(row['duration']/FRAME_SECONDS)]
        mel = F.pad(mel, (0, mel.shape[-1]%2))
        expected = encoder.features(mel[None].to(device))
        assert len(captured) == 1
        prefix = captured[0]
        actual = tail(prefix.to(device))
        difference = float((actual-expected).abs().max())
        cache_difference = float((expected.half().cpu()-row['condition']['features']).abs().max())
        if difference > 1e-6 or cache_difference > .004:
            raise RuntimeError(f'frontend reconstruction mismatch {row["id"]}: {difference}, {cache_difference}')
        row['condition']['prefix_features'] = prefix
        comparisons.append(dict(id=row['id'],tail_max_error=difference,original_half_cache_max_error=cache_difference))
        budget.check()
    handle.remove()
    payload = dict(schema=SCHEMA, original_data_sha256=manifest['data_sha256'], rows=rows,
        tail_config=tail.config, tail_state={k:v.cpu() for k,v in tail.state_dict().items()},
        comparisons=comparisons, source_sha256={p:sha256(p) for p in SOURCES})
    budget.save(payload, dest)
    write_json(dest.with_suffix('.json'),{k:v for k,v in payload.items() if k not in ('rows','tail_state')})
    print(json.dumps(dict(status='tail_prepared',rows=len(rows),sha256=sha256(dest),
        maximum_tail_error=max(r['tail_max_error'] for r in comparisons),
        maximum_old_cache_error=max(r['original_half_cache_max_error'] for r in comparisons),
        resources=budget.log_state())),flush=True)


@torch.no_grad()
def teacher(model, target, condition):
    device = next(model.parameters()).device
    d = model.decoder
    memory = d.prepare_memory(condition['features'].float().to(device),torch.zeros_like(condition['soft_onsets']).to(device))
    logits, hidden = d(target['tokens'][None].to(device), target['frames'][None].to(device), memory,
        condition['valid'].to(device), history=target['history'][None].to(device), return_hidden=True,
        input_condition=model.window_condition(target['bounds'].to(device))[None])
    indices = target['decisions'].to(device); h = hidden[0,indices]
    positive = target['stop_targets'].to(device)==0
    stopped = model.stop_head(h).squeeze(-1)>=0
    labels = target['layout_targets'].to(device); active = labels!=-100
    counts = Counter(events=int(positive.sum()),stops=int((~positive).sum()),
        false_stops=int((stopped&positive).sum()),correct_stops=int((stopped&~positive).sum()),
        layout_total=int(active.sum()),layout_correct=int((logits[0,active].argmax(-1)==labels[active]).sum()))
    if positive.any():
        lo,hi,allowed = window_intervals(target['lower_seconds'].to(device)[positive],
            target['bounds'].to(device)[indices[positive],1],memory.shape[1],condition['valid'].to(device))
        selected=model.frame_logits(h[positive],memory,allowed).argmax(-1);r=torch.arange(len(selected),device=device)
        times=lo[r,selected]+model.fraction(h[positive],memory[0,selected])*(hi[r,selected]-lo[r,selected])
        correct=(times-target['next_seconds'].to(device)[positive]).abs()<=.025
        counts.update(pointer_correct_25ms=int(correct.sum()),continue_and_time_correct_25ms=int((correct&~stopped[positive]).sum()))
    return dict(counts)


@torch.no_grad()
def evaluate(model, rows, args, budget, epoch, full):
    model.eval();outputs=[];teachers={}
    selected=rows if full else [r for r in rows if r['split']=='train' and int(r['id'].split('-')[-1]) in (0,1,2,7,12,15,23,31)]
    for row in selected:
        condition=model.audio_condition(row['condition'])
        result=generate(model,condition,120.,use_soft=False,greedy=True,max_tokens=512,check=budget.check)
        result.pop('trace',None);score(result,row)
        if full and row['split']!='fixed_probe':
            target=encode_window_target(row['route'],row['duration'])
            taught=teacher(model,target,condition)
            teachers.setdefault(row['split'],Counter()).update(taught)
            result['teacher']=taught
            clean=normalized_loss(WindowedAR.components(model,target,condition,use_soft=False),target)
            zero=dict(condition,features=torch.zeros_like(condition['features']))
            ablated=normalized_loss(WindowedAR.components(model,target,zero,use_soft=False),target)
            result['feature_zeroing_loss']=dict(clean=float(clean),zero=float(ablated),delta=float(ablated-clean))
        outputs.append(dict(id=row['id'],split=row['split'],**result))
    aggregate={}
    for split in sorted({r['split'] for r in outputs}):
        subset=[r for r in outputs if r['split']==split];positive=[r for r in subset if r['reference_onsets']]
        a=dict(examples=len(subset),positive_examples=len(positive),completed=sum(r['status']=='completed' for r in subset),
            onset_f1_25ms=sum(r['onset']['25']['f1'] for r in positive)/max(1,len(positive)),
            silence_extra_onsets=sum(r['generated_onsets'] for r in subset if not r['reference_onsets']),
            forced_windows=sum(r['forced_windows'] for r in subset))
        if split!='fixed_probe':
            a['joint_f1_25ms']=sum(r['joint_f1_25ms'] if r['status']=='completed' else 0. for r in positive)/max(1,len(positive))
        if split in teachers:a['teacher']=dict(teachers[split])
        aggregate[split]=a
    report=dict(epoch=epoch,variant=args.variant,full=full,aggregates=aggregate,rows=outputs,
        scope='Frozen 32-example constructed diagnostic, full audio. Positive-only onset macro; '
        'failed joint rollouts score zero. Feature zeroing is an intervention, NOT a silent waveform.')
    write_json(args.output/f'evaluation-e{epoch:04d}.json',report)
    return aggregate


def train(args,budget,device):
    manifest=read(args.data_root/'manifest.json')
    assert sha256(args.data_root/'conditions.pt')==manifest['data_sha256']
    data=torch.load(args.tail_data,map_location='cpu',weights_only=True) if args.variant=='encoder_tail' else None
    if data is not None:assert data['original_data_sha256']==manifest['data_sha256']
    rows=data['rows'] if data else torch.load(args.data_root/'conditions.pt',map_location='cpu',weights_only=True)
    train_rows=[r for r in rows if r['split']=='train'];assert len(train_rows)==32
    decoder_config=read('docs/evaluations/diffusion_ar_attribution_20261023.json')['ar']['config']
    torch.manual_seed(args.seed)
    model=VariantAR(decoder_config,variant=args.variant,tail_config=data['tail_config'] if data else None).to(device)
    if data:model.audio_tail.load_state_dict(data['tail_state'])
    model.decoder.gradient_checkpointing=True
    shared={k:v for k,v in model.state_dict().items() if not k.startswith(('decoder.lm_head.','audio_tail.'))}
    config=dict(variant=args.variant,condition='A',seed=args.seed,epochs=args.epochs,lr=args.lr,
        parameters=sum(p.numel() for p in model.parameters()),decoder_config=model.decoder_config,
        tail_config=data['tail_config'] if data else None,encoder_lr=args.encoder_lr if data else None,
        data_sha256=manifest['data_sha256'],tail_data_sha256=sha256(args.tail_data) if data else None,
        common_initial_state_sha256=state_hash(shared),source_sha256={p:sha256(p) for p in SOURCES},
        window_seconds=10.,history_noise_ms=0.,scope='32-example controlled diagnostic; no natural training')
    targets={r['id']:encode_window_target(r['route'],r['duration']) for r in train_rows}
    groups=[dict(params=[p for n,p in model.named_parameters() if not n.startswith('audio_tail.')],base_lr=args.lr)]
    if data:groups.append(dict(params=list(model.audio_tail.parameters()),base_lr=args.encoder_lr))
    optimizer=torch.optim.AdamW(groups,lr=args.lr);sampler=random.Random(args.seed)
    curves=[];steps=0;started=time.monotonic();torch.manual_seed(args.seed+1000)
    if (args.output/'model_latest.pt').exists():
        if not args.resume:raise ValueError('checkpoint exists; explicit --resume required')
        old=torch.load(args.output/'model_latest.pt',map_location='cpu',weights_only=True)
        assert old['config']==config,'resume source/config changed'
        model.load_state_dict(old['model']);restore_training(old['training_state'],optimizer,sampler)
        curves=old['curves'];steps=old['steps'];started-=curves[-1]['elapsed_seconds']
    best=min((r['objective'] for r in curves),default=math.inf)
    write_json(args.output/'training.json',dict(status='running',config=config,curves=curves))
    print(json.dumps(dict(status='validated',config=config,resources=budget.log_state())),flush=True)
    for epoch in range(len(curves),min(args.epochs,args.until_epoch or args.epochs)):
        order=list(train_rows);sampler.shuffle(order);model.train();stats=Counter();objective=0.
        for row in order:
            target=targets[row['id']]
            factor=min(1.,(steps+1)/max(1,.05*args.epochs*len(train_rows)))*(.1+.9*(1+math.cos(math.pi*steps/max(1,args.epochs*len(train_rows)-1)))/2)
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            optimizer.zero_grad(set_to_none=True)
            parts=model.components(target,row['condition'],use_soft=False);loss=normalized_loss(parts,target)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite stage-5 objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();steps+=1;objective+=float(loss.detach())
            for k,v in parts.items():stats[k]+=float(v.detach()) if torch.is_tensor(v) else v
            for k in COUNTS:stats[k]+=target[k]
            budget.check()
        record=dict(epoch=epoch+1,steps=steps,objective=objective/len(order),elapsed_seconds=time.monotonic()-started,
            frame_accuracy=stats['frame_correct']/max(1,stats['time_count']),
            conditional_time_mae_seconds=stats['conditional_offset_absolute_seconds']/max(1,stats['time_count']),
            stop_recall=stats['correct_stops']/max(1,stats['stop_count']),false_stop_rate=stats['false_stops']/max(1,stats['continue_count']),
            counts={k:int(stats[k]) for k in COUNTS},learning_rates=[g['lr'] for g in optimizer.param_groups])
        curves.append(record);last=epoch+1==min(args.epochs,args.until_epoch or args.epochs)
        if (epoch+1)%25==0 or last:
            saved=dict(schema=SCHEMA,config=config,epoch=epoch+1,steps=steps,curves=curves,
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},training_state=training_state(optimizer,sampler))
            budget.save(saved,args.output/'model_latest.pt')
            if record['objective']<best:
                best=record['objective'];budget.save({k:v for k,v in saved.items() if k!='training_state'},args.output/'model_best_fit.pt')
            del saved
            write_json(args.output/'training.json',dict(status='running',config=config,curves=curves,resources=budget.log_state()))
            print(json.dumps(record),flush=True)
        if (args.eval_every and (epoch+1)%args.eval_every==0) or last:
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                summary=evaluate(model,rows,args,budget,epoch+1,full=epoch+1==args.epochs)
            print(json.dumps(dict(stage='evaluation',epoch=epoch+1,aggregates=summary)),flush=True)
    write_json(args.output/'training.json',dict(status='completed' if epoch+1==args.epochs else 'phase_completed',
        config=config,curves=curves,resources=budget.log_state(),checkpoint_sha256=sha256(args.output/'model_latest.pt')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare-tail','train'))
    parser.add_argument('--variant',choices=('base','kind_position','encoder_tail','capacity'),default='base')
    parser.add_argument('--data-root',type=Path,default=Path('out/windowed_ar_20261031/controlled'))
    parser.add_argument('--tail-data',type=Path,default=Path('out/windowed_ar_20261031/controlled/tail_inputs.pt'))
    parser.add_argument('--model-root',type=Path,default=Path('checkpoints/timing_v1'))
    parser.add_argument('--samples',type=Path,default=Path('out/onset_attention_20261022/samples'))
    parser.add_argument('--fixed-probes',type=Path,default=Path('out/onset_attention_20261022/onsets/fixtures/fixtures.json'))
    parser.add_argument('--output',type=Path);parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--until-epoch',type=int);parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--encoder-lr',type=float,default=3e-5);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',default='cuda:0');add_resource_args(parser);parser.set_defaults(max_rss_gib=4.)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='train' and args.output is None:parser.error('--output required')
    output=args.output or args.tail_data.parent;output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device);budget=ResourceBudget(args,device,output)
    (prepare_tail if args.stage=='prepare-tail' else train)(args,budget,device)


if __name__=='__main__':main()

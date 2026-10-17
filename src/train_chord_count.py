"""Bounded frozen-encoder experiment for acoustic chord cardinality.

Split TRAIN by song into fit/tune, select the head using tune NLL, and only then
measure the untouched development VAL list. Wrong-song acoustic controls keep
onset-gap context unchanged. Count labels retain all original multi-press notes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from script.diagnose_decoder import sha256_file
from script.train_models import write_json
from src.data.chart_tokenizer import ChartTokenizer
from src.data.dataset import PhigrosDataset
from src.data.timing import TIMING_SCHEMA, check_checkpoint_timing
from src.models.chord_count import ChordCountHead, event_features, FEATURE_SCHEMA, FRAME_SECONDS
from src.models.transcriber import TranscriberNet
from src.resource_budget import ResourceBudget, add_resource_args
from src.train_utils import pad_or_trim


def song_split(songs, seed, tune_songs):
    if len(songs) != len(set(songs)) or not 1 <= tune_songs < len(songs):
        raise ValueError('unique songs and nonempty fit/tune sets required')
    shuffled = list(songs); random.Random(seed).shuffle(shuffled)
    tune = set(shuffled[:tune_songs])
    return [s for s in songs if s not in tune], [s for s in songs if s in tune]


@torch.no_grad()
def extract(listing, encoder, device, mode, budget):
    ds = PhigrosDataset(listing, dict(frame_ms=FRAME_SECONDS*1000, max_frame=4096),
                       cache_dir='data/cache_mel', augment=False)
    records = []
    tk = ChartTokenizer()
    for source, audio, _ in ds._entries:
        budget.check(force=True)
        song = Path(source).parent.name
        cache = ds._cache_path(audio)
        if cache is None or not cache.is_file():
            raise ValueError(f'missing existing mel cache: {song}')
        chart = json.loads(Path(source).read_text())
        bpm, offset = chart['judgeLineList'][0]['bpm'], chart.get('offset', 0)
        target_path = Path('data/gameplay_charts', song+'.json') if mode == 'freex' else Path(source)
        if mode == 'freex':
            target = json.loads(target_path.read_text())
            if target['judgeLineList'][0]['bpm'] != bpm or target.get('offset', 0) != offset:
                raise ValueError('gameplay timing differs from source')
            ticks = [n['time'] for n in target['judgeLineList'][0]['notesAbove']]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ticks = [n.tick for n in tk.decode_tokens(tk.encode_chart(str(source)))]
        duration = sf.info(audio).duration
        counts = Counter(t for t in ticks if 0 <= t*60/(32*bpm)+offset < duration)
        ordered = sorted(counts)
        if not ordered or max(counts.values()) > (64 if mode == 'freex' else 4):
            raise ValueError(f'empty chart or chord exceeds tokenizer vocabulary: {song}')
        mel = torch.from_numpy(np.load(cache, allow_pickle=False)).float()
        frames = math.ceil(mel.shape[1]/encoder.mel_per_frame)
        # Bound extraction to a maximum of 8192 audio feature frames (~380s).
        # Longer songs fail explicitly instead of silently truncating their targets.
        if frames > 8192:
            raise ValueError(f'song exceeds bounded encoder extraction: {song}, {frames} frames')
        mel = pad_or_trim(mel, frames*encoder.mel_per_frame).unsqueeze(0).to(device)
        encoded = encoder.features(mel)[0]
        times = [t*60/(32*bpm)+offset for t in ordered]
        features = event_features(encoded, times, FRAME_SECONDS).half().cpu()
        labels = torch.tensor([counts[t]-1 for t in ordered], dtype=torch.long)
        records.append(dict(song=song, x=features, y=labels,
            source_sha256=sha256_file(target_path), mel_sha256=sha256_file(cache)))
        del mel, encoded
        if len(records)%30 == 0:
            print(f'[extract] {listing}: {len(records)}/{len(ds)}', flush=True)
    return records


@torch.no_grad()
def evaluate(head, records, prior, device, *, wrong_audio=False):
    head.eval()
    rows = []
    for i, r in enumerate(records):
        x = r['x'].float()
        if wrong_audio:
            donor = records[(i+1)%len(records)]['x']
            idx = torch.linspace(0, len(donor)-1, len(x)).round().long()
            x[:, :-2] = donor[idx, :-2].float()  # keep recipient rhythm context
        probs = torch.cat([head(b.to(device)).softmax(-1).cpu() for b in x.split(2048)])
        y = r['y']; pred = probs.argmax(-1)
        classes = torch.arange(1, probs.shape[1]+1)
        expected = (probs*classes).sum(-1)
        chord = y >= 1
        rows.append(dict(song=r['song'], events=len(y),
            nll=float(-probs[torch.arange(len(y)), y].clamp_min(1e-12).log().mean()),
            prior_nll=float(-prior[y].log().mean()), accuracy=float((pred==y).float().mean()),
            count_mae=float((expected-(y+1)).abs().mean()),
            expected_note_ratio=float(expected.sum()/(y+1).sum()),
            chord_events=int(chord.sum()), detected_chords=int(((pred>=1)&chord).sum()),
            predicted_chord_events=int((pred>=1).sum()),
            target_histogram=torch.bincount(y, minlength=len(prior)).tolist(),
            predicted_histogram=torch.bincount(pred, minlength=len(prior)).tolist()))
    return dict(macro={key:float(np.mean([row[key] for row in rows])) for key in
                       ['nll','prior_nll','accuracy','count_mae','expected_note_ratio']}, songs=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['4k','freex'], required=True)
    p.add_argument('--data-list', default='data/train.txt')
    p.add_argument('--val-list', default='data/val.txt')
    p.add_argument('--encoder-ckpt', default='checkpoints/timing_v1/transcriber/transcriber_best.pt')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--tune-songs', type=int, default=27)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    add_resource_args(p)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('positive epochs, batch size and learning rate required')
    run=Path(args.run_dir); run.mkdir(parents=True, exist_ok=False)
    device=torch.device(args.device); budget=ResourceBudget(args, device, run)
    random.seed(args.seed); torch.manual_seed(args.seed)
    ck=torch.load(args.encoder_ckpt, map_location='cpu', weights_only=True)
    check_checkpoint_timing(ck, resume=True)
    encoder=TranscriberNet(**ck['config']); encoder.load_state_dict(ck.get('ema', ck['model']))
    encoder.to(device).eval().requires_grad_(False)
    encoder_config=ck['config']; del ck
    report=dict(config=vars(args), complete=False, started=time.time(),
                encoder_sha256=sha256_file(args.encoder_ckpt), timing_schema=TIMING_SCHEMA,
                feature_schema=FEATURE_SCHEMA, epochs=[])
    train=extract(args.data_list, encoder, device, args.mode, budget)
    fit_names,tune_names=song_split([r['song'] for r in train], args.seed, args.tune_songs)
    fit=[r for r in train if r['song'] in fit_names]; tune=[r for r in train if r['song'] in tune_names]
    report['split']=dict(fit=fit_names, tune=tune_names)
    report['fingerprints']={str(Path(p).resolve()):sha256_file(p) for p in [args.data_list,args.val_list,args.encoder_ckpt]}
    report['train_sources']=[{k:v for k,v in r.items() if k not in ['x','y']} for r in train]
    classes=64 if args.mode=='freex' else 4
    config=dict(input_dim=train[0]['x'].shape[1], n_classes=classes)
    head=ChordCountHead(**config).to(device)
    x=torch.cat([r['x'] for r in fit]); y=torch.cat([r['y'] for r in fit])
    hist=torch.bincount(y, minlength=classes); prior=(hist.float()+.05)/(len(y)+.05*classes)
    head.initialize_prior(hist.to(device))
    optimizer=torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    best=float('inf')
    print(f'[count] fit={len(fit)} tune={len(tune)} events={len(y)} parameters={sum(p.numel() for p in head.parameters())}',flush=True)
    for epoch in range(args.epochs):
        budget.check(force=True); head.train()
        order=torch.randperm(len(y)); total=0.
        for ids in order.split(args.batch_size):
            budget.check()
            logits=head(x[ids].float().to(device))
            loss=F.cross_entropy(logits,y[ids].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); optimizer.step()
            total+=float(loss.detach())*len(ids)
        scores=evaluate(head,tune,prior,device)
        metrics=dict(epoch=epoch+1,train_nll=total/len(y),tune=scores['macro'])
        report['epochs'].append(metrics)
        if scores['macro']['nll'] < best:
            best=scores['macro']['nll']; report['best_epoch']=epoch+1
            budget.save(dict(model=head.state_dict(),config=config,timing_schema=TIMING_SCHEMA,
                feature_schema=FEATURE_SCHEMA,encoder_config=encoder_config,
                encoder_path=str(Path(args.encoder_ckpt).resolve()),encoder_sha256=report['encoder_sha256'],
                run_config=vars(args),epoch=epoch+1,train_histogram=hist,prior=prior),run/'count_best.pt')
        write_json(run/'report.json',report)
        print(f'[count] {metrics}',flush=True)
    del x,y,optimizer
    ck=torch.load(run/'count_best.pt',map_location='cpu',weights_only=True); head.load_state_dict(ck['model']); del ck
    report['tune']=evaluate(head,tune,prior,device)
    report['tune_wrong_audio']=evaluate(head,tune,prior,device,wrong_audio=True)
    val=extract(args.val_list,encoder,device,args.mode,budget)
    if set(r['song'] for r in val)&set(r['song'] for r in train):
        raise ValueError('train and validation songs overlap')
    report['validation']=evaluate(head,val,prior,device)
    report['validation_wrong_audio']=evaluate(head,val,prior,device,wrong_audio=True)
    report['val_sources']=[{k:v for k,v in r.items() if k not in ['x','y']} for r in val]
    report.update(complete=True,finished=time.time(),resources=budget.log_state())
    write_json(run/'report.json',report)
    print(json.dumps({k:report[k]['macro'] for k in ['tune','tune_wrong_audio','validation','validation_wrong_audio']},indent=2),flush=True)


if __name__=='__main__':
    main()

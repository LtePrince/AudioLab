"""Train acoustic onset/time refinement from explicitly covered annotations.

Default: frozen existing encoder, fresh event head. No chart-derived labels
can enter this training path. Pseudo labels require an explicit opt-in.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from src.audio_understanding import SCHEMA, FEATURE_SCHEMA, FRAME_SECONDS
from src.audio_understanding.annotations import read_manifest, event_targets
from src.audio_understanding.inference import load_audio, sha256, code_version
from src.data.timing import TIMING_SCHEMA, check_checkpoint_timing
from src.models.transcriber import TranscriberNet
from src.resource_budget import ResourceBudget, add_resource_args
from src.train_rhythm import trainable_backbone


def acoustic_loss(logits, target, valid, offset_valid):
    weights = valid.to(logits.dtype)
    onset_sum = (F.binary_cross_entropy_with_logits(logits[:, 0], target[:, 0], reduction='none')*weights).sum()
    offset_mask = (offset_valid & valid).to(logits.dtype)*target[:, 0]
    offset_sum = ((logits[:, 1].sigmoid()-target[:, 1]).square()*offset_mask).sum()
    n, positives = weights.sum(), offset_mask.sum()
    loss = onset_sum/n.clamp_min(1) + .5*offset_sum/positives.clamp_min(1)
    return loss, torch.stack([onset_sum.detach(), offset_sum.detach(), n.detach(), positives.detach()])


def audit_splits(train, val, allow_pseudo=False):
    permitted = {'reviewed_acoustic', 'pseudo_acoustic'} if allow_pseudo else {'reviewed_acoustic'}
    domains = {r['label_source'] for r in train+val}
    if not domains <= permitted or len(domains) != 1:
        raise ValueError('one acoustic label source per experiment; pseudo requires --allow-pseudo-labels')
    for key in ('id', 'source_group', 'audio'):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f'train/validation leakage by {key}')
    hashes = {r['audio']: sha256(r['audio']) for r in train+val}
    if {hashes[r['audio']] for r in train} & {hashes[r['audio']] for r in val}:
        raise ValueError('train/validation duplicate audio bytes')
    return hashes


class AcousticDataset(Dataset):
    """All reviewed windows, including the end; at most one song cached in RAM."""
    def __init__(self, records, chunk_frames=1024, max_seconds=600, budget=None):
        if chunk_frames < 2 or chunk_frames % 2:
            raise ValueError('chunk_frames must be positive and even')
        self.records, self.chunk_frames, self.max_seconds = records, chunk_frames, max_seconds
        self.budget = budget
        self.entries, self.cache, self.collisions = [], None, 0
        for index, r in enumerate(records):
            if r['duration_seconds'] > max_seconds:
                raise ValueError('annotation exceeds duration budget')
            frames = math.ceil(r['duration_seconds']/FRAME_SECONDS)
            _, valid, _, collisions = event_targets(r, frames)
            self.collisions += collisions
            self.entries.extend((index, start) for start in range(0, frames, chunk_frames)
                                if valid[start:start+chunk_frames].any())
        if not self.entries:
            raise ValueError('no fully reviewed audio frames')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        song, start = self.entries[index]
        if self.cache is None or self.cache[0] != song:
            record = self.records[song]
            mel, duration = load_audio(record['audio'], self.max_seconds, budget=self.budget)
            if abs(duration-record['duration_seconds']) > 1/22050 + 1e-6:
                raise ValueError('annotation duration differs from decoded audio')
            target, valid, offset_valid, _ = event_targets(record, mel.shape[-1])
            self.cache = (song, mel, target, valid, offset_valid)
        values = []
        for value in self.cache[1:]:
            part = value[..., start:start+self.chunk_frames]
            values.append(F.pad(part, (0, self.chunk_frames-part.shape[-1])))
        return tuple(values)


class SongWindowSampler(Sampler):
    """Shuffle songs and their windows, decoding each song once per epoch."""
    def __init__(self, dataset):
        self.groups = {}
        for index, (song, _) in enumerate(dataset.entries):
            self.groups.setdefault(song, []).append(index)
        self.size = len(dataset)

    def __len__(self):
        return self.size

    def __iter__(self):
        songs = list(self.groups)
        for song_index in torch.randperm(len(songs)).tolist():
            windows = self.groups[songs[song_index]]
            for index in torch.randperm(len(windows)).tolist():
                yield windows[index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', required=True)
    p.add_argument('--val-manifest', required=True)
    p.add_argument('--encoder-ckpt', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--chunk-frames', type=int, default=1024)
    p.add_argument('--max-seconds', type=float, default=600)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--encoder-lr', type=float, default=0.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--allow-pseudo-labels', action='store_true')
    add_resource_args(p)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0 \
            or not math.isfinite(args.encoder_lr) or args.encoder_lr < 0:
        p.error('positive epochs, batch size and head lr; finite nonnegative encoder lr required')
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        p.error('max-seconds must be finite and positive')
    random.seed(args.seed); torch.manual_seed(args.seed)
    run = Path(args.run_dir)
    if run.exists() and any(run.iterdir()):
        raise ValueError('use an empty run-dir; existing experiments are immutable')
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    budget = ResourceBudget(args, device, run)
    train, val = read_manifest(args.train_manifest), read_manifest(args.val_manifest)
    hashes = audit_splits(train, val, args.allow_pseudo_labels)
    datasets = [AcousticDataset(rows, args.chunk_frames, args.max_seconds, budget=budget) for rows in (train, val)]
    loaders = [DataLoader(ds, batch_size=args.batch_size,
                         sampler=SongWindowSampler(ds) if i == 0 else None, num_workers=0)
               for i, ds in enumerate(datasets)]
    ck = torch.load(args.encoder_ckpt, map_location='cpu', weights_only=True)
    check_checkpoint_timing(ck)
    config = {**ck['config'], 'rhythm_head': True}
    if config.get('n_mels', 128) != 128 or config.get('mel_per_frame', 2) != 2:
        raise ValueError('encoder must use standard audio features')
    model = TranscriberNet(**config)
    # Transfer only the backbone/lane head; never silently reuse chart onset supervision.
    state = {k: v for k, v in ck.get('ema', ck['model']).items() if not k.startswith('rhythm_head.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith('rhythm_head.') for k in missing):
        raise ValueError('incompatible encoder checkpoint')
    del ck, state
    model.to(device)
    backbone = trainable_backbone(model, args.encoder_lr)
    groups = [dict(params=list(model.rhythm_head.parameters()), lr=args.lr)]
    if backbone:
        groups.append(dict(params=backbone, lr=args.encoder_lr))
    optimizer = torch.optim.AdamW(groups)
    provenance = dict(train_manifest_sha256=sha256(args.train_manifest),
                      val_manifest_sha256=sha256(args.val_manifest), audio_sha256=hashes,
                      encoder_sha256=sha256(args.encoder_ckpt), seed=args.seed,
                      implementation=code_version(),
                      collisions=dict(train=datasets[0].collisions, validation=datasets[1].collisions),
                      run_config=vars(args))
    (run/'run.json').write_text(json.dumps(provenance, indent=2)+'\n')
    best = math.inf
    for epoch in range(args.epochs):
        scores = {}
        for training, loader in zip((True, False), loaders):
            model.train(training and bool(backbone))
            model.rhythm_head.train(training)
            sums = torch.zeros(4, device=device)
            for mel, target, valid, offset_valid in loader:
                budget.check()
                with torch.set_grad_enabled(training and bool(backbone)):
                    features = model.features(mel.to(device))
                with torch.set_grad_enabled(training):
                    loss, parts = acoustic_loss(model.rhythm_from_features(features),
                        target.to(device), valid.to(device), offset_valid.to(device))
                if not torch.isfinite(loss):
                    raise ValueError('non-finite acoustic loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                sums += parts
            bce, mse, frames, positives = sums.tolist()
            scores['train_loss' if training else 'val_loss'] = bce/max(frames, 1)+.5*mse/max(positives, 1)
        payload = dict(model=model.state_dict(), config=config, timing_schema=TIMING_SCHEMA,
                       audio_schema=SCHEMA, feature_schema=FEATURE_SCHEMA,
                       label_source=train[0]['label_source'], provenance=provenance, epoch=epoch+1)
        if scores['val_loss'] < best:
            best = scores['val_loss']
            budget.save({**payload, **scores}, run/'audio_best.pt')
        budget.save({**payload, 'optimizer': optimizer.state_dict(), **scores}, run/'audio_last.pt')
        row = dict(epoch=epoch+1, **scores, **budget.log_state())
        with (run/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

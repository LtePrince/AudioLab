"""Train a small global onset/offset head on a frozen audio encoder.

Targets are unique audio-aligned gameplay onsets at mel-frame resolution;
lanes, positions and note types are deliberately absent from this objective.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import PhigrosDataset
from src.data.gameplay_miner import load_jsonl
from src.data.timing import TIMING_SCHEMA, check_checkpoint_timing
from src.models.transcriber import TranscriberNet
from src.resource_budget import ResourceBudget, add_resource_args
from src.train_utils import MetricLogger, pad_or_trim


def onset_targets(times_ms, frame_ms, frames):
    """Same-time chord notes count once; distinct within-frame offsets average."""
    target = torch.zeros(2, frames)
    counts = torch.zeros(frames)
    for time in sorted(set(times_ms)):
        position = time / frame_ms
        if not 0 <= position < frames:
            continue
        frame = math.floor(position)
        counts[frame] += 1
        target[1, frame] += position - frame
    target[0] = (counts > 0).float()
    target[1] /= counts.clamp_min(1)
    return target, int((counts - 1).clamp_min(0).sum())


class RhythmDataset(Dataset):
    def __init__(self, listing, max_mel=8192):
        self.base = PhigrosDataset(listing, dict(frame_ms=512 / 22050 * 2000, max_frame=max_mel // 2),
                                   cache_dir='data/cache_mel', augment=False)
        self.max_mel = max_mel
        self.cache = {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        source, audio, _ = self.base._entries[index]
        if index not in self.cache:
            chart = json.loads(Path(source).read_text())
            bpm, offset = chart['judgeLineList'][0]['bpm'], chart.get('offset', 0)
            notes = load_jsonl(Path('data/gameplay') / (Path(source).parent.name + '.jsonl'),
                               expected_timing=(bpm, offset))
            duration_ms = sf.info(audio).duration * 1000
            frame_ms = 512 / 22050 * 1000
            times = [n.tick * 60 / (32 * bpm) * 1000 + offset * 1000 for n in notes]
            target, collisions = onset_targets([t for t in times if 0 <= t < duration_ms],
                                                frame_ms, self.max_mel)
            valid = torch.arange(self.max_mel) < min(math.ceil(duration_ms / frame_ms), self.max_mel)
            self.cache[index] = (target, valid, collisions)
        target, valid, collisions = self.cache[index]
        mel = pad_or_trim(self.base._load_mel(audio), self.max_mel)
        return mel, target, valid, collisions


def rhythm_loss(logits, target, valid):
    valid = valid.to(logits.dtype)
    onset = target[:, 0] * valid
    bce_sum = (F.binary_cross_entropy_with_logits(logits[:, 0], target[:, 0], reduction='none') * valid).sum()
    offset_sum = ((logits[:, 1].sigmoid() - target[:, 1]).square() * onset).sum()
    n, positives = valid.sum(), onset.sum()
    loss = bce_sum / n.clamp_min(1) + 0.5 * offset_sum / positives.clamp_min(1)
    return loss, (bce_sum.detach(), offset_sum.detach(), n.detach(), positives.detach())


@torch.no_grad()
def validate(model, loader, device, budget):
    model.eval()
    sums = torch.zeros(4, device=device)
    for mel, target, valid, _ in loader:
        budget.check()
        _, parts = rhythm_loss(model.rhythm_logits(mel.to(device)), target.to(device), valid.to(device))
        sums += torch.stack(parts)
    bce, mse, frames, positives = sums.tolist()
    return dict(val_loss=bce / max(frames, 1) + 0.5 * mse / max(positives, 1),
                val_bce=bce / max(frames, 1), val_offset_mse=mse / max(positives, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-list', default='data/train.txt')
    p.add_argument('--val-list', default='data/val.txt')
    p.add_argument('--encoder-ckpt', default='checkpoints/timing_v1/transcriber/transcriber_best.pt')
    p.add_argument('--run-dir', default='checkpoints/rhythm_global_v1')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    add_resource_args(p)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error('epochs and batch size must be positive')
    random.seed(args.seed); torch.manual_seed(args.seed)
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    if (run / 'rhythm_best.pt').exists() and not args.resume:
        raise ValueError('run already exists; choose a new run-dir or --resume')
    device = torch.device(args.device)
    budget = ResourceBudget(args, device, run)
    ck = torch.load(args.encoder_ckpt, map_location='cpu', weights_only=True)
    check_checkpoint_timing(ck)
    config = {**ck['config'], 'rhythm_head': True}
    model = TranscriberNet(**config)
    state = ck.get('ema', ck['model'])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith('rhythm_head.') for key in missing):
        raise ValueError(f'incompatible encoder: missing={missing}, unexpected={unexpected}')
    del ck, state
    model.to(device).requires_grad_(False).eval()
    model.rhythm_head.requires_grad_(True)
    train = RhythmDataset(args.data_list)
    val = RhythmDataset(args.val_list)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW(model.rhythm_head.parameters(), lr=args.lr, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    T_max=args.epochs * len(loader), eta_min=args.lr / 100)
    start, best = 0, float('inf')
    if args.resume:
        ck = torch.load(run / 'rhythm_last.pt', map_location='cpu', weights_only=True)
        for key in ['epochs', 'lr', 'batch_size', 'data_list', 'val_list', 'encoder_ckpt', 'seed']:
            if ck['run_config'][key] != vars(args)[key]:
                raise ValueError(f'resume must preserve {key}')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer']); schedule.load_state_dict(ck['schedule'])
        start, best = ck['epoch'], ck['best_val']
        if 'rng_state' in ck:
            torch.set_rng_state(ck['rng_state'])
        del ck
    logger = MetricLogger(run / 'metrics.jsonl', stage='rhythm')
    print(f'[rhythm] training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters; encoder frozen', flush=True)
    for epoch in range(start, args.epochs):
        model.eval(); model.rhythm_head.train()
        for mel, target, valid, _ in loader:
            budget.check()
            with torch.no_grad():
                features = model.features(mel.to(device))
            loss, _ = rhythm_loss(model.rhythm_from_features(features), target.to(device), valid.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rhythm_head.parameters(), 1)
            optimizer.step(); schedule.step()
        scores = validate(model, val_loader, device, budget)
        payload = dict(model=model.state_dict(), config=config, timing_schema=TIMING_SCHEMA,
                       rhythm_schema='global-mel-frame-v1', run_config=vars(args), epoch=epoch + 1)
        if scores['val_loss'] < best:
            best = scores['val_loss']
            budget.save({**payload, **scores}, run / 'rhythm_best.pt')
        budget.save({**payload, 'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
                     'best_val': best, 'rng_state': torch.get_rng_state()}, run / 'rhythm_last.pt')
        logger.log(epoch=epoch + 1, **scores, **budget.log_state())
        print(f'[rhythm] epoch {epoch+1}/{args.epochs} {scores}, best={best:.6f}', flush=True)
    budget.save(payload, run / 'rhythm_final.pt') if start < args.epochs else None
    print(f'[rhythm] done: {run / "rhythm_best.pt"}', flush=True)


if __name__ == '__main__':
    main()

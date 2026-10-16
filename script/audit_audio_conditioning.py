"""Measure acoustic conditioning with identical teacher-forced token history.

Compare matched audio, reversed acoustic features, a different song, and zero
acoustic features. Decoder time embeddings and validity masks stay unchanged.
These are sensitivity diagnostics, not a perceptual quality or Hold-count goal.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from script.train_models import write_json
from src.data.chart_tokenizer import ChartTokenizer, PAD
from src.resource_budget import ResourceBudget, add_resource_args
from src.test_decoder import load_models
from src.train_decoder import TokenChartDataset
from src.train_utils import pad_or_trim


def token_scores(logits, targets, tk):
    """Mean bits per supervised token, split by action and NOTE semantics."""
    logp = logits.float().log_softmax(-1)
    losses = -logp.gather(-1, targets[:, None]).squeeze(-1) / math.log(2)
    first, end = tk.note(0, 1), tk.note(0, 1) + tk.n_lanes * 4
    notes = (targets >= first) & (targets < end)
    dt = (targets >= tk.dtb(1)) & (targets <= tk.dtt(31))
    dur = (targets >= tk.durb(1)) & (targets <= tk.durt(31))
    scores = {}
    for name, mask in [('all', targets != PAD), ('time', dt), ('note', notes), ('duration', dur)]:
        n = int(mask.sum())
        scores[name] = dict(count=n, bits=float(losses[mask].mean()) if n else None)
    if notes.any():
        block = logits[notes, first:end].reshape(-1, tk.n_lanes, 4)
        payload = targets[notes] - first
        for name, conditional, truth in [
            ('note_type', block.logsumexp(1), payload % 4),
            ('position', block.logsumexp(2), payload // 4),
        ]:
            scores[name] = dict(count=int(notes.sum()),
                bits=float(F.cross_entropy(conditional, truth) / math.log(2)))
    return scores


def alter_features(features, valid_length, mode, other=None, other_length=None):
    """Modify valid acoustic content; keep padding and later time embeddings."""
    result = features.clone()
    if mode == 'reversed':
        result[:, :valid_length] = features[:, :valid_length].flip(1)
    elif mode == 'wrong_song':
        if other is None or not other_length:
            raise ValueError('wrong_song requires nonempty other audio features')
        result[:, :valid_length] = F.interpolate(other[:, :other_length].transpose(1, 2),
            size=valid_length, mode='linear', align_corners=False).transpose(1, 2)
    elif mode == 'no_audio':
        result.zero_()
    elif mode != 'matched':
        raise ValueError(mode)
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-list', default='data/train.txt')
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-root', default='checkpoints/timing_v1')
    p.add_argument('--models', nargs='+', choices=['decoder', 'decoder_freex'],
                   default=['decoder', 'decoder_freex'])
    p.add_argument('--output', default='out/audio_conditioning')
    p.add_argument('--device', default='cuda:0')
    add_resource_args(p)
    args = p.parse_args()
    if args.samples < 1:
        p.error('samples must be positive')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    budget = ResourceBudget(args, device, output)
    report = dict(config=vars(args), models={})
    for name in args.models:
        enc, dec = load_models(str(Path(args.checkpoint_root) / name / 'decoder_best.pt'), device)
        tk = ChartTokenizer(n_lanes=dec.n_lanes)
        ds = TokenChartDataset(args.data_list, 'data/cache_mel', frame_ms=512 / 22050 * 2000,
            max_frame=4096, hop_length=512, sr=22050, n_mels=128,
            gameplay_dir='data/gameplay' if name == 'decoder_freex' else None)
        if len(ds) < 2:
            raise ValueError('need at least two songs for a wrong-audio control')
        indices = sorted(random.Random(args.seed).sample(range(len(ds)), min(args.samples, len(ds))))
        records = report['models'][name] = []
        for index in indices:
            budget.check(force=True)
            item = ds[index]
            other_index = (index + len(ds) // 2) % len(ds)
            other = ds[other_index]
            mel = pad_or_trim(item['mel'], 8192).unsqueeze(0).to(device)
            other_mel = pad_or_trim(other['mel'], 8192).unsqueeze(0).to(device)
            features, other_features = enc.features(mel), enc.features(other_mel)
            length = max(1, min(item['mel_frames'] // 2, 4096))
            other_length = max(1, min(other['mel_frames'] // 2, 4096))
            valid = torch.arange(4096, device=device)[None] < length
            tokens = item['tokens'].to(device)
            frames = item['frames'].to(device)
            row = dict(song=Path(ds.base._entries[index][0]).parent.name,
                       other_song=Path(ds.base._entries[other_index][0]).parent.name, conditions={})
            for mode in ['matched', 'reversed', 'wrong_song', 'no_audio']:
                budget.check()
                acoustic = alter_features(features, length, mode, other_features, other_length)
                memory = dec.prepare_memory(acoustic)
                logits = dec(tokens[None, :-1], frames[None, :-1], memory, valid)[0]
                row['conditions'][mode] = token_scores(logits, tokens[1:], tk)
            records.append(row)
            print(json.dumps(dict(model=name, **row)), flush=True)
            report['resources'] = budget.log_state()
            write_json(output / 'report.json', report)
        del enc, dec, ds, features, other_features, memory, logits
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

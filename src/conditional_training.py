"""4K training-only counterfactual pairs and condition-ranking objective.

Inspired by multimodal conditional preference learning (mDPO/CoPO), not a
reproduction: retain supervised CE, rank mean DT-token log probabilities,
and use no frozen-reference or response-preference objective. Generation and
all weak-onset inputs remain unchanged.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from src.audio_understanding import SAMPLE_RATE
from src.audio_understanding.evaluation import match_events
from src.data.chart_tokenizer import BOS, EOS, ChartTokenizer, TokenNote
from src.models.chart_history import history_features
from src.soft_onset import MEMORY_SECONDS

PAIR_SCHEMA = 'audiolab-waveform-shift-pairs-v1'
LENGTHS = (6., 12., 24., 40.)


def choose_pair(notes, bpm, offset, duration, total_seconds, rng: random.Random):
    """Choose a complete content crop; reject cut Holds/near-equivalent shifts.

    Waveform boundaries and paddings are integer samples. Neither source
    features nor the eval probes are cropped/reused to form these inputs.
    """
    tick_seconds = 60 / (32 * bpm)
    content_samples = round((total_seconds - 3.) * SAMPLE_RATE)
    total_samples = round(total_seconds * SAMPLE_RATE)
    if content_samples <= 0 or duration * SAMPLE_RATE < content_samples:
        raise ValueError('source too short for this complete counterfactual crop')
    events = [(n, n.tick*tick_seconds+offset,
               (n.tick+n.dur)*tick_seconds+offset) for n in notes]
    for _ in range(256):
        start = rng.randrange(max(1, math.floor(duration*SAMPLE_RATE)-content_samples+1))
        lo, hi = start/SAMPLE_RATE, (start+content_samples)/SAMPLE_RATE
        # A Hold already in progress is not a fresh note. Avoid either boundary
        # intersecting one, rather than silently dropping/clipping its label.
        if any(n.type == 3 and (a < lo < b or a < hi < b) for n,a,b in events):
            continue
        visible = [(n,a,b) for n,a,b in events if lo <= a < hi]
        if not visible:
            continue
        shifts = sorted(rng.sample(range(round(.2*SAMPLE_RATE), round(2.6*SAMPLE_RATE)), 2))
        if (shifts[1]-shifts[0])/SAMPLE_RATE < .35:
            continue
        clocks = [sorted({a-lo+s/SAMPLE_RATE for _,a,_ in visible}) for s in shifts]
        overlap = len(match_events(clocks[0],clocks[1],.025))/min(map(len,clocks))
        if overlap >= .5:
            continue
        return dict(start_sample=start,content_samples=content_samples,total_samples=total_samples,
                    shift_samples=shifts,overlap_25ms=overlap,source_note_count=len(visible))
    raise ValueError('no complete non-equivalent crop found; source not silently skipped')


def shifted_waveform(waveform, plan, shift_samples):
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError('expected one waveform with explicit channel dimension')
    start, length, total = (plan[k] for k in ('start_sample','content_samples','total_samples'))
    if start < 0 or start+length > waveform.shape[-1] or not 0 <= shift_samples <= total-length:
        raise ValueError('crop/padding outside waveform or output bounds')
    output = waveform.new_zeros(*waveform.shape[:-1],total)
    output[...,shift_samples:shift_samples+length] = waveform[...,start:start+length]
    return output


def shifted_targets(notes, bpm, offset, plan, shift_samples, *, n_lanes=4, residuals=None):
    """Re-encode all four note types at shifted physical audio times."""
    scale = 60/(32*bpm)
    lo = plan['start_sample']/SAMPLE_RATE
    hi = (plan['start_sample']+plan['content_samples'])/SAMPLE_RATE
    shift = shift_samples/SAMPLE_RATE
    selected, shifted_residuals = [], {}
    if n_lanes==64 and residuals is None:
        raise ValueError('free-x translation requires original coordinate residuals')
    for n in notes:
        a, b = n.tick*scale+offset, (n.tick+n.dur)*scale+offset
        if n.type == 3 and (a < lo < b or a < hi < b):
            raise ValueError('Hold crosses crop boundary')
        if lo <= a < hi:
            tick = round((a-lo+shift)/scale)
            selected.append(TokenNote(tick,n.lane,n.type,n.dur))
            if n_lanes==64:shifted_residuals[tick,n.lane]=residuals[n.tick,n.lane]
    tk = ChartTokenizer(n_lanes)
    tokens = torch.tensor(tk.encode_notes(selected))
    decoded = tk.decode_tokens(tokens.tolist(),strict=True)
    if len(decoded) != len(selected):
        raise ValueError('time quantization collided; no training notes silently discarded')
    if [(n.lane,n.type,n.dur) for n in decoded] != [(n.lane,n.type,n.dur) for n in selected]:
        raise ValueError('pure translation changed a note type, position, or Hold duration')
    frames = torch.tensor(tk.ticks_per_position(tokens.tolist()),dtype=torch.float32)*scale/MEMORY_SECONDS
    if any((n.tick+n.dur)*scale > plan['total_samples']/SAMPLE_RATE for n in decoded):
        raise ValueError('quantized target outside waveform')
    offsets=torch.zeros(len(tokens));mask=torch.zeros(len(tokens),dtype=torch.bool)
    if n_lanes==64:
        for i,(token,tick) in enumerate(zip(tokens.tolist(),tk.ticks_per_position(tokens.tolist()))):
            if tk.kind(token)=='NOTE':
                offsets[i]=shifted_residuals[tick,tk.value(token)//4];mask[i]=True
    return dict(tokens=tokens,frames=frames,offsets=offsets,offset_mask=mask,
                reference=dict(timebase_bpm=bpm,offset_seconds=0.))


def empty_targets(bpm):
    return dict(tokens=torch.tensor([BOS,EOS]),frames=torch.zeros(2),offsets=torch.zeros(2),
                offset_mask=torch.zeros(2,dtype=torch.bool),
                reference=dict(timebase_bpm=bpm,offset_seconds=0.))


def training_logits(decoder, target, condition, device):
    """Same target/prefix/clock for both candidate audio conditions; no leakage."""
    tokens = target['tokens'].to(device)
    frames = target['frames'].to(device)
    memory = decoder.prepare_memory(condition['features'].float().to(device),
                                    condition['soft_onsets'].to(device))
    history = history_features(target['tokens'][:-1],target['frames'][:-1],n_lanes=4,
        frames_per_beat=60/target['reference']['timebase_bpm']/MEMORY_SECONDS)[None].to(device)
    return decoder(tokens[None,:-1],frames[None,:-1],memory,
                   condition['valid'].to(device),history=history)[0]


def time_log_probability(logits, tokens):
    """Rank DT decisions, never reward forcing a particular lane/type ratio.

    DT is scored BEFORE it is consumed; its target time is not supplied as an
    input. EOS and NOTE are left to CE, since stopping/layout have multiple
    valid alternatives. A missing DT is explicit, not an artificial zero gap.
    """
    tk = ChartTokenizer(4)
    targets = tokens[1:].to(logits.device)
    mask = torch.tensor([tk.kind(t) in ('DTB','DTT') for t in tokens[1:].tolist()],device=logits.device)
    if not mask.any():
        raise ValueError('counterfactual positive has no time decision')
    values = logits.float().log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)
    return values[mask].mean(), int(mask.sum())


def paired_loss(decoder, target, positive, negative, device, *, rank_weight=0., beta=2.):
    """CE sum plus differentiable, reference-free conditional ranking.

    Restore RNG inside fork_rng for the second pass: identical dropout masks
    for equal-size correct/wrong conditions, with no effect on later updates.
    The returned CE is a SUM; the caller owns joint token normalization.
    """
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError('finite positive beta and nonnegative rank weight required')
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    cpu_rng = torch.get_rng_state()
    gpu_rng = [torch.cuda.get_rng_state(d) for d in devices]
    correct = training_logits(decoder,target,positive,device)
    targets = target['tokens'][1:].to(device)
    ce = F.cross_entropy(correct,targets,reduction='sum')
    details = dict(supervised_tokens=len(targets),ce_sum=float(ce.detach()))
    rank = ce.new_zeros(())
    if rank_weight:
        if positive['features'].shape != negative['features'].shape or not torch.equal(positive['valid'],negative['valid']):
            raise ValueError('counterfactual conditions must have equal lengths and masks')
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            for d,state in zip(devices,gpu_rng):torch.cuda.set_rng_state(state,d)
            wrong = training_logits(decoder,target,negative,device)
        p, count = time_log_probability(correct,target['tokens'])
        q, _ = time_log_probability(wrong,target['tokens'])
        rank = F.softplus(-beta*(p-q))
        details.update(time_tokens=count,matched_time_nll=float(-p.detach()),
                       mismatched_time_nll=float(-q.detach()),gap=float((p-q).detach()),
                       rank_loss=float(rank.detach()))
    return ce, rank*rank_weight, details

"""Experimental 4K / static free-x AR with supervised absolute-time pointers.

Versioned group protocol, not the product DT tokenizer: BOS / END_GROUP predict
stop or a time anchor; ANCHOR then predicts NOTE/DUR members and END_GROUP.
All audio stays visible. Dense onset evidence is an optional learned input.
"""
from __future__ import annotations

from collections import Counter
from itertools import groupby
import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.data.chart_tokenizer import BOS, EOS, ChartTokenizer, GrammarState, bin_to_hit
from src.models.chart_decoder import ChartDecoder, nucleus_probs
from src.models.chart_history import CausalChartHistory, history_features
from src.pattern_diagnostics import KINDS, LANE_X
from src.soft_onset import MEMORY_SECONDS

SCHEMA = 'audiolab-absolute-time-group-ar-v1'
FREEX_SCHEMA = 'audiolab-absolute-time-group-freex-ar-v1'
# Existing unused slots; their new meaning is bound to SCHEMA, never inferred
# by the legacy model loader or product tokenizer.
ANCHOR, END_GROUP = 113, 114
MIN_PROGRESS_SECONDS = 1e-6  # Numerical strict ordering, not musical spacing.


def residuals_by_note(route):
    """Read residuals at NOTE positions in the original 64-bin DT stream."""
    tk = ChartTokenizer(64)
    tokens = route['tokens'].tolist()
    offsets, mask = route['offsets'], route['offset_mask']
    if len(offsets) != len(tokens) or len(mask) != len(tokens):
        raise ValueError('free-x residual arrays must match the original token stream')
    result = {}
    for i, (token, tick) in enumerate(zip(tokens, tk.ticks_per_position(tokens))):
        is_note = tk.kind(token) == 'NOTE'
        if bool(mask[i]) != is_note:
            raise ValueError('free-x residual mask must mark exactly the NOTE positions')
        if is_note:
            value = float(offsets[i])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('free-x residual must be finite and normalized to [0,1]')
            key = (tick, tk.value(token)//4)
            if key in result:
                raise ValueError('duplicate free-x time/bin target')
            result[key] = value
    return result


def encode_target(route, duration, *, n_lanes=4):
    tk = ChartTokenizer(n_lanes)
    anchor, end_group = tk.cond(0), tk.cond(1)
    residuals = residuals_by_note(route) if n_lanes == 64 else {}
    bpm = route['reference']['timebase_bpm']
    offset = route['reference']['offset_seconds']
    scale = 60 / (32*bpm)
    notes = tk.decode_tokens(route['tokens'].tolist(), strict=True)
    tokens, seconds = [BOS], [0.]
    offsets, offset_mask = [0.], [False]
    decisions, next_seconds, stops = [], [], []
    groups = []
    for tick, members in groupby(notes, key=lambda n:n.tick):
        members = list(members)
        onset = tick*scale+offset
        if not 0 <= onset < duration or any(onset+n.dur*scale > duration+1e-6 for n in members):
            raise ValueError('reference onset/Hold exceeds physical audio bounds')
        decisions.append(len(tokens)-1);next_seconds.append(onset);stops.append(0.)
        groups.append(dict(time_seconds=onset, notes=members))
        tokens.append(anchor);seconds.append(onset)
        offsets.append(0.);offset_mask.append(False)
        for n in members:
            span = [tk.note(n.lane,n.type)]
            if n.type == 3:tk._emit_span(span,n.dur,tk.durb,tk.durt,16,31)
            tokens.extend(span);seconds.extend([onset]*len(span))
            offsets.extend([residuals.get((n.tick,n.lane),0.)]+[0.]*(len(span)-1))
            offset_mask.extend([n_lanes==64]+[False]*(len(span)-1))
        tokens.append(end_group);seconds.append(onset)
        offsets.append(0.);offset_mask.append(False)
    decisions.append(len(tokens)-1);next_seconds.append(0.);stops.append(1.)
    frames = torch.tensor(seconds,dtype=torch.float32)/MEMORY_SECONDS
    labels = torch.full((len(tokens),),-100,dtype=torch.long)
    for i in range(len(tokens)-1):
        if tokens[i] not in (BOS,end_group):labels[i]=tokens[i+1]
    history = history_features(tokens,frames,n_lanes=n_lanes,frames_per_beat=60/bpm/MEMORY_SECONDS,
                               offsets=offsets if n_lanes==64 else None)
    lower = [seconds[i]+(MIN_PROGRESS_SECONDS if tokens[i]==end_group else 0.) for i in decisions]
    result = dict(tokens=torch.tensor(tokens),frames=frames,history=history,layout_targets=labels,
        decisions=torch.tensor(decisions),next_seconds=torch.tensor(next_seconds,dtype=torch.float64),
        stop_targets=torch.tensor(stops),lower_seconds=torch.tensor(lower,dtype=torch.float64),duration=float(duration),
        time_count=len(groups),layout_count=int((labels!=-100).sum()),stop_count=1,
        continue_count=len(groups),reference_times=[g['time_seconds'] for g in groups],bpm=bpm)
    if n_lanes==64:
        result.update(offsets=torch.tensor(offsets),offset_mask=torch.tensor(offset_mask),
                      coordinate_count=sum(offset_mask))
    return result


def time_intervals(lower, length, duration, valid):
    """Candidate intervals include the remainder of the current audio frame.

    Future audio remains visible to cross attention. This mask only defines
    valid chronological output locations, like nonnegative DT did previously.
    """
    edges=torch.arange(length,device=lower.device,dtype=lower.dtype)*MEMORY_SECONDS
    lo=torch.maximum(edges[None],lower[:,None])
    hi=torch.minimum(edges[None]+MEMORY_SECONDS,lower.new_tensor(duration)).expand_as(lo)
    allowed=valid.reshape(1,length)&(hi>lo)
    return lo,hi,allowed


class AnchoredAR(nn.Module):
    def __init__(self, decoder_config, *, pointer_dim=64):
        super().__init__()
        self.decoder=ChartDecoder(**decoder_config)
        if self.decoder.n_lanes not in (4,64) or self.decoder.history_dim!=28 or self.decoder.soft_onset_dim!=2 \
                or self.decoder.cross_window is not None or self.decoder.cross_time_scale is not None:
            raise ValueError('anchored experiment requires global 4K/free-x decoder with causal history')
        if self.decoder.n_lanes==64 and self.decoder.offset_head is None:
            raise ValueError('free-x requires a continuous coordinate residual head')
        dim=decoder_config['hidden_dim']
        self.time_query=nn.Linear(dim,pointer_dim)
        self.time_key=nn.Linear(dim,pointer_dim)
        self.stop_head=nn.Linear(dim,1)
        self.subframe_head=nn.Sequential(nn.Linear(2*dim,128),nn.GELU(),nn.Linear(128,1))

    def frame_logits(self, hidden, memory, allowed):
        q=self.time_query(hidden);k=self.time_key(memory[0])
        return (q@k.T/math.sqrt(k.shape[-1])).masked_fill(~allowed,float('-inf'))

    def fraction(self, hidden, selected_memory):
        return self.subframe_head(torch.cat([hidden,selected_memory],-1)).squeeze(-1).sigmoid()

    def components(self,target,condition,*,use_soft):
        device=next(self.parameters()).device
        soft=condition['soft_onsets'].to(device)
        if not use_soft:soft=torch.zeros_like(soft)
        memory=self.decoder.prepare_memory(condition['features'].float().to(device),soft)
        logits,hidden=self.decoder(target['tokens'][None].to(device),target['frames'][None].to(device),
            memory,condition['valid'].to(device),history=target['history'][None].to(device),return_hidden=True)
        h=hidden[0,target['decisions'].to(device)]
        labels=target['stop_targets'].to(device)
        stop=F.binary_cross_entropy_with_logits(self.stop_head(h).squeeze(-1),labels,reduction='none')
        zero=hidden.sum()*0
        layout=(F.cross_entropy(logits[0],target['layout_targets'].to(device),ignore_index=-100,reduction='sum')
                if target['layout_count'] else zero)
        positive=labels==0
        frame_ce,subframe=zero,zero
        accuracy=offset_error=0.
        if positive.any():
            q=h[positive];lower=target['lower_seconds'].to(device)[positive]
            truth=target['next_seconds'].to(device)[positive]
            lo,hi,allowed=time_intervals(lower,memory.shape[1],target['duration'],condition['valid'].to(device))
            bins=torch.floor(truth/MEMORY_SECONDS).long()
            rows=torch.arange(len(bins),device=device)
            if not allowed[rows,bins].all():raise ValueError('reference pointer is not a valid future interval')
            scores=self.frame_logits(q,memory,allowed)
            frame_ce=F.cross_entropy(scores,bins,reduction='sum')
            fraction=self.fraction(q,memory[0,bins])
            a,b=lo[rows,bins],hi[rows,bins]
            target_fraction=(truth-a)/(b-a)
            if ((target_fraction < -1e-4)|(target_fraction > 1.0001)).any():
                raise ValueError('subframe target outside selected interval')
            subframe=(fraction-target_fraction).square().sum()
            accuracy=float((scores.argmax(-1)==bins).sum())
            offset_error=float(((a+fraction*(b-a)-truth).abs()).sum().detach())
        result = dict(frame_ce=frame_ce,subframe_mse=subframe,layout_ce=layout,
                    stop_bce=stop[~positive].sum(),continue_bce=stop[positive].sum(),
                    false_stops=float(((self.stop_head(h).squeeze(-1)>=0)&positive).sum()),
                    correct_stops=float(((self.stop_head(h).squeeze(-1)>=0)&~positive).sum()),
                    frame_correct=accuracy,conditional_offset_absolute_seconds=offset_error)
        if self.decoder.n_lanes==64:
            mask=target['offset_mask'].to(device)
            predicted=self.decoder.predict_offset(hidden)[0,mask]
            error=predicted-target['offsets'].to(device)[mask]
            result.update(coordinate_mse=error.square().sum(),
                          coordinate_absolute_screen=float(error.abs().sum().detach())*.9/64)
        return result


def normalized_loss(components, totals):
    return (components['frame_ce']/max(1,totals['time_count'])+
            components['subframe_mse']/max(1,totals['time_count'])+
            .25*components['layout_ce']/max(1,totals['layout_count'])+
            .5*(components['stop_bce']+components['continue_bce'])/
            max(1,totals['stop_count']+totals['continue_count'])+
            components.get('coordinate_mse',0.)/max(1,totals.get('coordinate_count',0)))


@torch.no_grad()
def generate(model,condition,bpm,*,use_soft,greedy=False,seed=42,max_tokens=12000,check=lambda:None):
    """Free rollout; predicted anchors stay continuous through exported notes."""
    started=time.monotonic();device=next(model.parameters()).device
    decoder=model.decoder;tk=ChartTokenizer(decoder.n_lanes);duration=float(condition['duration'])
    anchor,end_group=tk.cond(0),tk.cond(1)
    soft=condition['soft_onsets'].to(device)
    if not use_soft:soft=torch.zeros_like(soft)
    memory=decoder.prepare_memory(condition['features'].float().to(device),soft)
    valid=condition['valid'].to(device);caches=decoder.init_cache(memory)
    rng=torch.Generator(device=device).manual_seed(seed)
    history=CausalChartHistory(tk.n_lanes,60/bpm/MEMORY_SECONDS)
    token,clock=BOS,0.;notes=[];onsets=[];group=None;pending=None
    scale=60/(32*bpm);trace=[]

    def select(logits):
        probabilities=nucleus_probs(logits.float().softmax(-1),.95)
        return int(probabilities.argmax() if greedy else torch.multinomial(probabilities,1,generator=rng))

    def finish(status,error=None):
        # Derive benchmark events from exported notes, including for an
        # explicitly failed partial rollout; never count an unused anchor.
        exported=[n for n in notes if n['screen_x'] is not None]
        actual=sorted({n['time_seconds'] for n in exported})
        result=dict(status=status,greedy=greedy,seed=seed,seconds=time.monotonic()-started,
            notes=exported,onsets_seconds=actual,note_count=len(exported),generated_onsets=len(actual),
            types=dict(Counter(n['type'] for n in exported)),eos_clock_seconds=clock if status=='completed' else None,
            last_onset_seconds=actual[-1] if actual else None,
            termination='learned_stop' if status=='completed' else 'failed',trace=trace)
        if error:result['error']=error
        if len(exported)!=len(notes):result['incomplete_coordinate_notes']=len(notes)-len(exported)
        return result

    for position in range(max_tokens):
        if position%32==0:check()
        frame=clock/MEMORY_SECONDS
        state=torch.tensor([history.features(frame)],device=device)
        logits,hidden=decoder.step(torch.tensor([token],device=device),torch.tensor([frame],device=device),
            position,caches,memory.shape[1],valid,history=state)
        residual=None
        if tk.n_lanes==64 and tk.kind(token)=='NOTE':
            residual=float(decoder.predict_offset(hidden[:,None])[0,0])
            notes[-1]['screen_x']=bin_to_hit(notes[-1]['position'],residual)
            notes[-1]['residual']=residual
        history.observe(token,frame,residual)
        trace.append([int(token),clock])
        if token in (BOS,end_group):
            stop=model.stop_head(hidden).squeeze()
            if select(torch.stack([torch.zeros_like(stop),stop]))==1:
                return finish('completed')
            lower=torch.tensor([clock+(MIN_PROGRESS_SECONDS if token==end_group else 0.)],device=device,dtype=torch.float64)
            lo,hi,allowed=time_intervals(lower,memory.shape[1],duration,valid)
            if not allowed.any():
                return finish('failed','continued with no future audio interval')
            scores=model.frame_logits(hidden,memory,allowed)[0]
            index=select(scores)
            fraction=float(model.fraction(hidden,memory[0,index:index+1])[0])
            a,b=float(lo[0,index]),float(hi[0,index])
            # Float32 sigmoid may saturate at exactly one. Keep the selected
            # interval half-open without moving to a different frame or grid.
            clock=min(a+fraction*(b-a),math.nextafter(b,a))
            if not (float(lower[0])<=clock<duration):raise RuntimeError('nonfinite/out-of-bounds predicted time')
            onsets.append(clock);token=anchor
            group=GrammarState(tk);group.step(BOS);pending=None
            continue
        allowed=torch.zeros(tk.vocab_size,dtype=torch.bool,device=device)
        for candidate in range(50,anchor):
            if not group.is_allowed(candidate):continue
            kind=tk.kind(candidate)
            if kind=='NOTE':
                is_hold=tk.value(candidate)%4==2
                allowed[candidate]=not is_hold or clock+scale<=duration
            elif kind in ('DURB','DURT'):
                extra=tk.value(candidate)*(32 if kind=='DURB' else 1)*scale
                allowed[candidate]=pending is not None and notes[pending]['end_seconds']+extra<=duration
        allowed[end_group]=bool(group.used_lanes) and not group.in_dur
        if not allowed.any():return finish('failed','no valid group continuation')
        token=select(logits[0].masked_fill(~allowed,float('-inf')))
        if token==end_group:continue
        group.step(token);kind=tk.kind(token)
        if kind=='NOTE':
            lane,index=divmod(tk.value(token),4)
            notes.append(dict(time_seconds=clock,end_seconds=clock,position=lane,
                              type=KINDS[index+1],screen_x=LANE_X[lane] if tk.n_lanes==4 else None))
            pending=len(notes)-1 if index==2 else None
        else:
            notes[pending]['end_seconds']+=tk.value(token)*(32 if kind=='DURB' else 1)*scale
    return finish('failed','anchored decoder exhausted token budget before learned stop')

"""Experimental 4K AR with supervised absolute-time pointers.

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

from src.data.chart_tokenizer import BOS, EOS, ChartTokenizer, GrammarState
from src.models.chart_decoder import ChartDecoder, nucleus_probs
from src.models.chart_history import CausalChartHistory, history_features
from src.pattern_diagnostics import KINDS, LANE_X
from src.soft_onset import MEMORY_SECONDS

SCHEMA = 'audiolab-absolute-time-group-ar-v1'
# Existing unused slots; their new meaning is bound to SCHEMA, never inferred
# by the legacy model loader or product tokenizer.
ANCHOR, END_GROUP = 113, 114
MIN_PROGRESS_SECONDS = 1e-6  # Numerical strict ordering, not musical spacing.


def encode_target(route, duration):
    tk = ChartTokenizer(4)
    bpm = route['reference']['timebase_bpm']
    offset = route['reference']['offset_seconds']
    scale = 60 / (32*bpm)
    notes = tk.decode_tokens(route['tokens'].tolist(), strict=True)
    tokens, seconds = [BOS], [0.]
    decisions, next_seconds, stops = [], [], []
    groups = []
    for tick, members in groupby(notes, key=lambda n:n.tick):
        members = list(members)
        onset = tick*scale+offset
        if not 0 <= onset < duration or any(onset+n.dur*scale > duration+1e-6 for n in members):
            raise ValueError('reference onset/Hold exceeds physical audio bounds')
        decisions.append(len(tokens)-1);next_seconds.append(onset);stops.append(0.)
        groups.append(dict(time_seconds=onset, notes=members))
        tokens.append(ANCHOR);seconds.append(onset)
        for n in members:
            span = [tk.note(n.lane,n.type)]
            if n.type == 3:tk._emit_span(span,n.dur,tk.durb,tk.durt,16,31)
            tokens.extend(span);seconds.extend([onset]*len(span))
        tokens.append(END_GROUP);seconds.append(onset)
    decisions.append(len(tokens)-1);next_seconds.append(0.);stops.append(1.)
    frames = torch.tensor(seconds,dtype=torch.float32)/MEMORY_SECONDS
    labels = torch.full((len(tokens),),-100,dtype=torch.long)
    for i in range(len(tokens)-1):
        if tokens[i] not in (BOS,END_GROUP):labels[i]=tokens[i+1]
    history = history_features(tokens,frames,n_lanes=4,frames_per_beat=60/bpm/MEMORY_SECONDS)
    lower = [seconds[i]+(MIN_PROGRESS_SECONDS if tokens[i]==END_GROUP else 0.) for i in decisions]
    return dict(tokens=torch.tensor(tokens),frames=frames,history=history,layout_targets=labels,
        decisions=torch.tensor(decisions),next_seconds=torch.tensor(next_seconds,dtype=torch.float64),
        stop_targets=torch.tensor(stops),lower_seconds=torch.tensor(lower,dtype=torch.float64),duration=float(duration),
        time_count=len(groups),layout_count=int((labels!=-100).sum()),stop_count=1,
        continue_count=len(groups),reference_times=[g['time_seconds'] for g in groups],bpm=bpm)


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
        if self.decoder.n_lanes!=4 or self.decoder.history_dim!=28 or self.decoder.soft_onset_dim!=2 \
                or self.decoder.cross_window is not None or self.decoder.cross_time_scale is not None:
            raise ValueError('anchored experiment requires global 4K decoder with causal history')
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
        return dict(frame_ce=frame_ce,subframe_mse=subframe,layout_ce=layout,
                    stop_bce=stop[~positive].sum(),continue_bce=stop[positive].sum(),
                    false_stops=float(((self.stop_head(h).squeeze(-1)>=0)&positive).sum()),
                    correct_stops=float(((self.stop_head(h).squeeze(-1)>=0)&~positive).sum()),
                    frame_correct=accuracy,conditional_offset_absolute_seconds=offset_error)


def normalized_loss(components, totals):
    return (components['frame_ce']/max(1,totals['time_count'])+
            components['subframe_mse']/max(1,totals['time_count'])+
            .25*components['layout_ce']/max(1,totals['layout_count'])+
            .5*(components['stop_bce']+components['continue_bce'])/
            max(1,totals['stop_count']+totals['continue_count']))


@torch.no_grad()
def generate(model,condition,bpm,*,use_soft,greedy=False,seed=42,max_tokens=12000,check=lambda:None):
    """Free rollout; predicted anchors stay continuous through exported notes."""
    started=time.monotonic();device=next(model.parameters()).device
    decoder=model.decoder;tk=ChartTokenizer(4);duration=float(condition['duration'])
    soft=condition['soft_onsets'].to(device)
    if not use_soft:soft=torch.zeros_like(soft)
    memory=decoder.prepare_memory(condition['features'].float().to(device),soft)
    valid=condition['valid'].to(device);caches=decoder.init_cache(memory)
    rng=torch.Generator(device=device).manual_seed(seed)
    history=CausalChartHistory(4,60/bpm/MEMORY_SECONDS)
    token,clock=BOS,0.;notes=[];onsets=[];group=None;pending=None
    scale=60/(32*bpm);trace=[]

    def select(logits):
        probabilities=nucleus_probs(logits.float().softmax(-1),.95)
        return int(probabilities.argmax() if greedy else torch.multinomial(probabilities,1,generator=rng))

    for position in range(max_tokens):
        if position%32==0:check()
        frame=clock/MEMORY_SECONDS
        state=torch.tensor([history.features(frame)],device=device)
        logits,hidden=decoder.step(torch.tensor([token],device=device),torch.tensor([frame],device=device),
            position,caches,memory.shape[1],valid,history=state)
        history.observe(token,frame)
        trace.append([int(token),clock])
        if token in (BOS,END_GROUP):
            stop=model.stop_head(hidden).squeeze()
            if select(torch.stack([torch.zeros_like(stop),stop]))==1:
                return dict(status='completed',greedy=greedy,seed=seed,seconds=time.monotonic()-started,
                    notes=notes,onsets_seconds=onsets,note_count=len(notes),generated_onsets=len(onsets),
                    types=dict(Counter(n['type'] for n in notes)),eos_clock_seconds=clock,
                    last_onset_seconds=onsets[-1] if onsets else None,termination='learned_stop',trace=trace)
            lower=torch.tensor([clock+(MIN_PROGRESS_SECONDS if token==END_GROUP else 0.)],device=device,dtype=torch.float64)
            lo,hi,allowed=time_intervals(lower,memory.shape[1],duration,valid)
            if not allowed.any():
                return dict(status='failed',error='continued with no future audio interval',trace=trace)
            scores=model.frame_logits(hidden,memory,allowed)[0]
            index=select(scores)
            fraction=float(model.fraction(hidden,memory[0,index:index+1])[0])
            clock=float(lo[0,index])+fraction*float(hi[0,index]-lo[0,index])
            if not (float(lower[0])<=clock<duration):raise RuntimeError('nonfinite/out-of-bounds predicted time')
            onsets.append(clock);token=ANCHOR
            group=GrammarState(tk);group.step(BOS);pending=None
            continue
        allowed=torch.zeros(tk.vocab_size,dtype=torch.bool,device=device)
        for candidate in range(50,113):
            if not group.is_allowed(candidate):continue
            kind=tk.kind(candidate)
            if kind=='NOTE':
                is_hold=tk.value(candidate)%4==2
                allowed[candidate]=not is_hold or clock+scale<=duration
            elif kind in ('DURB','DURT'):
                extra=tk.value(candidate)*(32 if kind=='DURB' else 1)*scale
                allowed[candidate]=pending is not None and notes[pending]['end_seconds']+extra<=duration+1e-7
        allowed[END_GROUP]=bool(group.used_lanes) and not group.in_dur
        if not allowed.any():return dict(status='failed',error='no valid group continuation',trace=trace)
        token=select(logits[0].masked_fill(~allowed,float('-inf')))
        if token==END_GROUP:continue
        group.step(token);kind=tk.kind(token)
        if kind=='NOTE':
            lane,index=divmod(tk.value(token),4)
            notes.append(dict(time_seconds=clock,end_seconds=clock,position=lane,
                              type=KINDS[index+1],screen_x=LANE_X[lane]))
            pending=len(notes)-1 if index==2 else None
        else:
            notes[pending]['end_seconds']+=tk.value(token)*(32 if kind=='DURB' else 1)*scale
    return dict(status='failed',error='anchored decoder exhausted token budget before learned stop',trace=trace)

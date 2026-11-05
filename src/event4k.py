"""Synchronous 4x5 event AR. Window bounds affect output support only."""
from __future__ import annotations

from collections import Counter
import math
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.chart_windows import chart_windows
from src.data.chart_tokenizer import ChartTokenizer
from src.joint_audio_ar import AcousticBackbone
from src.models.attention import rope1d
from src.models.chart_decoder import DecoderBlock, time_embed
from src.pattern_diagnostics import KINDS, LANE_X
from src.soft_onset import MEMORY_SECONDS
from src.windowed_ar import window_intervals

SCHEMA = 'audiolab-synchronous-event4k-v1'
COLUMNS = ('Tap', 'Drag', 'Flick', 'HoldStart', 'HoldEnd')
TYPE_COLUMN = {1: 0, 2: 1, 3: 3, 4: 2}
COLUMN_TYPE = {v: k for k, v in TYPE_COLUMN.items()}
COUNTS = ('event_count', 'continue_count', 'stop_count')
TIME_EPS = 1e-9  # Floating-point canonicalization, not musical quantization.


def source_notes(target):
    """Read only canonical labels from the shared Mel cache, never its history."""
    tk = ChartTokenizer(4)
    anchors = {int(i)+1: float(t) for i, t, stop in
               zip(target['decisions'], target['next_seconds'], target['stop_targets']) if not stop}
    scale = 60 / (32 * float(target['bpm']))
    notes = []; clock = None; pending = None
    for index, token in enumerate(target['tokens'].tolist()):
        kind = tk.kind(token)
        if token == tk.cond(0):
            clock = anchors[index]; pending = None
        elif kind == 'NOTE':
            if clock is None: raise ValueError('NOTE without physical time')
            lane, type_index = divmod(tk.value(token), 4)
            notes.append(dict(time_seconds=clock, end_seconds=clock,
                              position=lane, type=KINDS[type_index+1], screen_x=LANE_X[lane]))
            pending = len(notes)-1 if type_index == 2 else None
        elif kind in ('DURB', 'DURT'):
            if pending is None: raise ValueError('duration without Hold')
            notes[pending]['end_seconds'] += tk.value(token)*(32 if kind == 'DURB' else 1)*scale
        elif token == tk.cond(1):
            clock = None; pending = None
    return notes


def encode_events(notes, duration, bpm, *, width=10.):
    if not math.isfinite(bpm) or bpm <= 0: raise ValueError('positive finite BPM required')
    windows = chart_windows(duration, width)
    inverse = {v:k for k,v in KINDS.items()}; groups = {}
    def add(t, lane, column):
        key = round(t / TIME_EPS)
        event = groups.setdefault(key, [[0]*5 for _ in range(4)])
        if event[lane][column]: raise ValueError('duplicate same-time/lane/type cannot fit a binary event')
        event[lane][column] = 1
    for note in notes:
        start, end, lane = float(note['time_seconds']), float(note['end_seconds']), int(note['position'])
        kind = inverse[note['type']]
        if not 0 <= lane < 4 or not 0 <= start < duration or not math.isfinite(end):
            raise ValueError('invalid note position/time')
        add(start, lane, TYPE_COLUMN[kind])
        if kind == 3:
            if not start < end <= duration + TIME_EPS: raise ValueError('invalid Hold interval')
            add(min(end,duration), lane, 4)
    keys = sorted(groups)
    times = torch.tensor([min(k*TIME_EPS,duration) for k in keys], dtype=torch.float64)
    events = torch.tensor([groups[k] for k in keys],dtype=torch.float32) if keys else torch.empty(0,4,5)
    # Validate pairing and ensure conversion cannot silently lose overlapping Holds.
    reconstructed = events_to_notes(times.tolist(), events)
    if len(reconstructed) != len(notes): raise ValueError('event codec changed note count')
    prefix = []; lower = []; upper = []; next_times = []; indices = []; index = 0
    for window in windows:
        def decision(next_index):
            previous = float(times[index-1]) if index else -TIME_EPS
            prefix.append(index); lower.append(max(window.start, previous+TIME_EPS))
            upper.append(window.end); indices.append(next_index)
            next_times.append(float(times[next_index]) if next_index >= 0 else 0.)
        while index < len(times) and (float(times[index]) < window.end or
                                     window.index == len(windows)-1 and float(times[index]) == duration):
            decision(index); index += 1
        decision(-1)
    if index != len(times): raise ValueError('unassigned events')
    return dict(events=events, times=times, prefix=torch.tensor(prefix),
        lower=torch.tensor(lower,dtype=torch.float64), upper=torch.tensor(upper,dtype=torch.float64),
        next_times=torch.tensor(next_times,dtype=torch.float64), event_indices=torch.tensor(indices),
        event_count=len(times), continue_count=len(times), stop_count=len(windows),
        duration=float(duration), bpm=float(bpm), width=float(width), reference=notes)


def events_to_notes(times, events, *, allow_open=False):
    notes = []; active = {}; previous = -math.inf
    events = torch.as_tensor(events).cpu()
    if events.ndim!=3 or events.shape[1:]!=(4,5) or not ((events==0)|(events==1)).all():
        raise ValueError('nonempty binary 4x5 event required')
    for t, event in zip(times, events.tolist(), strict=True):
        t = float(t)
        if not math.isfinite(t) or t <= previous: raise ValueError('strictly increasing event times required')
        if not any(any(row) for row in event):
            raise ValueError('nonempty binary 4x5 event required')
        previous = t
        for lane in range(4):
            if event[lane][4]:
                if lane not in active: raise ValueError('HoldEnd without HoldStart')
                notes[active.pop(lane)]['end_seconds'] = t
            for column in range(4):
                if not event[lane][column]: continue
                if column == 3 and lane in active: raise ValueError('overlapping Holds on the same lane')
                notes.append(dict(time_seconds=t,end_seconds=t,position=lane,
                    type=KINDS[COLUMN_TYPE[column]],screen_x=LANE_X[lane]))
                if column == 3: active[lane] = len(notes)-1
    if active and not allow_open: raise ValueError('unclosed Hold')
    return (notes, active) if allow_open else notes


class EventDecoder(nn.Module):
    def __init__(self, *, hidden_dim=256, depth=6, num_heads=8, mem_dim=256,
                 time_dim=128, dropout=.1, mlp_ratio=4., bpm_condition=True, pointer_dim=64):
        super().__init__()
        self.hidden_dim, self.time_dim = hidden_dim, time_dim
        self.head_dim, self.rope_theta = hidden_dim//num_heads, 10000.
        self.bpm_condition = bpm_condition; self.gradient_checkpointing = True
        self.event_proj = nn.Linear(20,hidden_dim,bias=False)
        self.bos = nn.Parameter(torch.zeros(hidden_dim))
        self.time_proj = nn.Linear(time_dim,hidden_dim)
        self.mem_proj = nn.Linear(mem_dim,hidden_dim)
        self.mem_time = nn.Linear(time_dim,hidden_dim)
        self.mem_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([DecoderBlock(hidden_dim,num_heads,mlp_ratio,dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.time_query = nn.Linear(hidden_dim,pointer_dim)
        self.time_key = nn.Linear(hidden_dim,pointer_dim)
        self.end_head = nn.Linear(hidden_dim,1)
        self.fraction_head = nn.Sequential(nn.Linear(2*hidden_dim,128),nn.GELU(),nn.Linear(128,1))
        self.event_head = nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,20))
        # Construct optional module last, with isolated RNG: common tensors match the no-BPM arm.
        with torch.random.fork_rng(devices=[]):
            self.bpm_proj = nn.Linear(1,hidden_dim,bias=False) if bpm_condition else None

    def tempo(self, bpm, like):
        value = torch.as_tensor(bpm,device=like.device,dtype=like.dtype)
        if not torch.isfinite(value).all() or (value<=0).any(): raise ValueError('positive finite BPM required')
        return 0. if self.bpm_proj is None else self.bpm_proj(torch.log2(value/120.).reshape(1,1))

    def embed(self, events, times, bpm, *, bos=False):
        x = self.event_proj(events.flatten(-2)) + self.time_proj(time_embed(times/MEMORY_SECONDS,self.time_dim))
        x = x + self.tempo(bpm,x)
        if bos: x[:,0] = x[:,0] + self.bos
        return x

    def memory(self, features):
        frames = torch.arange(features.shape[1],device=features.device,dtype=torch.float32)[None]
        return self.mem_norm(self.mem_proj(features)+self.mem_time(time_embed(frames,self.time_dim)))

    def hidden(self, events, times, memory, bpm):
        x = self.embed(events,times,bpm,bos=True)
        freqs = rope1d(torch.arange(x.shape[1],device=x.device).float(),self.head_dim,self.rope_theta)
        for block in self.blocks:
            x = checkpoint(block,x,freqs,memory,None,use_reentrant=False) if self.gradient_checkpointing and self.training and torch.is_grad_enabled() else block(x,freqs,memory,None)
        return self.norm(x)

    def step(self, event, time_seconds, position, caches, bpm):
        x = self.embed(event[None,None],event.new_tensor([[time_seconds]],dtype=torch.float64),bpm,bos=position==0)
        freqs = rope1d(event.new_tensor([position]),self.head_dim,self.rope_theta)
        for block, cache in zip(self.blocks,caches,strict=True): x = block.step(x,freqs,cache,None)
        return self.norm(x)[0]

    def time_logits(self, h, memory, allowed):
        keys = self.time_key(memory[0])
        scores = (self.time_query(h)@keys.T/math.sqrt(keys.shape[-1])).masked_fill(~allowed,-torch.inf)
        # END competes with this window's valid event times. No independent stop sigmoid.
        return torch.cat([scores,self.end_head(h)],-1)

    def fractions(self,h,audio):
        return self.fraction_head(torch.cat([h,audio],-1)).sigmoid().squeeze(-1)

    def event_logits(self,h,audio,seconds):
        timing = self.time_proj(time_embed(seconds/MEMORY_SECONDS,self.time_dim))
        return self.event_head(torch.cat([h,audio,timing],-1)).reshape(-1,4,5)


class Event4K(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.acoustic = AcousticBackbone(encoder_config)
        self.chart = EventDecoder(**decoder_config)

    def condition(self,mel,duration):
        length = math.ceil(duration/MEMORY_SECONDS)
        if mel.ndim!=2 or mel.shape[0]!=128 or mel.shape[1]!=length*2:
            raise ValueError('complete even-padded physical-clock Mel required')
        return dict(features=self.acoustic.features(mel[None].to(next(self.parameters()).device)),duration=duration)

    def optimizer_groups(self, *, decoder_lr, encoder_lr):
        if min(decoder_lr,encoder_lr)<=0 or not all(p.requires_grad for p in self.parameters()):
            raise ValueError('all parameters must train')
        return [dict(name='decoder',params=list(self.chart.parameters()),lr=decoder_lr),
                dict(name='encoder',params=list(self.acoustic.parameters()),lr=encoder_lr)]

    def components(self,target,mel,duration):
        c=self.chart; memory=c.memory(self.condition(mel,duration)['features']);device=memory.device
        events=target['events'].to(device);times=target['times'].to(device)
        inputs=torch.cat([events.new_zeros(1,4,5),events])[None]
        clocks=torch.cat([times.new_zeros(1),times])[None]
        hidden=c.hidden(inputs,clocks,memory,target['bpm'])[0]
        h=hidden[target['prefix'].to(device)];indices=target['event_indices'].to(device);positive=indices>=0
        lower,upper=target['lower'].to(device),target['upper'].to(device)
        lo,hi,allowed=window_intervals(lower,upper,memory.shape[1],torch.ones(memory.shape[1],dtype=torch.bool,device=device))
        logits=c.time_logits(h,memory,allowed)
        labels=torch.full_like(indices,memory.shape[1]);truth=target['next_times'].to(device)[positive]
        edges=torch.arange(memory.shape[1]+1,device=device,dtype=torch.float64)*MEMORY_SECONDS
        bins=(torch.searchsorted(edges,truth.contiguous(),right=True)-1).clamp(max=memory.shape[1]-1)
        rows=positive.nonzero().flatten()
        if len(rows) and not allowed[rows,bins].all(): raise ValueError('event time outside output support')
        labels[positive]=bins
        loss=F.cross_entropy(logits,labels,reduction='none')
        # Keep every head connected on silent views; no fabricated positive events.
        zero=sum(p.sum()*0 for p in c.fraction_head.parameters())+sum(p.sum()*0 for p in c.event_head.parameters())
        event_loss=fraction_loss=zero;exact=bit_correct=frame_correct=offset_error=0.
        if len(rows):
            selected=memory[0,bins];q=h[positive]
            fractions=c.fractions(q,selected);a,b=lo[rows,bins],hi[rows,bins]
            target_fraction=(truth-a)/(b-a)
            if ((target_fraction < -1e-8)|(target_fraction>1+1e-8)).any():raise ValueError('invalid event fraction')
            fraction_loss=(fractions-target_fraction).square().sum()
            output=c.event_logits(q,selected,truth);reference=events[indices[positive]]
            event_loss=F.binary_cross_entropy_with_logits(output,reference,reduction='sum')/20
            exact=float(((output>=0)==reference.bool()).flatten(1).all(1).sum())
            bit_correct=float(((output>=0)==reference.bool()).sum())
            frame_correct=float((logits[positive].argmax(-1)==bins).sum())
            offset_error=float((a+fractions*(b-a)-truth).abs().sum().detach())
        return dict(time_ce=loss[positive].sum(),end_ce=loss[~positive].sum(),event_bce=event_loss,
            fraction_mse=fraction_loss,event_exact=exact,event_bits_correct=bit_correct,frame_correct=frame_correct,
            end_correct=float((logits[~positive].argmax(-1)==memory.shape[1]).sum()),
            false_ends=float((logits[positive].argmax(-1)==memory.shape[1]).sum()),offset_error=offset_error)


def normalized_loss(parts,counts):
    return ((parts['time_ce']+parts['fraction_mse']+parts['event_bce'])/max(1,counts['event_count'])
            +parts['end_ce']/max(1,counts['stop_count']))


def sample_event(logits,active,*,greedy,rng):
    """Exact conditional product of Bernoulli probabilities over legal nonempty events.

    No quota, no lane ordering in the network, and no fallback argmax NOTE.
    Hold pairing is grammar only; it is never an input embedding.
    """
    bits=((torch.arange(32,device=logits.device)[:,None]>>torch.arange(5,device=logits.device))&1).float()
    distributions=[]; log_distributions=[]
    for lane in range(4):
        valid=(bits[:,4]==0) if lane not in active else ((bits[:,3]==0)|(bits[:,4]==1))
        scores=(bits*logits[lane]).sum(-1).masked_fill(~valid,-torch.inf)
        distributions.append(scores.softmax(-1));log_distributions.append(scores.log_softmax(-1))
    if greedy:
        chosen=[int(p.argmax()) for p in distributions]
        if not any(chosen):
            scores=torch.stack([p-p[0] for p in log_distributions])
            scores[:,0]=-torch.inf
            flat=int(scores.argmax());chosen[flat//32]=flat%32
    else:
        chosen=[];nonempty=False
        for lane,probability in enumerate(distributions):
            p=probability.clone()
            if not nonempty:
                empty_tail=math.prod(float(q[0]) for q in distributions[lane+1:])
                p[0]*=1-empty_tail
            if not torch.isfinite(p).all() or p.sum()<=0: raise RuntimeError('degenerate event distribution')
            value=int(torch.multinomial(p,1,generator=rng));chosen.append(value);nonempty|=value!=0
    return bits[torch.tensor(chosen,device=logits.device)]


@torch.no_grad()
def generate(model,mel,duration,bpm,*,greedy=True,seed=42,width=10.,max_events=12000,check=lambda:None):
    model.eval();start=time.monotonic();c=model.chart;condition=model.condition(mel,duration)
    memory=c.memory(condition['features']);caches=[b.init_cache(memory) for b in c.blocks]
    rng=torch.Generator(device=memory.device).manual_seed(seed)
    events=[];times=[];notes=[];active={};records=[];position=0
    h=c.step(memory.new_zeros(4,5),0.,position,caches,bpm)
    error=None
    for window in chart_windows(duration,width):
        count=0
        while True:
            check()
            if len(events)>=max_events:error='event_budget_exhausted';break
            lower=max(window.start,times[-1]+TIME_EPS if times else window.start)
            lo,hi,allowed=window_intervals(memory.new_tensor([lower],dtype=torch.float64),
                memory.new_tensor([window.end],dtype=torch.float64),memory.shape[1],
                torch.ones(memory.shape[1],dtype=torch.bool,device=memory.device))
            if not allowed.any():
                records.append(dict(index=window.index,start=window.start,end=window.end,termination='forced_no_remaining_interval',event_count=count));break
            logits=c.time_logits(h,memory,allowed)[0]
            index=int(logits.argmax()) if greedy else int(torch.multinomial(logits.softmax(-1),1,generator=rng))
            if index==memory.shape[1]:
                records.append(dict(index=window.index,start=window.start,end=window.end,termination='learned_end_window',event_count=count));break
            a,b=float(lo[0,index]),float(hi[0,index])
            fraction=float(c.fractions(h,memory[0,index:index+1])[0])
            t=min(a+fraction*(b-a),math.nextafter(b,a))
            if not lower<=t<window.end:raise RuntimeError('event outside output window')
            event=sample_event(c.event_logits(h,memory[0,index:index+1],memory.new_tensor([t],dtype=torch.float64))[0],active,greedy=greedy,rng=rng)
            for lane in range(4):
                if event[lane,4]:notes[active.pop(lane)]['end_seconds']=t
                for column in range(4):
                    if event[lane,column]:
                        notes.append(dict(time_seconds=t,end_seconds=t,position=lane,type=KINDS[COLUMN_TYPE[column]],screen_x=LANE_X[lane]))
                        if column==3:active[lane]=len(notes)-1
            events.append(event.cpu().tolist());times.append(t);count+=1;position+=1
            h=c.step(event,t,position,caches,bpm)
        if error:break
    if active:error=error or 'unclosed_holds'
    # Keep incomplete notes visible for diagnostics, explicitly marking their invalid pairing.
    for lane,index in active.items():notes[index]['unclosed_hold']=True
    forced=sum(w['termination']!='learned_end_window' for w in records)
    completed=error is None and not forced and len(records)==len(chart_windows(duration,width))
    onsets=sorted({n['time_seconds'] for n in notes})
    return dict(status='completed' if completed else 'failed',error=error,notes=notes,onsets_seconds=onsets,
        note_count=len(notes),generated_onsets=len(onsets),types=dict(Counter(n['type'] for n in notes)),
        event_times_seconds=times,events=events,event_count=len(events),unclosed_holds=len(active),
        hold_end_events=sum(sum(row[4] for row in e) for e in events),windows=records,forced_windows=forced,
        learned_window_end_fraction=sum(w['termination']=='learned_end_window' for w in records)/len(chart_windows(duration,width)),
        learned_song_eos=False,termination='window_scheduler_completed' if len(records)==len(chart_windows(duration,width)) else error,
        eos_clock_seconds=None,last_onset_seconds=onsets[-1] if onsets else None,seconds=time.monotonic()-start)

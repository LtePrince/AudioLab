"""Trainable output-window protocol with unchanged full-song audio visibility.

END_WINDOW is a supervised decision. The next window is scheduled externally;
neither scheduler traversal nor a forced boundary means learned song EOS.
"""
from __future__ import annotations

from collections import Counter
import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.anchored_ar import AnchoredAR, encode_target, time_intervals, MIN_PROGRESS_SECONDS
from src.chart_windows import chart_windows, partition_notes
from src.data.chart_tokenizer import BOS, ChartTokenizer, GrammarState, bin_to_hit
from src.models.chart_decoder import nucleus_probs, time_embed
from src.models.chart_history import CausalChartHistory, history_features
from src.pattern_diagnostics import KINDS, LANE_X
from src.soft_onset import MEMORY_SECONDS

SCHEMA='audiolab-global-audio-window-output-ar-v1'


def encode_window_target(route,duration,*,n_lanes=4,width=10.):
    original=encode_target(route,duration,n_lanes=n_lanes)
    tk=ChartTokenizer(n_lanes);end_group,end_window=tk.cond(1),tk.cond(2)
    windows=chart_windows(duration,width)
    groups=partition_notes([dict(time=t,index=i) for i,t in enumerate(original['reference_times'])],duration,width)
    tokens=[BOS];seconds=[0.];bounds=[[windows[0].start,windows[0].end]]
    offsets=[0.];offset_mask=[False]
    decisions=[];truth=[];stops=[];lower=[]

    def decision(t,stop,window):
        decisions.append(len(tokens)-1);truth.append(t);stops.append(stop)
        lower.append(max(window.start,seconds[-1]+(MIN_PROGRESS_SECONDS if tokens[-1]==end_group else 0.)))

    for window,members in zip(windows,groups):
        if window.index:
            tokens.append(end_window);seconds.append(window.start)
            bounds.append([window.start,window.end]);offsets.append(0.);offset_mask.append(False)
        for member in members:
            index=member['index'];onset=member['time'];decision(onset,0.,window)
            start=int(original['decisions'][index])+1
            end=int(original['decisions'][index+1])+1
            span=original['tokens'][start:end].tolist()
            tokens.extend(span);seconds.extend([onset]*len(span))
            bounds.extend([[window.start,window.end] for _ in span])
            offsets.extend(original['offsets'][start:end].tolist() if n_lanes==64 else [0.]*len(span))
            offset_mask.extend(original['offset_mask'][start:end].tolist() if n_lanes==64 else [False]*len(span))
        decision(0.,1.,window)
    frames=torch.tensor(seconds,dtype=torch.float32)/MEMORY_SECONDS
    labels=torch.full((len(tokens),),-100,dtype=torch.long)
    for i in range(len(tokens)-1):
        if tokens[i] not in (BOS,end_group,end_window):labels[i]=tokens[i+1]
    history=history_features(tokens,frames,n_lanes=n_lanes,
        frames_per_beat=60/original['bpm']/MEMORY_SECONDS,offsets=offsets if n_lanes==64 else None)
    result=dict(tokens=torch.tensor(tokens),frames=frames,history=history,layout_targets=labels,
        decisions=torch.tensor(decisions),next_seconds=torch.tensor(truth,dtype=torch.float64),
        stop_targets=torch.tensor(stops),lower_seconds=torch.tensor(lower,dtype=torch.float64),
        bounds=torch.tensor(bounds,dtype=torch.float64),duration=float(duration),window_width=float(width),
        time_count=original['time_count'],layout_count=int((labels!=-100).sum()),stop_count=len(windows),
        continue_count=original['continue_count'],reference_times=original['reference_times'],bpm=original['bpm'],
        windows=[dict(index=w.index,start=w.start,end=w.end,reference_groups=len(g)) for w,g in zip(windows,groups)])
    if n_lanes==64:
        result.update(offsets=torch.tensor(offsets),offset_mask=torch.tensor(offset_mask),coordinate_count=sum(offset_mask))
    return result


def window_intervals(lower,upper,length,valid):
    edges=torch.arange(length,device=lower.device,dtype=lower.dtype)*MEMORY_SECONDS
    lo=torch.maximum(edges[None],lower[:,None])
    hi=torch.minimum(edges[None]+MEMORY_SECONDS,upper[:,None])
    return lo,hi,valid.reshape(1,length)&(hi>lo)


class WindowedAR(AnchoredAR):
    def __init__(self,decoder_config,*,pointer_dim=64,width=10.):
        super().__init__(decoder_config,pointer_dim=pointer_dim)
        if not math.isfinite(width) or width<=0:raise ValueError('positive window width required')
        self.window_width=float(width)
        self.bounds_proj=nn.Linear(2*self.decoder.time_dim,decoder_config['hidden_dim'],bias=False)
        nn.init.zeros_(self.bounds_proj.weight)

    def window_condition(self,bounds):
        encoded=time_embed(bounds/MEMORY_SECONDS,self.decoder.time_dim)
        return self.bounds_proj(encoded.flatten(-2))

    def components(self,target,condition,*,use_soft):
        device=next(self.parameters()).device
        soft=condition['soft_onsets'].to(device)
        if not use_soft:soft=torch.zeros_like(soft)
        memory=self.decoder.prepare_memory(condition['features'].float().to(device),soft)
        bounds=target['bounds'].to(device)
        logits,hidden=self.decoder(target['tokens'][None].to(device),target['frames'][None].to(device),
            memory,condition['valid'].to(device),history=target['history'][None].to(device),return_hidden=True,
            input_condition=self.window_condition(bounds)[None])
        decision_indices=target['decisions'].to(device)
        h=hidden[0,decision_indices];labels=target['stop_targets'].to(device)
        stop_logits=self.stop_head(h).squeeze(-1)
        stop=F.binary_cross_entropy_with_logits(stop_logits,labels,reduction='none')
        zero=hidden.sum()*0
        layout=F.cross_entropy(logits[0],target['layout_targets'].to(device),ignore_index=-100,reduction='sum') \
            if target['layout_count'] else zero
        positive=labels==0;frame_ce=subframe=zero;accuracy=offset_error=0.
        if positive.any():
            q=h[positive];lower=target['lower_seconds'].to(device)[positive]
            truth=target['next_seconds'].to(device)[positive];upper=bounds[decision_indices[positive],1]
            lo,hi,allowed=window_intervals(lower,upper,memory.shape[1],condition['valid'].to(device))
            bins=torch.floor(truth/MEMORY_SECONDS).long();rows=torch.arange(len(bins),device=device)
            if not allowed[rows,bins].all():raise ValueError('reference pointer outside its output window')
            scores=self.frame_logits(q,memory,allowed);frame_ce=F.cross_entropy(scores,bins,reduction='sum')
            fraction=self.fraction(q,memory[0,bins]);a,b=lo[rows,bins],hi[rows,bins]
            target_fraction=(truth-a)/(b-a)
            if ((target_fraction<0)|(target_fraction>=1+1e-9)).any():raise ValueError('invalid subframe fraction')
            subframe=(fraction-target_fraction).square().sum()
            accuracy=float((scores.argmax(-1)==bins).sum())
            offset_error=float((a+fraction*(b-a)-truth).abs().sum().detach())
        result=dict(frame_ce=frame_ce,subframe_mse=subframe,layout_ce=layout,
            stop_bce=stop[~positive].sum(),continue_bce=stop[positive].sum(),
            false_stops=float(((stop_logits>=0)&positive).sum()),correct_stops=float(((stop_logits>=0)&~positive).sum()),
            frame_correct=accuracy,conditional_offset_absolute_seconds=offset_error)
        if self.decoder.n_lanes==64:
            mask=target['offset_mask'].to(device);predicted=self.decoder.predict_offset(hidden)[0,mask]
            error=predicted-target['offsets'].to(device)[mask]
            result.update(coordinate_mse=error.square().sum(),coordinate_absolute_screen=float(error.abs().sum().detach())*.9/64)
        return result


@torch.no_grad()
def generate(model,condition,bpm,*,use_soft,greedy=False,seed=42,max_tokens=12000,check=lambda:None):
    started=time.monotonic();device=next(model.parameters()).device
    decoder=model.decoder;tk=ChartTokenizer(decoder.n_lanes);duration=float(condition['duration'])
    anchor,end_group,end_window=tk.cond(0),tk.cond(1),tk.cond(2)
    windows=chart_windows(duration,model.window_width);window_index=0;window_records=[]
    soft=condition['soft_onsets'].to(device)
    if not use_soft:soft=torch.zeros_like(soft)
    memory=decoder.prepare_memory(condition['features'].float().to(device),soft)
    valid=condition['valid'].to(device);caches=decoder.init_cache(memory)
    rng=torch.Generator(device=device).manual_seed(seed)
    history=CausalChartHistory(tk.n_lanes,60/bpm/MEMORY_SECONDS)
    token,clock=BOS,0.;notes=[];group=None;pending=None;trace=[]
    scale=60/(32*bpm)

    def select(logits):
        probabilities=nucleus_probs(logits.float().softmax(-1),.95)
        return int(probabilities.argmax() if greedy else torch.multinomial(probabilities,1,generator=rng))

    def finish(error=None):
        exported=[n for n in notes if n['screen_x'] is not None]
        actual=sorted({n['time_seconds'] for n in exported})
        forced=sum(w['termination']!='learned_end_window' for w in window_records)
        completed=len(window_records)==len(windows) and not forced and error is None
        result=dict(status='completed' if completed else 'failed',greedy=greedy,seed=seed,
            seconds=time.monotonic()-started,notes=exported,onsets_seconds=actual,note_count=len(exported),
            generated_onsets=len(actual),types=dict(Counter(n['type'] for n in exported)),
            eos_clock_seconds=None,last_onset_seconds=actual[-1] if actual else None,
            termination='window_scheduler_completed' if len(window_records)==len(windows) else 'token_or_grammar_failure',
            learned_song_eos=False,windows=window_records,total_windows=len(windows),forced_windows=forced,
            scheduler_traversal_fraction=len(window_records)/len(windows),
            learned_window_end_fraction=sum(w['termination']=='learned_end_window' for w in window_records)/len(windows),
            trace=trace)
        if error or forced:result['error']=error or 'one or more windows required a forced boundary'
        if len(exported)!=len(notes):result['incomplete_coordinate_notes']=len(notes)-len(exported)
        return result

    for position in range(max_tokens):
        if position%32==0:check()
        window=windows[window_index];frame=clock/MEMORY_SECONDS
        state=torch.tensor([history.features(frame)],device=device)
        boundary=torch.tensor([[window.start,window.end]],device=device,dtype=torch.float64)
        logits,hidden=decoder.step(torch.tensor([token],device=device),torch.tensor([frame],device=device),
            position,caches,memory.shape[1],valid,history=state,input_condition=model.window_condition(boundary))
        residual=None
        if tk.n_lanes==64 and tk.kind(token)=='NOTE':
            residual=float(decoder.predict_offset(hidden[:,None])[0,0])
            notes[-1]['screen_x']=bin_to_hit(notes[-1]['position'],residual);notes[-1]['residual']=residual
        history.observe(token,frame,residual);trace.append([int(token),clock,window_index])
        if token in (BOS,end_group,end_window):
            stop=model.stop_head(hidden).squeeze()
            ended=select(torch.stack([torch.zeros_like(stop),stop]))==1
            lower=torch.tensor([max(window.start,clock+(MIN_PROGRESS_SECONDS if token==end_group else 0.))],device=device,dtype=torch.float64)
            upper=lower.new_tensor([window.end])
            lo,hi,allowed=window_intervals(lower,upper,memory.shape[1],valid)
            if ended or not allowed.any():
                window_records.append(dict(index=window_index,start=window.start,end=window.end,
                    decision_clock=clock,termination='learned_end_window' if ended else 'forced_no_remaining_interval',
                    note_count=sum(window.contains(n['time_seconds']) for n in notes)))
                window_index+=1
                if window_index==len(windows):return finish()
                clock=windows[window_index].start;token=end_window
                continue
            index=select(model.frame_logits(hidden,memory,allowed)[0])
            fraction=float(model.fraction(hidden,memory[0,index:index+1])[0]);a,b=float(lo[0,index]),float(hi[0,index])
            clock=min(a+fraction*(b-a),math.nextafter(b,a))
            if not float(lower[0])<=clock<window.end:raise RuntimeError('predicted onset outside current window')
            token=anchor;group=GrammarState(tk);group.step(BOS);pending=None
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
        if not allowed.any():return finish('no valid group continuation')
        token=select(logits[0].masked_fill(~allowed,float('-inf')))
        if token==end_group:continue
        group.step(token);kind=tk.kind(token)
        if kind=='NOTE':
            lane,index=divmod(tk.value(token),4)
            notes.append(dict(time_seconds=clock,end_seconds=clock,position=lane,type=KINDS[index+1],
                screen_x=LANE_X[lane] if tk.n_lanes==4 else None,window_index=window_index))
            pending=len(notes)-1 if index==2 else None
        else:notes[pending]['end_seconds']+=tk.value(token)*(32 if kind=='DURB' else 1)*scale
    return finish('windowed decoder exhausted token budget')

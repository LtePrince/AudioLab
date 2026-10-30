"""Inspect saved E20 gaps with exact rollout replay and bounded context probes.

No model/decoder changes, training, thresholding, grid snapping or export repair.
Short-history and cropped-audio probes are diagnostics, not benchmark results.
"""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import hashlib
import json
import math
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/anchored_ar.py').is_file())
sys.path.insert(0, str(ROOT))
import torch
from script.train_grounded_ar import encode_waveform, load_inputs
from src.anchored_ar import AnchoredAR, generate, time_intervals, MIN_PROGRESS_SECONDS
from src.audio_understanding import SAMPLE_RATE
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart_tokenizer import BOS, ChartTokenizer
from src.product.anchored_models import export_prediction
from src.resource_budget import ResourceBudget
from src.soft_onset import MEMORY_SECONDS

SOURCE = ROOT/'docs/evaluations/assets/phira_generalization10_20261029'
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def top_gaps(prediction, reference):
    times=prediction['onsets_seconds']
    gaps=[]
    for a,b in zip(times,times[1:]):
        if b-a<2:continue
        held_until=max(n['end_seconds'] for n in prediction['notes'] if n['time_seconds']<=a)
        count=sum(a<n['time_seconds']<b for n in reference)
        gaps.append(dict(start=a,end=b,seconds=b-a,reference_notes=count,
                         no_active_note_seconds=max(0,b-max(a,held_until))))
    return sorted(gaps,key=lambda g:g['seconds'],reverse=True)


def distribution(scores, current, allowed):
    scores=scores.detach().float().reshape(-1).cpu()
    permitted=allowed.detach().reshape(-1).cpu()
    p=scores.softmax(-1)
    t=torch.arange(len(p),dtype=torch.float64)*MEMORY_SECONDS
    local=permitted&(t<current+2)
    local_scores=scores.masked_fill(~local,-torch.inf)
    best_local=int(local_scores.argmax()) if local.any() else None
    values,indices=p.topk(min(5,int(permitted.sum())))
    return dict(top=[dict(frame=int(i),frame_start_seconds=float(t[i]),probability=float(v))
                     for v,i in zip(values,indices)],
        local_2s_mass=float(p[local].sum()),
        local_2s_best=None if best_local is None else dict(frame=best_local,
            frame_start_seconds=float(t[best_local]),probability=float(p[best_local]),
            winner_logit_margin=float(scores.max()-scores[best_local])),
        entropy_nats=float(-(p[p>0]*p[p>0].log()).sum()),
        allowed_frames=int(permitted.sum()),first_allowed_seconds=float(t[permitted][0]),
        last_allowed_seconds=float(t[permitted][-1]))


def audio_evidence(wave, view, gap):
    a,b=gap['start'],gap['end']
    segment=wave[...,math.ceil(a*SAMPLE_RATE):math.floor(b*SAMPLE_RATE)].float().reshape(-1)
    blocks=segment[:len(segment)//2205*2205].reshape(-1,2205)
    db=20*blocks.square().mean(-1).sqrt().clamp_min(1e-12).log10()
    first,last=math.ceil(a/MEMORY_SECONDS),math.floor(b/MEMORY_SECONDS)
    features=view['features'][0,first:last].float()
    soft=view['soft_onsets'][0,first:last,0]
    norm=features.norm(dim=-1)
    return dict(audio_rms_dbfs=float(20*segment.square().mean().sqrt().clamp_min(1e-12).log10()),
        blocks_100ms=len(db),fraction_blocks_below_minus60dbfs=float((db < -60).float().mean()),
        block_rms_dbfs_min=float(db.min()),block_rms_dbfs_median=float(db.median()),
        feature_frames=len(features),all_valid=bool(view['valid'][0,first:last].all()),
        all_finite=bool(torch.isfinite(features).all()),zero_feature_frames=int((norm==0).sum()),
        feature_l2_mean=float(norm.mean()),feature_l2_min=float(norm.min()),
        soft_probability_mean=float(soft.mean()),soft_probability_max=float(soft.max()),
        soft_frames_above_half=int((soft>.5).sum()))


@torch.inference_mode()
def inspect(model, view, bpm, use_soft, saved, budget):
    """Read-only method wrappers; replay must exactly match saved raw notes."""
    dec=model.decoder;tk=ChartTokenizer(dec.n_lanes)
    end_group=tk.cond(1)
    decision_clocks=[0.]+saved['onsets_seconds']
    gaps=top_gaps(saved,[])
    critical=max(gaps,key=lambda g:g['seconds'])
    recent=deque(maxlen=64);decisions=[];captured={};last_stop={}
    original_step,original_logits=dec.step,model.frame_logits

    def step(token,frame,pos,caches,T_mem,mem_valid,history=None):
        recent.append((token.detach().clone(),frame.detach().clone(),pos,history.detach().clone()))
        return original_step(token,frame,pos,caches,T_mem,mem_valid,history=history)

    def stop_hook(module,inputs,output):
        last_stop['probability']=float(output.sigmoid().reshape(-1)[0])

    def frame_logits(hidden,memory,allowed):
        scores=original_logits(hidden,memory,allowed)
        index=len(decisions);current=decision_clocks[index]
        chosen=int(scores.argmax(-1)[0])
        row=dict(index=index,current_seconds=current,selected_frame=chosen,
                 stop_probability=last_stop['probability'])
        if chosen*MEMORY_SECONDS-current>=2:
            row.update(distribution(scores,current,allowed))
        decisions.append(row)
        if abs(current-critical['start'])<1e-9:
            captured.update(memory=memory,allowed=allowed.clone(),hidden=hidden.clone(),suffix=list(recent),
                            original=row)
        return scores

    handle=model.stop_head.register_forward_hook(stop_hook)
    dec.step=step;model.frame_logits=frame_logits
    try:
        replay=generate(model,view,bpm,use_soft=use_soft,greedy=True,seed=42,max_tokens=12000,check=budget.check)
    finally:
        dec.step=original_step;model.frame_logits=original_logits;handle.remove()
    assert replay['notes']==saved['notes'], 'instrumented rollout changed raw notes'
    assert replay['status']==saved['status'] and replay.get('error')==saved.get('error')
    trace=replay['trace']
    anchors=[(t,trace[i+1][1]) for i,(token,t) in enumerate(trace[:-1]) if token in (BOS,end_group)]
    assert len(anchors)==len(decisions)
    for row,(a,b) in zip(decisions,anchors):
        assert a==row['current_seconds']
        row.update(next_seconds=b,jump_seconds=b-a)
    assert captured, 'largest gap decision missing'
    # Reset only the self-attention cache, replaying the last 64 identical
    # token/time/history-summary inputs with unchanged absolute positions and
    # unchanged global audio. This is a counterfactual, not a quality fix.
    memory=captured['memory'];cache=dec.init_cache(memory)
    valid=view['valid'].to(memory.device)
    for token,frame,pos,history in captured['suffix']:
        _,hidden=dec.step(token,frame,pos,cache,memory.shape[1],valid,history=history)
    stop=float(model.stop_head(hidden).sigmoid().reshape(-1)[0])
    scores=original_logits(hidden,memory,captured['allowed'])
    probe=distribution(scores,critical['start'],captured['allowed'])
    probe.update(stop_probability=stop,retained_tokens=len(captured['suffix']),
                 actual_action='stop' if stop>=.5 else 'continue',
                 interpretation='self-attention prefix truncated; dense causal history summaries, clock and whole audio unchanged')
    selected=int(scores.argmax(-1)[0]);frac=float(model.fraction(hidden,memory[0,selected:selected+1])[0])
    lower=hidden.new_tensor([critical['start']+MIN_PROGRESS_SECONDS],dtype=torch.float64)
    lo,hi,_=time_intervals(lower,len(valid[0]),view['duration'],valid)
    probe['pointer_next_seconds']=min(float(lo[0,selected])+frac*float(hi[0,selected]-lo[0,selected]),
                                      math.nextafter(float(hi[0,selected]),float(lo[0,selected])))
    return dict(exact_notes_and_status_reproduced=True,token_steps=len(trace),pointer_decisions=len(decisions),
        jumps=[r for r in decisions if r['jump_seconds']>=2],
        largest_gap_original=captured['original'],short_self_history_probe=probe,
        final_stop_probability=last_stop['probability'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['4k','freex'],required=True)
    parser.add_argument('--device',required=True);args=parser.parse_args()
    torch.set_num_threads(2);device=torch.device(args.device);torch.cuda.set_device(device)
    plan=read(SOURCE/'selection.json')
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for item in plan['frozen_frontends'].values():assert sha(item['path'])==item['sha256']
    options=SimpleNamespace(mode=args.mode,model_root=ROOT/'checkpoints/timing_v1',min_free_disk_gib=8,
        min_available_ram_gib=8,max_rss_gib=4,gpu_memory_fraction=.65,keep_checkpoints=1)
    budget=ResourceBudget(options,device,OUT);encoder,onset=load_inputs(options,device)
    proc=AudioGPUprocessor(device='cpu');results=[]
    with torch.inference_mode():
        for song in [s for s in plan['songs'] if s['id'] in (52417,40717)]:
            assert sha(ROOT/song['audio'])==song['audio_sha256']
            assert sha(ROOT/song['reference'])==song['reference_sha256']
            reference=read(ROOT/song['reference'])['notes']
            wave=proc.load_from_path(ROOT/song['audio'])
            view=encode_waveform(wave,encoder,onset,budget,device)
            for candidate in [c for c in plan['candidates'] if c['mode']==args.mode]:
                assert sha(ROOT/candidate['checkpoint'])==candidate['sha256']
                ck=torch.load(ROOT/candidate['checkpoint'],map_location='cpu',weights_only=True)
                model=AnchoredAR(ck['config']['decoder_config']).to(device).eval()
                model.load_state_dict(ck['model'],strict=True);del ck
                source=ROOT/'out/phira_generalization10_20261029'/str(song['id'])/candidate['id']/'result.json'
                saved=read(source);prediction=saved['prediction']
                gaps=top_gaps(prediction,reference)
                audit=inspect(model,view,song['clock']['bpm'],candidate['variant']=='soft',prediction,budget)
                evidence=audio_evidence(wave,view,gaps[0])
                exported,details=export_prediction(prediction,args.mode,song['clock']['bpm'],0.,view['duration'])
                assert len(exported)==len(prediction['notes'])
                a=gaps[0]['start']+1;start=round(a*SAMPLE_RATE);end=start+12*SAMPLE_RATE
                assert end<=math.floor(gaps[0]['end']*SAMPLE_RATE)
                cropped=encode_waveform(wave[...,start:end],encoder,onset,budget,device)
                crop_pred=generate(model,cropped,song['clock']['bpm'],use_soft=candidate['variant']=='soft',
                                   greedy=True,seed=42,max_tokens=12000,check=budget.check)
                crop=dict(start_seconds=start/SAMPLE_RATE,duration_seconds=cropped['duration'],
                    status=crop_pred['status'],error=crop_pred.get('error'),note_count=crop_pred['note_count'],
                    onsets_seconds=crop_pred['onsets_seconds'],types=crop_pred['types'],
                    interpretation='same checkpoint; missing audio segment re-encoded as independent 12s input; resets chart history and audio context/time origin; not an isolated history intervention')
                row=dict(song_id=song['id'],song=song['name'],run=candidate['id'],epoch=candidate['epoch'],
                    checkpoint_sha256=candidate['sha256'],source_result=str(source.relative_to(ROOT)),
                    source_result_sha256=sha(source),largest_gaps=gaps[:3],audio_and_features=evidence,
                    export_note_count=len(exported),raw_note_count=prediction['note_count'],
                    cutoff_changes=details.get('cutoff_changes'),diagnostic=audit,cropped_audio_probe=crop)
                results.append(row)
                write(OUT/f'{args.mode}.json',dict(schema='audiolab-gap-diagnosis-v1',experiment='E20',
                    selection_sha256=sha(SOURCE/'selection.json'),script_sha256=sha(__file__),
                    training_updates=0,results=results,resources=budget.log_state()))
                print(json.dumps(dict(song=song['name'],run=candidate['id'],gap=gaps[0],
                    exact_replay=True,pointer=audit['largest_gap_original'],audio=evidence,
                    short_history_next=audit['short_self_history_probe']['pointer_next_seconds'],
                    crop_notes=crop['note_count'],crop_status=crop['status']),ensure_ascii=False),flush=True)
                del model,cropped,crop_pred;gc.collect();torch.cuda.empty_cache()
            del wave,view;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()

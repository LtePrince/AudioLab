"""Replay unchanged controlled results and inspect true-history predictions."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/windowed_ar.py').exists())
sys.path.insert(0,str(ROOT))

from collections import Counter
import json
import statistics

import torch

from src.anchored_ar import AnchoredAR, encode_target, time_intervals
from src.audio_understanding.evaluation import match_events
from src.audio_understanding.inference import sha256
from src.model_benchmark import rhythm_bonus
from src.soft_onset import MEMORY_SECONDS
from src.windowed_ar import WindowedAR, encode_window_target, window_intervals


def read(p):return json.loads(p.read_text())


@torch.no_grad()
def teacher(model,target,condition,windowed):
    d=model.decoder
    memory=d.prepare_memory(condition['features'].float(),torch.zeros_like(condition['soft_onsets']))
    extra={'input_condition':model.window_condition(target['bounds'])[None]} if windowed else {}
    logits,hidden=d(target['tokens'][None],target['frames'][None],memory,condition['valid'],
                    history=target['history'][None],return_hidden=True,**extra)
    indices=target['decisions'];h=hidden[0,indices]
    stop_logits=model.stop_head(h).squeeze(-1);positive=target['stop_targets']==0
    predicted_stop=stop_logits>=0
    layout=target['layout_targets']!=-100
    counts=Counter(events=int(positive.sum()),stops=int((~positive).sum()),
        false_stops=int((predicted_stop & positive).sum()),correct_stops=int((predicted_stop & ~positive).sum()),
        layout_total=int(layout.sum()),layout_correct=int((logits[0,layout].argmax(-1)==target['layout_targets'][layout]).sum()))
    if positive.any():
        q=h[positive];lower=target['lower_seconds'][positive]
        if windowed:
            lo,hi,allowed=window_intervals(lower,target['bounds'][indices[positive],1],memory.shape[1],condition['valid'])
        else:lo,hi,allowed=time_intervals(lower,memory.shape[1],target['duration'],condition['valid'])
        selected=model.frame_logits(q,memory,allowed).argmax(-1);row=torch.arange(len(selected))
        fraction=model.fraction(q,memory[0,selected])
        predicted=lo[row,selected]+fraction*(hi[row,selected]-lo[row,selected])
        errors=(predicted-target['next_seconds'][positive]).abs()
        counts.update(pointer_correct_25ms=int((errors<=.025).sum()),
            continue_and_time_correct_25ms=int(((errors<=.025)&~predicted_stop[positive]).sum()),
            absolute_time_error_sum_seconds=float(errors.sum()))
    return counts


def main():
    torch.set_num_threads(2)
    base=ROOT/'out/windowed_ar_20261031';dataset=torch.load(base/'controlled/conditions.pt',map_location='cpu',weights_only=True)
    by_id={r['id']:r for r in dataset};protocol=read(ROOT/'benchmarks/v1/protocol.json')
    clock=dict(bpm=120.,beat_zero_seconds=0.,source='constructed physical clock',verified=True)
    results={}
    for arch in ('whole','window'):
        root=base/arch;checkpoint=torch.load(root/'model_latest.pt',map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==400
        model=(WindowedAR(checkpoint['config']['decoder_config']) if arch=='window' else AnchoredAR(checkpoint['config']['decoder_config'])).eval()
        model.load_state_dict(checkpoint['model']);evaluation=read(root/'evaluation-e0400.json')
        rows=[];teachers={}
        for prediction in evaluation['rows']:
            row=by_id[prediction['id']];ref_times=prediction['reference_onsets']
            ref_holds=[n for n in row['reference'] if n['type']=='hold']
            duration_errors=[];crossing=0;matched_crossing=0
            for lane in range(4):
                refs=[n for n in ref_holds if n['position']==lane]
                preds=[n for n in prediction['notes'] if n['type']=='hold' and n['position']==lane]
                pairs=match_events([n['time_seconds'] for n in refs],[n['time_seconds'] for n in preds],.025)
                for a,b in pairs:
                    r=min(refs,key=lambda n:abs(n['time_seconds']-a));p=min(preds,key=lambda n:abs(n['time_seconds']-b))
                    duration_errors.append(abs((r['end_seconds']-r['time_seconds'])-(p['end_seconds']-p['time_seconds']))*1000)
                    if int(r['time_seconds']//10)<int((r['end_seconds']-1e-9)//10):matched_crossing+=1
                crossing+=sum(int(n['time_seconds']//10)<int((n['end_seconds']-1e-9)//10) for n in refs)
            reference_grid=rhythm_bonus(ref_times,clock,protocol['rhythm'])['bonus_per_onset']
            prediction_grid=rhythm_bonus(prediction['onsets_seconds'],clock,protocol['rhythm'])['bonus_per_onset']
            target=(encode_window_target(row['route'],row['duration']) if arch=='window' else encode_target(row['route'],row['duration']))
            # Fixed-probe type/column labels are fictitious and MUST NOT support teacher claims.
            taught=teacher(model,target,row['condition'],arch=='window') if row['split']!='fixed_probe' else None
            if taught is not None:teachers.setdefault(row['split'],Counter()).update(taught)
            rows.append(dict(id=row['id'],split=row['split'],status=prediction['status'],
                onset=prediction['onset'],joint_f1_25ms=prediction.get('joint_f1_25ms'),
                reference_grid=reference_grid,prediction_grid=prediction_grid,
                reference_cross_window_holds=crossing,matched_cross_window_holds=matched_crossing,
                matched_hold_duration_errors_ms=duration_errors,teacher=dict(taught) if taught is not None else None,
                reference=ref_times,predictions=prediction['onsets_seconds'],
                forced_windows=prediction.get('forced_windows'),windows=prediction.get('windows')))
        summaries={}
        for split in sorted({r['split'] for r in rows}):
            all_rows=[r for r in rows if r['split']==split];positive=[r for r in all_rows if r['reference']]
            counters={key:sum(r['onset']['25'][key] for r in all_rows) for key in ('reference','predicted','matched')}
            s=dict(examples=len(all_rows),positive_examples=len(positive),completed=sum(r['status']=='completed' for r in all_rows),
                positive_macro_raw_onset_f1_25ms=statistics.fmean(r['onset']['25']['f1'] for r in positive),
                positive_macro_raw_onset_f1_50ms=statistics.fmean(r['onset']['50']['f1'] for r in positive),
                raw_onset_micro_25ms={**counters,'precision':counters['matched']/max(1,counters['predicted']),
                    'recall':counters['matched']/max(1,counters['reference'])},
                mean_reference_grid=statistics.fmean(r['reference_grid'] for r in positive),
                mean_prediction_grid_on_nonempty_outputs=statistics.fmean(r['prediction_grid'] for r in positive if r['prediction_grid'] is not None),
                silence_extra_onsets=sum(len(r['predictions']) for r in all_rows if not r['reference']))
            if split!='fixed_probe':
                t=teachers[split]
                s.update(positive_macro_joint_f1_25ms=statistics.fmean(r['joint_f1_25ms'] if r['status']=='completed' else 0. for r in positive),
                    teacher=dict(t),teacher_pointer_accuracy_25ms=t['pointer_correct_25ms']/t['events'],
                    teacher_continue_and_time_accuracy_25ms=t['continue_and_time_correct_25ms']/t['events'],
                    teacher_false_stop_rate=t['false_stops']/t['events'],teacher_stop_recall=t['correct_stops']/t['stops'],
                    teacher_layout_accuracy=t['layout_correct']/t['layout_total'],
                    reference_cross_window_holds=sum(r['reference_cross_window_holds'] for r in all_rows),
                    matched_cross_window_holds=sum(r['matched_cross_window_holds'] for r in all_rows))
                errors=[x for r in all_rows for x in r['matched_hold_duration_errors_ms']]
                s['matched_hold_duration_mae_ms']=statistics.fmean(errors) if errors else None
            if arch=='window':
                windows=[w for r in all_rows for w in r['windows']]
                s.update(windows=len(windows),forced_windows=sum(w['termination']!='learned_end_window' for w in windows),
                         learned_window_end_micro=sum(w['termination']=='learned_end_window' for w in windows)/len(windows))
            summaries[split]=s
        training=read(root/'training.json')
        results[arch]=dict(checkpoint=str((root/'model_latest.pt').relative_to(ROOT)),checkpoint_sha256=sha256(root/'model_latest.pt'),
            evaluation_sha256=sha256(root/'evaluation-e0400.json'),parameters=checkpoint['config']['parameters'],
            final_training=training['curves'][-1],resources=training['resources'],summaries=summaries,rows=rows)
        print(arch,json.dumps(summaries,ensure_ascii=False),flush=True)
    report=dict(schema='audiolab-controlled-window-attribution-v1',runs=results,
        protocol_sha256=sha256(ROOT/'benchmarks/v1/protocol.json'),
        metric_scope='Positive-only macro onset F1 follows existing benchmark; silence scored separately. '
                     'Raw metrics include failed prefixes; joint selection F1 counts failures as zero. '
                     'Teacher metrics use true chart history and are not free-generation scores.',
        rhythm_scope='Constructed notes use all 32 tick positions; grid bonus only, not an accuracy gate.')
    (Path(__file__).parent/'w0_w1_controlled_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()

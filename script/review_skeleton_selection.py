"""One .7 -> .8 selector comparison, reusing the frozen matched baseline."""
import argparse
import bisect
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from script.review_hold_repairs import digest, provenance, verified_source
from script.diagnose_patterns import selection_replay
from script.ablate_layout_features import check_output
from src.audio_understanding.inference import load_audio, make_result, code_version
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.data.timing import tick_seconds
from src.product.artifacts import chart_document, validate_chart, write_json, write_preview
from src.product.pipeline import JobControl, decode_notes, quantize_candidates
from src.resource_budget import ResourceBudget, add_resource_args


def difference(arrays, baseline, candidate):
    frames = {}
    for i, (t, valid) in enumerate(zip(arrays['event_times_seconds'], arrays['valid_event_time'])):
        if valid: frames.setdefault(float(t*1000)/1000, []).append(i)
    a={e['time_seconds']:e for e in baseline['events']}; b={e['time_seconds']:e for e in candidate['events']}
    return [dict(time_seconds=t, frame_indices=frames[t],
                 source_id='dense-'+','.join(str(i) for i in frames[t]),
                 baseline_event_id=a[t]['id'] if t in a else None,
                 candidate_event_id=b[t]['id'] if t in b else None,
                 confidence=(a.get(t) or b[t])['confidence'],
                 status='retained' if t in a and t in b else 'deleted' if t in a else 'added')
            for t in sorted(a.keys() | b.keys())]


def clips(changes, duration):
    deleted=[r for r in changes if r['status']=='deleted']
    if not deleted: raise ValueError('this fixed comparison has no deleted candidate')
    result=[]
    for index in sorted({0,len(deleted)//2,len(deleted)-1}):
        row=deleted[index];start=max(0,row['time_seconds']-4);end=min(duration,row['time_seconds']+4)
        if result and start<=result[-1]['end_seconds']:
            result[-1]['end_seconds']=end;result[-1]['source_ids'].append(row['source_id'])
        else: result.append(dict(start_seconds=start,end_seconds=end,source_ids=[row['source_id']]))
    return result


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);device=torch.device(args.device)
    budget=ResourceBudget(args,device,output);control=JobControl(output,budget,300)
    try:
        source=Path(args.source_job);old=Path(args.baseline_run)
        manifest,files=verified_source(source,'4k');verified_source(source,'freex')
        prior=json.loads((old/'report.json').read_text())
        if prior['status']!='completed' or prior['source']['manifest']['sha256']!=files['manifest']['sha256']:
            raise ValueError('baseline report does not match the frozen source job')
        evidence=json.loads((source/'audio_events.json').read_text())
        if evidence['selection']['threshold']!=.7 or evidence['selection']['min_interval_ms']!=35:
            raise ValueError('expected the frozen .7/35ms baseline')
        posterior=selection_replay(source,evidence,manifest['artifacts'][evidence['dense_file']]['sha256'])
        with np.load(source/evidence['dense_file'],allow_pickle=False) as z:
            arrays={k:z[k] for k in ('event_probability','event_times_seconds','valid_event_time')}
        candidate=make_result(arrays,evidence['duration_seconds'],dict(label_source=evidence['label_source']),threshold=.8,min_interval_ms=35)
        changes=difference(arrays,evidence,candidate);segments=clips(changes,evidence['duration_seconds'])
        reference=Path(args.reference_chart);ref=json.loads(reference.read_text())
        ref_times=sorted({tick_seconds(n['time'],line['bpm'])+ref['offset'] for line in ref['judgeLineList']
                          for n in line['notesAbove']+line['notesBelow']})
        for row in changes:
            i=bisect.bisect_left(ref_times,row['time_seconds']);near=ref_times[max(0,i-1):i+1]
            nearest=min(near,key=lambda t:abs(t-row['time_seconds']))
            row['nearest_reference_delta_seconds']=row['time_seconds']-nearest
        config=manifest['config'];bpm,offset,seed=config['bpm'],config['offset'],config['seed']
        ticks,quantization=quantize_candidates(candidate,bpm,offset)
        audio=Path(files['artifacts']['audio']['path']);mel,duration=load_audio(audio,max_seconds=600,budget=budget)
        report=dict(status='running',implementation=code_version(),source=files,baseline_report=provenance(old/'report.json'),
                    posterior=posterior,reference=provenance(reference),reference_semantics='same-song derived chart; not acoustic truth',
                    config=dict(baseline_threshold=.7,candidate_threshold=.8,min_interval_ms=35,bpm=bpm,offset=offset,seed=seed),
                    candidate_changes=changes,segments=segments,quantization=quantization,routes={},product_default_changed=False)
        write_json(output/'candidate_events.json',candidate)
        for condition in ('baseline','candidate'):
            folder=output/condition;folder.mkdir();(folder/audio.name).symlink_to(audio)
            (folder/'manifest.json').symlink_to('../report.json');routes={}
            for mode,subdir in [('4k','decoder'),('freex','decoder_freex')]:
                previous=next(r for r in prior['results'] if r['mode']==mode and r['condition']=='matched')
                if condition=='baseline':
                    for suffix in ('raw.json','json','events.json'):
                        item=previous['files'][suffix];path=old/item['path']
                        if digest(path)!=item['sha256']:raise ValueError('frozen baseline artifact changed')
                        shutil.copyfile(path,folder/f'{mode}.{suffix}')
                    checks=previous['checks'];decoding=previous['decoding']
                else:
                    ck=Path(args.model_root)/subdir/'decoder_best.pt'
                    if digest(ck)!=manifest['model_sha256'][mode]:raise ValueError('checkpoint changed')
                    control.update('candidate_'+mode)
                    raw,decoding=decode_notes(mel,duration,ticks,ck,mode,bpm,offset,seed,device,control,
                                              max_tokens=12000,encoder_mode='full',feature_condition='matched')
                    if decoding['feature_intervention']['source_features']!=previous['decoding']['feature_intervention']['source_features']:
                        raise ValueError('layout source features differ from baseline')
                    chart_document(raw,bpm,offset,folder/f'{mode}.raw.json')
                    end=(duration-offset)/tick_seconds(1,bpm)
                    final,repairs=constrain_4k_notes(raw,end) if mode=='4k' else constrain_free_notes(raw,end)
                    chart=chart_document(final,bpm,offset,folder/f'{mode}.json')
                    events,checks=validate_chart(chart,mode,duration,ticks)
                    write_json(folder/f'{mode}.events.json',events)
                events=json.loads((folder/f'{mode}.events.json').read_text())
                routes[mode]=dict(events=events['events'],chart_file=f'{mode}.json',events_file=f'{mode}.events.json')
                report['routes'][condition+'_'+mode]=dict(checks=checks,decoding=decoding,
                    files={suffix:provenance(folder/f'{mode}.{suffix}') for suffix in ('raw.json','json','events.json')})
                check_output(output,budget)
            write_preview(folder/'preview.html',title='原基线' if condition=='baseline' else '候选取舍对照',
                          audio_name=audio.name,routes=routes,duration=duration)
            page=folder/'preview.html'
            page.write_text(page.read_text().replace('</html>', '''<script>
const review=new URLSearchParams(location.hash.slice(1));
if(review.has('t')){const seek=()=>audio.currentTime=Number(review.get('t'));if(audio.readyState>=1)seek();else audio.addEventListener('loadedmetadata',seek,{once:true});}
if(review.has('mode')&&data.routes[review.get('mode')]){select.value=review.get('mode');changeMode();}
if(review.has('end'))audio.addEventListener('timeupdate',()=>{if(audio.currentTime>=Number(review.get('end'))&&!audio.paused)audio.pause();});
</script></html>'''))
        links=[]
        for i,segment in enumerate(segments,1):
            links.append(f'<p>片段{i}：{segment["start_seconds"]:.2f}–{segment["end_seconds"]:.2f}秒 ')
            for mode in ('4k','freex'):
                for condition,label in [('baseline','原基线'),('candidate','候选稿')]:
                    links.append(f'<a href="{condition}/preview.html#t={segment["start_seconds"]}&end={segment["end_seconds"]}&mode={mode}">{mode} {label}</a> ')
            links.append('</p>')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>候选取舍试听</title><style>body{font:18px system-ui;margin:36px;line-height:2}a{margin-right:16px}</style><h1>候选取舍试听</h1><p>唯一实验改动为取音分数阈值0.7→0.8，尚未判断哪份更好。</p><p>点击链接后播放，片段末尾暂停。比较删点是否漏掉音乐，以及之后保留点的布局变化；两份谱的后续采样历史会不同。</p>'+''.join(links)+'<p><a href="baseline/preview.html">原基线整曲</a><a href="candidate/preview.html">候选稿整曲</a><a href="report.json">候选来源与差异</a></p>')
        report.update(status='completed',resources=budget.log_state())
        write_json(output/'report.json',report);check_output(output,budget)
        if json.loads((output/'report.json').read_text())!=report:raise ValueError('report readback mismatch')
        control.update(status='completed');return report
    except BaseException as error:
        for p in [output/'report.json',output/'index.html',*output.glob('*/preview.html')]:p.unlink(missing_ok=True)
        control.update(status='failed',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-job','baseline-run','model-root','reference-chart','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');add_resource_args(p);p.set_defaults(max_rss_gib=4)
    result=run(p.parse_args());print(json.dumps(dict(status=result['status'],resources=result['resources'])))

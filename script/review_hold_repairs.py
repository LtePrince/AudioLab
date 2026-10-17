"""CPU replay of one Hold policy factor from twelve frozen product artifacts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import time

import soundfile as sf

from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.data.timing import tick_seconds
from src.product.artifacts import validate_chart, write_json, write_preview

SCHEMA='audiolab-hold-policy-replay-v1'
GIB=1024**3
KINDS={1:'tap',2:'drag',3:'hold',4:'flick'}


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): value.update(block)
    return value.hexdigest()


def provenance(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size)


def resources(output):
    # Audio/report symlinks refer to existing files; count only their own bytes.
    size=sum(p.lstat().st_size if p.is_symlink() else p.stat().st_size
             for p in output.rglob('*') if p.is_file() or p.is_symlink())
    available=next(int(line.split()[1])*1024/GIB for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))
    state=dict(output_bytes=size,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/GIB,
               disk_free_gib=shutil.disk_usage(output).free,available_ram_gib=available)
    state['disk_free_gib']/=GIB
    if size>10*1024**2 or state['peak_rss_gib']>4 or state['disk_free_gib']<8 or available<8:
        raise RuntimeError('replay exceeded 10 MiB output / 4 GiB RSS / 8 GiB reserve budget')
    return state


def hold_attribution(source, on_notes, off_notes, events, mode, end_tick, tick, min_gap):
    """Replay only the trigger lookup, after the shared spatial processing.

    Stable reverse-time iteration is part of static-v1: an earlier source index
    at the same tick may also have been seen. Report that case, not a strictly
    later-in-time claim. Types do not participate in the lookup.
    """
    visited=[];next_lane={};result=[]
    for index in sorted(range(len(source)),key=lambda i:source[i]['time'],reverse=True):
        original=source[index];a=on_notes[index];b=off_notes[index]
        if original['type']==3:
            if mode=='4k': neighbor=next_lane.get(a['positionX'])
            else:
                candidates=[j for j in visited if abs(events[index]['screen_x']-events[j]['screen_x'])<min_gap-1e-9]
                neighbor=min(candidates,key=lambda j:on_notes[j]['time']) if candidates else None
            limit=min(end_tick,on_notes[neighbor]['time']) if neighbor is not None else end_tick
            predicted=max(0,min(original['holdTime'],math.floor(limit-original['time']+1e-7)))
            if predicted!=a['holdTime']:
                raise ValueError('trigger lookup does not reproduce enabled Hold duration')
            extra=(b['holdTime']-a['holdTime'])*tick
            endpoint=(original['holdTime']-b['holdTime'])*tick
            trigger=None
            if extra>0:
                if neighbor is None: raise ValueError('future-rule loss has no trigger')
                event=events[neighbor]
                trigger=dict(id=event['id'],source_index=neighbor,type=event['type'],
                             source_type=KINDS[source[neighbor]['type']],
                             time_seconds=event['time_seconds'],screen_x=event['screen_x'],
                             positionX=on_notes[neighbor]['positionX'],
                             relation='same_lane' if mode=='4k' else 'nearby_after_spacing',
                             same_tick=original['time']==on_notes[neighbor]['time'],
                             distance_screen_x=abs(events[index]['screen_x']-event['screen_x']))
            result.append(dict(id=events[index]['id'],source_index=index,
                               original_duration_seconds=original['holdTime']*tick,
                               on_duration_seconds=a['holdTime']*tick,off_duration_seconds=b['holdTime']*tick,
                               endpoint_removed_seconds=endpoint,future_rule_removed_seconds=extra,
                               reason=('future_onset_and_audio_end' if endpoint>0 else 'future_onset') if extra>0 else
                                      ('audio_end' if endpoint>0 else 'unchanged'),trigger=trigger,
                               screen_x_after_spacing=events[index]['screen_x'],
                               screen_x_before_spacing=.5+original['positionX']*.05625))
        visited.append(index);next_lane[a['positionX']]=index
    return sorted(result,key=lambda row:row['source_index'])


def replay(raw, frozen_final, mode, duration, min_gap=.04):
    """A replays static-v1; B retains its spatial result and only omits future-hit clipping.

    Input notes are already exported. Reuse the fixed template and write JSON
    directly: passing them through save_phigros_notes would rescale floor twice.
    """
    if mode not in ('4k','freex') or len(raw.get('judgeLineList',[]))!=1:
        raise ValueError('one static 4K/free-x line required')
    line=raw['judgeLineList'][0]
    if line['notesBelow'] or not line['notesAbove']:
        raise ValueError('frozen product format requires nonempty notesAbove and empty notesBelow')
    source=line['notesAbove'];bpm=line['bpm'];offset=raw['offset']
    tick=tick_seconds(1,bpm);end_tick=(duration-offset)/tick
    on_notes,repair_summary=(constrain_4k_notes(source,end_tick) if mode=='4k' else
                             constrain_free_notes(source,end_tick,min_gap))
    on=deepcopy(raw);on['judgeLineList'][0]['notesAbove']=on_notes
    if on!=frozen_final:
        raise ValueError('enabled policy does not exactly reproduce the frozen final export')
    off=deepcopy(on);off_notes=off['judgeLineList'][0]['notesAbove']
    speed=line['speedEvents'][0]['value']
    for original,note in zip(source,off_notes):
        if original['type']==3:
            dur=max(0,min(original['holdTime'],math.floor(end_tick-original['time']+1e-7)))
            note.update(holdTime=dur,type=3 if dur>=1 else 1,speed=speed if dur>=1 else 1.)
    # The original speed template must cover every restored tail; no second factor.
    latest_tail=max(n['time']+n['holdTime'] for n in off_notes)
    if line['speedEvents'][0]['startTime']>0 or line['speedEvents'][0]['endTime']<latest_tail:
        raise ValueError('fixed source speed template does not cover the counterfactual tail')
    skeleton=sorted({n['time'] for n in source})
    on_events,on_checks=validate_chart(on,mode,duration,skeleton)
    off_events,off_checks=validate_chart(off,mode,duration,skeleton)
    if len(source)!=len(on_notes) or len(source)!=len(off_notes):
        raise ValueError('policy changed note identity/count')
    holds=hold_attribution(source,on_notes,off_notes,on_events['events'],mode,end_tick,tick,min_gap)
    by_index={row['source_index']:row for row in holds}
    changes=[]
    for index,(original,a,b) in enumerate(zip(source,on_notes,off_notes)):
        if any(a[key]!=b[key] for key in a if key not in ('holdTime','type','speed')):
            raise ValueError('counterfactual changed more than duration-derived fields')
        if a['time']!=original['time'] or a['floorPosition']!=original['floorPosition']:
            raise ValueError('replay changed the onset or already exported floorPosition')
        fields={key:dict(on=a[key],off=b[key]) for key in ('holdTime','type','speed') if a[key]!=b[key]}
        if fields and original['type']!=3:
            raise ValueError('non-Hold changed under Hold-only policy')
        if fields:
            if a['type']!=b['type'] and not (a['type']==1 and a['holdTime']==0 and b['type']==3 and b['holdTime']>=1):
                raise ValueError('type change is not derived from Hold-to-Tap conversion')
            changes.append(dict(id=on_events['events'][index]['id'],source_index=index,fields=fields,
                                time_seconds=on_events['events'][index]['time_seconds'],
                                on_end_seconds=on_events['events'][index]['end_seconds'],
                                off_end_seconds=off_events['events'][index]['end_seconds'],
                                on_duration_seconds=a['holdTime']*tick,off_duration_seconds=b['holdTime']*tick,
                                retained_duration_seconds=(b['holdTime']-a['holdTime'])*tick,
                                attribution=by_index[index]))
    if any(a['id']!=b['id'] or a['source_index']!=b['source_index'] or a['screen_x']!=b['screen_x']
           or a['time_seconds']!=b['time_seconds'] for a,b in zip(on_events['events'],off_events['events'])):
        raise ValueError('event identity/head/space changed')
    hold_ids=[i for i,n in enumerate(source) if n['type']==3]
    summary=dict(source_notes=len(source),source_holds=len(hold_ids),changed_notes=len(changes),
                 enabled_repair_summary=repair_summary,
                 source_hold_seconds=sum(source[i]['holdTime']*tick for i in hold_ids),
                 on_hold_seconds=sum(on_notes[i]['holdTime']*tick for i in hold_ids),
                 off_hold_seconds=sum(off_notes[i]['holdTime']*tick for i in hold_ids),
                 endpoint_removed_seconds=sum((source[i]['holdTime']-off_notes[i]['holdTime'])*tick for i in hold_ids),
                 future_rule_removed_seconds=sum(change['retained_duration_seconds'] for change in changes),
                 on_checks=on_checks,off_checks=off_checks,enabled_exactly_matches_frozen_export=True,
                 invariant_fields='source_index/id, note count, onset, position, floorPosition, line template',
                 candidate_basis='same frozen raw onset ticks; no new inference',
                 target_player_judgement='unverified',human_music_or_playability='unreviewed')
    return dict(on=on,off=off,on_events=on_events,off_events=off_events,summary=summary,changes=changes,holds=holds)


def verified_source(directory,mode):
    manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='completed' or mode not in manifest['routes']:
        raise ValueError('source product job is incomplete')
    route=manifest['routes'][mode]
    names=dict(raw=route['raw_file'],final=route['chart_file'],events=route['events_file'],
               repairs=f'{mode}.repairs.json',audio=manifest['audio']['file'],candidates='audio_events.json')
    files={}
    for key,name in names.items():
        if Path(name).name!=name: raise ValueError('unexpected source artifact path')
        files[key]=provenance(directory/name)
        if files[key]['sha256']!=manifest['artifacts'][name]['sha256']:
            raise ValueError(f'frozen source artifact changed: {name}')
    if files['audio']['sha256']!=manifest['audio']['sha256']:
        raise ValueError('audio provenance mismatch')
    if json.loads(Path(files['repairs']['path']).read_text())['profile']!='static-v1':
        raise ValueError('source repair policy is not static-v1')
    duration=manifest['audio']['duration_seconds']
    if abs(sf.info(files['audio']['path']).duration-duration)>2/22050:
        raise ValueError('audio duration differs from source manifest')
    return manifest,dict(manifest=provenance(manifest_path),artifacts=files)


def make_previews(output,results):
    """Two full-song views for one explicit difference case; audio is a symlink."""
    available=[row for row in results if row['status']=='completed' and row['changes']]
    if not available: return []
    chosen=max(available,key=lambda row:row['summary']['future_rule_removed_seconds'])
    pair=[row for row in results if row['status']=='completed' and row['group']==chosen['group'] and row['song']==chosen['song']]
    focus=max(chosen['changes'],key=lambda c:c['retained_duration_seconds'])
    previews=[]
    for policy,label in [('on','A · 现有持续处理'),('off','B · 仅按曲末限制持续')]:
        folder=output/'previews'/policy;folder.mkdir(parents=True)
        audio=folder/Path(chosen['source']['artifacts']['audio']['path']).name
        audio.symlink_to(chosen['source']['artifacts']['audio']['path'])
        (folder/'manifest.json').symlink_to('../../report.json')
        routes={}
        for row in pair:
            chart=json.loads((output/row['exports'][policy]['path']).read_text())
            notes=chart['judgeLineList'][0]['notesAbove']
            events,_=validate_chart(chart,row['mode'],row['duration_seconds'],{n['time'] for n in notes})
            events['source_audio_sha256']=row['source']['artifacts']['audio']['sha256']
            write_json(folder/f'{row["mode"]}.events.json',events)
            routes[row['mode']]=dict(events=events['events'],chart_file='../../'+row['exports'][policy]['path'],
                                    events_file=f'{row["mode"]}.events.json')
        write_preview(folder/'preview.html',title=f'{label} · {chosen["group"]}/{chosen["song"]} · 待试听对照',
                      audio_name=audio.name,routes=routes,duration=chosen['duration_seconds'])
        # Review-only hash links seek native audio time; product template stays unchanged.
        path=folder/'preview.html';text=path.read_text()
        extra="""<script>
const reviewParams=new URLSearchParams(location.hash.slice(1));
if(reviewParams.has('t')){const seekReview=()=>{const t=Number(reviewParams.get('t'));if(Number.isFinite(t))audio.currentTime=Math.max(0,Math.min(data.duration,t));};if(audio.readyState>=1)seekReview();else audio.addEventListener('loadedmetadata',seekReview,{once:true});}
if(reviewParams.has('mode')&&data.routes[reviewParams.get('mode')]){select.value=reviewParams.get('mode');changeMode();}
</script>\n"""
        path.write_text(text.replace('</html>',extra+'</html>'))
        previews.append(dict(policy=policy,path=str(path.relative_to(output)),group=chosen['group'],song=chosen['song'],
                             focus_mode=chosen['mode'],focus=focus,
                             suggested_start_seconds=max(0,focus['time_seconds']-1),
                             suggested_end_seconds=min(chosen['duration_seconds'],focus['off_end_seconds']+1),
                             audio_storage='symlink to unchanged source artifact'))
    return previews


def write_index(output,report):
    links=[]
    for preview in report['previews']:
        url=f'{preview["path"]}#t={preview["suggested_start_seconds"]:.9f}&mode={preview["focus_mode"]}'
        label='A：现有持续处理' if preview['policy']=='on' else 'B：仅按曲末限制持续'
        links.append(f'<a href="{html.escape(url)}">{label} · 跳到差异片段</a>')
    rows=[]
    for row in report['results']:
        label=html.escape(f'{row["group"]}/{row["song"]} {row["mode"]}')
        if row['status']!='completed': rows.append(f'<tr><td>{label}</td><td colspan="3">处理失败，见报告</td></tr>');continue
        summary=row['summary']
        rows.append(f'<tr><td>{label}</td><td>{summary["changed_notes"]}</td><td>{summary["future_rule_removed_seconds"]:.3f}s</td>'
                    f'<td><a download href="{row["exports"]["on"]["path"]}">A JSON</a> · '
                    f'<a download href="{row["exports"]["off"]["path"]}">B JSON</a></td></tr>')
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AudioLab 持续对照</title>'
        '<style>body{font:16px system-ui;max-width:980px;margin:36px auto;padding:20px;background:#111725;color:#eef3ff}'
        'a{color:#8cd8ff;margin-right:16px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #425474}</style>'
        '<h1>AudioLab · 持续处理对照</h1><p>两份谱的音符起点和位置相同，只比较持续时间处理。预览与导出供试听检查；实际判定与手感尚未确认。</p>'
        '<p>'+' '.join(links)+'</p><p>音频使用本机软链接，请保留原生成目录。产品默认保持原有处理。</p>'
        '<table><tr><th>来源与玩法</th><th>变化音符</th><th>B比A保留的持续</th><th>导出</th></tr>'+''.join(rows)+'</table>'
        '<p><a href="report.json" download>完整逐音与来源报告</a></p></html>')


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];started=time.monotonic()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    report=dict(schema=SCHEMA,status='running',expected_rows=12,results=[],
                implementation=dict(git_commit=commit,working_tree_dirty=dirty,source_sha256={path:digest(root/path) for path in
                    ['script/review_hold_repairs.py','src/data/chart_constraints.py','src/product/artifacts.py','src/product/preview.html']}),
                scope='same frozen outputs; only future-onset Hold clipping toggled; CPU replay, no new model inference',
                replay_config=dict(enabled_policy='static-v1',disabled_factor='future-onset Hold clipping',
                                   free_min_gap_screen_width=.04,endpoint_rounding='floor(end_tick-head_tick+1e-7)',
                                   trigger_geometry='after identical spatial processing',
                                   tie_order='existing stable reverse-time iteration; same tick may trigger'),
                product_default_changed=False,is_quality_holdout=False,config=vars(args))
    try:
        for group,location in [('manual',args.manual_root),('automatic',args.automatic_root)]:
            for song in range(3):
                directory=Path(location)/f'song-{song}'
                for mode in ('4k','freex'):
                    resources(output)
                    row=dict(group=group,song=f'song-{song}',mode=mode,status='running')
                    try:
                        manifest,source=verified_source(directory,mode)
                        row.update(source=source,source_commit=manifest['implementation']['git_commit'],
                                   source_config=manifest['config'],model_sha256=manifest['model_sha256'][mode],
                                   duration_seconds=manifest['audio']['duration_seconds'])
                        raw=json.loads(Path(source['artifacts']['raw']['path']).read_text())
                        final=json.loads(Path(source['artifacts']['final']['path']).read_text())
                        result=replay(raw,final,mode,row['duration_seconds'])
                        existing_events=json.loads(Path(source['artifacts']['events']['path']).read_text())
                        if result['on_events']['events']!=existing_events['events']:
                            raise ValueError('enabled event IDs/times differ from the frozen product event file')
                        row.update(summary=result['summary'],changes=result['changes'],holds=result['holds'],exports={})
                        folder=output/'charts'/f'{group}-song-{song}-{mode}';folder.mkdir(parents=True)
                        for policy in ('on','off'):
                            path=folder/f'{policy}.json';write_json(path,result[policy])
                            if json.loads(path.read_text())!=result[policy]: raise ValueError('export readback differs')
                            row['exports'][policy]=dict(path=str(path.relative_to(output)),sha256=digest(path),bytes=path.stat().st_size)
                        if any(digest(v['path'])!=v['sha256'] for v in [source['manifest'],*source['artifacts'].values()]):
                            raise ValueError('source changed during replay')
                        row['status']='completed'
                    except Exception as error:
                        row.update(status='failed',error=dict(type=type(error).__name__,message=str(error)))
                    report['results'].append(row)
        report['previews']=make_previews(output,report['results'])
        complete=sum(row['status']=='completed' for row in report['results'])
        report.update(status='completed' if complete==12 else 'completed_with_failures',completed_rows=complete,failed_rows=12-complete,
                      elapsed_seconds=time.monotonic()-started)
        report['by_mode']={mode:{key:sum(row['summary'][key] for row in report['results'] if row['mode']==mode and row['status']=='completed')
                         for key in ('source_notes','source_holds','changed_notes','source_hold_seconds','on_hold_seconds','off_hold_seconds',
                                     'endpoint_removed_seconds','future_rule_removed_seconds')} for mode in ('4k','freex')}
        write_index(output,report);report['resources']=resources(output)
        for _ in range(4):
            write_json(output/'report.json',report)
            if json.loads((output/'report.json').read_text())!=report: raise ValueError('final report readback differs')
            measured=resources(output)
            if measured['output_bytes']==report['resources']['output_bytes']: break
            report['resources']=measured
        else: raise ValueError('final output size did not converge')
        return report
    except BaseException as error:
        (output/'report.json').unlink(missing_ok=True)
        try: write_json(output/'failure.json',dict(schema=SCHEMA,status='failed',error=str(error),partial_results=report['results']))
        except OSError: pass
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manual-root',required=True);p.add_argument('--automatic-root',required=True)
    p.add_argument('--output',required=True)
    return p


if __name__=='__main__':
    result=run(parser().parse_args())
    print(json.dumps(dict(status=result['status'],completed_rows=result['completed_rows'],by_mode=result['by_mode'],resources=result['resources'])),flush=True)
    raise SystemExit(0 if result['status']=='completed' else 1)

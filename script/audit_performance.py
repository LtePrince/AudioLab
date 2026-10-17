"""Bounded CPU audit of original multi-line PGR interpretation and round trips."""
from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

from src.data.performance import PerformanceChart, content_hash, file_hash
from src.data.performance_reference import PhiraSourceReference, PHIRA_COMMIT

GIB = 1024**3


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    temporary.replace(path)


def budget(output):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB
    disk = shutil.disk_usage(output).free / GIB
    available = next(int(line.split()[1]) * 1024 / GIB for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if rss > 4 or disk < 8 or available < 8 or size > 50 * 1024**2:
        raise RuntimeError('performance audit exceeded 4 GiB RSS / 8 GiB reserves / 50 MiB output budget')
    return dict(peak_rss_gib=rss, disk_free_gib=disk, available_ram_gib=available, output_bytes=size)


def select_originals(data_list):
    """Two v1 and four v3 in source-list order; engineering coverage only."""
    groups = {1: [], 3: []}
    inspected = 0
    for raw in data_list.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        fields = raw.split(',')
        audio = (data_list.parent / fields[1]).resolve()
        original = audio.with_suffix('.json')
        source = json.loads(original.read_text())
        inspected += 1
        version = source.get('formatVersion')
        if version in groups and len(groups[version]) < (2 if version == 1 else 4):
            groups[version].append(dict(chart=str(original), audio=str(audio), version=version))
        if len(groups[1]) == 2 and len(groups[3]) == 4:
            break
    if len(groups[1]) != 2 or len(groups[3]) != 4:
        raise ValueError('engineering audit requires two original v1 and four original v3 charts in the list')
    return [*groups[1], *groups[3]], inspected


def comparison_metric(numerical_tolerance=1e-7):
    return dict(queries=0, comparable=0, canonical_unsupported=0, reference_unsupported=0,
                max_abs_error=0., max_abs_error_scope='queried_samples_only',numerical_tolerance=numerical_tolerance,
                differences_above_numerical_tolerance=0, examples=[])


def verify_reference_cache(root, cache):
    registry=json.loads((root/'docs/phira_reference.json').read_text())
    needed={'pgr','anim','object','tween','core'}
    results=[]
    for entry in registry['files']:
        stem=Path(entry['path']).stem
        if stem not in needed:continue
        path=cache/f'audiolab-phira-{stem}.rs'
        observed=file_hash(path)
        if observed!=entry['sha256']:raise ValueError(f'pinned reference source hash mismatch: {entry["path"]}')
        results.append(dict(path=entry['path'],sha256=observed,url=entry['url']))
    if len(results)!=len(needed):raise ValueError('reference registry is incomplete')
    return results


def compare(metric, canonical, reference, line_id, at):
    metric['queries'] += 1
    if canonical is None:
        metric['canonical_unsupported'] += 1
    if reference is None:
        metric['reference_unsupported'] += 1
    if canonical is None or reference is None:
        return
    metric['comparable'] += 1
    error = abs(canonical - reference)
    metric['max_abs_error'] = max(metric['max_abs_error'], error)
    if error > metric['numerical_tolerance']:
        metric['differences_above_numerical_tolerance'] += 1
        if len(metric['examples']) < 5:
            metric['examples'].append(dict(line_id=line_id, audio_seconds=at, canonical=canonical, reference=reference))


def audit_chart(chart, restored):
    reference = PhiraSourceReference(chart.source)
    summary = dict(source_lines=len(chart.lines), source_notes=len(chart.notes), source_holds=sum(n['type'] == 3 for n in chart.notes),
                   duration_seconds=chart.duration, version=chart.version, offset=chart.offset,
                   provenance=chart.provenance, line_classification=dict(), line_queries=0,
                   current_state_unsupported=0, scroll_unsupported=0,
                   reference_unsupported_reasons={},
                   canonical_roundtrip_identical=True, diagnostics=chart.diagnostics(),
                   reference_comparison={key: comparison_metric() for key in ['x','y','rotation','alpha','speed','scroll']},
                   hold_queries=dict(holds=0, samples=0, geometric_unsupported=0, state_unsupported=0,
                                     scroll_unsupported=0,
                                     samples_in_audio=0,samples_outside_audio=0,effective_holds=0,
                                     in_audio_geometric_unsupported=0,in_audio_state_unsupported=0,in_audio_scroll_unsupported=0,
                                     in_audio_sampled_outside_viewport=0,
                                     sampled_outside_viewport=0, holds_with_sampled_outside=0, jumps=0,
                                     query_budget_failures=[], continuous_safety_proof=False),
                   source_floor_comparison=comparison_metric(1e-4), line_details=[])
    classes = Counter()
    reference_unsupported = Counter()
    for index, line in enumerate(chart.lines):
        classification = chart.classify_line(line['id']);classes[classification['classification']] += 1
        low, high = chart.native_tick(line, 0.), chart.native_tick(line, chart.duration)
        boundaries = sorted({low, high, *(tick for track in line['tracks'].values() for tick in track.boundaries(low, high))})
        points = {0., chart.duration, *(chart.offset + tick * line['seconds_per_tick'] for tick in boundaries),
                  *(chart.offset + (a+b)/2 * line['seconds_per_tick'] for a,b in zip(boundaries,boundaries[1:]))}
        for note in line['notes']:
            points.update(t for t in [note['time_seconds'],note['end_seconds']] if 0 <= t <= chart.duration)
        detail = dict(id=line['id'], bpm=line['bpm'], source_notes=len(line['notes']), classification=classification,
                      queries=len(points), current_unsupported=0, scroll_unsupported=0, unsupported_examples=[])
        for at in sorted(points):
            state = chart.line_state(line['id'],at)
            roundtrip = restored.line_state(line['id'],at)
            if state != roundtrip:
                raise ValueError('canonical serialization changed a line query')
            summary['line_queries'] += 1
            if not state['current_state_supported']:
                summary['current_state_unsupported'] += 1;detail['current_unsupported'] += 1
            if not state['scroll_supported']:
                summary['scroll_unsupported'] += 1;detail['scroll_unsupported'] += 1
            if not state['supported'] and len(detail['unsupported_examples']) < 3:
                detail['unsupported_examples'].append(dict(audio_seconds=at, tracks={k:v['status'] for k,v in state['tracks'].items()},
                                                          scroll_status=state['scroll_integral']['status']))
            other = reference.line_state(index,at)
            reference_unsupported.update(f'{kind}:{axis["status"]}' for kind,axes in other['tracks'].items()
                                         for axis in axes if axis['value'] is None)
            for axis,key in enumerate(['x','y']):
                compare(summary['reference_comparison'][key], None if state['position'] is None else state['position'][axis],
                        None if other['position'] is None else other['position'][axis], line['id'],at)
            for key,field in [('rotation','rotation_degrees'),('alpha','alpha_source'),('speed','speed_source')]:
                compare(summary['reference_comparison'][key],state[field],other[field],line['id'],at)
            compare(summary['reference_comparison']['scroll'],state['scroll_integral']['value'],other['scroll_source'],line['id'],at)
        summary['line_details'].append(detail)
        for note in line['notes']:
            if 0 <= note['time_seconds'] <= chart.duration:
                floor = chart.line_state(line['id'],note['time_seconds'])['scroll_integral']['value']
                compare(summary['source_floor_comparison'],floor,note['source_floor_position'],line['id'],note['time_seconds'])
            if note['type'] != 3:
                continue
            hold = summary['hold_queries'];hold['holds'] += 1
            hold['effective_holds'] += note['end_seconds'] >= 0 and note['time_seconds'] <= chart.duration
            try:
                trajectory = chart.sample_hold(note['id'])
                rebuilt = restored.sample_hold(note['id'])
                if trajectory != rebuilt:
                    raise ValueError('canonical serialization changed a Hold trajectory')
            except ValueError as error:
                if 'sample budget' not in str(error):raise
                hold['query_budget_failures'].append(dict(note_id=note['id'],error=str(error)))
                continue
            hold['jumps'] += len(trajectory['jumps'])
            outside = False
            for sample in trajectory['samples']:
                hold['samples'] += 1
                hold['geometric_unsupported'] += sample['target_world'] is None
                hold['state_unsupported'] += not sample['line_state']['supported']
                hold['scroll_unsupported'] += not sample['line_state']['scroll_supported']
                inside=sample['line_state']['within_audio']
                hold['samples_in_audio'] += inside
                hold['samples_outside_audio'] += not inside
                if inside:
                    hold['in_audio_geometric_unsupported'] += sample['target_world'] is None
                    hold['in_audio_state_unsupported'] += not sample['line_state']['supported']
                    hold['in_audio_scroll_unsupported'] += not sample['line_state']['scroll_supported']
                    hold['in_audio_sampled_outside_viewport'] += sample['in_viewport'] is False
                if sample['in_viewport'] is False:
                    outside = True;hold['sampled_outside_viewport'] += 1
            hold['holds_with_sampled_outside'] += outside
    summary['line_classification'] = dict(classes)
    summary['reference_unsupported_reasons'] = dict(reference_unsupported)
    summary['diagnostic_counts'] = dict(Counter(row['code'] for row in summary['diagnostics']))
    summary['effective_diagnostic_counts'] = dict(Counter(row['code'] for row in summary['diagnostics'] if row['intersects_audio']))
    summary['status'] = 'interpreted_with_explicit_support_masks'
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-list',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--reference-cache',type=Path,help='directory containing pinned audiolab-phira-{pgr,anim,object,tween,core}.rs')
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();budget(output)
    data_list=Path(args.data_list).resolve()
    selection,inspected=select_originals(data_list)
    write_json(output/'selection.json',selection)
    root=Path(__file__).resolve().parents[1]
    verified=verify_reference_cache(root,args.reference_cache) if args.reference_cache else []
    report=dict(schema='audiolab-performance-interpretation-audit-v1',status='running',
                implementation=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                                    working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),
                                    source_sha256={str(path.relative_to(root)):file_hash(path) for path in
                                                   [root/'src/data/performance.py',root/'src/data/performance_reference.py',Path(__file__).resolve()]}),
                selection=dict(policy='first-two-v1-four-v3-in-list-order',list_sha256=file_hash(data_list),
                               header_documents_inspected=inspected,charts_selected=len(selection),is_quality_population_sample=False),
                reference=dict(commit=PHIRA_COMMIT,registry_sha256=file_hash(root/'docs/phira_reference.json'),
                               adapter='float64 source-derived keyframe semantics; not bit-exact f32',native_player_executed=False,
                               original_source_verified=bool(verified),verified_files=verified),results=[])
    for index,item in enumerate(selection):
        phase=time.monotonic()
        try:
            chart=PerformanceChart.from_file(item['chart'],item['audio'],split=data_list.stem)
            document=chart.to_document();canonical_content_sha256=content_hash(document)
            serialized=json.dumps(document,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            compressed=gzip.compress(serialized,compresslevel=6,mtime=0)
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+len(compressed)>45*1024**2:
                raise RuntimeError('canonical outputs would exceed reserved 45 MiB share of audit budget')
            canonical=output/f'chart-{index}.canonical.json.gz';canonical.write_bytes(compressed)
            with gzip.open(canonical,'rt') as source:reloaded=json.load(source)
            restored=PerformanceChart.from_document(reloaded)
            del document,serialized,compressed,reloaded
            row=audit_chart(chart,restored)
            row.update(canonical_file=canonical.name,canonical_file_sha256=file_hash(canonical),
                       canonical_file_bytes=canonical.stat().st_size,canonical_content_sha256=canonical_content_sha256,
                       serialization_roundtrip='written-gzip-json-reloaded-from-file')
        except (OSError,ValueError,KeyError,RuntimeError) as error:
            row=dict(status='failed',chart=item['chart'],error=dict(type=type(error).__name__,message=str(error)))
        row.update(elapsed_seconds=time.monotonic()-phase,resources=budget(output))
        report['results'].append(row);write_json(output/'audit.in_progress.json',report)
        print(json.dumps(dict(completed=index,status=row['status'],elapsed_seconds=row['elapsed_seconds'])),flush=True)
    report.update(status='audit_completed',elapsed_seconds=time.monotonic()-started,resources=budget(output),
                  parsed_charts=sum(row['status']!='failed' for row in report['results']),
                  failed_charts=sum(row['status']=='failed' for row in report['results']),
                  interpretation='roundtrip correctness, source-reference differences and query support; no learned performance model or playability claim')
    write_json(output/'audit.json',report)


if __name__=='__main__':main()

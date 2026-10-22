"""CPU-only geometry of frozen products, with existing C2 audio controls linked by SHA."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np

from script.review_hold_repairs import digest, provenance, resources, verified_source
from src.conditioning_audit import load_reference, validate_report as validate_conditioning
from src.data.chart_tokenizer import ChartTokenizer, bin_to_hit
from src.pattern_diagnostics import (SCHEMA, LANE_X, KINDS, candidate_coverage, chart_events,
                                     compare, centered_residual, distribution, summarize)
from src.audio_understanding.inference import make_result
from src.product.pipeline import quantize_candidates


def read_json(path):
    return json.loads(Path(path).read_text())


def compact(summary):
    """All aggregates use all events; only illustrative lists are bounded."""
    result = deepcopy(summary)
    instances = result['patterns'].pop('instances')
    # Retain examples of every class, rather than whichever class occurs first.
    result['patterns']['examples'] = {label: [r for r in instances if r['label'] == label][:3]
                                     for label in ('AAAA', 'ABAB', 'monotonic_run', 'unclassified')}
    result['chord_count'] = len(result['chords'])
    result['chords'] = result['chords'][:12]
    result['hold_examples'] = sorted(result.pop('holds'), key=lambda r: len(r['nearby_later_onsets']), reverse=True)[:8]
    result['example_policy'] = 'bounded examples only; counts and distributions include the complete chart'
    return result


def compact_comparison(value):
    result = deepcopy(value)
    rows = result.pop('raw_pattern_instances')
    changed = [r for r in rows if r['geometry_before'] != r['geometry_after']]
    result.update(paired_pattern_instances=len(rows), geometry_changed_instances=len(changed),
                  changed_pattern_examples=changed[:12])
    return result


def reference_events(source, mode):
    meta = source['reference']
    prov = meta['provenance']
    for value in prov.values():
        if isinstance(value, dict) and 'path' in value:
            if digest(value['path']) != value['sha256']:
                raise ValueError('C2 reference source changed')
    reference = load_reference(prov['reference_chart']['path'], source['features']['duration_seconds'],
                               gameplay_path=prov['gameplay']['path'] if mode == 'freex' else None,
                               source_root=prov.get('relative_source_root'))
    if reference.metadata != meta:
        raise ValueError('reference codec/provenance differs from the frozen C2 report')
    tk = ChartTokenizer(4 if mode == '4k' else 64)
    tokens = reference.tokens.tolist()
    ticks = tk.ticks_per_position(tokens)
    offsets = {(tick, tk.value(token)//4): offset for tick, token, offset in
               zip(ticks, tokens, reference.offsets.tolist()) if tk.kind(token) == 'NOTE'}
    scale = 60/(32*meta['timebase_bpm'])
    result = []
    for index, note in enumerate(tk.decode_tokens(tokens, strict=True)):
        x = LANE_X[note.lane] if mode == '4k' else bin_to_hit(note.lane, offsets[(note.tick, note.lane)])
        result.append(dict(id=f'reference-{mode}-{index:06d}', source_index=index, tick=note.tick,
                           time_seconds=note.tick*scale+meta['offset_seconds'],
                           end_seconds=(note.tick+note.dur)*scale+meta['offset_seconds'],
                           screen_x=x, position=note.lane, type=KINDS[note.type]))
    return result, meta


def selection_replay(directory, evidence, expected_sha):
    """Read three small NPY members only, never the saved encoded features."""
    name = evidence.get('dense_file')
    if not name:
        return dict(status='unavailable', reason='no saved pre-threshold arrays')
    if Path(name).name != name:
        raise ValueError('unexpected dense artifact path')
    path = directory/name
    file = provenance(path)
    if file['sha256'] != expected_sha or file['sha256'] != evidence['dense_sha256']:
        raise ValueError('dense posterior SHA mismatch')
    names = ('event_probability', 'event_times_seconds', 'valid_event_time')
    with zipfile.ZipFile(path) as archive:
        if any(archive.getinfo(key+'.npy').file_size > 1024**2 for key in names):
            raise ValueError('posterior member exceeds 1 MiB diagnostic budget')
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in names}
    scores, times, valid = (arrays[key] for key in names)
    if scores.ndim != 1 or scores.shape != times.shape or scores.shape != valid.shape or valid.dtype != np.bool_ \
            or not np.isfinite(scores).all() or not np.isfinite(times).all() or np.any((scores < 0) | (scores > 1)) \
            or not np.array_equal(valid, times < evidence['duration_seconds']) or np.any(times[valid] < 0):
        raise ValueError('invalid saved posterior arrays')
    selected = make_result(arrays, evidence['duration_seconds'], dict(label_source=evidence['label_source']),
                           threshold=evidence['selection']['threshold'],
                           min_interval_ms=evidence['selection']['min_interval_ms'])
    if selected['events'] != evidence['events']:
        raise ValueError('saved selector does not reproduce selected events exactly')
    above = int(np.count_nonzero(scores[valid] >= evidence['selection']['threshold']))
    return dict(status='reproduced', source=file, loaded_members=list(names), frame_count=len(scores),
                valid_frames=int(valid.sum()), invalid_tail_frames=int((~valid).sum()),
                valid_score_distribution=distribution(scores[valid]),
                below_threshold_frames=int(valid.sum())-above, at_or_above_threshold_frames=above,
                selected_events=len(selected['events']), nms_or_duplicate_time_removed=above-len(selected['events']),
                interpretation='uncalibrated chart-onset frame scores; not weak-beat labels or acoustic recall')


def audit_product(directory, mode, conditioning):
    manifest, source = verified_source(directory, mode)
    matches = [r for r in conditioning['results'] if r['status'] == 'completed'
               and r['audio']['sha256'] == source['artifacts']['audio']['sha256']
               and r['model'] == ('decoder' if mode == '4k' else 'decoder_freex')]
    if len(matches) != 1:
        raise ValueError('product has no unique same-audio C2 reference')
    ref = matches[0]
    if manifest['model_sha256'][mode] != conditioning['checkpoints'][ref['model']]['sha256']:
        raise ValueError('product and audio-control weights differ')
    raw = chart_events(read_json(source['artifacts']['raw']['path']), mode)
    final = chart_events(read_json(source['artifacts']['final']['path']), mode)
    stored = read_json(source['artifacts']['events']['path'])['events']
    if len(final) != len(stored) or any(any(event[key] != frozen[key] for key in
            ('id', 'source_index', 'type', 'time_seconds', 'end_seconds', 'screen_x'))
            for event, frozen in zip(final, stored)):
        raise ValueError('computed events differ from the frozen product events')
    evidence = read_json(source['artifacts']['candidates']['path'])
    ticks, quantization = quantize_candidates(evidence, manifest['config']['bpm'], manifest['config']['offset'])
    if any(manifest['audio'][key] != value for key, value in quantization.items()):
        raise ValueError('candidate quantization differs from the frozen product')
    scores = [e['confidence'] for e in evidence['events']]
    posterior = selection_replay(directory, evidence, manifest['artifacts'][evidence['dense_file']]['sha256']) \
                if evidence.get('dense_file') else dict(status='unavailable', reason='no saved pre-threshold arrays')
    coverage = {phase: candidate_coverage(events, ticks, scores) for phase, events in [('raw', raw), ('final', final)]}
    row = dict(source=source, source_commit=manifest['implementation']['git_commit'],
               source_config=manifest['config'], model_sha256=manifest['model_sha256'][mode],
               source_sampling={key: manifest['routes'][mode][key] for key in
                                ('seed', 'temperature', 'top_p', 'sampling_policy', 'encoder')},
               duration_seconds=manifest['audio']['duration_seconds'],
               reference_key=dict(model=ref['model'], index=ref['input']['index']),
               candidate_selection=evidence['selection'], selected_events=len(scores), selection_replay=posterior,
               quantization=quantization, coverage=coverage,
               raw=compact(summarize(raw)), final=compact(summarize(final)),
               repair=compact_comparison(compare(raw, final)))
    if mode == 'freex':
        center = centered_residual(raw)
        row['residual_center_counterfactual'] = dict(
            scope='diagnostic geometry only; no export, sampling, repairs, or product-default change',
            summary=compact(summarize(center)), changes=compact_comparison(compare(raw, center)),
            maximum_possible_displacement_screen_width=.9/64/2,
            quality_interpretation='not an improvement score; geometry matching the chosen bin template is expected')
    if any(digest(v['path']) != v['sha256'] for v in [source['manifest'], *source['artifacts'].values()]):
        raise ValueError('source changed during diagnostics')
    return row, ref


def validate_report(report):
    if report['schema'] != SCHEMA or len(report['results']) != report['expected_rows']:
        raise ValueError('invalid report denominator/schema')
    count = sum(row['status'] == 'completed' for row in report['results'])
    if (report['completed_rows'], report['failed_rows']) != (count, report['expected_rows']-count):
        raise ValueError('invalid completion denominator')
    for row in report['results']:
        if row['status'] == 'failed':
            if not row.get('error'): raise ValueError('missing error')
            continue
        if row['repair']['matched_ids'] != row['raw']['note_count'] or row['raw']['note_count'] != row['final']['note_count']:
            raise ValueError('identity denominator mismatch')
        for phase in ('raw', 'final'):
            data = row[phase]; pattern = data['patterns']
            if sum(pattern['counts'].values()) != pattern['singleton_windows'] or \
                    pattern['singleton_windows']+pattern['chord_interrupted_windows'] != pattern['total_four_onset_windows']:
                raise ValueError('pattern denominator mismatch')
            if sum(int(size)*n for size, n in data['chord_size_counts'].items()) != data['note_count']:
                raise ValueError('chord denominator mismatch')
            coverage = row['coverage'][phase]
            if sum(coverage['notes_per_candidate'].values()) != coverage['unique_candidate_ticks']:
                raise ValueError('candidate denominator mismatch')


def write_report(output, report):
    """No completed marker survives failed readback or final resource checks."""
    validate_report(report)
    path = output/'report.json'
    try:
        for _ in range(4):
            path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
            restored = read_json(path); validate_report(restored)
            if restored != report: raise ValueError('report JSON readback mismatch')
            measured = resources(output)
            if measured['output_bytes'] == report['resources']['output_bytes']:
                return
            report['resources'] = measured
        raise ValueError('final report size did not converge')
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def run(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); resources(output)
    c2 = read_json(args.conditioning_report)
    if digest(args.conditioning_report) != args.conditioning_sha256:
        raise ValueError('conditioning report SHA differs from the specified frozen evidence')
    validate_conditioning(c2)
    root = Path(__file__).resolve().parents[1]
    report = dict(schema=SCHEMA, status='running', expected_rows=12, results=[], references=[],
                  implementation=dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
                    source_sha256={name: digest(root/name) for name in
                        ('src/pattern_diagnostics.py', 'script/diagnose_patterns.py', 'src/conditioning_audit.py',
                         'src/data/chart_tokenizer.py', 'src/product/pipeline.py', 'script/review_hold_repairs.py',
                         'src/audio_understanding/inference.py', 'src/models/rhythm_skeleton.py')}),
                  config=vars(args), product_default_changed=False, is_quality_holdout=False,
                  units=dict(time='audio_seconds', position='fraction_of_screen_width'),
                  interpretation='descriptive reused-development samples; no symmetry/quality score, acoustic truth or human playtest',
                  audio_conditioning=dict(source=provenance(args.conditioning_report),
                    scope='reuse of full-song teacher-forcing controls, not new generated-output interventions',
                    summaries=c2['model_summaries']))
    reference_keys = set()
    for group, location in [('manual', args.manual_root), ('automatic', args.automatic_root)]:
        for song in range(3):
            for mode in ('4k', 'freex'):
                resources(output)
                row = dict(group=group, song=f'song-{song}', mode=mode, status='running')
                try:
                    values, ref = audit_product(Path(location)/f'song-{song}', mode, c2)
                    key = (ref['model'], ref['input']['index'])
                    if key not in reference_keys:
                        events, metadata = reference_events(ref, mode)
                        report['references'].append(dict(model=key[0], index=key[1], mode=mode,
                                                          metadata=metadata, summary=compact(summarize(events))))
                        reference_keys.add(key)
                    row.update(values, status='completed')
                except Exception as error:
                    row.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
                report['results'].append(row)
    completed = sum(r['status'] == 'completed' for r in report['results'])
    report.update(status='completed' if completed == 12 else 'completed_with_failures',
                  completed_rows=completed, failed_rows=12-completed, elapsed_seconds=time.monotonic()-started,
                  resources=resources(output))
    write_report(output, report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manual-root', 'automatic-root', 'conditioning-report', 'conditioning-sha256', 'output'):
        p.add_argument('--'+name, required=True)
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    print(json.dumps({key: result[key] for key in ('status', 'completed_rows', 'failed_rows', 'elapsed_seconds', 'resources')}))
    raise SystemExit(0 if result['status'] == 'completed' else 1)

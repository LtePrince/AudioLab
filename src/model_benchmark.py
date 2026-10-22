"""Replay-only model benchmark. Never changes notes, selectors or decoding."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import statistics

from src.audio_understanding.annotations import contains, intervals, reviewed_regions
from src.audio_understanding.evaluation import evaluate_record, match_events
from src.pattern_diagnostics import BIN_WIDTH, LANE_X, patterns

SCHEMA = 'audiolab-benchmark-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def unique(values):
    values = [float(x) for x in values]
    if any(not math.isfinite(x) for x in values):
        raise ValueError('non-finite time')
    return sorted(set(round(x, 6) for x in values))


def rhythm_bonus(times, clock, profile):
    """Circular phase distance; 32 == 0. No negative points or quota."""
    if clock is None:
        return {'status': 'unavailable', 'reason': 'no declared beat clock'}
    bpm, origin = float(clock['bpm']), float(clock['beat_zero_seconds'])
    tolerance = float(profile['tolerance_ticks'])
    if (not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(origin)
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('invalid beat clock or rhythm tolerance')
    phases = sorted({float(t) % 32 for t in profile['phase_ticks']})
    if not phases or any(not math.isfinite(p) for p in phases):
        raise ValueError('invalid rhythm profile')
    rows, streak, longest = [], 0, 0
    times = unique(times)
    for time in times:
        phase = ((time-origin)*bpm/60*32) % 32
        distance, nearest = min((min(abs(phase-p), 32-abs(phase-p)), p) for p in phases)
        bonus = max(0., 1.-distance/tolerance)
        # Seconds are stored at microsecond precision, not rounded to a beat grid.
        if distance < 1e-4:
            bonus = 1.
        streak = streak+1 if bonus > 0 else 0
        longest = max(longest, streak)
        rows.append((nearest, bonus))
    points = sum(b for _, b in rows)
    gaps = Counter(round((b-a)*bpm/60*32, 3) for a, b in zip(times, times[1:]))
    return dict(status='measured', clock_source=clock['source'],
                clock_independently_verified=bool(clock.get('verified', False)),
                unique_onsets=len(rows), matched_onsets=sum(b > 0 for _, b in rows),
                bonus_points=points, bonus_per_onset=points/len(rows) if rows else None,
                matched_phase_counts={str(int(p)): sum(q == p and b > 0 for q, b in rows) for p in phases},
                most_common_ioi_ticks=[dict(ticks=k, count=v) for k, v in gaps.most_common(16)],
                longest_matched_run=longest, profile_phases=phases,
                interpretation='bonus only; nonmatches are not errors; no musical-quality pass threshold')


def pattern_metrics(notes):
    """Geometry of consecutive onsets, with chords breaking singleton chains."""
    groups = {}
    events = []
    for i, n in enumerate(notes):
        t = round(n['time_seconds'], 6)
        groups.setdefault(t, []).append(n)
        events.append(dict(n, tick=t, id=str(i), type=n['type']))
    measured = patterns(events)
    runs, current, previous = [], 0, None
    for t, group in sorted(groups.items()):
        pos = group[0]['position'] if len(group) == 1 else None
        if pos is not None and pos == previous:
            current += 1
        else:
            if current >= 2:
                runs.append(current)
            current = 1 if pos is not None else 0
        previous = pos
    if current >= 2:
        runs.append(current)
    # Short/long here denotes chain length, not a universal gameplay definition.
    hold_pairs = []
    for i, a in enumerate(notes):
        if a['type'] != 'hold':
            continue
        for j, b in enumerate(notes):
            if j <= i or b['type'] != 'hold' or a['position'] != b['position']:
                continue
            earlier, later = sorted((a, b), key=lambda x: x['time_seconds'])
            gap = later['time_seconds']-earlier['end_seconds']
            if gap < -1e-6 or abs(gap) <= 1e-6:
                hold_pairs.append(dict(first=i, second=j, gap_seconds=gap))
    return dict(note_count=len(notes), unique_onsets=len(groups),
                type_counts=dict(Counter(n['type'] for n in notes)),
                chord_sizes=dict(Counter(str(len(g)) for g in groups.values())),
                four_onset_patterns=measured['counts'],
                singleton_windows=measured['singleton_windows'],
                chord_interrupted_windows=measured['chord_interrupted_windows'],
                jack_runs_2_to_3=sum(2 <= n <= 3 for n in runs),
                jack_runs_4_plus=sum(n >= 4 for n in runs),
                longest_jack_run=max(runs, default=0),
                hold_overlap_pairs=sum(p['gap_seconds'] < -1e-6 for p in hold_pairs),
                hold_touching_pairs=sum(abs(p['gap_seconds']) <= 1e-6 for p in hold_pairs),
                hold_examples=hold_pairs[:20],
                interpretation='descriptive; no desired type/pattern/chord quota or two-finger cap')


def score_case(case, prediction, protocol, mode=None):
    duration = float(case['duration_seconds'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid duration')
    regions = reviewed_regions(case)
    if not regions:
        raise ValueError('explicit reviewed coverage required')
    reference = unique(case['events_seconds'])
    if any(not contains(t, case['coverage']) for t in reference):
        raise ValueError('reference outside annotated coverage')
    base = dict(id=case['id'], source_group=case['source_group'],
                label_source=case['label_source'], split=case['split'],
                case_sha256=fingerprint(case), gates={})
    if prediction is None or prediction.get('status') != 'completed':
        return dict(base, status='missing' if prediction is None else 'failed',
                    error=None if prediction is None else prediction.get('error'),
                    gates={'generation_completed': False})
    times = unique(prediction['onsets_seconds'])
    outside = [t for t in times if not 0 <= t < duration]
    valid = [t for t in times if 0 <= t < duration]
    metric = evaluate_record(case, valid)
    # Out-of-duration output cannot disappear from the precision denominator.
    for ms in protocol['event_tolerances_ms']:
        m = metric[f'at_{ms}ms']
        m['false_positives'] += len(outside)
        count = metric['prediction_count']+len(outside)
        m['precision'] = m['matched']/count if count else (0. if metric['reference_count'] else 1.)
        m['f1'] = 2*m['matched']/(count+metric['reference_count']) if count+metric['reference_count'] else 1.
        m['false_positives_per_minute'] = m['false_positives']*60/metric['evaluated_seconds']
    metric['prediction_count_including_outside'] = metric['prediction_count']+len(outside)
    n_pred = metric['prediction_count_including_outside']
    metric['timing_100ms']['unmatched_prediction_fraction'] = (
        (n_pred-metric['at_100ms']['matched'])/n_pred if n_pred else 0.)
    metric.pop('evaluation_data')
    reference = [t for t in reference if contains(t, regions)]
    reviewed_prediction = [t for t in valid if contains(t, regions)]
    notes = prediction.get('notes')
    if notes is not None and unique(n['time_seconds'] for n in notes) != times:
        raise ValueError('notes/onsets disagree')
    bad_notes = []
    for i, n in enumerate(notes or []):
        a, b, x = n['time_seconds'], n['end_seconds'], n['screen_x']
        pos = n['position']
        geometry_ok = True
        if mode in ('4k', 'freex'):
            geometry_ok = (isinstance(pos, int) and 0 <= pos < (4 if mode == '4k' else 64))
            if geometry_ok:
                geometry_ok = (abs(x-LANE_X[pos]) <= 1e-8 if mode == '4k' else
                               .05+pos*BIN_WIDTH-1e-8 <= x <= .05+(pos+1)*BIN_WIDTH+1e-8)
        if (not all(math.isfinite(v) for v in (a, b, x)) or not 0 <= a < duration
                or not a <= b <= duration or not .05 <= x <= .95
                or not geometry_ok
                or n['type'] not in ('tap', 'drag', 'hold', 'flick')
                or (n['type'] == 'hold' and b <= a) or (n['type'] != 'hold' and b != a)):
            bad_notes.append(i)
    base['gates'] = dict(generation_completed=True, onset_bounds=not outside)
    if notes is not None:
        base['gates']['note_bounds_and_types'] = not bad_notes
    if reference:
        base['gates']['nonempty_positive'] = bool(reviewed_prediction)
    if case.get('contract') == 'exact_onsets':
        m = metric[f'at_{protocol["contract_tolerance_ms"]}ms']
        base['gates']['isolated_positive_contract'] = m['false_negatives'] == m['false_positives'] == 0
    elif case.get('contract') == 'silence':
        base['gates']['silence_no_events'] = not times
    silent = intervals(case.get('silent_regions', []), duration)
    silent_count = sum(contains(t, silent) for t in valid)
    if silent:
        base['gates']['reviewed_silence_no_events'] = silent_count == 0
    clock = case.get('clock')
    rhythm = rhythm_bonus(reviewed_prediction, clock, protocol['rhythm'])
    if case.get('segments'):
        segments = case['segments']
        intervals([[r['start'], r['end']] for r in segments], duration)
    else:
        width = protocol['rhythm']['window_beats']*60/clock['bpm'] if clock else 8.
        segments = [dict(id=f'window-{i}', start=i*width, end=min((i+1)*width, duration))
                    for i in range(math.ceil(duration/width))]
    segment_rows = []
    for seg in segments:
        lo, hi = seg['start'], seg['end']
        ref = [t for t in reference if lo <= t < hi]
        pred = [t for t in reviewed_prediction if lo <= t < hi]
        row = dict(seg, reference_onsets=len(ref), predicted_onsets=len(pred),
                   rhythm=rhythm_bonus(pred, clock, protocol['rhythm']))
        if notes is not None:
            ns = [n for n in notes if lo <= n['time_seconds'] < hi]
            row['note_count'] = len(ns)
            row['types'] = dict(Counter(n['type'] for n in ns))
        segment_rows.append(row)
    pairs = match_events(reference, reviewed_prediction, .05)
    # Tail coverage uses selected reference events, not an assumption of notes at EOF.
    tail = reference[math.floor(.75*len(reference)):] if reference else []
    tail_matched = sum(a in tail for a, _ in pairs)
    weak = [t for t in unique(case.get('weak_events_seconds', [])) if contains(t, regions)]
    eos = prediction.get('eos_clock_seconds')
    result = dict(base, status='scored', events=metric,
                  out_of_audio_onsets=outside, invalid_note_indices=bad_notes,
                  silence=dict(annotated_seconds=sum(b-a for a, b in silent), events=silent_count),
                  weak_recall_50ms=(sum(a in weak for a, _ in pairs)/len(weak) if weak else None),
                  rhythm=rhythm, segments=segment_rows,
                  coverage=dict(reference_nonempty_segments=sum(r['reference_onsets'] > 0 for r in segment_rows),
                    empty_positive_segments=sum(r['reference_onsets'] > 0 and r['predicted_onsets'] == 0 for r in segment_rows),
                    last_quarter_reference_recall_50ms=tail_matched/len(tail) if tail else None,
                    first_reference_seconds=reference[0] if reference else None,
                    first_prediction_seconds=valid[0] if valid else None,
                    onset_count_ratio=len(reviewed_prediction)/len(reference) if reference else None,
                    eos_clock_seconds=eos, eos_overshoot_seconds=max(0., eos-duration) if eos is not None else None),
                  layout=pattern_metrics(notes) if notes is not None and not bad_notes else None)
    # Exact designated patterns only: no expectation that every song contains them.
    result['pattern_targets'] = []
    for target in case.get('pattern_targets', []):
        selected = [n for n in notes or [] if target['start'] <= n['time_seconds'] < target['end']]
        stats = pattern_metrics(selected) if selected else None
        count = (stats or {}).get('four_onset_patterns', {}).get(target['label'], 0)
        result['pattern_targets'].append(dict(target, matched_windows=count,
                                              status='measured' if notes is not None else 'unavailable'))
    result['contract_status'] = 'pass' if all(base['gates'].values()) else 'fail'
    return result


def score_run(run, protocol):
    cases, predictions = run['cases'], run['predictions']
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)) or set(predictions)-set(ids):
        raise ValueError('duplicate cases or unknown prediction IDs')
    groups = [(c['source_group'], c['label_source'], c['split']) for c in cases]
    if len(groups) != len(set(groups)):
        raise ValueError('one case per source/label/split; merge excerpts before song macro scoring')
    missing_probes = sorted(set(protocol['required_probes'])-set(ids))
    rows = [score_case(c, predictions.get(c['id']), protocol, run['metadata'].get('mode')) for c in cases]
    groups = {}
    for row in rows:
        groups.setdefault((row['label_source'], row['split']), []).append(row)
    summary = []
    for (labels, split), selected in groups.items():
        scores = [r for r in selected if r['status'] == 'scored']
        positive = [r for r in scores if r['events']['reference_count']]
        songs = {}
        for r in positive:
            songs.setdefault(r['source_group'], []).append(r)
        macro = statistics.fmean(statistics.fmean(r['events']['at_50ms']['f1'] for r in rs)
                                 for rs in songs.values()) if songs else None
        summary.append(dict(label_source=labels, split=split, expected_cases=len(selected),
                            scored_cases=len(scores), missing_or_failed=len(selected)-len(scores),
                            positive_f1_50ms_macro=macro,
                            positive_reference_onsets=sum(r['events']['reference_count'] for r in positive),
                            empty_positive_cases=sum(r['events']['prediction_count_including_outside'] == 0 for r in positive),
                            failed_contracts=sum(r.get('contract_status') == 'fail' for r in scores)))
    failures = sum(r.get('contract_status') == 'fail' or r['status'] == 'failed' for r in rows)
    incomplete = bool(missing_probes) or any(r['status'] == 'missing' for r in rows)
    return dict(id=run['id'], metadata=run['metadata'], cases_sha256=fingerprint(cases),
                predictions_sha256=fingerprint(predictions), missing_probe_definitions=missing_probes,
                contract_status='fail' if failures else ('incomplete' if incomplete else 'pass'),
                coverage_status='incomplete' if incomplete else 'complete',
                summary=summary, rows=rows)


def evaluate(bundle, protocol):
    if bundle['schema'] != SCHEMA or protocol['schema'] != SCHEMA or protocol['ticks_per_beat'] != 32:
        raise ValueError('unsupported benchmark schema/timebase')
    if len({r['id'] for r in bundle['runs']}) != len(bundle['runs']) or not bundle['runs']:
        raise ValueError('nonempty, unique run IDs required')
    return dict(schema=SCHEMA, protocol_sha256=fingerprint(protocol),
                input_sha256=fingerprint(bundle), runs=[score_run(r, protocol) for r in bundle['runs']],
                overall_quality='not_certified; no independent natural test or human playability score')

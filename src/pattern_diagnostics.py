"""Descriptive chart geometry, without a musical-quality score or repairs.

All positions use screen width; all elapsed times use audio seconds. Pattern
labels refer to discrete lanes/bins, not inferred human composition intent.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import statistics

BIN_WIDTH = .9 / 64
LANE_X = (.2890625, .4296875, .5703125, .7109375)
KINDS = {1: 'tap', 2: 'drag', 3: 'hold', 4: 'flick'}
SCHEMA = 'audiolab-pattern-diagnostics-v1'


def distribution(values):
    values = sorted(float(x) for x in values)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('non-finite diagnostic value')
    return dict(count=len(values), mean=statistics.fmean(values) if values else None,
                median=statistics.median(values) if values else None,
                min=values[0] if values else None, max=values[-1] if values else None)


def position_bin(x, mode):
    if mode == '4k':
        distances = [abs(x - value) for value in LANE_X]
        if min(distances) > 1e-8:
            raise ValueError('4K position is not a fixed lane')
        return distances.index(min(distances))
    if mode != 'freex' or not .05 - 1e-9 <= x < .95 + 1e-9:
        raise ValueError('unsupported free-x position')
    return min(63, max(0, int((x - .05) / BIN_WIDTH)))


def chart_events(chart, mode):
    if chart.get('formatVersion') != 3 or len(chart.get('judgeLineList', [])) != 1:
        raise ValueError('one static v3 product line required')
    line = chart['judgeLineList'][0]
    if line.get('notesBelow'):
        raise ValueError('product notesBelow must be empty')
    bpm, offset = float(line['bpm']), float(chart['offset'])
    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(offset):
        raise ValueError('invalid clock')
    scale = 60 / (32 * bpm)
    events = []
    for index, note in enumerate(line['notesAbove']):
        tick, length = float(note['time']), float(note['holdTime'])
        x = .5 + float(note['positionX']) * .05625
        if not all(math.isfinite(v) for v in (tick, length, x)) or length < 0:
            raise ValueError('invalid note')
        kind = KINDS[note['type']]
        if kind != 'hold' and length != 0:
            raise ValueError('non-Hold has duration')
        events.append(dict(id=f'{mode}-note-{index:06d}', source_index=index,
                           tick=tick, time_seconds=tick*scale+offset,
                           end_seconds=(tick+length)*scale+offset, screen_x=x,
                           position=position_bin(x, mode), type=kind))
    return events


def onset_groups(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event['tick']].append(event)
    return [sorted(grouped[key], key=lambda e: (e['screen_x'], e['id'])) for key in sorted(grouped)]


def label_pattern(positions):
    if len(set(positions)) == 1:
        return 'AAAA'
    if positions[0] == positions[2] and positions[1] == positions[3]:
        return 'ABAB'
    steps = [b-a for a, b in zip(positions, positions[1:])]
    if all(d > 0 for d in steps) or all(d < 0 for d in steps):
        return 'monotonic_run'
    return 'unclassified'


def geometry(events, label):
    xs = [e['screen_x'] for e in events]
    result = dict(step_screen_width=[b-a for a, b in zip(xs, xs[1:])])
    if label == 'AAAA':
        result['same_role_deviation_screen_width'] = max(xs)-min(xs)
    elif label == 'ABAB':
        result['same_role_deviation_screen_width'] = max(abs(xs[0]-xs[2]), abs(xs[1]-xs[3]))
    elif label == 'monotonic_run':
        steps = [abs(x) for x in result['step_screen_width']]
        result['step_std_screen_width'] = statistics.pstdev(steps)
        mean = statistics.fmean(steps)
        result['step_cv'] = statistics.pstdev(steps)/mean if mean else None
    return result


def patterns(events):
    groups = onset_groups(events)
    rows = []
    for start in range(max(0, len(groups)-3)):
        window = groups[start:start+4]
        if any(len(group) != 1 for group in window):
            continue  # Never skip a chord to manufacture a four-note pattern.
        notes = [group[0] for group in window]
        pos = [e['position'] for e in notes]
        ioi = [b['time_seconds']-a['time_seconds'] for a, b in zip(notes, notes[1:])]
        label = label_pattern(pos)
        rows.append(dict(ids=[e['id'] for e in notes], positions=pos,
                         types=[e['type'] for e in notes], label=label,
                         start_seconds=notes[0]['time_seconds'], end_seconds=notes[-1]['time_seconds'],
                         ioi_seconds=ioi, ioi_cv=statistics.pstdev(ioi)/statistics.fmean(ioi),
                         geometry=geometry(notes, label)))
    return dict(total_four_onset_windows=max(0, len(groups)-3), singleton_windows=len(rows),
                chord_interrupted_windows=max(0, len(groups)-3)-len(rows),
                instances=rows, counts=dict(Counter(r['label'] for r in rows)))


def summarize(events):
    groups = onset_groups(events)
    pattern = patterns(events)
    chords, holds = [], []
    for group in groups:
        if len(group) < 2:
            continue
        xs = [e['screen_x'] for e in group]
        gaps = [b-a for a, b in zip(xs, xs[1:])]
        # Uniform spacing is a measurement here, not an intended target.
        chords.append(dict(ids=[e['id'] for e in group], time_seconds=group[0]['time_seconds'],
                           size=len(group), span_screen_width=xs[-1]-xs[0],
                           gaps_screen_width=gaps,
                           gap_std_screen_width=statistics.pstdev(gaps) if len(group) >= 3 else None))
    for hold in events:
        if hold['type'] != 'hold':
            continue
        nearby = [dict(id=e['id'], type=e['type'], distance_screen_width=abs(e['screen_x']-hold['screen_x']),
                       time_seconds=e['time_seconds']) for e in events
                  if hold['time_seconds'] < e['time_seconds'] < hold['end_seconds']
                  and abs(e['screen_x']-hold['screen_x']) < .04-1e-9]
        holds.append(dict(id=hold['id'], duration_seconds=hold['end_seconds']-hold['time_seconds'],
                          nearby_later_onsets=nearby))
    # Exact discrete lane/bin+type sequences only; no claim of phrase recognition.
    motifs = Counter(tuple(zip(row['positions'], row['types'])) for row in pattern['instances'])
    repeated = [dict(sequence=[list(p) for p in key], windows=count) for key, count in motifs.most_common()
                if count > 1]
    metrics = {}
    for label in ('AAAA', 'ABAB', 'monotonic_run', 'unclassified'):
        selected = [r for r in pattern['instances'] if r['label'] == label]
        metrics[label] = dict(count=len(selected), ioi_cv=distribution(r['ioi_cv'] for r in selected),
                             geometry={field: distribution(r['geometry'][field] for r in selected
                                                          if field in r['geometry']) for field in
                                       ('same_role_deviation_screen_width', 'step_std_screen_width', 'step_cv')})
    return dict(note_count=len(events), onset_count=len(groups),
                type_counts=dict(Counter(e['type'] for e in events)),
                chord_size_counts=dict(Counter(str(len(g)) for g in groups)),
                onset_position_combinations=dict(Counter(','.join(str(e['position']) for e in g) for g in groups)),
                patterns=pattern, pattern_metrics=metrics,
                repeated_discrete_four_note_motifs=dict(distinct=len(motifs), repeated_distinct=len(repeated),
                    repeated_windows=sum(r['windows'] for r in repeated), examples=repeated[:12],
                    overlapping_windows=True, rhythm_in_signature=False),
                chords=chords,
                chord_geometry=dict(count=len(chords),
                    span_screen_width=distribution(c['span_screen_width'] for c in chords),
                    sorted_gap_screen_width=distribution(g for c in chords for g in c['gaps_screen_width']),
                    three_or_more_gap_std_screen_width=distribution(c['gap_std_screen_width'] for c in chords
                                                                    if c['gap_std_screen_width'] is not None)),
                holds=holds,
                hold_summary=dict(count=len(holds), duration_seconds=sum(h['duration_seconds'] for h in holds),
                                  with_nearby_later_onsets=sum(bool(h['nearby_later_onsets']) for h in holds),
                                  nearby_later_pairs=sum(len(h['nearby_later_onsets']) for h in holds)))


def compare(raw, final):
    if len(raw) != len(final) or len({e['id'] for e in raw}) != len(raw):
        raise ValueError('note count/identity mismatch')
    by_id = {e['id']: e for e in final}
    if set(by_id) != {e['id'] for e in raw}:
        raise ValueError('note identity mismatch')
    changes = []
    for before in raw:
        after = by_id[before['id']]
        if before['time_seconds'] != after['time_seconds'] or before['tick'] != after['tick']:
            raise ValueError('repair changed onset')
        dx = after['screen_x']-before['screen_x']
        dt = after['end_seconds']-before['end_seconds']
        if dx or dt or before['type'] != after['type']:
            changes.append(dict(id=before['id'], position_delta_screen_width=dx,
                                end_delta_seconds=dt, type_before=before['type'], type_after=after['type']))
    matched = []
    for row in patterns(raw)['instances']:
        after = [by_id[identity] for identity in row['ids']]
        matched.append(dict(ids=row['ids'], label_before=row['label'],
                            label_after=label_pattern([e['position'] for e in after]),
                            geometry_before=row['geometry'], geometry_after=geometry(after, row['label'])))
    chord_changes = []
    chord_count = 0
    for group in onset_groups(raw):
        if len(group) < 2:
            continue
        chord_count += 1
        # Preserve original ID order: repair may cross bins or reorder positions.
        xs = [e['screen_x'] for e in group]
        ys = [by_id[e['id']]['screen_x'] for e in group]
        before_gaps = [b-a for a, b in zip(xs, xs[1:])]
        after_gaps = [b-a for a, b in zip(ys, ys[1:])]
        if before_gaps != after_gaps:
            chord_changes.append(dict(ids=[e['id'] for e in group], gaps_before=before_gaps, gaps_after=after_gaps,
                                      same_id_order=True))
    return dict(note_count=len(raw), matched_ids=len(by_id), changes=changes,
                moved_notes=sum(r['position_delta_screen_width'] != 0 for r in changes),
                shortened_notes=sum(r['end_delta_seconds'] < 0 for r in changes),
                position_displacement=distribution(abs(r['position_delta_screen_width']) for r in changes
                                                  if r['position_delta_screen_width'] != 0),
                removed_duration_seconds=-sum(min(0, r['end_delta_seconds']) for r in changes),
                raw_pattern_instances=matched, paired_chords=chord_count,
                changed_chord_geometry=chord_changes,
                pattern_class_changes=sum(r['label_before'] != r['label_after'] for r in matched))


def centered_residual(events):
    """Counterfactual measurements only. No export, repair, or token resampling."""
    result = deepcopy(events)
    for event in result:
        event['screen_x'] = .05 + (event['position']+.5)*BIN_WIDTH
    return result


def candidate_coverage(events, candidate_ticks, candidate_scores):
    counts = Counter(e['tick'] for e in events)
    expected = set(candidate_ticks)
    return dict(unique_candidate_ticks=len(expected), unique_output_ticks=len(counts),
                missing_ticks=sorted(expected-counts.keys()), extra_ticks=sorted(counts.keys()-expected),
                notes_per_candidate=dict(Counter(str(counts[t]) for t in expected)),
                selected_confidence=distribution(candidate_scores),
                confidence_semantics='uncalibrated model score; not acoustic recall',
                pre_threshold_candidates='see row selection_replay; coverage itself uses selected events only')

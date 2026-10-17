"""Unique onset protocol: maximum cardinality, then minimum absolute error."""
from __future__ import annotations

import bisect
import math

import numpy as np

from .annotations import contains, reviewed_regions, intervals

PROTOCOL = 'unique-us-max-cardinality-min-absolute-error-v1'


def match_events(reference, prediction, tolerance):
    """Sparse monotone assignment via a prefix-best tree (O(E log P)).

    Absolute distance on a line admits an optimal non-crossing assignment.
    Batch updates per reference prevent matching one reference multiple times.
    Prefix queries exclude the current prediction, preventing prediction reuse.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    if any(not math.isfinite(t) for t in [*reference, *prediction]):
        raise ValueError('event times must be finite')
    ref, pred = sorted(set(round(t, 6) for t in reference)), sorted(set(round(t, 6) for t in prediction))
    # (match count, negative total error, previous node, ref index, pred index)
    nodes = [(0, 0., 0, -1, -1)]
    tree = [0] * (len(pred) + 1)

    def better(a, b):
        return a if nodes[a][:2] > nodes[b][:2] else b

    def query(end):
        best = 0
        while end:
            best = better(tree[end], best)
            end -= end & -end
        return best

    for i, time in enumerate(ref):
        updates = []
        lo = bisect.bisect_left(pred, time-tolerance-1e-12)
        hi = bisect.bisect_right(pred, time+tolerance+1e-12)
        for j in range(lo, hi):
            parent = query(j)
            nodes.append((nodes[parent][0]+1, nodes[parent][1]-abs(pred[j]-time), parent, i, j))
            updates.append((j+1, len(nodes)-1))
        for index, node in updates:
            while index < len(tree):
                tree[index] = better(node, tree[index])
                index += index & -index
    node, pairs = query(len(pred)), []
    while node:
        _, _, parent, i, j = nodes[node]
        pairs.append((ref[i], pred[j]))
        node = parent
    return pairs[::-1]


def evaluate_record(record, predictions):
    regions = reviewed_regions(record)
    duration = sum(b-a for a, b in regions)
    pred = sorted(set(round(t, 6) for t in predictions))
    if any(not math.isfinite(t) or not 0 <= t < record['duration_seconds'] for t in pred):
        raise ValueError('prediction outside audio duration')
    ref = [t for t in record['events_seconds'] if contains(t, regions)]
    ignored = sum(not contains(t, regions) for t in pred)
    pred = [t for t in pred if contains(t, regions)]
    scores = dict(id=record['id'], source_group=record['source_group'], label_source=record['label_source'],
                  reference_count=len(ref), prediction_count=len(pred), ignored_predictions=ignored,
                  evaluated_seconds=duration)
    shift = record.get('source_start_seconds', 0.)
    scores['evaluation_data'] = dict(regions=[[a+shift, b+shift] for a, b in regions],
                                    reference=[t+shift for t in ref], prediction=[t+shift for t in pred])
    for ms in (25, 50, 100):
        pairs = [pair for region in regions
                 for pair in match_events([t for t in ref if contains(t, [region])],
                                          [t for t in pred if contains(t, [region])], ms/1000)]
        tp = len(pairs)
        precision = tp / len(pred) if pred else (1. if not ref else 0.)
        recall = tp / len(ref) if ref else 1.
        scores[f'at_{ms}ms'] = dict(precision=precision, recall=recall,
            f1=2*tp/(len(ref)+len(pred)) if ref or pred else 1., matched=tp,
            false_positives=len(pred)-tp, false_negatives=len(ref)-tp,
            false_positives_per_minute=(len(pred)-tp)*60/duration)
        if ms == 100:
            errors = np.array([p-r for r, p in pairs])*1000
            scores['timing_100ms'] = dict(
                signed_mean_ms=float(errors.mean()) if tp else None,
                median_absolute_ms=float(np.median(abs(errors))) if tp else None,
                p95_absolute_ms=float(np.percentile(abs(errors), 95)) if tp else None,
                unmatched_reference_fraction=(len(ref)-tp)/len(ref) if ref else 0.,
                unmatched_prediction_fraction=(len(pred)-tp)/len(pred) if pred else 0.)
    return scores


def aggregate(rows, seed=42, bootstrap_samples=2000):
    if not rows or len({r['label_source'] for r in rows}) != 1:
        raise ValueError('evaluate acoustic, pseudo and chart labels in separate reports')
    # Merge disjoint excerpts on the source timeline BEFORE computing song F1.
    # Reject overlapping coverage: repeated difficulties/annotations must not
    # increase a song's weight. Contiguous excerpts also match across the seam.
    groups = sorted({r['source_group'] for r in rows})
    songs = []
    for group in groups:
        parts = [r['evaluation_data'] for r in rows if r['source_group'] == group]
        regions = [region for part in parts for region in part['regions']]
        duration = max(b for _, b in regions)
        regions = intervals(regions, duration)
        song = dict(id=group, source_group=group, label_source=rows[0]['label_source'],
                    duration_seconds=duration, coverage=regions,
                    events_seconds=[t for part in parts for t in part['reference']])
        songs.append(evaluate_record(song, [t for part in parts for t in part['prediction']]))
    rng = np.random.default_rng(seed)
    result = dict(protocol=PROTOCOL, label_source=rows[0]['label_source'],
                  song_count=len(groups), record_count=len(rows), seed=seed, per_record=rows,
                  bootstrap_samples=bootstrap_samples, per_song=songs, macro_by_song={})
    for ms in (25, 50):
        for metric in ('precision', 'recall', 'f1', 'false_positives_per_minute'):
            values = np.array([r[f'at_{ms}ms'][metric] for r in songs])
            samples = [rng.choice(values, len(values), replace=True).mean() for _ in range(bootstrap_samples)]
            result['macro_by_song'][f'{metric}_{ms}ms'] = dict(mean=float(values.mean()),
                ci95=[float(v) for v in np.percentile(samples, [2.5, 97.5])] if len(groups)>1 else None)
    return result

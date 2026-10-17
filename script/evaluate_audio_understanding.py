"""Evaluate frozen audio candidates against one explicit annotation domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audio_understanding import SCHEMA
from src.audio_understanding.annotations import read_manifest
from src.audio_understanding.evaluation import evaluate_record, aggregate
from src.audio_understanding.inference import sha256, result_to_ticks


def evaluate(manifest, predictions, output):
    records = read_manifest(manifest)
    if len({r['label_source'] for r in records}) != 1:
        raise ValueError('annotation domains must be evaluated separately')
    listing = Path(predictions)
    # Explicit id -> path mapping avoids unsafe IDs as filenames and silent skips.
    mapping = json.loads(listing.read_text())
    if set(mapping) != {r['id'] for r in records}:
        raise ValueError('prediction index must cover exactly every annotation id')
    if Path(output).exists():
        raise ValueError('report already exists; use a fresh output')
    rows, fingerprints, settings = [], {}, []
    for record in records:
        path = listing.parent / mapping[record['id']]
        result = json.loads(path.read_text())
        if result.get('schema') != SCHEMA:
            raise ValueError('incompatible prediction schema')
        result_to_ticks(result, 120)  # validate the versioned absolute time contract
        if result.get('audio_sha256') != sha256(record['audio']):
            raise ValueError('prediction is for different audio bytes')
        if abs(result['duration_seconds']-record['duration_seconds']) > 1/22050 + 1e-6:
            raise ValueError('prediction/annotation durations differ')
        settings.append({k: result[k] for k in ('checkpoint_sha256', 'label_source', 'selection', 'inference')})
        if settings[-1] != settings[0]:
            raise ValueError('all songs require frozen weights, thresholds and inference settings')
        rows.append(evaluate_record(record, [e['time_seconds'] for e in result['events']]))
        fingerprints[record['id']] = dict(result_sha256=sha256(path), audio_sha256=result['audio_sha256'])
    report = aggregate(rows)
    report.update(manifest_sha256=sha256(manifest), predictions=fingerprints,
                  frozen_settings=settings[0], acceptance='not_assessed',
                  note='Development protocol; candidate product thresholds are not approved acceptance criteria.')
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--predictions', required=True, help='JSON object: annotation id -> result JSON path')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = evaluate(args.manifest, args.predictions, args.output)
    print(json.dumps(report['macro_by_song'], indent=2))


if __name__ == '__main__':
    main()

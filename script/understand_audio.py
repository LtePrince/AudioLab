"""Audio -> shared features, dense event probabilities and selected candidates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from src.audio_understanding.inference import load_audio, load_model, infer_mel, make_result, sha256, code_version
from src.models.rhythm_skeleton import select_onsets
from src.resource_budget import ResourceBudget, add_resource_args


def analyze(audio, checkpoint, output, *, device='cpu', threshold=.7, min_interval_ms=35,
            chunk_frames=1024, context_frames=128, max_seconds=600, budget=None, loaded=None):
    select_onsets([], [], threshold, min_interval_ms)
    output = Path(output)
    dense = output.with_suffix('.npz')
    if output.suffix != '.json' or output.exists() or dense.exists():
        raise ValueError('use a fresh .json output path (and matching .npz)')
    start = time.monotonic()
    if budget:
        budget.check(force=True)
    model, metadata = loaded or load_model(checkpoint, device)
    mel, duration = load_audio(audio, max_seconds, budget=budget)
    arrays = infer_mel(model, mel, duration, chunk_frames=chunk_frames, context_frames=context_frames,
                       check=budget.check if budget else None)
    result = make_result(arrays, duration, metadata, threshold=threshold, min_interval_ms=min_interval_ms)
    result.update(audio_sha256=sha256(audio), audio_name=Path(audio).name, implementation=code_version(),
                  inference=dict(chunk_frames=chunk_frames, context_frames=context_frames,
                                 max_seconds=max_seconds, device=str(device),
                                 version='disjoint-core-with-context-v1'),
                  dense_file=dense.name, feature_shape=list(arrays['features'].shape))
    output.parent.mkdir(parents=True, exist_ok=True)
    if budget:
        budget.check(force=True, extra_bytes=sum(a.nbytes for a in arrays.values())+1024*1024)
    temporary = dense.with_suffix('.npz.tmp')
    json_tmp = output.with_suffix('.json.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        result['dense_sha256'] = sha256(temporary)
        result['elapsed_seconds'] = time.monotonic()-start
        if budget:
            result['resources'] = budget.log_state()
        json_tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(temporary, dense)
        os.replace(json_tmp, output)  # JSON is the completion marker.
    finally:
        temporary.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, default=.7)
    p.add_argument('--min-interval-ms', type=float, default=35)
    p.add_argument('--chunk-frames', type=int, default=1024)
    p.add_argument('--context-frames', type=int, default=128)
    p.add_argument('--max-seconds', type=float, default=600)
    add_resource_args(p)
    args = p.parse_args()
    directory = Path(args.output).parent
    directory.mkdir(parents=True, exist_ok=True)
    budget = ResourceBudget(args, torch.device(args.device), directory)
    result = analyze(args.audio, args.checkpoint, args.output, device=args.device,
                     threshold=args.threshold, min_interval_ms=args.min_interval_ms,
                     chunk_frames=args.chunk_frames, context_frames=args.context_frames,
                     max_seconds=args.max_seconds, budget=budget)
    print(json.dumps(dict(output=args.output, events=len(result['events']),
                         label_source=result['label_source'], elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()

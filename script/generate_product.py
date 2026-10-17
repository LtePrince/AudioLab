"""Generate baseline charts and an offline audio-synchronised visual preview."""
from __future__ import annotations

import argparse
import json
import signal

import torch

from src.product.pipeline import JobCancelled, run_job
from src.resource_budget import add_resource_args


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', required=True, help='WAV or OGG input')
    p.add_argument('--bpm', required=True, type=float, help='known BPM; tempo inference is not implemented')
    p.add_argument('--mode', choices=['4k', 'freex', 'both'], default='both')
    p.add_argument('--output', required=True, help='new job directory; existing directories are never overwritten')
    p.add_argument('--model-root', default='checkpoints/timing_v1')
    p.add_argument('--audio-checkpoint', help='optional shared event checkpoint; defaults to model-root transcriber')
    p.add_argument('--device', default='cpu')
    p.add_argument('--offset', type=float, default=0, help='nonnegative chart delay in audio seconds')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-seconds', type=float, default=600)
    p.add_argument('--timeout-seconds', type=float, default=900, help='engineering timeout, not a speed promise')
    p.add_argument('--max-decoder-steps', type=int, default=12000)
    p.add_argument('--cancel-file', help='touch this path to cancel; defaults to OUTPUT/CANCEL')
    add_resource_args(p)
    p.set_defaults(max_rss_gib=4)
    return p


def main():
    args = parser().parse_args()
    torch.set_num_threads(2)
    def cancel(_signal, _frame):
        raise JobCancelled('termination requested')
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        report = run_job(args)
        print(json.dumps(dict(status=report['status'], output=args.output,
                              elapsed_seconds=report['elapsed_seconds'],
                              notes={mode: data['note_count'] for mode, data in report['routes'].items()})))
    except (KeyboardInterrupt, JobCancelled):
        raise SystemExit(130)
    except Exception as error:
        print(json.dumps(dict(status='failed', error=str(error)), ensure_ascii=False))
        raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()

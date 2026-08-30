"""eval_onset_f1.py — Chart-quality evaluation against reference charts.

Given a directory of generated charts named ``<song_dir>.json`` and a data
list, computes per-song:
  - time-only onset F1 @ ±tol (default 92.9 ms = 2 chart frames)
  - the same F1 for time-SHIFTED copies of the generated chart (its own
    chance-level control: structure/density preserved, alignment destroyed)
  - lane-aware F1
  - note-type distribution (generated vs reference)

Reports means, the uplift over the shifted control and the per-song win rate
(fraction of songs whose aligned F1 beats their own shifted control).

Usage
-----
    uv run python script/eval_onset_f1.py --gen-dir out/eval_tsc --list data/val.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


X_UNIT_W = 0.05625   # positionX unit in screen widths (line at x=0.5)


def onsets(path: str, shift_ms: float = 0.0) -> list[tuple[float, int, int, float]]:
    """(time_ms, lane, type, screen_x) per note; screen_x = 0.5 + positionX·0.05625."""
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for l in d["judgeLineList"]:
        bpm = float(l["bpm"])
        for n in l["notesAbove"] + l["notesBelow"]:
            t = n["time"] / 32.0 * 60000.0 / bpm + shift_ms
            lane = min(max(round((n["positionX"] + 3.75) / 2.5), 0), 3)
            out.append((t, lane, int(n["type"]), 0.5 + n["positionX"] * X_UNIT_W))
    return sorted(out)


def f1(gen, ref, tol: float, lane_aware: bool, pos_tol: float | None = None) -> float:
    """Greedy one-to-one onset matching within ±tol ms.
    lane_aware: lanes must agree (4k).  pos_tol: |Δscreen_x| ≤ pos_tol (free-x)."""
    import bisect
    ref_t = [r[0] for r in ref]
    used = [False] * len(ref)
    tp = 0
    for g in gen:
        lo = bisect.bisect_left(ref_t, g[0] - tol)
        hi = bisect.bisect_right(ref_t, g[0] + tol)
        best, bd = -1, tol + 1
        for i in range(lo, hi):
            if used[i] or (lane_aware and ref[i][1] != g[1]):
                continue
            if pos_tol is not None and abs(ref[i][3] - g[3]) > pos_tol:
                continue
            d = abs(ref[i][0] - g[0])
            if d < bd:
                best, bd = i, d
        if best >= 0:
            used[best] = True
            tp += 1
    p = tp / max(len(gen), 1)
    r = tp / max(len(ref), 1)
    return 2 * p * r / max(p + r, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description="Onset-F1 evaluation vs reference charts.")
    ap.add_argument("--gen-dir", required=True, help="dir of <song_dir>.json generated charts")
    ap.add_argument("--list", default="data/val.txt")
    ap.add_argument("--tol-ms", type=float, default=92.9)
    ap.add_argument("--shifts", default="500,-400", help="control shifts (ms)")
    ap.add_argument("--ref-dir", default=None,
                    help="use <song_dir>.json under this dir as references instead of the "
                         "list's chart files (free-x: data/gameplay_charts)")
    ap.add_argument("--pos-tol", type=float, default=0.05,
                    help="position tolerance (screen fraction) for pos-F1")
    args = ap.parse_args()

    base = Path(args.list).resolve().parent
    shifts = [float(s) for s in args.shifts.split(",")]
    rows, types_g, types_r = [], np.zeros(5), np.zeros(5)
    missing = 0
    for line in open(args.list):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel = line.split(",")[0]
        song = Path(rel).parent.name
        gen_path = Path(args.gen_dir) / f"{song}.json"
        if not gen_path.exists():
            missing += 1
            continue
        ref_path = (Path(args.ref_dir) / f"{song}.json") if args.ref_dir else (base / rel)
        if not ref_path.exists():
            missing += 1
            continue
        ref = onsets(str(ref_path)); gen = onsets(str(gen_path))
        aligned = f1(gen, ref, args.tol_ms, False)
        control = float(np.mean([f1(onsets(str(gen_path), s), ref, args.tol_ms, False)
                                 for s in shifts]))
        lane = f1(gen, ref, args.tol_ms, True)
        pos  = f1(gen, ref, args.tol_ms, False, pos_tol=args.pos_tol)
        rows.append((aligned, control, lane, pos))
        for _, _, t, _ in gen: types_g[t] += 1
        for _, _, t, _ in ref: types_r[t] += 1

    if not rows:
        print("[ERROR] no generated charts matched the list", file=sys.stderr); sys.exit(1)
    R = np.array(rows)
    up = (R[:, 0].mean() - R[:, 1].mean()) / max(R[:, 1].mean(), 1e-9)
    print(f"songs={len(rows)} (missing {missing})  tol=±{args.tol_ms:.0f}ms")
    print(f"time-F1   {R[:,0].mean():.3f}   shifted-control {R[:,1].mean():.3f}   "
          f"uplift {100*up:+.0f}%   win-rate {100*(R[:,0] > R[:,1]).mean():.0f}%")
    print(f"lane-F1   {R[:,2].mean():.3f}   pos-F1(±{args.pos_tol:.2f} screen) {R[:,3].mean():.3f}")
    names = ["", "Tap", "Drag", "Hold", "Flick"]
    def pct(v): return "  ".join(f"{names[i]}={100*v[i]/max(v.sum(),1):4.1f}%" for i in range(1, 5))
    print(f"types gen  {pct(types_g)}   (n={int(types_g.sum())})")
    print(f"types ref  {pct(types_r)}   (n={int(types_r.sum())})")


if __name__ == "__main__":
    main()

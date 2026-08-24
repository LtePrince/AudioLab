"""plot_chart.py — Render a Phigros chart as a piano-roll image (P7-1).

Time runs left→right, folded into rows (like sheet-music lines); each row
shows the 4 lanes bottom-to-top.  Note types are colour-coded, Holds are
drawn as bars spanning their duration, and beat gridlines (from the chart's
BPM) make rhythm alignment visible at a glance.

With ``--ref`` a second chart is drawn INSIDE each lane band: reference in
the upper half, the main chart in the lower half — generated-vs-real
alignment can be judged lane by lane without playing anything.

Usage
-----
    uv run python script/plot_chart.py gen.json -o gen.png
    uv run python script/plot_chart.py gen.json --ref real.json -o cmp.png
    uv run python script/plot_chart.py gen.json --row-seconds 15 --start 30 --end 90
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# CJK song titles: fall back to a system font that has the glyphs
from matplotlib import font_manager as _fm
_CJK = [f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback")
        if any(f == x.name for x in _fm.fontManager.ttflist)]
if _CJK:
    plt.rcParams["font.family"] = ["DejaVu Sans", *_CJK]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", *_CJK]

# ── palette (dark, game-flavoured) ──────────────────────────────────────────
BG      = "#14171c"
LANE_BG = ("#1a1e24", "#20242b")           # alternating lane bands
GRID_B  = "#2e3540"                        # beat line
GRID_M  = "#465064"                        # measure (4-beat) line
TYPE_STYLE = {
    1: dict(color="#4fc3f7", label="Tap"),
    2: dict(color="#ffd54f", label="Drag"),
    3: dict(color="#81c784", label="Hold"),
    4: dict(color="#f06292", label="Flick"),
}
NUM_LANES = 4


def load_notes(path: str) -> tuple[list[dict], float, float]:
    """Return (notes, bpm, duration_s); note = {t, dur, lane, type} in seconds."""
    d = json.load(open(path, encoding="utf-8"))
    bpm = float(d["judgeLineList"][0]["bpm"])
    out = []
    for line in d["judgeLineList"]:
        lbpm = float(line["bpm"])
        for n in line["notesAbove"] + line["notesBelow"]:
            t   = n["time"] / 32.0 * 60.0 / lbpm
            dur = n["holdTime"] / 32.0 * 60.0 / lbpm
            lane = min(max(round((n["positionX"] + 3.75) / 2.5), 0), NUM_LANES - 1)
            out.append({"t": t, "dur": dur, "lane": lane, "type": int(n["type"])})
    dur_s = max((n["t"] + n["dur"] for n in out), default=1.0)
    return out, bpm, dur_s


def _draw_note(ax, n: dict, y0: float, h: float, alpha: float = 1.0) -> None:
    """Draw one note inside the vertical band [y0, y0+h] at its time."""
    st = TYPE_STYLE[n["type"]]
    yc = y0 + h / 2
    if n["type"] == 3 and n["dur"] > 0:                    # Hold: body bar + head
        ax.add_patch(Rectangle((n["t"], y0 + h * 0.18), n["dur"], h * 0.64,
                               facecolor=st["color"], alpha=0.45 * alpha,
                               edgecolor=st["color"], linewidth=0.6, zorder=3))
        ax.plot([n["t"]], [yc], marker="s", ms=3.4, color=st["color"],
                alpha=alpha, zorder=4)
    else:
        marker = {1: "s", 2: "D", 4: "^"}[n["type"]] if n["type"] != 3 else "s"
        ms     = {1: 3.8, 2: 2.9, 4: 4.2}.get(n["type"], 3.4)
        ax.plot([n["t"]], [yc], marker=marker, ms=ms, color=st["color"],
                alpha=alpha, linestyle="none", zorder=4)


def plot_chart(
    main_path:   str,
    ref_path:    str | None,
    out_path:    str,
    row_seconds: float = 20.0,
    start:       float = 0.0,
    end:         float | None = None,
    title:       str | None = None,
) -> None:
    notes, bpm, dur = load_notes(main_path)
    ref_notes = None
    if ref_path:
        ref_notes, ref_bpm, ref_dur = load_notes(ref_path)
        dur = max(dur, ref_dur)
    end  = min(end, dur) if end else dur
    span = end - start
    rows = max(1, math.ceil(span / row_seconds))

    fig_h = rows * 1.55 + 0.9
    fig, axes = plt.subplots(rows, 1, figsize=(15, fig_h), squeeze=False)
    fig.patch.set_facecolor(BG)
    beat = 60.0 / bpm

    for r in range(rows):
        ax = axes[r][0]
        t0 = start + r * row_seconds
        t1 = min(t0 + row_seconds, end)
        ax.set_facecolor(BG)
        ax.set_xlim(t0, t0 + row_seconds)
        ax.set_ylim(0, NUM_LANES)

        for k in range(NUM_LANES):                          # lane bands
            ax.add_patch(Rectangle((t0, k), row_seconds, 1,
                                   facecolor=LANE_BG[k % 2], zorder=0))
            if ref_notes is not None:                       # split line
                ax.plot([t0, t0 + row_seconds], [k + 0.5, k + 0.5],
                        color=BG, linewidth=0.7, zorder=2)

        b0 = math.ceil(t0 / beat)                           # beat grid
        for b in range(b0, int(t1 / beat) + 1):
            x = b * beat
            major = (b % 4 == 0)
            ax.axvline(x, color=GRID_M if major else GRID_B,
                       linewidth=0.9 if major else 0.5, zorder=1)

        for n in notes:                                     # main: lower half
            if t0 - 0.5 <= n["t"] <= t1 + 0.5 or (n["type"] == 3 and n["t"] < t1 and n["t"] + n["dur"] > t0):
                if ref_notes is not None:
                    _draw_note(ax, n, n["lane"] + 0.03, 0.44)
                else:
                    _draw_note(ax, n, n["lane"] + 0.08, 0.84)
        if ref_notes is not None:                           # ref: upper half
            for n in ref_notes:
                if t0 - 0.5 <= n["t"] <= t1 + 0.5 or (n["type"] == 3 and n["t"] < t1 and n["t"] + n["dur"] > t0):
                    _draw_note(ax, n, n["lane"] + 0.53, 0.44, alpha=0.85)

        ax.set_yticks([k + 0.5 for k in range(NUM_LANES)])
        ax.set_yticklabels([f"L{k}" for k in range(NUM_LANES)],
                           color="#8b96a5", fontsize=7)
        ax.tick_params(axis="x", colors="#8b96a5", labelsize=7)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_color("#2a2f37")
        ax.set_xlabel("")

    # ── legend / title ──────────────────────────────────────────────────
    handles = [plt.Line2D([], [], marker={1: "s", 2: "D", 3: "s", 4: "^"}[t],
                          linestyle="none", ms=6, color=TYPE_STYLE[t]["color"],
                          label=TYPE_STYLE[t]["label"]) for t in (1, 2, 3, 4)]
    n_main = len([n for n in notes if start <= n["t"] <= end])
    parts = [title or Path(main_path).stem,
             f"bpm {bpm:.0f}", f"{n_main} notes"]
    if ref_notes is not None:
        parts.append("each lane: ▲upper=reference  ▼lower=generated")
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False,
               labelcolor="#c8d0da", fontsize=8, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("  ·  ".join(parts), color="#dfe6ee", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"[plot] {rows} rows × {row_seconds:.0f}s → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render a Phigros chart as a piano-roll PNG.")
    p.add_argument("chart", help="chart JSON to render (drawn in the lower half if --ref)")
    p.add_argument("--ref", default=None,
                   help="reference chart JSON — drawn in each lane's upper half")
    p.add_argument("-o", "--out", default=None, help="output PNG (default: <chart>.png)")
    p.add_argument("--row-seconds", type=float, default=20.0)
    p.add_argument("--start", type=float, default=0.0, help="start time (s)")
    p.add_argument("--end",   type=float, default=None, help="end time (s)")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    out = args.out or str(Path(args.chart).with_suffix(".png"))
    if not Path(args.chart).exists():
        print(f"[ERROR] not found: {args.chart}", file=sys.stderr)
        sys.exit(1)
    plot_chart(args.chart, args.ref, out, row_seconds=args.row_seconds,
               start=args.start, end=args.end, title=args.title)


if __name__ == "__main__":
    main()

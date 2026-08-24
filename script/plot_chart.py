"""plot_chart.py — Render a Phigros chart as a vertical piano-roll image (P7-1).

Time flows BOTTOM → TOP (the falling-note direction), folded into vertical
strips arranged left→right (each strip covers --row-seconds).  With --ref,
every strip shows EIGHT lanes: the reference chart's 4 lanes on the left,
the generated chart's 4 lanes on the right, so the two charts can be read
side by side at the same instant by scanning horizontally.

Notes are drawn as game-like bars: Tap = wide blue bar, Drag = narrow amber
bar, Hold = green body spanning its duration with a solid head, Flick = pink
bar with a ▲.  Horizontal gridlines mark beats (thin) and 4-beat measures
(thick), computed from the chart BPM.

Usage
-----
    uv run python script/plot_chart.py gen.json -o gen.png
    uv run python script/plot_chart.py gen.json --ref real.json -o cmp.png
    uv run python script/plot_chart.py gen.json --ref real.json --start 60 --end 100
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
from matplotlib.patches import Polygon, Rectangle

# CJK song titles: fall back to a system font that has the glyphs
from matplotlib import font_manager as _fm
_CJK = [f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback")
        if any(f == x.name for x in _fm.fontManager.ttflist)]
if _CJK:
    plt.rcParams["font.family"] = ["DejaVu Sans", *_CJK]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", *_CJK]

# ── palette (dark, game-flavoured) ──────────────────────────────────────────
BG      = "#14171c"
LANE_BG = ("#1a1e24", "#20242b")
GRID_B  = "#2b323d"                        # beat line
GRID_M  = "#49546a"                        # measure (4-beat) line
DIVIDER = "#0c0e11"
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


def _draw_note(ax, n: dict, x0: float, seg: float) -> None:
    """Draw one note as a horizontal bar at its time (y) in lane column x0..x0+1."""
    st = TYPE_STYLE[n["type"]]
    h  = seg / 110.0                                        # bar thickness (time units)
    if n["type"] == 3 and n["dur"] > 0:                     # Hold: body + head
        # enforce a minimum VISUAL body length so short holds stay
        # distinguishable from taps at coarse zoom levels
        body = max(n["dur"], h * 1.8)
        ax.add_patch(Rectangle((x0 + 0.16, n["t"]), 0.68, body,
                               facecolor=st["color"], alpha=0.32,
                               edgecolor=st["color"], linewidth=0.7, zorder=3))
        ax.add_patch(Rectangle((x0 + 0.10, n["t"] - h / 2), 0.80, h,
                               facecolor=st["color"], zorder=4))
    elif n["type"] == 2:                                    # Drag: narrow bar
        ax.add_patch(Rectangle((x0 + 0.26, n["t"] - h / 2), 0.48, h,
                               facecolor=st["color"], zorder=4))
    elif n["type"] == 4:                                    # Flick: bar + ▲
        ax.add_patch(Rectangle((x0 + 0.10, n["t"] - h / 2), 0.80, h,
                               facecolor=st["color"], zorder=4))
        ax.add_patch(Polygon([(x0 + 0.38, n["t"] + h * 0.7),
                              (x0 + 0.62, n["t"] + h * 0.7),
                              (x0 + 0.50, n["t"] + h * 1.9)],
                             closed=True, facecolor=st["color"], zorder=5))
    else:                                                   # Tap: wide bar
        ax.add_patch(Rectangle((x0 + 0.10, n["t"] - h / 2), 0.80, h,
                               facecolor=st["color"], zorder=4))


def _draw_half(ax, notes: list[dict], x_base: float, t0: float, t1: float,
               seg: float) -> int:
    n_drawn = 0
    for n in notes:
        if (t0 - 0.5 <= n["t"] <= t1 + 0.5) or \
           (n["type"] == 3 and n["t"] < t1 and n["t"] + n["dur"] > t0):
            _draw_note(ax, n, x_base + n["lane"], seg)
            if t0 <= n["t"] <= t1:
                n_drawn += 1
    return n_drawn


def plot_chart(
    main_path:   str,
    ref_path:    str | None,
    out_path:    str,
    row_seconds: float = 20.0,
    start:       float = 0.0,
    end:         float | None = None,
    title:       str | None = None,
    main_label:  str = "生成",
    ref_label:   str = "真实",
) -> None:
    notes, bpm, dur = load_notes(main_path)
    ref_notes = None
    if ref_path:
        ref_notes, _, ref_dur = load_notes(ref_path)
        dur = max(dur, ref_dur)
    end    = min(end, dur) if end else dur
    span   = end - start
    strips = max(1, math.ceil(span / row_seconds))
    beat   = 60.0 / bpm

    # x geometry per strip: [0,4) ref lanes · gap · [4.6,8.6) gen lanes
    two   = ref_notes is not None
    gap   = 0.6
    width = (2 * NUM_LANES + gap) if two else NUM_LANES

    fig_w = strips * (2.1 if two else 1.15) + 0.7
    fig, axes = plt.subplots(1, strips, figsize=(fig_w, 10.5), squeeze=False)
    fig.patch.set_facecolor(BG)

    total_main = 0
    for s in range(strips):
        ax = axes[0][s]
        t0 = start + s * row_seconds
        t1 = min(t0 + row_seconds, end)
        ax.set_facecolor(BG)
        ax.set_xlim(0, width)
        ax.set_ylim(t0, t0 + row_seconds)          # bottom → top = earlier → later

        halves = [(0.0, ref_notes), (NUM_LANES + gap, notes)] if two \
                 else [(0.0, notes)]
        for x_base, _ in halves:                    # lane bands
            for k in range(NUM_LANES):
                ax.add_patch(Rectangle((x_base + k, t0), 1, row_seconds,
                                       facecolor=LANE_BG[k % 2], zorder=0))
        if two:                                     # divider between the halves
            ax.add_patch(Rectangle((NUM_LANES, t0), gap, row_seconds,
                                   facecolor=DIVIDER, zorder=1))

        for b in range(math.ceil(t0 / beat), int(t1 / beat) + 1):   # beat grid
            y = b * beat
            major = (b % 4 == 0)
            for x_base, _ in halves:
                ax.plot([x_base, x_base + NUM_LANES], [y, y],
                        color=GRID_M if major else GRID_B,
                        linewidth=1.0 if major else 0.45, zorder=2)

        for x_base, ns in halves:
            drawn = _draw_half(ax, ns, x_base, t0, t1, row_seconds)
            if ns is notes:
                total_main += drawn

        # labels: half names on top, lane numbers at bottom
        if two:
            ax.text(NUM_LANES / 2, t0 + row_seconds * 1.006, ref_label,
                    ha="center", va="bottom", color="#8b96a5", fontsize=8)
            ax.text(NUM_LANES + gap + NUM_LANES / 2, t0 + row_seconds * 1.006,
                    main_label, ha="center", va="bottom",
                    color="#8b96a5", fontsize=8)
        xt = ([x + 0.5 for x in range(NUM_LANES)]
              + ([NUM_LANES + gap + x + 0.5 for x in range(NUM_LANES)] if two else []))
        ax.set_xticks(xt)
        ax.set_xticklabels([str(k) for k in range(NUM_LANES)] * (2 if two else 1),
                           color="#5f6a78", fontsize=6)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", colors="#8b96a5", labelsize=7)
        if s == 0:
            ax.set_ylabel("time (s)  ↑", color="#8b96a5", fontsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2a2f37")

    handles = [plt.Line2D([], [], marker="s", linestyle="none", ms=7,
                          color=TYPE_STYLE[t]["color"], label=TYPE_STYLE[t]["label"])
               for t in (1, 2, 3, 4)]
    parts = [title or Path(main_path).stem, f"bpm {bpm:.0f}",
             f"{total_main} notes ({main_label})"]
    fig.legend(handles=handles, loc="lower right", ncol=4, frameon=False,
               labelcolor="#c8d0da", fontsize=8, bbox_to_anchor=(0.995, 0.002))
    fig.suptitle("  ·  ".join(parts), color="#dfe6ee", fontsize=10,
                 x=0.01, ha="left")
    if two:
        fig.text(0.01, 0.006, f"每列:左 4 轨 = {ref_label} · 右 4 轨 = {main_label} · 时间从下往上",
                 color="#8b96a5", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"[plot] {strips} strips × {row_seconds:.0f}s → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a Phigros chart as a vertical piano-roll PNG.")
    p.add_argument("chart", help="chart JSON to render (right half if --ref)")
    p.add_argument("--ref", default=None,
                   help="reference chart JSON — drawn as the LEFT 4 lanes")
    p.add_argument("-o", "--out", default=None, help="output PNG (default: <chart>.png)")
    p.add_argument("--row-seconds", type=float, default=20.0,
                   help="seconds per vertical strip")
    p.add_argument("--start", type=float, default=0.0, help="start time (s)")
    p.add_argument("--end",   type=float, default=None, help="end time (s)")
    p.add_argument("--title", default=None)
    p.add_argument("--main-label", default="生成")
    p.add_argument("--ref-label",  default="真实")
    args = p.parse_args()

    out = args.out or str(Path(args.chart).with_suffix(".png"))
    if not Path(args.chart).exists():
        print(f"[ERROR] not found: {args.chart}", file=sys.stderr)
        sys.exit(1)
    plot_chart(args.chart, args.ref, out, row_seconds=args.row_seconds,
               start=args.start, end=args.end, title=args.title,
               main_label=args.main_label, ref_label=args.ref_label)


if __name__ == "__main__":
    main()

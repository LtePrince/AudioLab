"""
src/data/gameplay_miner.py
──────────────────────────
Mine the GAMEPLAY layer out of original (multi-line) Phigros charts.

For every note, evaluate its judgment line's move/rotate events at the hit
tick and recover where the note is actually judged on screen — turning the
messy 24-line presentation format into a clean supervised target
(docs/freeform_design.md §1):

    (tick, type, dur, hit_x, hit_y, theta, …)   per note

Verified feasible on 60 charts: 99.97% of recovered hit points fall inside
the screen.

Coordinate conventions
~~~~~~~~~~~~~~~~~~~~~~
- Screen coordinates are normalised: bottom-left (0,0) → top-right (1,1),
  16:9 aspect assumed (official 1920×1080 reference).
- 1 positionX unit = 0.05625 screen WIDTHS along the line direction
  (108 px @1920).  Because width ≠ height, the vertical component uses
  0.05625 × 16/9 = 0.1 screen heights:
      hit_x = line_x + posX·0.05625·cosθ
      hit_y = line_y + posX·0.1·sinθ
- formatVersion 3 move events carry (start,end)=(x) and (start2,end2)=(y);
  formatVersion 1 packs both into one number: x = trunc(v/1000)/880,
  y = (v mod 1000)/520.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["GameplayNote", "mine_chart", "X_UNIT_W", "X_UNIT_H"]

X_UNIT_W = 0.05625          # positionX unit in screen-width fraction
X_UNIT_H = 0.05625 * 16 / 9  # …and in screen-height fraction (= 0.1)


@dataclass
class GameplayNote:
    """One mined note: gameplay target + presentation attributes for P3."""
    tick:    int
    type:    int     # 1 Tap / 2 Drag / 3 Hold / 4 Flick
    dur:     int     # hold duration in ticks (0 otherwise)
    pos_x:   float   # raw positionX (line-local, X units)
    hit_x:   float   # judgment point, screen-normalised
    hit_y:   float
    theta:   float   # line angle at hit, degrees in [-180, 180)
    line_id: int
    above:   bool    # from notesAbove (approach side)
    moving:  bool    # line translating at hit time
    rotating: bool   # line rotating at hit time
    tail_x:  float   # hold tail judgment point (== hit for non-holds)
    tail_y:  float


# ─────────────────────────────────────────────────────────────────────────────
# Event evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_v1(v: float) -> tuple[float, float]:
    return math.trunc(v / 1000.0) / 880.0, (v % 1000.0) / 520.0


def _eval_scalar(events: list[dict], t: float, default: float) -> tuple[float, bool]:
    """Piecewise-linear value at *t*; returns (value, changing_at_t).

    Outside all events, clamps to the nearest event endpoint (official charts
    normally cover (-999999, 1e9), this is a robustness fallback).
    """
    if not events:
        return default, False
    for e in events:
        if e["startTime"] <= t <= e["endTime"]:
            f = (t - e["startTime"]) / max(e["endTime"] - e["startTime"], 1e-9)
            return (e["start"] + f * (e["end"] - e["start"]),
                    abs(e["end"] - e["start"]) > 1e-9)
    first = min(events, key=lambda e: e["startTime"])
    last  = max(events, key=lambda e: e["endTime"])
    return (first["start"] if t < first["startTime"] else last["end"]), False


def _eval_xy(events: list[dict], t: float, fmt: int) -> tuple[float, float, bool]:
    """Line (x, y) at *t* plus whether it is translating at *t*."""
    if not events:
        return 0.5, 0.5, False
    for e in events:
        if e["startTime"] <= t <= e["endTime"]:
            f = (t - e["startTime"]) / max(e["endTime"] - e["startTime"], 1e-9)
            if fmt == 1:
                x0, y0 = _unpack_v1(e["start"])
                x1, y1 = _unpack_v1(e["end"])
            else:
                x0, y0 = e["start"], e["start2"]
                x1, y1 = e["end"],   e["end2"]
            moving = abs(x1 - x0) > 1e-9 or abs(y1 - y0) > 1e-9
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0), moving
    first = min(events, key=lambda e: e["startTime"])
    last  = max(events, key=lambda e: e["endTime"])
    e = first if t < first["startTime"] else last
    key = "start" if e is first else "end"
    if fmt == 1:
        x, y = _unpack_v1(e[key])
    else:
        x, y = e[key], e[key + "2"]
    return x, y, False


def _hit_point(lx: float, ly: float, theta_deg: float, pos_x: float) -> tuple[float, float]:
    th = math.radians(theta_deg)
    return (lx + pos_x * X_UNIT_W * math.cos(th),
            ly + pos_x * X_UNIT_H * math.sin(th))


# ─────────────────────────────────────────────────────────────────────────────
# Mining
# ─────────────────────────────────────────────────────────────────────────────

def mine_chart(json_path: str | Path) -> list[GameplayNote]:
    """Mine one original chart into its gameplay-layer note list."""
    d = json.load(open(json_path, encoding="utf-8"))
    fmt = int(d.get("formatVersion", 3))
    out: list[GameplayNote] = []

    for line_id, line in enumerate(d["judgeLineList"]):
        mv, rot = line["judgeLineMoveEvents"], line["judgeLineRotateEvents"]
        for above, notes in ((True, line["notesAbove"]), (False, line["notesBelow"])):
            for n in notes:
                t     = float(n["time"])
                dur   = int(round(n["holdTime"])) if int(n["type"]) == 3 else 0
                lx, ly, moving   = _eval_xy(mv, t, fmt)
                theta, rotating  = _eval_scalar(rot, t, 0.0)
                theta = ((theta + 180.0) % 360.0) - 180.0
                hx, hy = _hit_point(lx, ly, theta, n["positionX"])

                if dur > 0:                       # hold tail follows the line
                    t2 = t + dur
                    lx2, ly2, _ = _eval_xy(mv, t2, fmt)
                    th2, _      = _eval_scalar(rot, t2, 0.0)
                    tx, ty = _hit_point(lx2, ly2, th2, n["positionX"])
                else:
                    tx, ty = hx, hy

                out.append(GameplayNote(
                    tick=int(round(t)), type=int(n["type"]), dur=dur,
                    pos_x=float(n["positionX"]),
                    hit_x=hx, hit_y=hy, theta=theta,
                    line_id=line_id, above=above,
                    moving=moving, rotating=rotating,
                    tail_x=tx, tail_y=ty,
                ))

    out.sort(key=lambda g: (g.tick, g.hit_x))
    return out


def save_jsonl(notes: list[GameplayNote], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for g in notes:
            fh.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[GameplayNote]:
    return [GameplayNote(**json.loads(l))
            for l in open(path, encoding="utf-8") if l.strip()]

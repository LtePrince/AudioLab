"""mine_gameplay.py — Mine all original charts into gameplay-layer datasets.

For every song directory: read the ORIGINAL chart (via info.txt, same
convention as convert_chart_4k.py), recover each note's judgment-time screen
position and line attributes, and write ``data/gameplay/<song>.jsonl``.
Prints the dataset-wide distribution report that grounds the free-x model
and the ergonomic constraints (docs/freeform_design.md).

Usage
-----
    uv run python script/mine_gameplay.py [--data-dir data/chart] [--out-dir data/gameplay]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.data.gameplay_miner import mine_chart, save_jsonl


def _parse_info(info_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Mine gameplay layer from original charts.")
    p.add_argument("--data-dir", default="data/chart")
    p.add_argument("--out-dir",  default="data/gameplay")
    args = p.parse_args()

    dirs = sorted(d for d in Path(args.data_dir).iterdir() if d.is_dir())
    all_notes, n_fail = [], 0
    per_song_offscreen = []

    for d in dirs:
        info = d / "info.txt"
        if not info.exists():
            n_fail += 1; print(f"[FAIL] {d.name}: no info.txt"); continue
        chart_fn = _parse_info(info).get("Chart")
        src = d / chart_fn if chart_fn else None
        if not src or not src.exists():
            n_fail += 1; print(f"[FAIL] {d.name}: source chart missing"); continue
        try:
            notes = mine_chart(src)
        except Exception as exc:
            n_fail += 1; print(f"[FAIL] {d.name}: {type(exc).__name__}: {exc}")
            continue
        save_jsonl(notes, Path(args.out_dir) / f"{d.name}.jsonl")
        all_notes.extend(notes)
        off = np.mean([not (0 <= g.hit_x <= 1 and 0 <= g.hit_y <= 1) for g in notes])
        per_song_offscreen.append((off, d.name))

    N = all_notes
    hx = np.array([g.hit_x for g in N]); hy = np.array([g.hit_y for g in N])
    th = np.array([abs(g.theta) for g in N])
    print(f"\n=== {len(dirs)} songs → {len(dirs) - n_fail} mined ({n_fail} failed), "
          f"{len(N)} notes → {args.out_dir}/ ===")
    print(f"hit_x: p1={np.percentile(hx,1):.3f} p50={np.percentile(hx,50):.3f} "
          f"p99={np.percentile(hx,99):.3f}   出屏={100*(( hx<0)|(hx>1)).mean():.2f}%")
    print(f"hit_y: p1={np.percentile(hy,1):.3f} p50={np.percentile(hy,50):.3f} "
          f"p99={np.percentile(hy,99):.3f}")
    print(f"线状态@hit: 移动 {100*np.mean([g.moving for g in N]):.1f}%  "
          f"旋转 {100*np.mean([g.rotating for g in N]):.1f}%  "
          f"倾斜>5° {100*(th>5).mean():.1f}%")
    holds = [g for g in N if g.dur > 0]
    if holds:
        drift = np.array([abs(g.tail_x - g.hit_x) + abs(g.tail_y - g.hit_y) for g in holds])
        print(f"Hold: {len(holds)} 个,尾部判定点漂移>2%屏 {100*(drift>0.02).mean():.1f}% "
              f"(演出合成器约束项)")
    # 手速统计:同曲相邻音符 |Δhit_x| / Δtick (人体工学约束的数据依据)
    worst = sorted(per_song_offscreen, reverse=True)[:3]
    print("出屏最多的歌:", ", ".join(f"{n}({100*o:.1f}%)" for o, n in worst if o > 0) or "无")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

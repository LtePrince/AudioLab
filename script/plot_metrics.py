"""plot_metrics.py — Render train/val curves from metrics.jsonl files.

Each training stage (vae / wave / dit) appends one JSON line per epoch to
``<ckpt_dir>/<stage>/metrics.jsonl``.  This script plots train vs val loss
(plus lr on a twin axis) for every stage it finds.

Usage::

    uv run python script/plot_metrics.py                       # scan checkpoints/
    uv run python script/plot_metrics.py --ckpt-dir checkpoints --out curves.png
    uv run python script/plot_metrics.py checkpoints/dit/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VAL_KEYS = ("val_loss", "val_recon")   # first present key is the val curve


def _load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # A resumed run re-logs the epochs after its restart point; keep the LAST
    # record per epoch so curves do not zigzag backwards.
    by_epoch: dict = {}
    for r in records:
        by_epoch[r.get("epoch")] = r
    return [by_epoch[e] for e in sorted(k for k in by_epoch if k is not None)]


def _plot_stage(ax, records: list[dict], title: str) -> None:
    epochs = [r.get("epoch") for r in records]
    train  = [r.get("train_loss") for r in records]
    ax.plot(epochs, train, label="train", color="tab:blue")

    val_key = next((k for k in VAL_KEYS if any(k in r for r in records)), None)
    if val_key:
        val_pts = [(r["epoch"], r[val_key]) for r in records if val_key in r]
        ax.plot(*zip(*val_pts), label=val_key, color="tab:orange")
        best = min(val_pts, key=lambda p: p[1])
        ax.scatter([best[0]], [best[1]], color="tab:red", zorder=5,
                   label=f"best {best[1]:.4f} @ ep{best[0]}")

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if any("lr" in r for r in records):
        ax2 = ax.twinx()
        lr_pts = [(r["epoch"], r["lr"]) for r in records if "lr" in r]
        ax2.plot(*zip(*lr_pts), color="tab:green", alpha=0.35, linewidth=1)
        ax2.set_ylabel("lr", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green", labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics.jsonl training curves.")
    parser.add_argument("paths", nargs="*",
                        help="explicit metrics.jsonl files (default: scan --ckpt-dir)")
    parser.add_argument("--ckpt-dir", default="checkpoints",
                        help="root checkpoint dir to scan for */metrics.jsonl")
    parser.add_argument("--out", default="checkpoints/curves.png",
                        help="output PNG path")
    args = parser.parse_args()

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        files = sorted(Path(args.ckpt_dir).glob("*/metrics.jsonl"))
    files = [f for f in files if f.exists()]
    if not files:
        print(f"[ERROR] no metrics.jsonl found (searched {args.ckpt_dir}/*/)",
              file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 4.2),
                             squeeze=False)
    for ax, f in zip(axes[0], files):
        records = _load(f)
        stage   = records[0].get("stage", f.parent.name) if records else f.parent.name
        _plot_stage(ax, records, f"{stage}  ({f})")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[plot] {len(files)} stage(s) → {out}")


if __name__ == "__main__":
    main()

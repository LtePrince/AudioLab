"""verify_tokenizer.py — Full-dataset lossless round-trip check for the AR tokenizer.

For every chart in a data list:
  source notes → encode → strict grammar decode → compare field-by-field
  (tick, lane, type, hold-ticks) against the canonicalised source.

"Canonicalised" = sorted by (tick, lane), duplicate (tick, lane) dropped
(first wins — the same convention as the training array codec) and lane
quantised from positionX.  Any mismatch is a hard failure.

Also reports token-sequence statistics (the AR decoder's context budget).

Usage
-----
    uv run python script/verify_tokenizer.py                   # data/data.txt
    uv run python script/verify_tokenizer.py --data-list data/train.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.data.chart2array import Phigros4kConvertor
from src.data.chart_tokenizer import ChartTokenizer, TokenNote


def source_notes(json_path: str) -> list[TokenNote]:
    """Canonicalised source notes (same convention as ChartTokenizer.encode_chart)."""
    d = json.load(open(json_path, encoding="utf-8"))
    canon: dict[tuple[int, int], TokenNote] = {}
    for line in d["judgeLineList"]:
        for n in line["notesAbove"] + line["notesBelow"]:
            tick = int(round(n["time"]))
            lane = Phigros4kConvertor._lane_from_x(n["positionX"])
            if (tick, lane) in canon:
                continue
            type_ = int(n["type"])
            dur   = int(round(n["holdTime"])) if type_ == 3 else 0
            if type_ == 3 and dur < 1:
                type_, dur = 1, 0
            canon[(tick, lane)] = TokenNote(tick, lane, type_, dur)
    return [canon[k] for k in sorted(canon)]


def main() -> None:
    p = argparse.ArgumentParser(description="Round-trip verify the AR chart tokenizer.")
    p.add_argument("--data-list", default="data/data.txt")
    args = p.parse_args()

    entries = [l.strip().split(",")[0] for l in open(args.data_list)
               if l.strip() and not l.lstrip().startswith("#")]
    base = Path(args.data_list).resolve().parent

    tk = ChartTokenizer()
    lengths, note_counts = [], []
    n_fail = 0
    n_degenerate_holds = 0     # legacy pre-"convert bug of holdtime==0" data:
    n_dup_notes        = 0     # counted and tolerated; anything else fails
    for i, rel in enumerate(entries, 1):
        path = str(base / rel)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                src  = source_notes(path)
                toks = tk.encode_chart(path)
                back = tk.decode_tokens(toks, strict=True)
            unexpected = []
            for w in caught:
                msg = str(w.message)
                if "zero-duration Hold" in msg:
                    n_degenerate_holds += 1
                elif "duplicate (tick,lane)" in msg:
                    n_dup_notes += 1
                else:
                    unexpected.append(msg)
            if unexpected:
                raise RuntimeError("; ".join(unexpected))
        except Exception as exc:
            n_fail += 1
            print(f"[FAIL] {rel}: {type(exc).__name__}: {exc}")
            continue

        if back != src:
            n_fail += 1
            bad = next((a, b) for a, b in zip(src, back) if a != b) \
                if len(src) == len(back) else (f"len {len(src)}", f"len {len(back)}")
            print(f"[FAIL] {rel}: mismatch — src={bad[0]} vs decoded={bad[1]}")
            continue
        # canonical idempotence: re-encoding the decode must be identical
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reencoded = tk.encode_notes(back)
        if reencoded != toks:
            n_fail += 1
            print(f"[FAIL] {rel}: encoding not canonical/idempotent")
            continue
        lengths.append(len(toks))
        note_counts.append(len(src))

    L = np.array(lengths)
    print(f"\n=== {len(entries)} charts: {len(lengths)} lossless, {n_fail} FAILED ===")
    if n_degenerate_holds or n_dup_notes:
        print(f"    tolerated legacy patterns: {n_degenerate_holds} zero-duration "
              f"Holds → Tap (pre-'convert bug of holdtime==0' data), "
              f"{n_dup_notes} duplicate notes dropped")
    if len(L):
        print(f"token 长度: p50={np.percentile(L,50):.0f}  p90={np.percentile(L,90):.0f}  "
              f"p99={np.percentile(L,99):.0f}  max={L.max()}  "
              f"tokens/note={L.sum()/max(sum(note_counts),1):.2f}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

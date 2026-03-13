"""4k_verify.py — Verify that a Phigros chart meets 4k fixed-lane requirements.

Uses ``parse_phigros_file`` (same entry point as the training pipeline) to load
the chart into a ``PhigrosChart`` object, then inspects the **raw**
``judge_lines`` fields — NOT ``flatten()`` which would convert any chart into a
valid 4k representation and therefore pass everything.

Three invariants:

1. **Single judge line** — ``len(chart.judge_lines) == 1``
2. **Fixed lane positions** — every note's ``positionX`` must be within *tol*
   of one of the four canonical lane centres ``(-3.75, -1.25, 1.25, 3.75)``.
3. **Constant scroll speed** — all non-zero ``speedEvent.value`` entries are
   equal to ``expected_speed``; no mid-chart speed changes.

Usage
-----
Verify a single chart::

    uv run python script/4k_verify.py data/chart/Eltaw_IN/generate.json

Batch-verify all ``generate.json`` files under ``data/chart/``::

    uv run python script/4k_verify.py --data-dir data/chart

Options
-------
--data-dir DIR   Scan this directory instead of a single file.
--speed VALUE    Expected scroll speed value (default: 3.0).
--tol VALUE      Tolerance for positionX matching (default: 0.01).
--fail-fast      Stop after the first failure.
--verbose        Print OK lines as well as failures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path so ``src`` is importable when running as a script
# from the AudioLab project root.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent   # …/AudioLab
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.chart2array import parse_phigros_file, LANE_CENTERS as _LC  # noqa: E402

# Canonical lane centres produced by save_phigros_file
LANE_CENTERS: tuple[float, ...] = tuple(_LC)   # (-3.75, -1.25, 1.25, 3.75)


# ---------------------------------------------------------------------------
# Single-file verification
# ---------------------------------------------------------------------------

class VerifyResult:
    """Accumulates check failures for one chart file."""

    def __init__(self, path: Path) -> None:
        self.path     = path
        self.failures: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def summary(self) -> str:
        if self.ok:
            return f"OK    {self.path}"
        lines = [f"FAIL  {self.path}"]
        for f in self.failures:
            lines.append(f"        • {f}")
        return "\n".join(lines)


def _nearest_lane(x: float) -> tuple[float, float]:
    """Return (nearest centre, distance) for positionX *x*."""
    best = min(LANE_CENTERS, key=lambda c: abs(c - x))
    return best, abs(best - x)


def verify_file(path: Path, expected_speed: float = 3.0, tol: float = 0.01) -> VerifyResult:
    """Run all 4k checks on *path* and return a :class:`VerifyResult`.

    Loads with ``parse_phigros_file`` (same entry point as the training
    pipeline), then inspects ``chart.judge_lines`` — the raw
    ``judgeLineList`` field — without calling ``flatten()``.
    """
    result = VerifyResult(path)

    # ── load via data pipeline (parse_phigros_file, NOT flatten) ────────────
    try:
        chart = parse_phigros_file(str(path))
    except Exception as exc:
        result.fail(f"Cannot load chart: {exc}")
        return result

    lines = chart.judge_lines       # raw list[JudgeLine], no flatten applied

    # ── check 1: single judge line ──────────────────────────────────────────
    if len(lines) != 1:
        result.fail(f"Expected 1 judge line, found {len(lines)}")
        # Can still check the first line if it exists
        if not lines:
            return result
    line = lines[0]

    # ── check 2: fixed lane positions ───────────────────────────────────────
    all_notes: list[dict] = list(line["notesAbove"]) + list(line["notesBelow"])
    if not all_notes:
        result.fail("Judge line has no notes")
    else:
        bad_positions: list[tuple[int, float, float]] = []   # (idx, x, dist)
        seen_xs: set[float] = set()
        for i, note in enumerate(all_notes):
            x = float(note["positionX"])
            _, dist = _nearest_lane(x)
            seen_xs.add(round(x, 4))
            if dist > tol:
                bad_positions.append((i, x, dist))

        if bad_positions:
            sample = bad_positions[:5]
            detail = ", ".join(f"note[{i}] x={x:.4f} (off by {d:.4f})" for i, x, d in sample)
            extra  = f" … and {len(bad_positions) - 5} more" if len(bad_positions) > 5 else ""
            result.fail(f"{len(bad_positions)} note(s) off-lane: {detail}{extra}")

        actual_lanes = len(seen_xs)
        if actual_lanes > 4:
            result.fail(
                f"More than 4 distinct positionX values found ({actual_lanes}): "
                + ", ".join(f"{x:.4f}" for x in sorted(seen_xs))
            )

    # ── check 3: constant scroll speed ──────────────────────────────────────
    speed_events: list[dict] = line["speedEvents"]
    if not speed_events:
        result.fail("No speedEvents found")
    else:
        # Active events: value != 0.0  (the trailing zero-speed segment is OK)
        active = [e for e in speed_events if e["value"] != 0.0]
        if not active:
            result.fail("All speedEvents have value=0 (no active scroll speed)")
        else:
            speeds = {round(e["value"], 6) for e in active}
            if len(speeds) > 1:
                result.fail(
                    f"Multiple non-zero speed values found: {sorted(speeds)} "
                    "(chart has mid-song speed changes)"
                )
            else:
                actual_speed = next(iter(speeds))
                if abs(actual_speed - expected_speed) > 1e-4:
                    result.fail(
                        f"Speed value {actual_speed} ≠ expected {expected_speed}"
                    )

    return result


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def verify_dir(
    chart_root: Path,
    expected_speed: float,
    tol: float,
    fail_fast: bool,
    verbose: bool,
) -> tuple[int, int]:
    """Verify all generate.json files under *chart_root*.

    Returns ``(n_ok, n_fail)``.
    """
    gen_files = sorted(chart_root.rglob("generate.json"))
    if not gen_files:
        print(f"No generate.json files found under {chart_root}", file=sys.stderr)
        return 0, 0

    n_ok = n_fail = 0
    for f in gen_files:
        r = verify_file(f, expected_speed=expected_speed, tol=tol)
        if r.ok:
            n_ok += 1
            if verbose:
                print(r.summary())
        else:
            n_fail += 1
            print(r.summary())
            if fail_fast:
                break

    return n_ok, n_fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that Phigros generate.json files meet 4k format requirements."
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help="Path to a single generate.json (or any .json chart file). "
             "Omit to use --data-dir batch mode.",
    )
    parser.add_argument(
        "--data-dir", default="data/chart",
        help="Root directory to scan recursively for generate.json files. "
             "Used only when no positional path is given. Default: data/chart",
    )
    parser.add_argument(
        "--speed", type=float, default=3.0,
        help="Expected speedEvent value (default: 3.0).",
    )
    parser.add_argument(
        "--tol", type=float, default=0.01,
        help="Maximum allowed deviation from a LANE_CENTER (default: 0.01).",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop after the first failure.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print OK results as well as failures.",
    )
    args = parser.parse_args()

    if args.path:
        # Single-file mode
        p = Path(args.path)
        if not p.exists():
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)
        r = verify_file(p, expected_speed=args.speed, tol=args.tol)
        print(r.summary())
        sys.exit(0 if r.ok else 1)

    # Batch mode
    chart_root = Path(args.data_dir)
    if not chart_root.is_dir():
        print(f"[ERROR] Directory not found: {chart_root}", file=sys.stderr)
        sys.exit(1)

    n_ok, n_fail = verify_dir(
        chart_root,
        expected_speed=args.speed,
        tol=args.tol,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
    )
    total = n_ok + n_fail
    print(f"\n=== {total} charts verified: {n_ok} OK, {n_fail} FAILED ===")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

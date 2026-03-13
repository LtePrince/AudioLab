"""convert_chart_4k.py — Batch 4k conversion for all Phigros charts.

Scans ``data/chart/`` for every song directory, reads the original multi-line
Phigros JSON (identified via ``info.txt``), converts it to a single-judge-line
4-lane chart, and writes ``generate.json`` in the same directory.

Run from the AudioLab root::

    uv run python script/convert_chart_4k.py [--data-dir data/chart] [--force]

Options
-------
--data-dir   Path to the chart root directory.  Default: ``data/chart``.
--force      Re-convert even when ``generate.json`` already exists.
--dry-run    Print what would be done without writing any files.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_info(info_path: Path) -> dict[str, str]:
    """Parse ``info.txt`` into a key→value dict (ignores comment lines)."""
    result: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _version_from_dir(dir_name: str) -> str:
    """Extract the difficulty tag from a directory like ``SongName_IN``."""
    parts = dir_name.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else "IN"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_one(chart_dir: Path, force: bool = False, dry_run: bool = False) -> str:
    """Convert a single chart directory.

    Returns one of: ``'skipped'``, ``'done'``, ``'error:<message>'``.
    """
    info_path = chart_dir / "info.txt"
    if not info_path.exists():
        return "error:no info.txt"

    info     = _parse_info(info_path)
    chart_fn = info.get("Chart")
    if not chart_fn:
        return "error:Chart key missing in info.txt"

    src_json = chart_dir / chart_fn
    if not src_json.exists():
        return f"error:source JSON not found ({chart_fn})"

    out_json = chart_dir / "generate.json"
    if out_json.exists() and not force:
        return "skipped"

    if dry_run:
        return "done(dry)"

    # ---- actual conversion ----
    # Import here so the script remains importable without side-effects
    from src.data.chart2array import Phigros4kConvertor, parse_phigros_file

    FRAME_MS  = 512 / 22050 / 4 * 8 * 1000   # ≈ 46.44 ms/frame
    MAX_FRAME = 4096

    version = _version_from_dir(chart_dir.name)

    try:
        chart = parse_phigros_file(str(src_json), version=version)
        conv  = Phigros4kConvertor(frame_ms=FRAME_MS, max_frame=MAX_FRAME)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress lane-collision warnings
            flat = conv.flatten(chart)

        conv.save_phigros_file(
            note_array=flat.note_array,
            bpm=chart.bpm,
            output_path=str(out_json),
            offset=chart.offset,
        )
    except Exception as exc:  # noqa: BLE001
        return f"error:{exc}"

    return "done"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert Phigros charts to single-line 4k format."
    )
    parser.add_argument(
        "--data-dir", default="data/chart",
        help="Root directory containing per-song chart folders. Default: data/chart",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert even when generate.json already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing files.",
    )
    args = parser.parse_args()

    chart_root = Path(args.data_dir)
    if not chart_root.is_dir():
        print(f"[ERROR] chart directory not found: {chart_root}", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(p for p in chart_root.iterdir() if p.is_dir())
    total = len(dirs)
    done = skipped = errors = 0

    print(f"Found {total} chart directories under {chart_root}/")
    if args.dry_run:
        print("(dry-run mode — no files will be written)\n")

    for idx, d in enumerate(dirs, 1):
        status = convert_one(d, force=args.force, dry_run=args.dry_run)
        tag    = d.name

        if status == "skipped":
            skipped += 1
        elif status.startswith("error:"):
            errors += 1
            print(f"[{idx:>4}/{total}] FAIL  {tag}  — {status[6:]}")
        else:
            done += 1
            if args.dry_run:
                print(f"[{idx:>4}/{total}] WOULD {tag}")
            # Only print every 10th success to keep output manageable
            elif done % 10 == 0 or idx == total:
                print(f"[{idx:>4}/{total}] OK    ({done} converted so far)")

    print(f"\n=== Done: {done} converted, {skipped} skipped, {errors} errors ===")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

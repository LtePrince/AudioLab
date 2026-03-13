"""datalist.py — Generate data/data.txt from converted chart directories.

Scans ``data/chart/`` for all directories that contain **both** a
``generate.json`` (4k-converted chart) and at least one ``.ogg`` audio file,
then writes each valid entry as a line in ``data/data.txt``::

    chart/<dir>/generate.json,chart/<dir>/<id>.ogg,<version>

Directories that are missing ``generate.json`` or have no ``.ogg`` are
skipped with a warning.  Run ``script/convert_chart_4k.py`` first to produce
the ``generate.json`` files.

Run from the AudioLab root::

    uv run python script/datalist.py [--data-dir data/chart] [--out data/data.txt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _version_from_dir(dir_name: str) -> str:
    """Extract difficulty tag from a directory like ``SongName_IN``."""
    parts = dir_name.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else "IN"


def _scan_entry(chart_dir: Path, chart_root: Path) -> str | None:
    """Return the data.txt line for *chart_dir*, or ``None`` if not ready.

    The returned path uses POSIX-style relative paths from the repo root
    (i.e. relative to the parent of *chart_root*).
    """
    gen_json = chart_dir / "generate.json"
    if not gen_json.exists():
        return None

    ogg_files = sorted(chart_dir.glob("*.ogg"))
    if not ogg_files:
        return None

    ogg   = ogg_files[0]                   # pick first if multiple exist
    root  = chart_root.parent              # AudioLab root
    rel_j = gen_json.relative_to(root).as_posix()
    rel_a = ogg.relative_to(root).as_posix()
    ver   = _version_from_dir(chart_dir.name)
    return f"{rel_j},{rel_a},{ver}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/data.txt from chart directories that have generate.json."
    )
    parser.add_argument(
        "--data-dir", default="data/chart",
        help="Root directory containing per-song chart folders. Default: data/chart",
    )
    parser.add_argument(
        "--out", default="data/data.txt",
        help="Output path for the data list file. Default: data/data.txt",
    )
    args = parser.parse_args()

    chart_root = Path(args.data_dir)
    if not chart_root.is_dir():
        print(f"[ERROR] chart directory not found: {chart_root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p for p in chart_root.iterdir() if p.is_dir())
    total  = len(dirs)
    ready  = []
    no_gen = []
    no_ogg = []

    for d in dirs:
        if not (d / "generate.json").exists():
            no_gen.append(d.name)
            continue
        if not list(d.glob("*.ogg")):
            no_ogg.append(d.name)
            continue
        entry = _scan_entry(d, chart_root)
        if entry:
            ready.append(entry)

    out_path.write_text("\n".join(ready) + ("\n" if ready else ""), encoding="utf-8")

    print(f"Scanned {total} directories under {chart_root}/")
    print(f"  Written  : {len(ready)} entries  →  {out_path}")
    if no_gen:
        print(f"  Skipped  : {len(no_gen)} (no generate.json — run convert_chart_4k.py first)")
    if no_ogg:
        print(f"  Skipped  : {len(no_ogg)} (no .ogg audio file)")
    print("Done.")


if __name__ == "__main__":
    main()

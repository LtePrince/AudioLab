"""test_chart2array.py — P1 smoke tests
---------------------------------------
Verifies PhigrosChart loading, Phigros4kConvertor flatten/chart_to_array/
array_to_notes, and read-only mutation guard.

Run from AudioLab root:
    uv run python test/data/test_chart2array.py
"""

from __future__ import annotations

import numpy as np

from src.data.chart2array import (
    Phigros4kConvertor,
    parse_phigros_file,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Eltaw.json is the original multi-line Phigros chart (23 judge lines).
# It is used here as the raw input to the 4k conversion pipeline.
# test_generate_json() will produce generate.json in the same directory.
_JSON     = "data/example/json/Eltaw.json"
_GEN_JSON = "data/example/json/generate.json"   # output of the 4k conversion
FRAME_MS  = 512 / 22050 / 4 * 8 * 1000   # ≈ 46.44 ms
MAX_FRAME = 4096


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_load_chart() -> None:
    """PhigrosChart loads correctly and read-only properties are accessible."""
    chart = parse_phigros_file(_JSON, version="IN")
    print(repr(chart))
    print(f"  format_version : {chart.format_version}")
    print(f"  bpm            : {chart.bpm}")
    print(f"  offset         : {chart.offset}")
    print(f"  judge_lines    : {len(chart.judge_lines)}")
    print(f"  total_notes    : {chart.total_note_count}")
    print(f"  first note     : {chart.judge_lines[0]['notesAbove'][0]}")
    print(f"  for_batch()    : {chart.for_batch()}")
    assert chart.format_version == 3, "Expected formatVersion=3"
    assert chart.bpm > 0, "BPM must be positive"
    print("[1] load_chart ✓")


def test_mutation_guard() -> None:
    """PhigrosChart must raise AttributeError on attribute assignment."""
    chart = parse_phigros_file(_JSON, version="IN")
    try:
        chart.bpm = 120.0   # type: ignore[misc]
        raise AssertionError("[FAIL] mutation was not blocked")
    except AttributeError:
        print("[2] mutation correctly blocked ✓")


def test_flatten() -> None:
    """Flatten produces correct shapes for note_array and valid_flag."""
    chart      = parse_phigros_file(_JSON, version="IN")
    conv       = Phigros4kConvertor(frame_ms=FRAME_MS, max_frame=MAX_FRAME)
    flat_chart = conv.flatten(chart)

    print(repr(flat_chart))
    print(f"  note_array shape : {flat_chart.note_array.shape}")
    print(f"  valid_flag shape : {flat_chart.valid_flag.shape}")
    print(f"  valid frames     : {int(flat_chart.valid_flag.sum())}")
    print(
        f"  encoded onsets   : {flat_chart.encoded_onset_count}"
        f"  (original: {chart.total_note_count})"
    )
    print(f"  first flat note  : {flat_chart.flat_notes[0]}")

    from src.data.chart2array import NUM_CHANNELS
    assert flat_chart.note_array.shape == (NUM_CHANNELS, MAX_FRAME), \
        f"note_array shape mismatch: {flat_chart.note_array.shape}"
    assert flat_chart.valid_flag.shape == (MAX_FRAME,), \
        f"valid_flag shape mismatch: {flat_chart.valid_flag.shape}"
    print("[3] flatten shapes ✓")


def test_chart_to_array() -> None:
    """chart_to_array shortcut must match flatten() output."""
    chart      = parse_phigros_file(_JSON, version="IN")
    conv       = Phigros4kConvertor(frame_ms=FRAME_MS, max_frame=MAX_FRAME)
    flat_chart = conv.flatten(chart)
    arr2, _vf2 = conv.chart_to_array(chart)

    assert arr2.shape == flat_chart.note_array.shape, \
        f"shape mismatch: {arr2.shape} vs {flat_chart.note_array.shape}"
    assert np.allclose(arr2, flat_chart.note_array), "chart_to_array output differs from flatten()"
    print(f"[4] chart_to_array matches flatten()  shape={arr2.shape} ✓")


def test_array_to_notes() -> None:
    """array_to_notes round-trip returns a non-empty list of valid notes."""
    chart      = parse_phigros_file(_JSON, version="IN")
    conv       = Phigros4kConvertor(frame_ms=FRAME_MS, max_frame=MAX_FRAME)
    flat_chart = conv.flatten(chart)
    recovered  = conv.array_to_notes(flat_chart.note_array, bpm=chart.bpm)

    print(f"  decoded notes : {len(recovered)}")
    print(f"  first decoded : {recovered[0]}")
    assert len(recovered) > 0, "array_to_notes returned empty list"
    print("[5] array_to_notes round-trip ✓")


def test_generate_json() -> None:
    """Flatten original Eltaw.json and write generate.json to the same dir.

    The output is a single-judge-line Phigros-format JSON that can be loaded
    back with ``parse_phigros_file`` and must have exactly one judge line.
    """
    import json as _json, warnings
    chart = parse_phigros_file(_JSON, version="IN")
    conv  = Phigros4kConvertor(frame_ms=FRAME_MS, max_frame=MAX_FRAME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # suppress collision warnings
        flat = conv.flatten(chart)

    conv.save_phigros_file(
        note_array=flat.note_array,
        bpm=chart.bpm,
        output_path=_GEN_JSON,
        offset=chart.offset,
    )

    # Verify the output is readable and well-formed
    with open(_GEN_JSON) as f:
        data = _json.load(f)
    assert data["formatVersion"] == 3
    lines = data["judgeLineList"]
    assert len(lines) == 1, f"expected 1 judge line, got {len(lines)}"
    notes = lines[0]["notesAbove"]
    assert len(notes) == flat.encoded_onset_count, (
        f"note count mismatch: {len(notes)} vs {flat.encoded_onset_count}"
    )

    # Round-trip: reload with parse_phigros_file
    gen_chart = parse_phigros_file(_GEN_JSON, version="IN")
    assert gen_chart.total_note_count == flat.encoded_onset_count
    assert abs(gen_chart.bpm - chart.bpm) < 1e-3

    print(f"[6] generate.json written → {_GEN_JSON}")
    print(f"    judge lines : {len(lines)},  notes : {len(notes)} ✓")


def test_save_load_flat_array() -> None:
    """save_flat_array / load_flat_array round-trip preserves arrays and meta."""
    import tempfile, os
    chart = parse_phigros_file(_JSON, version="IN")
    conv  = Phigros4kConvertor(frame_ms=FRAME_MS, max_frame=MAX_FRAME)
    flat  = conv.flatten(chart)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.npz")
        Phigros4kConvertor.save_flat_array(flat, path)
        assert os.path.exists(path), "npz file was not created"

        na, vf, bpm, offset = Phigros4kConvertor.load_flat_array(path)

    from src.data.chart2array import NUM_CHANNELS
    assert na.shape == (NUM_CHANNELS, MAX_FRAME), f"note_array shape mismatch: {na.shape}"
    assert vf.shape == (MAX_FRAME,),              f"valid_flag shape mismatch: {vf.shape}"
    assert na.dtype.name == "float32"
    assert vf.dtype.name == "float32"
    assert np.allclose(na, flat.note_array), "note_array mismatch after round-trip"
    assert np.allclose(vf, flat.valid_flag), "valid_flag mismatch after round-trip"
    assert abs(bpm - flat.bpm) < 1e-6,     f"bpm mismatch: {bpm} vs {flat.bpm}"
    print(f"[7] save/load flat array  shape={na.shape}  bpm={bpm:.1f} ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_load_chart()
    test_mutation_guard()
    test_flatten()
    test_chart_to_array()
    test_array_to_notes()
    test_generate_json()
    test_save_load_flat_array()
    print("\n=== All chart2array tests passed ===")

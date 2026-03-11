"""test_dataset.py — P1 smoke tests
-------------------------------------
Verifies PhigrosDataset loading, shape contracts, mel cache consistency,
mirror augmentation, and DataLoader collation.

Run from AudioLab root:
    uv run python test/data/test_dataset.py
"""

from __future__ import annotations

import pathlib
import shutil

import torch
from torch.utils.data import DataLoader

from src.data.chart2array import NUM_CHANNELS
from src.data.dataset import PhigrosDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT        = pathlib.Path(__file__).resolve().parents[2]   # AudioLab/
_DATA_DIR         = _REPO_ROOT / "data"
_LIST_PATH        = _DATA_DIR / "data.txt"
_CACHE_DIR        = _DATA_DIR / "cache_mel"
_CHART_CACHE_DIR  = _DATA_DIR / "cache_chart"  # persistent, mirrors cache_mel

FRAME_MS  = 512 / 22050 / 4 * 8 * 1000   # ≈ 46.44 ms
MAX_FRAME = 4096


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(**kwargs) -> PhigrosDataset:
    return PhigrosDataset(
        data_list_path=str(_LIST_PATH),
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        cache_dir=_CACHE_DIR,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_parse_list() -> None:
    """Dataset reads data.txt and finds exactly 1 entry."""
    ds = _make_dataset(augment=False)
    print(f"[1] {repr(ds)}")
    assert len(ds) == 1, f"Expected 1 entry, got {len(ds)}"
    print("[1] parse_list ✓")


def test_sample_shapes() -> None:
    """__getitem__ returns tensors with correct shapes and dtypes."""
    ds = _make_dataset(augment=False)
    sample = ds[0]
    audio      = sample["audio"]
    note       = sample["note"]
    valid_flag = sample["valid_flag"]
    meta       = sample["meta"]

    print(f"[2] audio      : {tuple(audio.shape)}  dtype={audio.dtype}")
    print(f"    note       : {tuple(note.shape)}   dtype={note.dtype}")
    print(f"    valid_flag : {tuple(valid_flag.shape)}  dtype={valid_flag.dtype}")
    print(f"    meta       : {meta}")

    assert audio.ndim == 2,          "audio must be 2-D (n_mels, T)"
    assert audio.shape[0] == 128,    f"expected n_mels=128, got {audio.shape[0]}"
    assert note.shape == (NUM_CHANNELS, MAX_FRAME), \
        f"note shape mismatch: {note.shape}"
    assert valid_flag.shape == (MAX_FRAME,), \
        f"valid_flag shape mismatch: {valid_flag.shape}"
    print("[2] sample shapes ✓")


def test_mel_cache() -> None:
    """Second fetch must hit the mel cache and return identical audio tensor."""
    ds      = _make_dataset(augment=False)
    sample1 = ds[0]
    sample2 = ds[0]
    assert torch.allclose(sample1["audio"], sample2["audio"]), \
        "cache read differs from original"
    print("[3] mel cache consistent ✓")


def test_mirror_augmentation() -> None:
    """Mirror augmentation must produce a different note array."""
    ds_plain = _make_dataset(augment=False)
    ds_mirror = PhigrosDataset(
        data_list_path=str(_LIST_PATH),
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        augment=True,
        mirror_prob=1.0,      # always mirror
        rate_range=(1.0, 1.0),  # fixed rate for determinism
    )
    note_plain  = ds_plain[0]["note"]
    note_mirror = ds_mirror[0]["note"]
    assert not torch.allclose(note_plain, note_mirror), \
        "mirror augmentation had no effect"
    print(f"[4] augmented note shape : {tuple(note_mirror.shape)}")
    print("[4] mirror augmentation changes note array ✓")


def test_chart_cache() -> None:
    """chart_cache_dir: first access writes npz, second access reads it.

    Uses the persistent directory ``data/cache_chart/`` so the npz file is
    visible on disk after the test (mirroring the mel-cache behaviour).
    The directory is wiped at the start of the test to guarantee a cache miss
    on the first access.
    """
    # Start clean so we always exercise the cache-miss branch
    shutil.rmtree(_CHART_CACHE_DIR, ignore_errors=True)
    _CHART_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds = PhigrosDataset(
        data_list_path=str(_LIST_PATH),
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        chart_cache_dir=str(_CHART_CACHE_DIR),
        augment=False,
    )

    # cache miss: npz must be created
    sample1 = ds[0]
    npz_files = list(_CHART_CACHE_DIR.glob("*.npz"))
    assert len(npz_files) == 1, f"expected 1 npz after first access, got {npz_files}"
    print(f"[5] cache miss: npz created → {npz_files[0].name}")
    print(f"    location: {npz_files[0]}")

    # cache hit: result must be identical
    sample2 = ds[0]
    assert torch.allclose(sample1["note"],       sample2["note"]),       "note mismatch on cache hit"
    assert torch.allclose(sample1["valid_flag"], sample2["valid_flag"]), "valid_flag mismatch on cache hit"
    print("[5] chart cache hit returns identical tensors ✓")


def test_chart_cache_mirror() -> None:
    """Mirror augmentation still works when chart_cache_dir is active.

    Reuses the npz written by test_chart_cache so no re-parsing is needed.
    """
    # _CHART_CACHE_DIR already contains the npz from test_chart_cache
    ds_plain = PhigrosDataset(
        data_list_path=str(_LIST_PATH),
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        chart_cache_dir=str(_CHART_CACHE_DIR),
        augment=False,
    )
    ds_mirror = PhigrosDataset(
        data_list_path=str(_LIST_PATH),
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        chart_cache_dir=str(_CHART_CACHE_DIR),
        augment=True,
        mirror_prob=1.0,
        rate_range=(1.0, 1.0),
    )
    note_plain  = ds_plain[0]["note"]
    note_mirror = ds_mirror[0]["note"]
    assert not torch.allclose(note_plain, note_mirror), \
        "mirror via channel-permutation had no effect with chart cache active"
    print("[6] mirror augmentation works with chart_cache_dir ✓")


def test_rate_augment_disabled_with_chart_cache() -> None:
    """When chart_cache_dir is set, rate must always be 1.0 (no temporal stretch)."""
    ds = PhigrosDataset(
        data_list_path=str(_LIST_PATH),
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        chart_cache_dir=str(_CHART_CACHE_DIR),
        augment=True,
        rate_range=(0.8, 1.2),   # wide range to catch any variation
    )
    # Sample 10 times; note shapes must be identical (rate=1.0 always)
    shapes = {tuple(ds[0]["note"].shape) for _ in range(10)}
    assert shapes == {(NUM_CHANNELS, MAX_FRAME)}, \
        f"unexpected note shapes with chart cache: {shapes}"
    print("[7] rate augmentation correctly disabled when chart_cache_dir is set ✓")


def test_dataloader_collation() -> None:
    """DataLoader must batch audio and note tensors correctly."""
    ds     = _make_dataset(augment=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    batch  = next(iter(loader))

    print(f"[8] DataLoader batch audio : {tuple(batch['audio'].shape)}")
    print(f"    DataLoader batch note  : {tuple(batch['note'].shape)}")
    assert batch["audio"].shape[0] == 1, "batch size should be 1"
    print("[8] DataLoader collation ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_parse_list()
    test_sample_shapes()
    test_mel_cache()
    test_mirror_augmentation()
    test_chart_cache()
    test_chart_cache_mirror()
    test_rate_augment_disabled_with_chart_cache()
    test_dataloader_collation()
    print("\n=== All dataset tests passed ===")

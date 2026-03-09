"""
dataset.py  –  P1-7
-------------------
PyTorch Dataset for Phigros chart generation training.

Data-list format
----------------
A plain-text CSV file (no header).  Each non-empty, non-comment line:

    <json_path>,<audio_path>[,<version>]

Paths may be absolute **or** relative to the list file's own directory.
``version`` defaults to ``""`` when omitted.

Example line::

    chart/Eltaw_IN/9752727302241212.json,audio/Eltaw.ogg,IN

Sample returned by ``__getitem__``
-----------------------------------
::

    {
        "audio"     : Tensor (n_mels, T_audio),   # log-mel spectrogram
        "note"      : Tensor (NUM_CHANNELS, max_frame),
        "valid_flag": Tensor (max_frame,),
        "meta"      : {json_path, audio_path, bpm, offset, version},
    }

Data flow
---------
::

    data_list.txt
        └─ PhigrosDataset.__getitem__()
                ├─ AudioGPUprocessor  ──►  log-mel  (n_mels, T_audio)
                └─ parse_phigros_file()
                        └─ Phigros4kConvertor.flatten()
                                └─ PhigrosFlatChart
                                    ├─ .note_array  (NUM_CHANNELS, max_frame)
                                    └─ .valid_flag  (max_frame,)
"""

from __future__ import annotations

import hashlib
import os
import random
import warnings
from pathlib import Path
from typing import Optional
import pathlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.chart2array import (
    NUM_CHANNELS,
    Phigros4kConvertor,
    parse_phigros_file,
)
from src.data.audio2mel import AudioGPUprocessor


# ===========================================================================
# PhigrosDataset
# ===========================================================================

class PhigrosDataset(Dataset):
    """PyTorch ``Dataset`` for Phigros → 4-lane note-array training pairs.

    Parameters
    ----------
    data_list_path:
        Path to the data-list text file (see module docstring for format).
    convertor_params:
        Keyword arguments forwarded to :class:`Phigros4kConvertor`.
        **Required** keys: ``frame_ms`` (float), ``max_frame`` (int).
        Optional keys: ``mirror`` (bool), ``rate`` (float),
        ``offset_ms`` (float), ``from_logits`` (bool).
    cache_dir:
        Directory for caching pre-computed log-mel ``.npy`` files.
        Pass ``None`` (default) to disable caching.
    augment:
        When ``True``, randomly apply mirror-flip and rate-stretch on every
        ``__getitem__`` call.
    mirror_prob:
        Probability of applying the lane-mirror augmentation.
        Only active when ``augment=True``.  Default ``0.5``.
    rate_range:
        ``(min_rate, max_rate)`` for uniform random rate sampling.
        Only active when ``augment=True``.  Default ``(0.9, 1.1)``.
    sr:
        Target sample rate for audio loading.  Default ``22050``.
    n_fft:
        FFT window size for mel computation.  Default ``2048``.
    hop_length:
        STFT hop length for mel computation.  Default ``512``.
    n_mels:
        Number of mel filter banks.  Default ``128``.
    device:
        Torch device string (``"cuda"``, ``"cpu"``, …) for the
        :class:`AudioGPUprocessor`.  ``None`` auto-selects CUDA if available.
    """

    def __init__(
        self,
        data_list_path: str | Path,
        convertor_params: dict,
        *,
        cache_dir: Optional[str | Path] = None,
        augment: bool = False,
        mirror_prob: float = 0.5,
        rate_range: tuple[float, float] = (0.9, 1.1),
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. Parse data list                                                  #
        # ------------------------------------------------------------------ #
        data_list_path = Path(data_list_path).resolve()
        if not data_list_path.exists():
            raise FileNotFoundError(f"Data list not found: {data_list_path}")
        self._list_dir = data_list_path.parent
        self._entries  = self._parse_list(data_list_path)

        if len(self._entries) == 0:
            warnings.warn(
                f"PhigrosDataset: no valid entries in {data_list_path}",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------ #
        # 2. Store convertor parameters (needed for per-item augmentation)   #
        # ------------------------------------------------------------------ #
        _cp = dict(convertor_params)          # shallow copy
        self._frame_ms    = float(_cp.pop("frame_ms"))
        self._max_frame   = int(_cp.pop("max_frame"))
        self._base_mirror = bool(_cp.pop("mirror",      False))
        self._base_rate   = float(_cp.pop("rate",       1.0))
        self._offset_ms   = float(_cp.pop("offset_ms",  0.0))
        self._from_logits = bool(_cp.pop("from_logits", False))

        if _cp:
            warnings.warn(
                f"PhigrosDataset: unrecognised convertor_params keys: {list(_cp)}",
                UserWarning,
                stacklevel=2,
            )

        # Default (non-augmented) convertor instance – re-used across items
        # when augment=False to avoid repeated object allocation.
        self._convertor = Phigros4kConvertor(
            frame_ms=self._frame_ms,
            max_frame=self._max_frame,
            mirror=self._base_mirror,
            rate=self._base_rate,
            offset_ms=self._offset_ms,
            from_logits=self._from_logits,
        )

        # ------------------------------------------------------------------ #
        # 3. Augmentation settings                                            #
        # ------------------------------------------------------------------ #
        self.augment     = augment
        self.mirror_prob = mirror_prob
        self.rate_range  = tuple(rate_range)

        # ------------------------------------------------------------------ #
        # 4. Audio processor                                                  #
        # ------------------------------------------------------------------ #
        self._sr         = sr
        self._n_fft      = n_fft
        self._hop_length = hop_length
        self._n_mels     = n_mels
        self._audio_proc = AudioGPUprocessor(
            sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            device=device,
        )

        # ------------------------------------------------------------------ #
        # 5. Cache directory                                                  #
        # ------------------------------------------------------------------ #
        if cache_dir is not None:
            self._cache_dir: Optional[Path] = Path(cache_dir).resolve()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _parse_list(self, list_path: Path) -> list[tuple[str, str, str]]:
        """Parse the data-list file.

        Returns
        -------
        list of ``(json_path_abs, audio_path_abs, version)`` 3-tuples.
        Lines starting with ``#`` and blank lines are skipped.
        Entries whose JSON file does not exist are skipped with a warning.
        """
        entries: list[tuple[str, str, str]] = []
        for lineno, raw in enumerate(list_path.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if not parts[0]:
                continue

            json_path  = self._resolve(parts[0])
            audio_path = self._resolve(parts[1]) if len(parts) > 1 else ""
            version    = parts[2]                if len(parts) > 2 else ""

            if not Path(json_path).exists():
                warnings.warn(
                    f"PhigrosDataset line {lineno}: JSON not found → {json_path}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            entries.append((json_path, audio_path, version))

        return entries

    def _resolve(self, path_str: str) -> str:
        """Resolve *path_str* relative to the list file's parent directory."""
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((self._list_dir / p).resolve())

    def _cache_path(self, audio_path: str) -> Optional[Path]:
        """Return the cache ``.npy`` path for *audio_path*, or ``None`` if
        the cache directory is not set."""
        if self._cache_dir is None:
            return None
        h    = hashlib.sha1(audio_path.encode()).hexdigest()[:8]
        stem = Path(audio_path).stem
        return self._cache_dir / f"{stem}_{h}.npy"

    def _load_mel(self, audio_path: str) -> torch.Tensor:
        """Load log-mel spectrogram for *audio_path*.

        Returns ``Tensor`` of shape ``(n_mels, T_audio)`` on CPU.
        Reads from the disk cache when available; writes to cache after
        first computation.  Returns a zero tensor if the audio file is
        missing.
        """
        npy_path = self._cache_path(audio_path)

        # ---- try cache first ----
        if npy_path is not None and npy_path.exists():
            arr = np.load(str(npy_path))
            return torch.from_numpy(arr).float()   # (n_mels, T)

        # ---- audio file missing ----
        if not audio_path or not Path(audio_path).exists():
            warnings.warn(
                f"PhigrosDataset: audio not found: {audio_path!r}; "
                "returning zero mel tensor.",
                UserWarning,
                stacklevel=2,
            )
            return torch.zeros(self._n_mels, 1, dtype=torch.float32)

        # ---- compute from raw audio ----
        waveform = self._audio_proc.load_from_path(audio_path)   # (1, C, T_wav)
        mel      = self._audio_proc.forward(waveform)            # (1, n_mels, T)
        mel      = mel.squeeze(0).cpu()                          # (n_mels, T)

        # ---- save to cache ----
        if npy_path is not None:
            np.save(str(npy_path), mel.numpy())

        return mel.float()

    def _make_convertor(self, mirror: bool, rate: float) -> Phigros4kConvertor:
        """Instantiate a new :class:`Phigros4kConvertor` with the given
        augmentation parameters but the same base configuration."""
        return Phigros4kConvertor(
            frame_ms=self._frame_ms,
            max_frame=self._max_frame,
            mirror=mirror,
            rate=rate,
            offset_ms=self._offset_ms,
            from_logits=self._from_logits,
        )

    # ---------------------------------------------------------------------- #
    # Dataset protocol                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict:
        """Fetch one training sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict
            ``audio``      : ``Tensor (n_mels, T_audio)`` — log-mel spectrogram.
            ``note``       : ``Tensor (NUM_CHANNELS, max_frame)`` — encoded chart.
            ``valid_flag`` : ``Tensor (max_frame,)`` — 1 up to last onset, 0 after.
            ``meta``       : DataLoader-safe dict with ``json_path``, ``audio_path``,
                             ``bpm``, ``offset``, ``version``.
        """
        json_path, audio_path, version = self._entries[idx]

        # ---- 1. Log-mel spectrogram ----
        audio = self._load_mel(audio_path)   # (n_mels, T_audio)

        # ---- 2. Phigros chart ----
        chart = parse_phigros_file(json_path, audio_path=audio_path, version=version)

        # ---- 3. Choose convertor (augmented or default) ----
        if self.augment:
            mirror    = random.random() < self.mirror_prob
            rate      = random.uniform(self.rate_range[0], self.rate_range[1])
            convertor = self._make_convertor(mirror, rate)
        else:
            rate      = self._base_rate
            convertor = self._convertor

        # ---- 3b. Time-stretch mel to match note rate ----
        # note 时间轴已按 adjusted_ms = original_ms / rate 压缩，
        # mel 需同步拉伸：new_T = round(T / rate)
        if rate != 1.0:
            n_mels, T = audio.shape
            new_T = max(1, int(round(T / rate)))
            # cv2.resize: (width=new_T, height=n_mels)
            audio_np = cv2.resize(
                audio.numpy().reshape(n_mels, T, 1),
                (new_T, n_mels),
                interpolation=cv2.INTER_LINEAR,
            ).reshape(n_mels, new_T)
            audio = torch.from_numpy(audio_np).float()

        # ---- 4. Flatten chart → note_array + valid_flag ----
        flat = convertor.flatten(chart)

        note       = torch.from_numpy(flat.note_array).float()   # (NUM_CHANNELS, T)
        valid_flag = torch.from_numpy(flat.valid_flag).float()   # (T,)
        meta       = flat.for_batch()

        return {
            "audio":      audio,
            "note":       note,
            "valid_flag": valid_flag,
            "meta":       meta,
        }

    def __repr__(self) -> str:
        return (
            f"PhigrosDataset("
            f"n={len(self)}, "
            f"frame_ms={self._frame_ms:.2f}, "
            f"max_frame={self._max_frame}, "
            f"n_mels={self._n_mels}, "
            f"augment={self.augment}, "
            f"cache={'on' if self._cache_dir else 'off'})"
        )


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    _REPO_ROOT  = pathlib.Path(__file__).resolve().parents[2]   # AudioLab/
    _DATA_DIR   = _REPO_ROOT / "data"
    _LIST_PATH  = _DATA_DIR / "data.txt"
    _CACHE_DIR  = _DATA_DIR / "cache_mel"

    FRAME_MS  = 512 / 22050 / 4 * 8 * 1000   # ≈ 46.44 ms
    MAX_FRAME = 4096

    # ---- 1. Use data/data.txt directly ----
    list_path = str(_LIST_PATH)
    print(f"[1] data list : {list_path}")

    # ---- 2. Instantiate dataset (no augmentation, disk cache) ----
    dataset = PhigrosDataset(
        data_list_path=list_path,
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        cache_dir=_CACHE_DIR,
        augment=False,
    )
    print(f"[2] {repr(dataset)}")
    assert len(dataset) == 1, f"Expected 1 entry, got {len(dataset)}"

    # ---- 3. Fetch sample, verify shapes ----
    sample = dataset[0]
    audio      = sample["audio"]
    note       = sample["note"]
    valid_flag = sample["valid_flag"]
    meta       = sample["meta"]

    print(f"[3] audio      : {tuple(audio.shape)}  dtype={audio.dtype}")
    print(f"    note       : {tuple(note.shape)}   dtype={note.dtype}")
    print(f"    valid_flag : {tuple(valid_flag.shape)}  dtype={valid_flag.dtype}")
    print(f"    meta       : {meta}")

    assert audio.ndim == 2,      "audio must be 2-D (n_mels, T)"
    assert audio.shape[0] == 128, f"expected n_mels=128, got {audio.shape[0]}"
    assert note.shape    == (NUM_CHANNELS, MAX_FRAME), \
        f"note shape mismatch: {note.shape}"
    assert valid_flag.shape == (MAX_FRAME,), \
        f"valid_flag shape mismatch: {valid_flag.shape}"
    print("[3] shape assertions passed ✓")

    # ---- 4. Fetch again – should hit mel cache ----
    sample2 = dataset[0]
    assert torch.allclose(sample["audio"], sample2["audio"]), \
        "cache read differs from original"
    print("[4] mel cache read consistent ✓")

    # ---- 5. Instantiate with augmentation, fetch sample ----
    dataset_aug = PhigrosDataset(
        data_list_path=list_path,
        convertor_params={"frame_ms": FRAME_MS, "max_frame": MAX_FRAME},
        augment=True,
        mirror_prob=1.0,    # always mirror for deterministic test
        rate_range=(1.0, 1.0),   # fixed rate
    )
    sample_aug = dataset_aug[0]
    print(f"[5] augmented note shape : {tuple(sample_aug['note'].shape)}")
    # mirrored note array should differ from non-mirrored
    assert not torch.allclose(sample["note"], sample_aug["note"]), \
        "mirror augmentation had no effect"
    print("[5] mirror augmentation changes note array ✓")

    # ---- 6. DataLoader collation sanity check ----
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    batch  = next(iter(loader))
    print(f"[6] DataLoader batch audio : {tuple(batch['audio'].shape)}")
    print(f"    DataLoader batch note  : {tuple(batch['note'].shape)}")
    assert batch["audio"].shape[0] == 1, "batch size should be 1"
    print("[6] DataLoader collation ✓")

    print("\n=== All smoke tests passed ===")

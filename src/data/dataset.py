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
                ├─ _load_mel()        ──►  log-mel  (n_mels, T_audio)      [mel_cache npy]
                └─ _load_note_array()
                        ├─ [chart_cache hit]  npz → note_array + valid_flag
                        └─ [cache miss / disabled]
                                parse_phigros_file()
                                └─ Phigros4kConvertor.flatten()
                                        └─ PhigrosFlatChart
                                            ├─ .note_array  (NUM_CHANNELS, max_frame)
                                            └─ .valid_flag  (max_frame,)
                                [save to chart_cache if enabled]

Note: when ``chart_cache_dir`` is set, note arrays are cached as compressed
``.npz`` files (base form: rate=1.0, mirror=False).  Mirror augmentation is
applied in-memory via a channel permutation.  Rate augmentation changes frame
timings and **cannot** be reconstructed from a cached array, so it is
automatically disabled whenever ``chart_cache_dir`` is active.
"""

from __future__ import annotations

import hashlib
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.chart2array import (
    NUM_CHANNELS,
    NUM_LANES,
    Phigros4kConvertor,
    parse_phigros_file,
)

# Channel permutation for mirror augmentation applied to a pre-encoded note_array.
# Mirror swaps lane 0↔3 and lane 1↔2; each of the 5 channel groups is reversed.
# Groups: is_start[0-3], start_off[4-7], is_holding[8-11], end_offset[12-15], note_type[16-19]
_MIRROR_PERM: list[int] = [
    3, 2, 1, 0,
    7, 6, 5, 4,
    11, 10, 9, 8,
    15, 14, 13, 12,
    19, 18, 17, 16,
]
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
        Pass ``None`` (default) to disable mel caching.
    chart_cache_dir:
        Directory for caching pre-encoded note-array ``.npz`` files
        (base form: rate=1.0, mirror=False).  When set, JSON parsing and
        ``Phigros4kConvertor.flatten()`` are skipped on cache hits, and
        rate augmentation is automatically disabled (mirror still works).
        Pass ``None`` (default) to disable chart caching.
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
        chart_cache_dir: Optional[str | Path] = None,
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
        # AudioGPUprocessor must always run on CPU inside DataLoader workers:
        # CUDA cannot be re-initialized in forked subprocesses (Linux default
        # start method = fork).  The training loop moves tensors to GPU after
        # collation, and mel results are cached on disk anyway.
        self._audio_proc = AudioGPUprocessor(
            sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            device="cpu",
        )

        # ------------------------------------------------------------------ #
        # 5. Mel cache directory                                              #
        # ------------------------------------------------------------------ #
        if cache_dir is not None:
            self._cache_dir: Optional[Path] = Path(cache_dir).resolve()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

        # ------------------------------------------------------------------ #
        # 6. Chart array cache directory                                      #
        # ------------------------------------------------------------------ #
        if chart_cache_dir is not None:
            self._chart_cache_dir: Optional[Path] = Path(chart_cache_dir).resolve()
            self._chart_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._chart_cache_dir = None

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

    def _chart_cache_path(self, json_path: str) -> Optional[Path]:
        """Return the cache ``.npz`` path for *json_path*, or ``None`` if
        chart caching is disabled."""
        if self._chart_cache_dir is None:
            return None
        h    = hashlib.sha1(json_path.encode()).hexdigest()[:8]
        stem = Path(json_path).stem
        return self._chart_cache_dir / f"{stem}_{h}.npz"

    def _load_note_array(
        self,
        json_path:  str,
        audio_path: str,
        version:    str,
        mirror:     bool,
        convertor:  Optional["Phigros4kConvertor"] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(note_array, valid_flag)`` for the given chart.

        If chart caching is enabled:
          - Cache hit:  load from ``.npz``, apply mirror in-memory.
          - Cache miss: parse JSON, flatten (rate=1.0), save to cache,
            apply mirror in-memory.
        If chart caching is disabled: parse JSON with current convertor
        settings (rate + mirror both applied during flatten).

        Returns
        -------
        note_array : ndarray (NUM_CHANNELS, max_frame)  float32
        valid_flag : ndarray (max_frame,)               float32
        """
        npz_path = self._chart_cache_path(json_path)

        if npz_path is not None:
            # --- chart cache path ---
            if not npz_path.exists():
                # First access: parse JSON with base convertor (rate=1.0, mirror=False)
                chart = parse_phigros_file(json_path, audio_path=audio_path, version=version)
                flat  = self._convertor.flatten(chart)   # base convertor: rate=1.0 mirror=False
                Phigros4kConvertor.save_flat_array(flat, str(npz_path))
                note_array = flat.note_array
                valid_flag = flat.valid_flag
            else:
                note_array, valid_flag, _, _ = Phigros4kConvertor.load_flat_array(str(npz_path))

            # Apply mirror augmentation in-memory (channel permutation, no re-parse needed)
            if mirror:
                note_array = note_array[_MIRROR_PERM, :]

            return note_array, valid_flag

        # --- no chart cache: use convertor directly (supports rate augmentation) ---
        conv  = convertor if convertor is not None else self._convertor
        chart = parse_phigros_file(json_path, audio_path=audio_path, version=version)
        flat  = conv.flatten(chart)   # uses whatever mirror/rate is set
        return flat.note_array, flat.valid_flag

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

        # ---- 2 & 3. Augmentation parameters ----
        # When chart_cache_dir is active, rate augmentation is disabled because
        # the cached array encodes frame positions at rate=1.0 and cannot be
        # re-indexed post-hoc.  Mirror is still applied as a channel permutation.
        chart_cache_active = self._chart_cache_dir is not None

        if self.augment:
            mirror = random.random() < self.mirror_prob
            rate   = 1.0 if chart_cache_active else random.uniform(self.rate_range[0], self.rate_range[1])
        else:
            mirror = self._base_mirror
            rate   = self._base_rate

        # ---- 3b. Time-stretch mel when rate != 1.0 (only if not using chart cache) ----
        if rate != 1.0:
            n_mels, T = audio.shape
            new_T = max(1, int(round(T / rate)))
            audio_np = cv2.resize(
                audio.numpy().reshape(n_mels, T, 1),
                (new_T, n_mels),
                interpolation=cv2.INTER_LINEAR,
            ).reshape(n_mels, new_T)
            audio = torch.from_numpy(audio_np).float()

        # Update default convertor's mirror/rate so flatten() uses correct settings
        # (only relevant when chart_cache is NOT active)
        if not chart_cache_active and (mirror != self._base_mirror or rate != self._base_rate):
            convertor = self._make_convertor(mirror, rate)
        else:
            convertor = self._convertor

        # ---- 4. Load note_array + valid_flag (with chart caching) ----
        note_array, valid_flag_arr = self._load_note_array(
            json_path, audio_path, version, mirror=mirror, convertor=convertor
        )

        note       = torch.from_numpy(note_array).float()       # (NUM_CHANNELS, max_frame)
        valid_flag = torch.from_numpy(valid_flag_arr).float()   # (max_frame,)

        # Build minimal meta from JSON path (bpm/offset not available without parsing
        # when chart cache is active; use 0.0 as placeholder)
        meta = {
            "json_path":  json_path,
            "audio_path": audio_path,
            "bpm":        0.0,
            "offset":     0.0,
            "version":    version,
        }

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
            f"mel_cache={'on' if self._cache_dir else 'off'}, "
            f"chart_cache={'on' if self._chart_cache_dir else 'off'})"
        )


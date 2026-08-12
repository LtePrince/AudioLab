"""test_audio2mel.py — P1 smoke tests
--------------------------------------
Verifies AudioCPUprocessor and AudioGPUprocessor output shapes and
CPU→GPU npy round-trip.

Run from AudioLab root:
    uv run python test/data/test_audio2mel.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from src.data.audio2mel import (
    AudioCPUprocessor,
    AudioGPUprocessor,
    chart_frames_to_mel_frames,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]   # AudioLab/
_AUDIO     = str(_REPO_ROOT / "data" / "example" / "audio" / "Eltaw.ogg")
_NPY       = str(_REPO_ROOT / "data" / "example" / "audio" / "Eltaw.npy")

N_MELS = 128


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_cpu_process_shape() -> None:
    """AudioCPUprocessor.process() returns (n_mels, Frames)."""
    proc = AudioCPUprocessor(n_mels=N_MELS)
    mel  = proc.process(_AUDIO, debug_mode=False)
    print(f"[1] CPU log-mel : {mel.shape}")
    assert mel.ndim == 2,           "expected 2-D array (n_mels, Frames)"
    assert mel.shape[0] == N_MELS,  f"expected n_mels={N_MELS}, got {mel.shape[0]}"
    print("[1] cpu_process_shape ✓")


def test_cpu_save_and_reload() -> None:
    """AudioCPUprocessor.save() writes a .npy that can be reloaded identically."""
    proc = AudioCPUprocessor(n_mels=N_MELS)
    mel  = proc.process(_AUDIO, debug_mode=False)
    proc.save(mel, _NPY)
    reloaded = np.load(_NPY)
    assert np.allclose(mel, reloaded), "reloaded npy differs from original"
    print(f"[2] CPU saved to {_NPY}")
    print("[2] cpu_save_and_reload ✓")


def test_gpu_forward_shape() -> None:
    """AudioGPUprocessor.forward() returns (1, n_mels, Frames)."""
    proc     = AudioGPUprocessor(n_mels=N_MELS)
    waveform = proc.load_from_path(_AUDIO)
    print(f"[3] GPU waveform : {waveform.shape}")
    out = proc.forward(waveform)
    print(f"[3] GPU forward  : {out.shape}")
    assert out.ndim == 3,          "expected 3-D tensor (Batch, n_mels, Frames)"
    assert out.shape[0] == 1,      "batch dim should be 1"
    assert out.shape[1] == N_MELS, f"expected n_mels={N_MELS}, got {out.shape[1]}"
    print("[3] gpu_forward_shape ✓")


def test_gpu_load_mel_spec() -> None:
    """AudioGPUprocessor.load_mel_spec() loads .npy saved by CPU processor."""
    proc     = AudioGPUprocessor(n_mels=N_MELS)
    mel_spec = proc.load_mel_spec(_NPY)
    print(f"[4] GPU mel from npy : {mel_spec.shape}")
    assert mel_spec.ndim == 3,          "expected 3-D tensor (1, n_mels, Frames)"
    assert mel_spec.shape[0] == 1,      "batch dim should be 1"
    assert mel_spec.shape[1] == N_MELS, f"expected n_mels={N_MELS}, got {mel_spec.shape[1]}"
    assert isinstance(mel_spec, torch.Tensor), "expected torch.Tensor"
    print("[4] gpu_load_mel_spec ✓")


def test_chart_frames_to_mel_frames() -> None:
    """Chart→mel frame conversion covers the same duration on both axes.

    With the default pipeline (sr=22050, hop=512, frame_ms≈46.44) one chart
    frame spans exactly two mel frames, so 4096 chart frames ≡ 8192 mel
    frames.  Padding mel to 4096 (the old bug) would cover only half the song.
    """
    sr, hop   = 22050, 512
    frame_ms  = hop / sr / 4 * 8 * 1000          # ≈ 46.44 ms (train.py default)

    n_mel = chart_frames_to_mel_frames(4096, frame_ms, hop, sr)
    assert n_mel == 8192, f"expected 8192 mel frames for 4096 chart frames, got {n_mel}"

    # duration equivalence must hold for arbitrary (frame_ms, hop, sr) combos
    for n_chart, fms, h, s in [
        (4096, 46.44, 512, 22050),
        (2048, 23.22, 512, 22050),
        (1000, 10.0,  256, 16000),
    ]:
        n_mel      = chart_frames_to_mel_frames(n_chart, fms, h, s)
        chart_dur  = n_chart * fms / 1000.0
        mel_dur    = n_mel * h / s
        assert abs(chart_dur - mel_dur) <= h / s, (
            f"duration mismatch: chart {chart_dur:.3f}s vs mel {mel_dur:.3f}s "
            f"(n_chart={n_chart}, frame_ms={fms}, hop={h}, sr={s})"
        )
    print("[5] chart_frames_to_mel_frames ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cpu_process_shape()
    test_cpu_save_and_reload()
    test_gpu_forward_shape()
    test_gpu_load_mel_spec()
    test_chart_frames_to_mel_frames()
    print("\n=== All audio2mel tests passed ===")

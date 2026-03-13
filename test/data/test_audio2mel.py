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

from src.data.audio2mel import AudioCPUprocessor, AudioGPUprocessor

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cpu_process_shape()
    test_cpu_save_and_reload()
    test_gpu_forward_shape()
    test_gpu_load_mel_spec()
    print("\n=== All audio2mel tests passed ===")

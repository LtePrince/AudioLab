"""test_wave.py — P3 smoke tests
--------------------------------
Run from AudioLab root:
    uv run python test/condition/test_wave.py
"""

from __future__ import annotations

import torch

from src.condition.wave import AudioWaveEncoder, DEFAULT_WAVE_CONFIG
from src.encoder.encoder import DEFAULT_VAE_CONFIG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B      = 2
N_MELS = 128
T_MEL  = 16384   # ~190 s at hop=512/sr=22050  (≥ chart frames × audio_stride)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MULT         = DEFAULT_WAVE_CONFIG["channel_mult"]
AUDIO_STRIDE  = 2 ** (len(_MULT) - 1)                   # 16
T_COND        = T_MEL // AUDIO_STRIDE                    # 1024
OUT_CH        = DEFAULT_WAVE_CONFIG["out_channels"]      # 256

VAE_STRIDE    = 2 ** (len(DEFAULT_VAE_CONFIG["channel_mult"]) - 1)   # 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_encoder() -> AudioWaveEncoder:
    return AudioWaveEncoder(**DEFAULT_WAVE_CONFIG).to(DEVICE)

def _make_mel() -> torch.Tensor:
    return torch.randn(B, N_MELS, T_MEL, device=DEVICE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape() -> None:
    """Output must be (B, out_channels, T_mel // audio_stride)."""
    enc = _make_encoder()
    mel = _make_mel()
    with torch.no_grad():
        out = enc(mel)
    expected = (B, OUT_CH, T_COND)
    assert out.shape == expected, f"[1] Shape mismatch: {out.shape} ≠ {expected}"
    print(f"[1] output shape   : {tuple(out.shape)}  ✓")


def test_stride_property() -> None:
    """Encoder.stride must match chart VAE stride."""
    enc = _make_encoder()
    assert enc.stride == VAE_STRIDE, (
        f"[2] audio_stride={enc.stride} ≠ vae_stride={VAE_STRIDE}  "
        "— temporal axes won't align!"
    )
    print(f"[2] stride         : {enc.stride}  (== VAE stride {VAE_STRIDE})  ✓")


def test_backward() -> None:
    """Gradient must flow through the encoder."""
    enc = _make_encoder()
    mel = _make_mel()
    out = enc(mel)
    loss = out.mean()
    loss.backward()
    first_grad = next(enc.parameters()).grad
    assert first_grad is not None, "[3] No gradient computed"
    print(f"[3] backward pass  : ✓  (first param grad norm={first_grad.norm().item():.4f})")


def test_variable_length() -> None:
    """Encoder must handle arbitrary T as long as T is divisible by audio_stride."""
    enc = _make_encoder()
    for T in [AUDIO_STRIDE * k for k in [64, 128, 256]]:
        mel = torch.randn(1, N_MELS, T, device=DEVICE)
        with torch.no_grad():
            out = enc(mel)
        assert out.shape == (1, OUT_CH, T // AUDIO_STRIDE), \
            f"[4] Shape error for T={T}: {out.shape}"
    print(f"[4] variable-length: ✓  (tested T = {[AUDIO_STRIDE*k for k in [64,128,256]]})")


def test_param_count() -> None:
    enc   = _make_encoder()
    total = sum(p.numel() for p in enc.parameters())
    print(f"[5] total params   : {total:>10,}")
    enc.summary(N_MELS, T_MEL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device        : {DEVICE}")
    print(f"Input mel     : (B={B}, n_mels={N_MELS}, T_mel={T_MEL})")
    print(f"Expected out  : (B={B}, out_ch={OUT_CH}, T_cond={T_COND})")
    print(f"audio_stride  : {AUDIO_STRIDE}  |  vae_stride : {VAE_STRIDE}\n")

    test_output_shape()
    test_stride_property()
    test_backward()
    test_variable_length()
    test_param_count()

    print("\n=== All P3 smoke tests passed ===")

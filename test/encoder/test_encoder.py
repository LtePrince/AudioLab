"""test_encoder.py — P2 smoke tests
-----------------------------------
Verifies shapes, loss computation, backward pass and parameter count
for ChartVAE / ChartReconLoss.

Run from AudioLab root:
    uv run python test/encoder/test_encoder.py
"""

from __future__ import annotations

import torch

from src.data.chart2array import NUM_CHANNELS, NUM_LANES
from src.encoder.encoder import (
    ChartReconLoss,
    ChartVAE,
    DEFAULT_VAE_CONFIG,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B      = 2        # batch size
T      = 4096     # time frames

_MULT  = DEFAULT_VAE_CONFIG["channel_mult"]
STRIDE = 2 ** (len(_MULT) - 1)     # 16  (4 downsamplings)
T_LAT  = T // STRIDE                # 256 latent time frames
Z_CH   = DEFAULT_VAE_CONFIG["z_channels"]   # 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_vae() -> ChartVAE:
    return ChartVAE(DEFAULT_VAE_CONFIG).to(DEVICE)


def _make_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal synthetic note_array + valid_flag."""
    x          = torch.zeros(B, NUM_CHANNELS, T, device=DEVICE)
    valid_flag = torch.ones(B, T, device=DEVICE)

    # Scatter synthetic notes into the first half of each lane
    for k in range(NUM_LANES):
        onsets = torch.randint(0, T // 2, (30,))
        x[:, k + 0,  onsets] = 1.0                        # is_start
        x[:, k + 4,  onsets] = torch.rand(30, device=DEVICE)   # start_off
        nt = torch.tensor([0.25, 0.50, 0.75, 1.00] * 8, device=DEVICE)[:30]
        x[:, k + 16, onsets] = nt                         # note_type
        # Hold body for 10 random runs of length 3
        for _ in range(10):
            t0 = int(torch.randint(0, T // 2 - 5, ()))
            x[:, k + 8,  t0 : t0 + 3] = 1.0              # is_holding
            x[:, k + 12, t0 + 2]      = torch.rand(1, device=DEVICE)  # end_offset

    valid_flag[:, T // 2:] = 0.0   # zero-out second half
    return x, valid_flag


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_encoder_shape() -> None:
    """Encoder raw output must be (B, 2*z_channels, T_lat)."""
    vae = _make_vae()
    x, _ = _make_input()
    with torch.no_grad():
        h = vae.encoder(x)
    assert h.shape == (B, Z_CH * 2, T_LAT), \
        f"[1] Encoder shape mismatch: {h.shape}  expected {(B, Z_CH*2, T_LAT)}"
    print(f"[1] encoder output : {tuple(h.shape)}  ✓")


def test_decoder_shape() -> None:
    """Decoder must map (B, z, T_lat) → (B, NUM_CHANNELS, T)."""
    vae = _make_vae()
    z   = torch.randn(B, Z_CH, T_LAT, device=DEVICE)
    with torch.no_grad():
        recon = vae.decode(z)
    assert recon.shape == (B, NUM_CHANNELS, T), \
        f"[2] Decoder shape mismatch: {recon.shape}"
    print(f"[2] decoder output : {tuple(recon.shape)}  ✓")


def test_forward_shapes() -> None:
    """Full VAE forward pass: recon + posterior shapes."""
    vae  = _make_vae()
    x, _ = _make_input()
    with torch.no_grad():
        recon, post = vae(x, sample=False)   # deterministic (mode)
    assert recon.shape == x.shape, \
        f"[3] Recon shape mismatch: {recon.shape}"
    assert post.mean.shape == (B, Z_CH, T_LAT), \
        f"[3] Posterior mean shape: {post.mean.shape}"
    print(f"[3] recon shape    : {tuple(recon.shape)}  ✓")
    print(f"    latent shape   : {tuple(post.mean.shape)}  ✓")


def test_loss_values() -> None:
    """ChartReconLoss: scalar tensor + float log dict."""
    vae     = _make_vae()
    loss_fn = ChartReconLoss()
    x, vf   = _make_input()
    with torch.no_grad():
        recon, _ = vae(x)
    loss, log = loss_fn(x, recon, vf)
    assert loss.shape == (), f"[4] Loss must be scalar, got shape {loss.shape}"
    assert all(isinstance(v, float) for v in log.values()), \
        "[4] Log values must be Python floats"
    print(f"[4] recon loss     : {loss.item():.6f}  ✓")
    for k, v in log.items():
        print(f"    {k:<20s}: {v:.6f}")


def test_compute_loss_backward() -> None:
    """compute_loss() returns total loss; backward must not raise."""
    vae     = _make_vae()
    loss_fn = ChartReconLoss()
    x, vf   = _make_input()
    loss, log = vae.compute_loss(x, vf, loss_fn)
    assert "kl_loss"    in log, "[5] kl_loss missing from log"
    assert "total_loss" in log, "[5] total_loss missing from log"
    loss.backward()   # must not raise
    print(f"[5] total_loss     : {log['total_loss']:.6f}  ✓  "
          f"(recon + {log['kl_loss']:.2e} * kl)")
    print("    backward pass  : ✓")


def test_kl_magnitude() -> None:
    """KL from random init should be small (near 0 for zero-mean encoder)."""
    vae  = _make_vae()
    x, _ = _make_input()
    with torch.no_grad():
        post = vae.encode(x)
        kl   = post.kl().item()
    # From random init the KL is not huge (< 10 is a sensible sanity bound)
    assert kl < 50.0, f"[6] KL suspiciously large: {kl:.4f}"
    print(f"[6] KL (init)      : {kl:.6f}  ✓")


def test_param_count() -> None:
    """Report parameter counts for inspection."""
    vae = _make_vae()
    enc = sum(p.numel() for p in vae.encoder.parameters())
    dec = sum(p.numel() for p in vae.decoder.parameters())
    tot = sum(p.numel() for p in vae.parameters())
    print(f"[7] encoder params : {enc:>10,}")
    print(f"    decoder params : {dec:>10,}")
    print(f"    total params   : {tot:>10,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"Input  : (B={B}, C={NUM_CHANNELS}, T={T})")
    print(f"Latent : (B={B}, z={Z_CH}, T_lat={T_LAT})   [stride={STRIDE}]\n")

    test_encoder_shape()
    test_decoder_shape()
    test_forward_shapes()
    test_loss_values()
    test_compute_loss_backward()
    test_kl_magnitude()
    test_param_count()

    print("\n=== All P2 smoke tests passed ===")

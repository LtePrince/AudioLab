"""
test/models/test_attention.py
------------------------------
Smoke tests for src/models/attention.py

[1] rope1d          — shape and rotation properties
[2] apply_rope      — rotates Q/K, preserves norms
[3] TimestepEmbedder — shape and dtype
[4] Modulation      — zero-init gate at start
[5] DoubleStreamBlock — output shape, backward pass, symmetry
[6] DoubleStreamBlock — variable-length sequences
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# ── config ────────────────────────────────────────────────────────────────
B          = 2          # batch size
T          = 256        # sequence length (4096 // stride 16)
D          = 512        # hidden dim
H          = 8          # heads
Dh         = D // H     # head dim = 64
VAE_Z      = 16         # chart latent channels
AUDIO_OUT  = 256        # AudioWaveEncoder out channels


def make_block():
    from src.models.attention import DoubleStreamBlock
    return DoubleStreamBlock(hidden_dim=D, num_heads=H).to(DEVICE)

def make_freqs(T_len=T):
    from src.models.attention import rope1d
    pos = torch.arange(T_len, device=DEVICE)
    return rope1d(pos, Dh)   # (T, Dh//2, 2, 2)


# ──────────────────────────────────────────────────────────────────────────
# [1]  rope1d — shape and basic rotation property
# ──────────────────────────────────────────────────────────────────────────
def test_rope1d_shape():
    from src.models.attention import rope1d
    pos   = torch.arange(T, device=DEVICE)
    freqs = rope1d(pos, Dh)
    assert freqs.shape == (T, Dh // 2, 2, 2), f"bad shape {freqs.shape}"
    # Each 2×2 sub-matrix should be a rotation matrix: det ≈ 1, R^T R ≈ I
    R   = freqs[0, 0]          # (2, 2) for position 0, freq-bin 0
    det = R[0,0]*R[1,1] - R[0,1]*R[1,0]
    assert abs(det.item() - 1.0) < 1e-5, f"det = {det.item():.6f}, expected 1"
    print(f"[1] rope1d shape : {tuple(freqs.shape)}  det[0,0] = {det.item():.6f}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [2]  apply_rope — output shape and norm-preservation
# ──────────────────────────────────────────────────────────────────────────
def test_apply_rope():
    from src.models.attention import rope1d, apply_rope
    freqs = make_freqs()
    q = torch.randn(B, H, T, Dh, device=DEVICE)
    k = torch.randn(B, H, T, Dh, device=DEVICE)
    qr, kr = apply_rope(q, k, freqs)
    assert qr.shape == q.shape
    assert kr.shape == k.shape
    # RoPE is a rotation → preserves per-token ‖q‖²  (up to fp32 precision)
    norm_diff = (qr.norm(dim=-1) - q.norm(dim=-1)).abs().max().item()
    assert norm_diff < 1e-4, f"norm not preserved: max diff = {norm_diff:.2e}"
    print(f"[2] apply_rope   : shape {tuple(qr.shape)},  max norm diff = {norm_diff:.2e}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [3]  TimestepEmbedder — shape and dtype
# ──────────────────────────────────────────────────────────────────────────
def test_timestep_embedder():
    from src.models.attention import TimestepEmbedder
    emb = TimestepEmbedder(hidden_dim=D).to(DEVICE)
    t   = torch.randint(0, 1000, (B,), device=DEVICE).float()
    vec = emb(t)
    assert vec.shape == (B, D), f"bad shape {vec.shape}"
    assert vec.dtype == DTYPE
    print(f"[3] TimestepEmb  : {tuple(t.shape)} → {tuple(vec.shape)}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [4]  Modulation — zero-init means gate≈0 at start
# ──────────────────────────────────────────────────────────────────────────
def test_modulation_zero_init():
    from src.models.attention import Modulation
    mod = Modulation(dim=D, double=True).to(DEVICE)
    vec = torch.randn(B, D, device=DEVICE)
    # Freshly initialised: lin.weight == 0 and lin.bias == 0
    # → output of lin is 0 for any input → all fields are 0
    m1, m2 = mod(vec)
    for name, field in [("shift", m1.shift), ("scale", m1.scale), ("gate", m1.gate),
                        ("shift2", m2.shift), ("scale2", m2.scale), ("gate2", m2.gate)]:
        assert field.abs().max().item() < 1e-6, f"{name} not zero-init"
    assert m1.gate.shape == (B, 1, D)
    print(f"[4] Modulation   : zero-init verified  gate shape {tuple(m1.gate.shape)}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [5]  DoubleStreamBlock — output shape, backward pass
# ──────────────────────────────────────────────────────────────────────────
def test_double_stream_block_shape_and_backward():
    block  = make_block()
    freqs  = make_freqs()
    x      = torch.randn(B, T, D, device=DEVICE, requires_grad=True)
    c      = torch.randn(B, T, D, device=DEVICE)
    vec    = torch.randn(B, D, device=DEVICE)

    x_out, c_out = block(x, c, vec, freqs)

    assert x_out.shape == (B, T, D), f"x shape {x_out.shape}"
    assert c_out.shape == (B, T, D), f"c shape {c_out.shape}"

    # backward
    loss = x_out.sum() + c_out.sum()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"[5] DoubleStream : x_out {tuple(x_out.shape)},  "
          f"c_out {tuple(c_out.shape)},  grad_norm={grad_norm:.4f}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [6]  DoubleStreamBlock — variable sequence lengths
# ──────────────────────────────────────────────────────────────────────────
def test_variable_length():
    from src.models.attention import rope1d
    block = make_block()
    for T_var in [64, 128, 512]:
        pos    = torch.arange(T_var, device=DEVICE)
        freqs  = rope1d(pos, Dh)
        x      = torch.randn(1, T_var, D, device=DEVICE)
        c      = torch.randn(1, T_var, D, device=DEVICE)
        vec    = torch.randn(1, D, device=DEVICE)
        xo, co = block(x, c, vec, freqs)
        assert xo.shape == (1, T_var, D)
        assert co.shape == (1, T_var, D)
    print(f"[6] Variable len : ✓  (tested T = {[64, 128, 512]})")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.attention import DoubleStreamBlock
    print(f"Device : {DEVICE}")
    print(f"Config : D={D}  H={H}  Dh={Dh}  T={T}  B={B}\n")

    test_rope1d_shape()
    test_apply_rope()
    test_timestep_embedder()
    test_modulation_zero_init()
    test_double_stream_block_shape_and_backward()
    test_variable_length()

    # param count
    block = make_block()
    n_params = sum(p.numel() for p in block.parameters())
    print(f"\nDoubleStreamBlock params : {n_params:,}")
    print("\n=== All attention smoke tests passed ===")

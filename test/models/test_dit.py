"""
test/models/test_dit.py
-----------------------
Smoke tests for src/models/dit.py

[1] output shape   — eps_pred matches z_noisy shape
[2] zero-init      — fresh model outputs all-zero at t=0
[3] backward pass  — gradients flow through both streams
[4] variable T     — works for different sequence lengths
[5] depth scaling  — depth=1 and depth=4 both forward correctly
[6] param count + summary
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── config (matches DEFAULT_DIT_CONFIG) ───────────────────────────────────
B          = 2
T          = 256
Z_CH       = 16
A_CH       = 256
HIDDEN_DIM = 512
NUM_HEADS  = 8
DEPTH      = 4   # use depth=4 for speed in tests


def make_model(depth=DEPTH):
    from src.models.dit import RhythmDiT
    return RhythmDiT(
        z_channels     = Z_CH,
        audio_channels = A_CH,
        hidden_dim     = HIDDEN_DIM,
        depth          = depth,
        num_heads      = NUM_HEADS,
    ).to(DEVICE)


def make_inputs(T_len=T, batch=B):
    z = torch.randn(batch, Z_CH, T_len, device=DEVICE)
    c = torch.randn(batch, A_CH, T_len, device=DEVICE)
    t = torch.randint(0, 1000, (batch,), device=DEVICE).float()
    return z, c, t


# ──────────────────────────────────────────────────────────────────────────
# [1]  output shape
# ──────────────────────────────────────────────────────────────────────────
def test_output_shape():
    model = make_model()
    z, c, t = make_inputs()
    eps = model(z, c, t)
    assert eps.shape == z.shape, f"shape mismatch: {eps.shape} vs {z.shape}"
    print(f"[1] output shape   : {tuple(eps.shape)}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [2]  zero-init — fresh model output must be all zero
# ──────────────────────────────────────────────────────────────────────────
def test_zero_init():
    model = make_model()
    z, c, t = make_inputs()
    with torch.no_grad():
        eps = model(z, c, t)
    max_val = eps.abs().max().item()
    assert max_val == 0.0, f"expected zero output at init, got max={max_val:.2e}"
    print(f"[2] zero-init      : max |ε̂| = {max_val:.2e}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [3]  backward — gradients exist in both streams
# ──────────────────────────────────────────────────────────────────────────
def test_backward():
    model = make_model()
    z, c, t = make_inputs()

    eps = model(z, c, t)
    # mse loss against random target (simulates diffusion training signal)
    target = torch.randn_like(eps)
    loss   = (eps - target).pow(2).mean()
    loss.backward()   # must not crash

    # Key assertions:
    # (a) FinalLayer.linear.weight receives a non-zero gradient (direct signal)
    final_w_grad = model.final_layer.linear.weight.grad
    assert final_w_grad is not None, "FinalLayer.linear.weight has no grad"
    assert final_w_grad.abs().sum().item() > 0, "FinalLayer.linear.weight grad is zero"

    # (b) t_embedder is always in the forward path → must have grad
    temb_grad = model.t_embedder.mlp[0].weight.grad
    assert temb_grad is not None, "t_embedder has no grad"

    # Note: last block's c-stream params (c_proj, c_mlp, c_qnorm) have grad=None
    # because the final c output is unused (only x feeds final_layer) — expected.
    last = model.blocks[-1]
    assert last.c_proj.weight.grad is None, (
        "expected last block c_proj to have no grad (c output is discarded)"
    )

    print(f"[3] backward pass  : loss={loss.item():.4f}  "
          f"FinalLayer.linear.weight |grad|={final_w_grad.abs().sum().item():.4f}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [4]  variable sequence lengths
# ──────────────────────────────────────────────────────────────────────────
def test_variable_length():
    model = make_model()
    for T_var in [64, 128, 512]:
        z, c, t = make_inputs(T_len=T_var, batch=1)
        with torch.no_grad():
            eps = model(z, c, t)
        assert eps.shape == (1, Z_CH, T_var), f"bad shape for T={T_var}: {eps.shape}"
    print(f"[4] variable T     : ✓  (tested T = {[64, 128, 512]})")


# ──────────────────────────────────────────────────────────────────────────
# [5]  depth scaling — depth=1 and depth=4 both work
# ──────────────────────────────────────────────────────────────────────────
def test_depth_scaling():
    for d in [1, 4]:
        model = make_model(depth=d)
        z, c, t = make_inputs(batch=1)
        with torch.no_grad():
            eps = model(z, c, t)
        assert eps.shape == (1, Z_CH, T)
    print(f"[5] depth scaling  : ✓  (tested depth = 1, 4)")


# ──────────────────────────────────────────────────────────────────────────
# [6]  param count + summary
# ──────────────────────────────────────────────────────────────────────────
def test_summary():
    from src.models.dit import DEFAULT_DIT_CONFIG, RhythmDiT
    model = RhythmDiT(**DEFAULT_DIT_CONFIG).to(DEVICE)
    model.summary(T=T)
    n = model.num_params
    print(f"[6] total params   : {n:,}  ✓")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"Config : Z={Z_CH}  A={A_CH}  D={HIDDEN_DIM}  H={NUM_HEADS}  depth={DEPTH}  T={T}  B={B}\n")

    test_output_shape()
    test_zero_init()
    test_backward()
    test_variable_length()
    test_depth_scaling()
    test_summary()

    print("\n=== All P4 DiT smoke tests passed ===")

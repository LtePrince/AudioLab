"""
test/diffusion/test_schedule.py
--------------------------------
Smoke tests for src/diffusion/schedule.py

[1] make_beta_schedule — linear: 单调递增，值域 (0,1)
[2] make_beta_schedule — cosine: 单调递增，值域 (0,1)
[3] alphas_bar        — 从 ≈1 单调递减到 ≈0
[4] q_sample          — 输出形状；t=0 时接近 x0；t=T-1 时接近纯噪声
[5] snr               — t=0 时 SNR >> 1；t=T-1 时 SNR << 1
[6] posterior_var     — 非负，在 t=1 时为 0（因为 alphas_bar_prev[0]=1）
[7] summary           — 不崩溃，打印输出
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T      = 1000
B      = 4
C      = 16
L      = 256   # latent sequence length


def make_schedule(schedule="cosine"):
    from src.diffusion.schedule import NoiseSchedule
    return NoiseSchedule(schedule=schedule, T=T).to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────
# [1]  make_beta_schedule — linear
# ──────────────────────────────────────────────────────────────────────────
def test_linear_betas():
    from src.diffusion.schedule import make_beta_schedule
    betas = make_beta_schedule("linear", T=T)
    assert betas.shape == (T,)
    assert betas.dtype == np.float64
    assert (betas > 0).all() and (betas < 1).all()
    # 单调不严格递增（sqrt 插值后平方保证单调）
    assert (np.diff(betas) >= 0).all()
    print(f"[1] linear betas   : min={betas.min():.2e}  max={betas.max():.2e}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [2]  make_beta_schedule — cosine
# ──────────────────────────────────────────────────────────────────────────
def test_cosine_betas():
    from src.diffusion.schedule import make_beta_schedule
    betas = make_beta_schedule("cosine", T=T)
    assert betas.shape == (T,)
    assert (betas > 0).all() and (betas <= 0.999).all()
    print(f"[2] cosine betas   : min={betas.min():.2e}  max={betas.max():.2e}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [3]  alphas_bar — 单调递减 [≈1, ≈0]
# ──────────────────────────────────────────────────────────────────────────
def test_alphas_bar():
    sch = make_schedule("cosine")
    ab  = sch.alphas_bar.cpu()
    assert ab[0]  > 0.99,  f"alphas_bar[0]={ab[0]:.4f} should be ≈1"
    assert ab[-1] < 0.01,  f"alphas_bar[-1]={ab[-1]:.6f} should be ≈0"
    assert (ab[:-1] >= ab[1:]).all(), "alphas_bar must be non-increasing"
    print(f"[3] alphas_bar     : t=0→{ab[0]:.4f}  t=499→{ab[499]:.4f}  t=999→{ab[-1]:.6f}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [4]  q_sample — 形状；极端时间步语义
# ──────────────────────────────────────────────────────────────────────────
def test_q_sample():
    sch  = make_schedule("cosine")
    x0   = torch.randn(B, C, L, device=DEVICE)
    eps  = torch.randn_like(x0)

    # 形状
    t_mid = torch.full((B,), T // 2, device=DEVICE)
    xt, noise_out = sch.q_sample(x0, t_mid, noise=eps)
    assert xt.shape    == x0.shape, f"xt shape {xt.shape}"
    assert noise_out is eps,        "should return the provided noise tensor"

    # t=0 时 x_t ≈ x0  (alphas_bar[0] ≈ 1)
    t0 = torch.zeros(B, dtype=torch.long, device=DEVICE)
    xt0, _ = sch.q_sample(x0, t0)
    err0 = (xt0 - x0).abs().mean().item()
    assert err0 < 0.1, f"t=0: xt should ≈ x0, mean abs diff={err0:.4f}"

    # t=T-1 时 SNR 极低，x_t 以噪声为主
    tT = torch.full((B,), T - 1, dtype=torch.long, device=DEVICE)
    xtT, noiseT = sch.q_sample(x0, tT)
    # 与纯噪声的相似度应远高于与 x0 的相似度
    diff_noise = (xtT - noiseT).abs().mean().item()
    diff_x0    = (xtT - x0).abs().mean().item()
    assert diff_noise < diff_x0, \
        f"t=T-1: xt should be closer to noise than x0; diff_noise={diff_noise:.4f}, diff_x0={diff_x0:.4f}"

    print(f"[4] q_sample       : shape {tuple(xt.shape)},  "
          f"err@t=0={err0:.4f},  diff_noise/x0={diff_noise:.3f}/{diff_x0:.3f}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [5]  snr — 单调递减，t=0 >> 1，t=T-1 << 1
# ──────────────────────────────────────────────────────────────────────────
def test_snr():
    sch  = make_schedule("cosine")
    t0   = torch.tensor([0],     device=DEVICE)
    tT   = torch.tensor([T - 1], device=DEVICE)
    snr0 = sch.snr(t0).item()
    snrT = sch.snr(tT).item()
    assert snr0 > 100,  f"SNR at t=0 should be >> 1, got {snr0:.2f}"
    assert snrT < 0.01, f"SNR at t=T-1 should be << 1, got {snrT:.4f}"
    print(f"[5] snr            : t=0→{snr0:.1f}  t={T-1}→{snrT:.5f}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [6]  posterior_variance — 非负
# ──────────────────────────────────────────────────────────────────────────
def test_posterior_variance():
    sch = make_schedule("cosine")
    pv  = sch.posterior_variance.cpu()
    assert (pv >= 0).all(), "posterior_variance must be non-negative"
    # t=1 时 alphas_bar_prev[1]=alphas_bar[0]≈1 → posterior_var[1] ≈ 0
    assert pv[0].item() < 1e-6, f"posterior_var[0]={pv[0].item():.2e} should be ≈0"
    print(f"[6] posterior_var  : min={pv.min():.2e}  max={pv.max():.4f}  pv[0]={pv[0]:.2e}  ✓")


# ──────────────────────────────────────────────────────────────────────────
# [7]  summary — 不崩溃
# ──────────────────────────────────────────────────────────────────────────
def test_summary():
    sch = make_schedule("cosine")
    sch.summary()
    print(f"[7] summary        : ✓")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device : {DEVICE}\nT={T}  B={B}  C={C}  L={L}\n")
    test_linear_betas()
    test_cosine_betas()
    test_alphas_bar()
    test_q_sample()
    test_snr()
    test_posterior_variance()
    test_summary()
    print("\n=== All schedule smoke tests passed ===")

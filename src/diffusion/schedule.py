"""
src/diffusion/schedule.py
─────────────────────────
DDPM noise schedule (pure math, no nn.Module).

Public API
~~~~~~~~~~
  make_beta_schedule(schedule, T, ...)  → betas  (T,) float64 ndarray
  NoiseSchedule                         → lightweight container for all precomputed buffers
    .q_sample(x0, t, noise)            → x_t   forward diffusion (used during training)
    .snr(t)                            → SNR_t  signal-to-noise ratio (for loss weighting)

Supported schedules
~~~~~~~~~~~~~~~~~~~
  linear : β_t linearly interpolated from β_start to β_end (original DDPM)
  cosine : α̅_t = cos²((t/T + s)/(1+s) · π/2), smoother (Improved DDPM)
           converges faster for short sequences (T=256 latent); recommended for this project

Math notation
~~~~~~~~~~~~~
  T          : total diffusion steps, default 1000
  β_t        : per-step noise level, monotonically increasing, range ≈ [1e-4, 2e-2]
  α_t        = 1 - β_t
  ᾱ_t        = ∏_{s=1}^{t} α_s       (cumulative product, 1 → 0)
  √ᾱ_t       : signal retention coefficient
  √(1-ᾱ_t)  : noise mixing coefficient

  x_t = √ᾱ_t · x₀  +  √(1-ᾱ_t) · ε,   ε ~ N(0, I)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor

__all__ = ["make_beta_schedule", "NoiseSchedule", "DEFAULT_SCHEDULE_CONFIG"]

DEFAULT_SCHEDULE_CONFIG = dict(
    schedule   = "cosine",  # "linear" or "cosine"
    T          = 1000,      # total diffusion steps
    beta_start = 1e-4,      # linear schedule starting β
    beta_end   = 2e-2,      # linear schedule ending β
    cosine_s   = 8e-3,      # cosine schedule offset (prevents β₁ ≈ 0)
)


# ────────────────────────────────────────────────────────────────────────────
# β schedule construction
# ────────────────────────────────────────────────────────────────────────────

def make_beta_schedule(
    schedule:   Literal["linear", "cosine"] = "cosine",
    T:          int   = 1000,
    beta_start: float = 1e-4,
    beta_end:   float = 2e-2,
    cosine_s:   float = 8e-3,
) -> np.ndarray:
    """Compute the β schedule as a float64 numpy array of shape (T,).

    Args:
        schedule  : "linear" or "cosine"
        T         : number of diffusion steps
        beta_start: (linear only) first β value
        beta_end  : (linear only) last  β value
        cosine_s  : (cosine only) small offset to prevent β₁ ≈ 0

    Returns:
        betas : (T,) float64 ndarray, values in (0, 0.999]
    """
    if schedule == "linear":
        # Linearly interpolate √β_start → √β_end then square
        # (consistent with Mug-Diffusion)
        betas = (
            torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end),
                           T, dtype=torch.float64) ** 2
        ).numpy()

    elif schedule == "cosine":
        # α̅_t = cos²((t/T + s) / (1+s) · π/2)  →  β_t = 1 - α̅_t / α̅_{t-1}
        # Reference: Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models"
        steps = torch.arange(T + 1, dtype=torch.float64)
        f     = torch.cos((steps / T + cosine_s) / (1.0 + cosine_s) * math.pi / 2.0) ** 2
        f     = f / f[0]                     # normalise so that α̅_0 = 1
        betas = 1.0 - f[1:] / f[:-1]        # β_t = 1 - α̅_t / α̅_{t-1}
        betas = betas.clamp(0.0, 0.999).numpy()

    else:
        raise ValueError(f"Unknown schedule '{schedule}'. Choose 'linear' or 'cosine'.")

    assert betas.shape == (T,), f"Expected ({T},), got {betas.shape}"
    assert (betas > 0).all() and (betas < 1).all(), "betas must be in (0, 1)"
    return betas


# ────────────────────────────────────────────────────────────────────────────
# NoiseSchedule
# ────────────────────────────────────────────────────────────────────────────

class NoiseSchedule:
    """Lightweight container for all precomputed DDPM buffers (not an nn.Module).

    All buffers are moved at once via ``to(device)``.
    During training only ``q_sample()`` is needed; ``sampler.py`` reads the
    buffers directly during DDIM inference.

    Args:
        schedule  : "linear" or "cosine"
        T         : total steps (default 1000)
        beta_start: linear schedule start value
        beta_end  : linear schedule end value
        cosine_s  : cosine schedule offset
    """

    def __init__(
        self,
        schedule:   str   = "cosine",
        T:          int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 2e-2,
        cosine_s:   float = 8e-3,
    ):
        self.T = T
        betas = make_beta_schedule(
            schedule=schedule, T=T,
            beta_start=beta_start, beta_end=beta_end, cosine_s=cosine_s,
        )                                                 # (T,) float64

        alphas      = 1.0 - betas                        # α_t
        alphas_bar  = np.cumprod(alphas)                 # ᾱ_t
        alphas_bar_prev = np.append(1.0, alphas_bar[:-1])  # ᾱ_{t-1}, ᾱ_0 = 1

        def f32(x): return torch.tensor(x, dtype=torch.float32)

        # ── core buffers ───────────────────────────────────────────────
        self.betas                    = f32(betas)          # β_t
        self.alphas_bar               = f32(alphas_bar)     # ᾱ_t
        self.alphas_bar_prev          = f32(alphas_bar_prev)
        self.sqrt_alphas_bar          = f32(np.sqrt(alphas_bar))        # √ᾱ_t
        self.sqrt_one_minus_alphas_bar = f32(np.sqrt(1.0 - alphas_bar)) # √(1-ᾱ_t)
        self.log_one_minus_alphas_bar = f32(np.log(1.0 - alphas_bar))   # log(1-ᾱ_t)
        # posterior variance σ_t² = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)  (DDPM eq. 7)
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.posterior_variance       = f32(posterior_var)
        self.posterior_log_var_clipped = f32(
            np.log(np.maximum(posterior_var, 1e-20))     # avoid log(0)
        )
        # DDPM posterior mean coefficients
        self.posterior_mean_coef1 = f32(
            betas * np.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
        )  # coefficient for x₀
        self.posterior_mean_coef2 = f32(
            (1.0 - alphas_bar_prev) * np.sqrt(alphas) / (1.0 - alphas_bar)
        )  # coefficient for x_t

    # ── device management ────────────────────────────────────────────

    def to(self, device) -> "NoiseSchedule":
        """Move all buffers to *device* and return self (chainable)."""
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                setattr(self, k, v.to(device))
        return self

    @property
    def device(self):
        return self.betas.device

    # ── core API ──────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:    Tensor,
        t:     Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward diffusion: add t steps of noise to x₀ to produce x_t.

        x_t = √ᾱ_t · x₀  +  √(1-ᾱ_t) · ε

        Args:
            x0    : (B, C, T)  clean latent
            t     : (B,)       integer timesteps in [0, T)
            noise : (B, C, T)  optional; sampled from N(0,I) if None

        Returns:
            x_t   : (B, C, T)  noisy latent
            noise : (B, C, T)  noise actually used (prediction target during training)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # index buffer by timestep t and broadcast to (B, 1, 1)
        sqrt_ab   = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    def snr(self, t: Tensor) -> Tensor:
        """Signal-to-noise ratio at timestep t:  SNR_t = ᾱ_t / (1 - ᾱ_t).

        Args:
            t : (B,) or scalar

        Returns:
            snr : same shape as t
        """
        ab = self.alphas_bar[t]
        return ab / (1.0 - ab)

    # ── utilities ────────────────────────────────────────────────────

    def summary(self):
        """Print key statistics of the noise schedule."""
        ab = self.alphas_bar.cpu().numpy()
        betas = self.betas.cpu().numpy()
        T = self.T
        print(f"NoiseSchedule  T={T}")
        print(f"  β  : min={betas.min():.2e}  max={betas.max():.2e}")
        print(f"  ᾱ  : t=0 → {ab[0]:.4f}  t={T//4} → {ab[T//4]:.4f}  "
              f"t={T//2} → {ab[T//2]:.4f}  t={T-1} → {ab[-1]:.6f}")
        snr0 = ab[0]   / (1 - ab[0])
        snr_half = ab[T//2] / (1 - ab[T//2])
        snrT = ab[-1]  / (1 - ab[-1])
        print(f"  SNR: t=0 → {snr0:.1f}  t={T//2} → {snr_half:.3f}  t={T-1} → {snrT:.4f}")

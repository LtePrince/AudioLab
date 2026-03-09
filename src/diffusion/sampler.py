"""
src/diffusion/sampler.py
------------------------
DDIM inference sampler (P6-1 / P6-3).

Core idea
~~~~~~~~~
DDPM requires T denoising steps (e.g. 1000).  DDIM accelerates inference by
skipping to a subsequence of S << T steps (typically 50-100), and is fully
deterministic when eta=0.

Per-step update (Song et al., 2020):
    x_{t-1} = √ᾱ_{t-1} · x̂₀(xₜ)
             + √(1-ᾱ_{t-1} - σₜ²) · εθ(xₜ,t)
             + σₜ · ε

    x̂₀(xₜ) = (xₜ - √(1-ᾱₜ) · εθ) / √ᾱₜ      ← predicted x₀ at current step

    σₜ      = η · √((1-ᾱ_{t-1}) / (1-ᾱₜ) · (1-ᾱₜ/ᾱ_{t-1}))
              eta=0 → fully deterministic DDIM;  eta=1 → reduces to DDPM

Classifier-Free Guidance (P6-3):
    ε_cfg = ε_uncond + scale · (ε_cond - ε_uncond)

Public API
~~~~~~~~~~
    DDIMSampler.sample(
        dit, audio_c, shape, steps, eta, cfg_scale, audio_c_uncond
    ) → z_0 : (B, C, T)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from src.diffusion.schedule import NoiseSchedule

__all__ = ["DDIMSampler"]


def _make_ddim_timesteps(T: int, S: int) -> np.ndarray:
    """Uniformly select S DDIM sub-steps, ordered from large t to small t.

    Returns:
        ddim_steps : (S,) int, **descending** order (high → low),
                     values in [1, T] inclusive (consistent with Mug-Diffusion)
    """
    c     = T // S
    steps = np.arange(0, T, c) + 1          # [1, 1+c, 1+2c, ...]
    steps = steps[:S]                        # take exactly S steps
    return steps[::-1].copy()               # descending: denoise from T toward 0


class DDIMSampler:
    """DDIM sampler (no neural network weights; pure sampling logic).

    Example usage::

        sampler = DDIMSampler(schedule)
        z0 = sampler.sample(
            dit       = rhythm_dit,
            audio_c   = audio_features,   # (B, a_ch, T)
            shape     = (B, z_ch, T),
            steps     = 50,
            eta       = 0.0,              # 0=deterministic, 1=DDPM
            cfg_scale = 3.0,              # 1.0=no CFG
            audio_c_uncond = None,        # None=use zero vector as uncond
        )

    Args:
        schedule : a ``NoiseSchedule`` instance (already moved to device via ``.to()``)
    """

    def __init__(self, schedule: NoiseSchedule):
        self.schedule = schedule
        self.T        = schedule.T

    # ------------------------------------------------------------------
    # Core: single DDIM denoising step
    # ------------------------------------------------------------------

    def _ddim_step(
        self,
        x_t:     Tensor,        # (B, C, L)  current noisy latent
        eps_t:   Tensor,        # (B, C, L)  noise predicted by DiT
        ab_t:    float,         # ᾱ_t   signal retention at current step
        ab_prev: float,         # ᾱ_{t-1}
        eta:     float,
    ) -> Tensor:
        """Execute one DDIM denoising step and return x_{t-1}."""
        # 1. predict x̂₀ from x_t and ε̂
        sqrt_ab    = ab_t  ** 0.5
        sqrt_1mab  = (1.0 - ab_t)  ** 0.5
        pred_x0    = (x_t - sqrt_1mab * eps_t) / sqrt_ab
        pred_x0    = pred_x0.clamp(-10.0, 10.0)   # numerical stability

        # 2. DDIM σ_t
        sigma_t = eta * ((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)) ** 0.5

        # 3. direction term (residual pointing toward x_t)
        dir_xt = (1.0 - ab_prev - sigma_t ** 2) ** 0.5 * eps_t

        # 4. stochastic term
        noise  = sigma_t * torch.randn_like(x_t) if sigma_t > 0 else 0.0

        # 5. x_{t-1}
        x_prev = ab_prev ** 0.5 * pred_x0 + dir_xt + noise
        return x_prev

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        dit,                                       # RhythmDiT instance
        audio_c:       Tensor,                     # (B, a_ch, T_seq)
        shape:         tuple[int, int, int],       # (B, z_ch, T_seq)
        steps:         int          = 50,
        eta:           float        = 0.0,
        cfg_scale:     float        = 1.0,
        audio_c_uncond: Optional[Tensor] = None,   # None → zero vector
        x_T:           Optional[Tensor] = None,    # initial noise; None → sample N(0,I)
        callback:      Optional[Callable[[int, Tensor], None]] = None,
        show_progress: bool         = True,
    ) -> Tensor:
        """Start from pure noise and return the clean z₀ after DDIM denoising.

        Args:
            dit           : ``RhythmDiT`` model (eval mode / no_grad)
            audio_c       : (B, a_ch, T_seq)  audio condition
            shape         : target latent shape (B, z_ch, T_seq)
            steps         : number of DDIM denoising steps (50-100 recommended)
            eta           : stochasticity (0=deterministic DDIM, 1=DDPM)
            cfg_scale     : CFG guidance strength (1.0=no guidance, >1 enhances condition)
            audio_c_uncond: unconditional audio features; None → all-zeros
            x_T           : starting noise; None → sample from N(0,I)
            callback      : called after each step as callback(step_idx, x_t)
            show_progress : whether to show a tqdm progress bar

        Returns:
            z_0 : (B, z_ch, T_seq)  predicted clean latent
        """
        device = self.schedule.device
        B, C, L = shape

        # ── initial noise ─────────────────────────────────────────────
        x = x_T if x_T is not None else torch.randn(shape, device=device)

        # ── CFG: prepare unconditional batch ──────────────────────────
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            if audio_c_uncond is None:
                audio_c_uncond = torch.zeros_like(audio_c)
            # concatenate [uncond, cond]
            audio_c_cat = torch.cat([audio_c_uncond, audio_c], dim=0)   # (2B, a_ch, L)

        # ── DDIM sub-step sequence (descending) ───────────────────────
        ddim_steps = _make_ddim_timesteps(self.T, steps)  # (S,) descending

        ab = self.schedule.alphas_bar.cpu().numpy()        # (T,) numpy

        iterator = tqdm(ddim_steps, desc="DDIM", disable=not show_progress)

        for i, t_val in enumerate(iterator):
            t_prev = ddim_steps[i + 1] if i + 1 < len(ddim_steps) else 0

            ab_t    = float(ab[t_val - 1])   # ᾱ_t  (buffer is 0-indexed, steps are 1-indexed)
            ab_prev = float(ab[t_prev - 1]) if t_prev > 0 else 1.0

            # timestep tensor
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)

            # ── DiT forward ───────────────────────────────────────────
            if use_cfg:
                x_in = torch.cat([x, x], dim=0)                       # (2B, C, L)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)         # (2B,)
                eps_out = dit(x_in, audio_c_cat, t_in)                 # (2B, C, L)
                eps_uncond, eps_cond = eps_out.chunk(2, dim=0)
                eps_t = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_t = dit(x, audio_c, t_tensor)                      # (B, C, L)

            # ── single DDIM step ──────────────────────────────────────
            x = self._ddim_step(x, eps_t, ab_t, ab_prev, eta)

            if callback is not None:
                callback(i, x)

        return x

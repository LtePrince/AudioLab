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

Prefix conditioning (variable-length windowed inference):
    ``sample(..., prefix_z0=z_known)`` treats the first ``L`` latent tokens as
    KNOWN: at every denoising step the prefix region of x_t is replaced by
    q_sample(z_known, t) (RePaint-style, single pass), so the freely-generated
    suffix is denoised while attending to a consistent, already-decided
    context.  ``sample_windowed()`` uses this to chain overlapping windows
    over arbitrarily long songs.

Public API
~~~~~~~~~~
    DDIMSampler.sample(
        dit, audio_c, shape, steps, eta, cfg_scale, audio_c_uncond, prefix_z0
    ) → z_0 : (B, C, T)
    DDIMSampler.sample_windowed(
        dit, audio_c, z_channels, window, overlap, ...
    ) → z_0 : (B, C, T_total)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from src.diffusion.schedule import NoiseSchedule

__all__ = ["DDIMSampler", "make_window_starts"]


def _make_ddim_timesteps(T: int, S: int) -> np.ndarray:
    """Uniformly select S DDIM sub-steps, ordered from large t to small t.

    Returns:
        ddim_steps : (S,) int, **descending** order (high → low),
                     values in [1, T] inclusive (consistent with Mug-Diffusion)
    """
    c     = max(T // S, 1)                  # S > T degenerates to S = T
    steps = np.arange(0, T, c) + 1          # [1, 1+c, 1+2c, ...]
    steps = steps[:S]                        # take exactly S steps
    return steps[::-1].copy()               # descending: denoise from T toward 0


def make_window_starts(T_total: int, window: int, overlap: int) -> list[int]:
    """Window start offsets covering ``[0, T_total)`` with ≥ *overlap* reuse.

    Windows advance by ``window - overlap``; the final window is left-shifted
    to end exactly at ``T_total`` (so every window has the full ``window``
    length — the length the model was trained on), which can only INCREASE
    its overlap with the previously generated region.

    Requires ``T_total >= window`` and ``0 <= overlap < window``.
    """
    if not 0 <= overlap < window:
        raise ValueError(f"overlap must be in [0, window); got {overlap} vs {window}")
    if T_total < window:
        raise ValueError(f"T_total {T_total} < window {window}: no windowing needed")
    starts: list[int] = []
    s = 0
    while s + window < T_total:
        starts.append(s)
        s += window - overlap
    starts.append(T_total - window)
    return starts


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
        prefix_z0:     Optional[Tensor] = None,    # known clean latent prefix
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
            prefix_z0     : (B, z_ch, L_p) with L_p < T_seq — KNOWN clean latent
                            for the first L_p tokens.  At every step the prefix
                            region is re-noised from prefix_z0 to the current
                            noise level (RePaint single pass), so the suffix is
                            generated in a context consistent with it; the
                            returned prefix equals prefix_z0 exactly.
            callback      : called after each step as callback(step_idx, x_t)
            show_progress : whether to show a tqdm progress bar

        Returns:
            z_0 : (B, z_ch, T_seq)  predicted clean latent
        """
        device = self.schedule.device
        B, C, L = shape

        L_p = 0
        if prefix_z0 is not None:
            L_p = prefix_z0.shape[-1]
            if not 0 < L_p < L:
                raise ValueError(
                    f"prefix_z0 length {L_p} must be in (0, T_seq={L})"
                )
            prefix_z0 = prefix_z0.to(device)

        # ── initial noise ─────────────────────────────────────────────
        # clone: prefix injection writes in place and must not mutate the
        # caller's x_T tensor
        x = x_T.clone().to(device) if x_T is not None else torch.randn(shape, device=device)

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

            # ── prefix conditioning: re-noise the known region to ᾱ_t ─
            # x enters this iteration at noise level ᾱ_t, so the known
            # prefix must be injected at exactly that level.
            if L_p > 0:
                x[:, :, :L_p] = (
                    ab_t ** 0.5 * prefix_z0
                    + (1.0 - ab_t) ** 0.5 * torch.randn_like(prefix_z0)
                )

            # Timestep passed to the DiT uses the TRAINING convention:
            # train.py samples t ∈ [0, T-1] and indexes alphas_bar[t], so the
            # noise level ab[t_val-1] corresponds to conditioning t = t_val - 1.
            t_tensor = torch.full((B,), t_val - 1, device=device, dtype=torch.float32)

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

        # the known prefix is returned verbatim (fully denoised by definition)
        if L_p > 0:
            x[:, :, :L_p] = prefix_z0

        return x

    # ------------------------------------------------------------------
    # Windowed sampling for songs longer than the trained token length
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_windowed(
        self,
        dit,
        audio_c:        Tensor,                    # (B, a_ch, T_total)
        z_channels:     int,
        window:         int,
        overlap:        int,
        steps:          int          = 50,
        eta:            float        = 0.0,
        cfg_scale:      float        = 1.0,
        show_progress:  bool         = True,
    ) -> Tensor:
        """Generate a latent longer than the trained window by chaining
        overlapping prefix-conditioned windows.

        Window k sees RoPE positions 0..window-1 — exactly the positional
        range the model was trained on — while its first ``overlap`` (or
        more, for the end-aligned final window) tokens are frozen to the
        previously generated content via ``prefix_z0``, so the seams stay
        coherent without any positional extrapolation.

        Args:
            dit        : RhythmDiT (eval)
            audio_c    : (B, a_ch, T_total) audio condition over the WHOLE song,
                         already on the latent token grid
            z_channels : latent channels (16)
            window     : tokens per window — use the trained length (256)
            overlap    : tokens of context carried into each next window
            steps/eta/cfg_scale : as in :meth:`sample` (applied per window)

        Returns:
            z_0 : (B, z_channels, T_total)
        """
        B, _, T_total = audio_c.shape
        starts = make_window_starts(T_total, window, overlap)

        z_full: Optional[Tensor] = None
        for k, s in enumerate(starts):
            a_win = audio_c[:, :, s : s + window]
            if show_progress:
                print(f"[windowed] window {k + 1}/{len(starts)}  "
                      f"tokens [{s}, {s + window})")

            if z_full is None:
                z_win = self.sample(
                    dit, a_win, (B, z_channels, window),
                    steps=steps, eta=eta, cfg_scale=cfg_scale,
                    show_progress=show_progress,
                )
                z_full = z_win
            else:
                L_p = z_full.shape[-1] - s      # overlap with generated region
                # With overlap=0 every seam has L_p=0 — windows become
                # independent tiles with no coherence guarantee (opt-in).
                # With overlap>0, L_p >= overlap for every window.
                prefix = z_full[:, :, s : s + L_p] if L_p > 0 else None
                z_win = self.sample(
                    dit, a_win, (B, z_channels, window),
                    steps=steps, eta=eta, cfg_scale=cfg_scale,
                    prefix_z0=prefix,
                    show_progress=show_progress,
                )
                z_full = torch.cat([z_full, z_win[:, :, L_p:]], dim=-1)

        assert z_full is not None and z_full.shape[-1] == T_total
        return z_full

"""
src/models/dit.py
-----------------
RhythmDiT: 1-D MMDiT diffusion transformer for Phigros rhythm generation.

Pipeline context
~~~~~~~~~~~~~~~~
    ChartVAE.encode  →  z  (B, z_ch=16,  T=256)   ← noisy latent input
    AudioWaveEncoder →  c  (B, a_ch=256, T=256)   ← audio condition
    RhythmDiT        →  ε̂  (B, z_ch=16,  T=256)   ← predicted noise
    ChartVAE.decode  →  chart (B, 20, 4096)

Architecture
~~~~~~~~~~~~
    1.  x  = Linear(z_ch  → D)(z_noisy.T)   chart tokens   (B, T, D)
    2.  c  = Linear(a_ch  → D)(audio_c.T)   audio tokens   (B, T, D)
    3.  vec = TimestepEmbedder(t)             cond vector    (B, D)
    4.  freqs = rope1d(0..T-1, head_dim)      RoPE matrices  (T, Dh//2, 2, 2)
    5.  for block in blocks: x, c = block(x, c, vec, freqs)
    6.  ε̂ = FinalLayer(x, vec).T             (B, z_ch, T)

Notes
~~~~~
- Both streams are updated in every block (full bidirectional cross-attention).
  Only the chart stream x feeds the noise-prediction output head.
- The audio stream c is treated as a learnable conditioning context.
  Its tokens shift / scale as the model learns which audio frames matter.
- RoPE uses the same positions for x and c because both are time-aligned
  (identical stride = 16 from VAE and AudioWaveEncoder).
- FinalLayer is zero-initialised → ε̂ = 0 at the very start of training,
  which makes training loss well-defined from step 1 (DiT §3.4).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.models.attention import (
    DoubleStreamBlock,
    TimestepEmbedder,
    rope1d,
)

__all__ = ["RhythmDiT", "DEFAULT_DIT_CONFIG"]


DEFAULT_DIT_CONFIG = dict(
    z_channels     = 16,       # ChartVAE z_channels
    audio_channels = 256,      # AudioWaveEncoder out_channels
    hidden_dim     = 512,      # transformer width D
    depth          = 12,       # number of DoubleStreamBlocks
    num_heads      = 8,        # attention heads H  (hidden_dim must be divisible)
    mlp_ratio      = 4.0,      # FFN hidden-dim multiplier
    dropout        = 0.0,      # FFN dropout
    rope_theta     = 10000.0,  # RoPE base frequency
)


# ---------------------------------------------------------------------------
# adaLN-Zero Final Layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """adaLN-Zero output projection.

    Applies a modulated LayerNorm then projects from ``hidden_dim`` to
    ``out_channels``.  Both the modulation MLP and the output linear are
    **zero-initialised**, so the model predicts exactly zero noise at
    the very beginning of training (stabilises early gradient flow).

    Args:
        hidden_dim  : model width D
        out_channels: number of output channels  (= z_channels)
    """

    def __init__(self, hidden_dim: int, out_channels: int):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mod    = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        self.linear = nn.Linear(hidden_dim, out_channels, bias=True)

        # zero-init  →  zero output at initialisation
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """
        x   : (B, T, D)
        vec : (B, D)
        →     (B, T, out_channels)
        """
        shift, scale = self.mod(vec).unsqueeze(1).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1.0 + scale) + shift)


# ---------------------------------------------------------------------------
# RhythmDiT
# ---------------------------------------------------------------------------

class RhythmDiT(nn.Module):
    """MMDiT-based 1-D diffusion transformer for rhythm chart generation.

    Args:
        z_channels     : chart latent channels   (default 16)
        audio_channels : audio cond channels     (default 256)
        hidden_dim     : model width D            (default 512)
        depth          : number of DoubleStreamBlocks (default 12)
        num_heads      : attention heads H        (default 8)
        mlp_ratio      : FFN expansion factor     (default 4.0)
        dropout        : FFN dropout              (default 0.0)
        rope_theta     : RoPE base frequency      (default 10000.0)
    """

    def __init__(
        self,
        z_channels:     int   = 16,
        audio_channels: int   = 256,
        hidden_dim:     int   = 512,
        depth:          int   = 12,
        num_heads:      int   = 8,
        mlp_ratio:      float = 4.0,
        dropout:        float = 0.0,
        rope_theta:     float = 10000.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.z_channels     = z_channels
        self.hidden_dim     = hidden_dim
        self.num_heads      = num_heads
        self.head_dim       = hidden_dim // num_heads
        self.rope_theta     = rope_theta

        # ── input projections ─────────────────────────────────────────
        self.x_proj = nn.Linear(z_channels,     hidden_dim, bias=True)
        self.c_proj = nn.Linear(audio_channels, hidden_dim, bias=True)

        # ── timestep conditioning ─────────────────────────────────────
        self.t_embedder = TimestepEmbedder(hidden_dim)

        # ── transformer body ──────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DoubleStreamBlock(
                hidden_dim = hidden_dim,
                num_heads  = num_heads,
                mlp_ratio  = mlp_ratio,
                dropout    = dropout,
            )
            for _ in range(depth)
        ])

        # ── output head ───────────────────────────────────────────────
        self.final_layer = FinalLayer(hidden_dim, z_channels)

    # ------------------------------------------------------------------

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------

    def forward(
        self,
        z_noisy: Tensor,
        audio_c: Tensor,
        t:       Tensor,
    ) -> Tensor:
        """
        Args:
            z_noisy : (B, z_ch, T)   noisy chart latent
            audio_c : (B, a_ch, T)   audio condition (from AudioWaveEncoder)
            t       : (B,)           diffusion timesteps (float, e.g. 0..1000)

        Returns:
            eps_pred : (B, z_ch, T)  predicted noise — same shape as z_noisy
        """
        # (B, C, T) → (B, T, C) → linear project to (B, T, D)
        x = self.x_proj(rearrange(z_noisy, "B C T -> B T C"))   # (B, T, D)
        c = self.c_proj(rearrange(audio_c,  "B C T -> B T C"))   # (B, T, D)

        # timestep vector  (B, D)
        vec = self.t_embedder(t)

        # RoPE frequencies for this sequence length
        T_seq = x.shape[1]
        pos   = torch.arange(T_seq, device=x.device).float()
        freqs = rope1d(pos, self.head_dim, self.rope_theta)       # (T, Dh//2, 2, 2)

        # double-stream transformer blocks
        for block in self.blocks:
            x, c = block(x, c, vec, freqs)

        # output head  →  (B, T, z_ch)  →  (B, z_ch, T)
        x = self.final_layer(x, vec)
        return rearrange(x, "B T C -> B C T")

    # ------------------------------------------------------------------

    def summary(self, T: int = 256):
        """Run one forward pass and print a parameter summary line."""
        device = next(self.parameters()).device
        a_ch   = self.c_proj.in_features
        z      = torch.zeros(1, self.z_channels, T, device=device)
        c      = torch.zeros(1, a_ch,             T, device=device)
        t_in   = torch.zeros(1,                      device=device)
        with torch.no_grad():
            out = self.forward(z, c, t_in)
        depth  = len(self.blocks)
        print(
            f"RhythmDiT  "
            f"z({self.z_channels},{T}) + c({a_ch},{T})  →  ε({out.shape[1]},{out.shape[2]})  "
            f"[depth={depth}  D={self.hidden_dim}  H={self.num_heads}]  "
            f"params={self.num_params:,}"
        )

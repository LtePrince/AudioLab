"""
src/models/attention.py
-----------------------
Building blocks for RhythmDiT (P4):

  rope1d()           : precompute 1-D RoPE rotation matrices
  apply_rope()       : apply RoPE to Q and K tensors
  timestep_embedding(): sinusoidal embedding for diffusion timesteps
  TimestepEmbedder   : sinusoidal → MLP → hidden_dim vector
  RMSNorm            : Root Mean Square normalisation (for QK norm)
  ModulationOut      : dataclass holding (shift, scale, gate)
  Modulation         : adaLN-Zero module — derives modulation params from vec
  DoubleStreamBlock  : MMDiT-style double-stream attention block

Notation used throughout:
  B   = batch size
  T   = sequence length  (chart frames = audio frames, both stride-aligned)
  D   = hidden dimension
  H   = number of attention heads
  Dh  = D // H   (head dimension)

References:
  - Open-Sora  opensora/models/mmdit/layers.py  (DoubleStreamBlock / Modulation)
  - Flux        (origin of MMDiT double-stream design)
  - DiT         (adaLN-Zero initialisation)
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


__all__ = [
    "rope1d",
    "apply_rope",
    "timestep_embedding",
    "TimestepEmbedder",
    "RMSNorm",
    "ModulationOut",
    "Modulation",
    "DoubleStreamBlock",
]


# ---------------------------------------------------------------------------
# 1-D Rotary Position Embedding
# ---------------------------------------------------------------------------

def rope1d(pos: Tensor, dim: int, theta: float = 10000.0) -> Tensor:
    """Precompute 1-D RoPE rotation matrices.

    Each consecutive pair of head-dim features is rotated by the angle
    θ_d = pos / theta^(2d/dim).  The result is stored as explicit 2×2
    rotation matrices so ``apply_rope`` can work on real-valued tensors
    (no complex-number casting needed).

    Args:
        pos  : (T,) position indices — long or float
        dim  : head dimension, **must be even**
        theta: base frequency (default 10 000)

    Returns:
        freqs : (T, dim//2, 2, 2) float32
                  freqs[t, d] = [[cos θ,  -sin θ],
                                 [sin θ,   cos θ]]
    """
    assert dim % 2 == 0, f"head dim must be even, got {dim}"
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)                      # (dim//2,)
    ang   = torch.einsum("t,d->td", pos.double(), omega) # (T, dim//2)
    cos_a, sin_a = ang.cos(), ang.sin()
    # Stack as [[cos, -sin], [sin, cos]] and reshape to 2×2
    freqs = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=-1)  # (T, dim//2, 4)
    return freqs.reshape(*ang.shape, 2, 2).float()               # (T, dim//2, 2, 2)


def apply_rope(q: Tensor, k: Tensor, freqs: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K.

    The rotation is applied pair-wise to head features:
        [x_{2i}, x_{2i+1}] ← R(θ_i) · [x_{2i}, x_{2i+1}]

    Args:
        q, k  : (B, H, T, Dh) — real-valued query / key tensors
        freqs : (T, Dh//2, 2, 2) — from ``rope1d()``

    Returns:
        q_rot, k_rot : (B, H, T, Dh) — same dtype as inputs
    """
    def _rotate(x: Tensor) -> Tensor:
        # (B, H, T, Dh) → (B, H, T, Dh//2, 1, 2)
        x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
        # freqs: (T, Dh//2, 2, 2)
        #   col 0 = [cos, sin]  ← multiplied by x component 0
        #   col 1 = [-sin, cos] ← multiplied by x component 1
        # result: (B, H, T, Dh//2, 2)
        x_rot = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
        return x_rot.reshape(*x.shape).type_as(x)

    return _rotate(q), _rotate(k)


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal embedding for diffusion timesteps (same as DiT / DDPM).

    Args:
        t         : (B,) float timesteps in [0, T_max]
        dim       : output dimension
        max_period: controls minimum frequency

    Returns:
        emb : (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )                                                     # (half,)
    args = t[:, None].float() * freqs[None]              # (B, half)
    emb  = torch.cat([args.cos(), args.sin()], dim=-1)   # (B, dim or dim-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → 2-layer MLP → hidden_dim vector.

    Args:
        hidden_dim: model width D
        freq_dim  : sinusoidal frequency dimension (default 256)
    """

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t : (B,)  →  vec : (B, hidden_dim)"""
        return self.mlp(timestep_embedding(t, self.freq_dim))


# ---------------------------------------------------------------------------
# RMSNorm  (for QK normalisation inside attention heads)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm — no centering, no bias.

    Preferred over LayerNorm for per-head QK normalisation because
    it has fewer parameters and is numerically stable at fp16/bf16.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps   = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.scale).to(dtype)


# ---------------------------------------------------------------------------
# adaLN-Zero Modulation
# ---------------------------------------------------------------------------

@dataclass
class ModulationOut:
    """Output of one Modulation slot: shift, scale, gate — all (B, 1, D)."""
    shift: Tensor
    scale: Tensor
    gate:  Tensor


class Modulation(nn.Module):
    """adaLN-Zero: derive (shift, scale, gate) × 1 or 2 from a condition vector.

    The output linear is zero-initialised, so at the start of training:
      shift=0, scale=0  →  modulate(x, 0, 0) = x        (identity)
      gate=0            →  x + 0 * branch(x) = x         (skip)
    This makes early training more stable (DiT, §3.4).

    Args:
        dim   : model width D
        double: if True, returns TWO ModulationOut
                  (first for pre-attention, second for pre-FFN)
    """

    def __init__(self, dim: int, double: bool = True):
        super().__init__()
        self.double = double
        n = 6 if double else 3
        self.lin = nn.Linear(dim, n * dim, bias=True)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, ...]:
        """vec : (B, D)  →  1 or 2 ModulationOut, each field is (B, 1, D)"""
        # silu then linear, unsqueeze for seq-dim broadcast
        chunks = self.lin(F.silu(vec)).unsqueeze(1).chunk(
            6 if self.double else 3, dim=-1
        )
        if self.double:
            return ModulationOut(*chunks[:3]), ModulationOut(*chunks[3:])
        return (ModulationOut(*chunks),)


# ---------------------------------------------------------------------------
# Double Stream Block  (MMDiT core)
# ---------------------------------------------------------------------------

class DoubleStreamBlock(nn.Module):
    """MMDiT-style double-stream attention block.

    Both token sequences — chart latent ``x`` and audio condition ``c`` —
    project their own Q, K, V.  The K and V of both streams are **concatenated**
    before the attention softmax, so every chart token can attend to every
    audio token and vice-versa.  The two streams are modulated independently by
    the diffusion timestep vector ``vec``.

    Follows Open-Sora's ``DoubleStreamBlock`` structure (Flux lineage) but:
      • Uses ``F.scaled_dot_product_attention`` (PyTorch SDPA) instead of
        flash_attn — no extra dependency, still Flash-Attention-compatible.
      • Uses ``RMSNorm`` (simple, no liger_kernel) for QK normalisation.
      • No ``processor`` indirection — single clean ``forward()``.

    Args:
        hidden_dim : model width D  (must be divisible by num_heads)
        num_heads  : number of attention heads H
        mlp_ratio  : FFN hidden-dim multiplier (default 4.0)
        dropout    : dropout inside FFN (default 0.0)
        qkv_bias   : bias in QKV projections (default True)

    Forward signature::

        x, c  : (B, T, D)         chart latent / audio condition tokens
        vec   : (B, D)            timestep embedding from TimestepEmbedder
        freqs : (T, Dh//2, 2, 2)  RoPE matrices from rope1d()
        →  x, c : (B, T, D)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads:  int,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
        qkv_bias:   bool  = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        mlp_dim = int(hidden_dim * mlp_ratio)

        # ── chart stream (x) ──────────────────────────────────────────
        self.x_mod   = Modulation(hidden_dim, double=True)
        self.x_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.x_qkv   = nn.Linear(hidden_dim, 3 * hidden_dim, bias=qkv_bias)
        self.x_qnorm = RMSNorm(self.head_dim)
        self.x_knorm = RMSNorm(self.head_dim)
        self.x_proj  = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.x_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.x_mlp   = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim, bias=True),
            nn.Dropout(dropout),
        )

        # ── audio stream (c) ──────────────────────────────────────────
        self.c_mod   = Modulation(hidden_dim, double=True)
        self.c_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.c_qkv   = nn.Linear(hidden_dim, 3 * hidden_dim, bias=qkv_bias)
        self.c_qnorm = RMSNorm(self.head_dim)
        self.c_knorm = RMSNorm(self.head_dim)
        self.c_proj  = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.c_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.c_mlp   = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim, bias=True),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    def _project_qkv(
        self,
        qkv_linear: nn.Linear,
        x_normed:   Tensor,
        q_norm:     RMSNorm,
        k_norm:     RMSNorm,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Linear project → split → per-head reshape → QK RMSNorm.

        Returns q, k, v each shaped (B, H, T, Dh).
        """
        q, k, v = rearrange(
            qkv_linear(x_normed),
            "B T (three H Dh) -> three B H T Dh",
            three=3,
            H=self.num_heads,
        )
        q = q_norm(q)
        k = k_norm(k)
        return q, k, v

    @staticmethod
    def _modulate(norm: nn.LayerNorm, mod: ModulationOut, x: Tensor) -> Tensor:
        """Apply adaLN modulation:  norm(x) * (1 + scale) + shift."""
        return norm(x) * (1.0 + mod.scale) + mod.shift

    # ------------------------------------------------------------------
    def forward(
        self,
        x:     Tensor,
        c:     Tensor,
        vec:   Tensor,
        freqs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x     : (B, T, D)          noisy chart latent tokens
            c     : (B, T, D)          audio condition tokens
            vec   : (B, D)             timestep embedding
            freqs : (T, Dh//2, 2, 2)   RoPE matrices — precomputed by rope1d()

        Returns:
            x, c  : (B, T, D)
        """
        # ── adaLN-Zero modulation params ──────────────────────────────
        x_mod1, x_mod2 = self.x_mod(vec)   # pre-attn / pre-FFN for chart
        c_mod1, c_mod2 = self.c_mod(vec)   # pre-attn / pre-FFN for audio

        # ── QKV projections ───────────────────────────────────────────
        xq, xk, xv = self._project_qkv(
            self.x_qkv, self._modulate(self.x_norm1, x_mod1, x),
            self.x_qnorm, self.x_knorm,
        )
        cq, ck, cv = self._project_qkv(
            self.c_qkv, self._modulate(self.c_norm1, c_mod1, c),
            self.c_qnorm, self.c_knorm,
        )

        # ── RoPE (same positions for both streams — they are aligned) ─
        xq, xk = apply_rope(xq, xk, freqs)
        cq, ck = apply_rope(cq, ck, freqs)

        # ── shared KV: cat on T dim ───────────────────────────────────
        K = torch.cat([xk, ck], dim=2)   # (B, H, 2T, Dh)
        V = torch.cat([xv, cv], dim=2)

        # ── attention (SDPA — Flash-Attention-compatible) ─────────────
        x_attn = F.scaled_dot_product_attention(xq, K, V)   # (B, H, T, Dh)
        c_attn = F.scaled_dot_product_attention(cq, K, V)

        # ── merge heads ───────────────────────────────────────────────
        x_attn = rearrange(x_attn, "B H T Dh -> B T (H Dh)")
        c_attn = rearrange(c_attn, "B H T Dh -> B T (H Dh)")

        # ── residual  +  gate  (attention branch) ─────────────────────
        x = x + x_mod1.gate * self.x_proj(x_attn)
        c = c + c_mod1.gate * self.c_proj(c_attn)

        # ── residual  +  gate  (FFN branch) ───────────────────────────
        x = x + x_mod2.gate * self.x_mlp(self._modulate(self.x_norm2, x_mod2, x))
        c = c + c_mod2.gate * self.c_mlp(self._modulate(self.c_norm2, c_mod2, c))

        return x, c

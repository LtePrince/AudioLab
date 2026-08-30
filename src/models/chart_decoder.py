"""
src/models/chart_decoder.py
───────────────────────────
Autoregressive chart decoder (osuT5-style): generates the token sequence of
docs/ar_tokenizer_design.md conditioned on audio memory from the
transcription encoder (TranscriberNet.features).

Why AR on top of the transcriber
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The per-frame transcriber nails WHEN (audio-determined) but has no output-
side context, so design semantics (Drag streams, Hold+Flick combos, type
marginals) collapse.  Here every token is conditioned on the tokens already
generated (p(y_t | y_<t, audio)) — sequence-level structure is expressible —
while cross-attention to the encoder keeps the alignment.

Alignment mechanics
~~~~~~~~~~~~~~~~~~~
- decoder tokens carry a sinusoidal embedding of their MUSICAL time converted
  to chart frames (ticks → seconds via bpm → frames); memory frames carry the
  same embedding of their frame index.  Cross-attention can therefore match
  "where am I in the song" directly (docs §4).
- self-attention is causal with RoPE over token positions.

API
~~~
    dec = ChartDecoder(**DEFAULT_DECODER_CONFIG)
    logits = dec(tokens, tok_frames, memory, mem_valid)   # (B, L, vocab)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data.chart_tokenizer import VOCAB_SIZE
from src.models.attention import RMSNorm, apply_rope, rope1d, timestep_embedding

__all__ = ["DEFAULT_DECODER_CONFIG", "ChartDecoder", "time_embed"]


DEFAULT_DECODER_CONFIG = dict(
    vocab_size = VOCAB_SIZE,
    hidden_dim = 256,
    depth      = 6,
    num_heads  = 8,
    mlp_ratio  = 4.0,
    dropout    = 0.1,
    mem_dim    = 256,       # TranscriberNet hidden_dim
    time_dim   = 128,       # sinusoidal time-embedding width
    rope_theta = 10000.0,
)


def time_embed(frames: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a (…,) float tensor of chart-frame times."""
    shape = frames.shape
    emb = timestep_embedding(frames.reshape(-1).float(), dim, max_period=20000.0)
    return emb.reshape(*shape, dim)


class DecoderBlock(nn.Module):
    """pre-LN: causal self-attn (RoPE, QK-RMSNorm) → cross-attn → MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.h, self.dh = num_heads, dim // num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv   = nn.Linear(dim, 3 * dim)
        self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)
        self.proj  = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.q_x   = nn.Linear(dim, dim)
        self.kv_x  = nn.Linear(dim, 2 * dim)
        self.proj_x = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, freqs: Tensor, mem: Tensor,
                mem_mask: Tensor | None) -> Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.qn(q), self.kn(k), freqs)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.permute(0, 2, 1, 3).reshape(B, L, D))

        T = mem.shape[1]
        qx = self.q_x(self.norm2(x)).reshape(B, L, self.h, self.dh).transpose(1, 2)
        kx, vx = self.kv_x(mem).reshape(B, T, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qx, kx, vx, attn_mask=mem_mask)
        x = x + self.proj_x(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.norm3(x))


class ChartDecoder(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = 256,
                 depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, mem_dim: int = 256, time_dim: int = 128,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.head_dim, self.rope_theta, self.time_dim = hidden_dim // num_heads, rope_theta, time_dim
        self.tok_emb   = nn.Embedding(vocab_size, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.mem_proj  = nn.Linear(mem_dim, hidden_dim)
        self.mem_time  = nn.Linear(time_dim, hidden_dim)
        self.mem_norm  = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.drop      = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.lm_head  = nn.Linear(hidden_dim, vocab_size)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def prepare_memory(self, features: Tensor) -> Tensor:
        """(B, T, mem_dim) encoder features → decoder memory with time embedding."""
        B, T, _ = features.shape
        frames = torch.arange(T, device=features.device).float().expand(B, T)
        mem = self.mem_proj(features) + self.mem_time(time_embed(frames, self.time_dim))
        return self.mem_norm(mem)

    def forward(self, tokens: Tensor, tok_frames: Tensor, memory: Tensor,
                mem_valid: Tensor | None = None) -> Tensor:
        """
        tokens     : (B, L) long
        tok_frames : (B, L) float — musical time of each token in chart frames
        memory     : (B, T, D) from prepare_memory()
        mem_valid  : (B, T) bool — True on real (non-padded) audio frames
        → logits (B, L, vocab)
        """
        B, L = tokens.shape
        x = self.tok_emb(tokens) + self.time_proj(time_embed(tok_frames, self.time_dim))
        x = self.drop(x)
        freqs = rope1d(torch.arange(L, device=x.device).float(), self.head_dim, self.rope_theta)
        mask = None
        if mem_valid is not None:
            mask = mem_valid[:, None, None, :]           # (B,1,1,T) broadcast over heads/queries
        for blk in self.blocks:
            x = blk(x, freqs, memory, mask)
        return self.lm_head(self.norm_out(x))


# ─────────────────────────────────────────────────────────────────────────────
# Constrained autoregressive generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_tokens(
    decoder:     ChartDecoder,
    memory:      Tensor,          # (1, T, D) prepared memory
    mem_valid:   Tensor,          # (1, T) bool
    bpm:         float,
    frame_ms:    float,
    audio_frames: int,            # real audio length in chart frames (time cutoff)
    temperature: float = 1.0,
    top_p:       float = 0.95,
    max_tokens:  int   = 6000,
    generator:   torch.Generator | None = None,
) -> list[int]:
    """Sample a grammar-valid token sequence.

    Every step: grammar mask from GrammarState (chord lane exclusivity,
    mandatory hold durations, chain rules) ∧ time cutoff (once the musical
    clock passes the end of the audio only EOS — or the tokens needed to
    close an open duration — remain legal) → temperature / top-p sampling.
    """
    from src.data.chart_tokenizer import (
        BOS, EOS, TICKS_PER_BEAT, ChartTokenizer, GrammarState,
    )
    T = ChartTokenizer
    device = memory.device
    frames_per_tick = 60.0 / (32.0 * bpm) * 1000.0 / frame_ms
    kinds = [T.kind(i) for i in range(decoder.lm_head.out_features)]
    values = [T.value(i) for i in range(decoder.lm_head.out_features)]

    tokens, frames = [BOS], [0.0]
    gs = GrammarState(); gs.step(BOS)
    cur_tick = 0
    for _ in range(max_tokens):
        tok_t = torch.tensor([tokens], device=device)
        frm_t = torch.tensor([frames], device=device, dtype=torch.float32)
        logits = decoder(tok_t, frm_t, memory, mem_valid)[0, -1].float()

        allowed = torch.tensor([gs.is_allowed(i) for i in range(len(kinds))], device=device)
        past_end = cur_tick * frames_per_tick >= audio_frames
        if past_end:
            # close any open duration, otherwise only EOS
            closing = torch.tensor(
                [k in ("DURB", "DURT") if gs.in_dur else k == "EOS" for k in kinds],
                device=device)
            if (allowed & closing).any():
                allowed = allowed & closing
        logits = logits.masked_fill(~allowed, float("-inf"))

        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if 0.0 < top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
            probs = probs / probs.sum()
        nxt = int(torch.multinomial(probs, 1, generator=generator).item())

        tokens.append(nxt); gs.step(nxt)
        if kinds[nxt] == "DTB":
            cur_tick += values[nxt] * TICKS_PER_BEAT
        elif kinds[nxt] == "DTT":
            cur_tick += values[nxt]
        frames.append(cur_tick * frames_per_tick)
        if nxt == EOS:
            break
    if tokens[-1] != EOS:
        tokens.append(EOS)
    return tokens

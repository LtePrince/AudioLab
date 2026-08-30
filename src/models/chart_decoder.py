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

    def init_cache(self, mem: Tensor) -> dict:
        """Per-song cache: cross-attention K/V computed once; self-attention K/V grow."""
        B, T, D = mem.shape
        kx, vx = self.kv_x(mem).reshape(B, T, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return {"kx": kx, "vx": vx, "k": None, "v": None}

    def step(self, x: Tensor, freqs: Tensor, cache: dict, mem_mask: Tensor | None) -> Tensor:
        """x: (B,1,D) new token; freqs: (1,Dh//2,2,2) RoPE for its position."""
        B, _, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(B, 1, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.qn(q), self.kn(k), freqs)
        cache["k"] = k if cache["k"] is None else torch.cat([cache["k"], k], dim=2)
        cache["v"] = v if cache["v"] is None else torch.cat([cache["v"], v], dim=2)
        a = F.scaled_dot_product_attention(q, cache["k"], cache["v"])      # past+current only
        x = x + self.proj(a.permute(0, 2, 1, 3).reshape(B, 1, D))
        qx = self.q_x(self.norm2(x)).reshape(B, 1, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(qx, cache["kx"], cache["vx"], attn_mask=mem_mask)
        x = x + self.proj_x(a.transpose(1, 2).reshape(B, 1, D))
        return x + self.mlp(self.norm3(x))

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
                 rope_theta: float = 10000.0, n_lanes: int = 4,
                 factorized: bool = False, offset_head: bool = False,
                 cross_window: int | None = None, token_dropout: float = 0.0):
        """
        n_lanes      : position slots in the NOTE block (4 = 4k lanes, 64 = free-x bins)
        factorized   : NOTE embeddings = E_type[t] + E_pos[bin]  (shares parameters
                       across the 4×n_lanes fused tokens — docs/freeform_design.md P1)
        offset_head  : regress the sub-bin position offset from the NOTE token's
                       hidden state (anchor+offset; not part of the autoregression)
        cross_window : LOCAL cross-attention — a token whose musical clock is at
                       frame f may only attend memory frames |f' - f| <= window
                       (None = global).  The alignment is known from the clock, so
                       the model should not have to discover it; round 1 (global
                       attention) collapsed into an unconditional chart LM
                       (mismatched audio changed val NLL by 0.3%).
        token_dropout: word-dropout on the token embedding (time embedding kept)
                       during training, so the history cannot fully explain the
                       next token and the audio path receives gradient pressure.
        """
        super().__init__()
        self.head_dim, self.rope_theta, self.time_dim = hidden_dim // num_heads, rope_theta, time_dim
        self.cross_window, self.token_dropout = cross_window, token_dropout
        self.n_lanes, self.factorized = n_lanes, factorized
        self._note0, self._n_note = 50, 4 * n_lanes
        self.tok_emb   = nn.Embedding(vocab_size, hidden_dim)
        if factorized:
            self.type_emb = nn.Embedding(4, hidden_dim)
            self.pos_emb  = nn.Embedding(n_lanes, hidden_dim)
        self.offset_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, 1)) if offset_head else None
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
                mem_valid: Tensor | None = None, return_hidden: bool = False):
        """
        tokens     : (B, L) long
        tok_frames : (B, L) float — musical time of each token in chart frames
        memory     : (B, T, D) from prepare_memory()
        mem_valid  : (B, T) bool — True on real (non-padded) audio frames
        → logits (B, L, vocab)
        """
        B, L = tokens.shape
        tok = self.embed(tokens)
        if self.training and self.token_dropout > 0:
            keep = (torch.rand(B, L, 1, device=tok.device) >= self.token_dropout).to(tok.dtype)
            tok = tok * keep
        x = tok + self.time_proj(time_embed(tok_frames, self.time_dim))
        x = self.drop(x)
        freqs = rope1d(torch.arange(L, device=x.device).float(), self.head_dim, self.rope_theta)
        mask = self.cross_mask(tok_frames, memory.shape[1], mem_valid)
        for blk in self.blocks:
            x = blk(x, freqs, memory, mask)
        h = self.norm_out(x)
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        return logits

    def init_cache(self, memory: Tensor) -> list[dict]:
        return [blk.init_cache(memory) for blk in self.blocks]

    def step(self, token: Tensor, frame: Tensor, pos: int, caches: list[dict],
             T_mem: int, mem_valid: Tensor | None):
        """One incremental decoding step.
        token (B,) long · frame (B,) float · pos = sequence index of this token.
        Returns (logits (B,V), hidden (B,D))."""
        x = self.embed(token[:, None]) + self.time_proj(time_embed(frame[:, None], self.time_dim))
        freqs = rope1d(torch.tensor([pos], device=x.device).float(), self.head_dim, self.rope_theta)
        mask = self.cross_mask(frame[:, None], T_mem, mem_valid)
        for blk, cache in zip(self.blocks, caches):
            x = blk.step(x, freqs, cache, mask)
        h = self.norm_out(x)
        return self.lm_head(h)[:, 0], h[:, 0]

    def cross_mask(self, tok_frames: Tensor, T: int, mem_valid: Tensor | None):
        """Boolean cross-attention mask (B,1,L,T): valid memory ∧ local window."""
        if mem_valid is None and self.cross_window is None:
            return None
        B = tok_frames.shape[0]
        mask = torch.ones(B, 1, 1, T, dtype=torch.bool, device=tok_frames.device)
        if mem_valid is not None:
            mask = mask & mem_valid[:, None, None, :]
        if self.cross_window is not None:
            fidx = torch.arange(T, device=tok_frames.device).float()
            local = (tok_frames[:, None, :, None] - fidx[None, None, None, :]).abs() <= self.cross_window
            mask = mask & local                                # (B,1,L,T)
            # a token whose window falls entirely off the valid memory (e.g. past
            # the audio end) must keep at least one key → allow the nearest frame
            empty = ~mask.any(dim=-1, keepdim=True)            # (B,1,L,1)
            if empty.any():
                nearest = tok_frames.clamp(0, T - 1).round().long()  # (B,L)
                fallback = torch.zeros_like(mask)
                fallback.scatter_(-1, nearest[:, None, :, None], True)
                mask = mask | (fallback & empty)
        return mask

    def embed(self, tokens: Tensor) -> Tensor:
        """Token embeddings; the NOTE block is factorised when enabled."""
        if not self.factorized:
            return self.tok_emb(tokens)
        note_block = (self.pos_emb.weight[:, None, :] + self.type_emb.weight[None, :, :]
                      ).reshape(self._n_note, -1)                  # (4·n_lanes, D), bin-major
        table = torch.cat([self.tok_emb.weight[: self._note0], note_block,
                           self.tok_emb.weight[self._note0 + self._n_note:]], dim=0)
        return F.embedding(tokens, table)

    def predict_offset(self, hidden: Tensor) -> Tensor:
        """Sub-bin offset ∈ (0,1) from NOTE-token hidden states (B, L, D) → (B, L)."""
        assert self.offset_head is not None
        return torch.sigmoid(self.offset_head(hidden)).squeeze(-1)


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
    tokenizer    = None,          # ChartTokenizer instance (default: 4-lane)
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
    T = tokenizer if tokenizer is not None else ChartTokenizer()
    assert T.vocab_size == decoder.lm_head.out_features, "tokenizer/decoder vocab mismatch"
    device = memory.device
    frames_per_tick = 60.0 / (32.0 * bpm) * 1000.0 / frame_ms
    kinds = [T.kind(i) for i in range(decoder.lm_head.out_features)]
    values = [T.value(i) for i in range(decoder.lm_head.out_features)]

    tokens, frames = [BOS], [0.0]
    offsets: dict[int, float] = {}       # token position → sub-bin offset (free-x)
    gs = GrammarState(T); gs.step(BOS)
    cur_tick = 0
    caches = decoder.init_cache(memory)
    T_mem = memory.shape[1]
    for _ in range(max_tokens):
        pos = len(tokens) - 1
        logits, hidden = decoder.step(
            torch.tensor([tokens[-1]], device=device),
            torch.tensor([frames[-1]], device=device, dtype=torch.float32),
            pos, caches, T_mem, mem_valid)
        logits = logits[0].float()
        if decoder.offset_head is not None and kinds[tokens[-1]] == "NOTE":
            offsets[pos] = float(decoder.predict_offset(hidden[:, None])[0, 0])

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
    if decoder.offset_head is not None:
        return tokens, offsets
    return tokens

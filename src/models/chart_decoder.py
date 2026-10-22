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
from torch.utils.checkpoint import checkpoint

from src.data.chart_tokenizer import VOCAB_SIZE
from src.models.attention import RMSNorm, apply_rope, rope1d, timestep_embedding
from src.models.chart_history import CausalChartHistory, HISTORY_DIM

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
                 cross_window: int | None = None, token_dropout: float = 0.0,
                 soft_onset_dim: int = 0, history_dim: int = 0,
                 relation_auxiliary: bool = False):
        """
        n_lanes      : position slots in the NOTE block (4 = 4k lanes, 64 = free-x bins)
        factorized   : NOTE embeddings = E_type[t] + E_pos[bin]  (shares parameters
                       across the 4×n_lanes fused tokens — docs/freeform_design.md P1)
        offset_head  : regress the sub-bin position offset from the NOTE token's
                       hidden state (anchor+offset). Legacy mode has no residual
                       feedback; history mode feeds past predicted residuals to
                       later input tokens, never back into the current NOTE.
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
        if soft_onset_dim not in (0, 2):
            raise ValueError('soft_onset_dim must be 0 (legacy) or 2 (dense probability/offset)')
        self.soft_onset_dim = soft_onset_dim
        # Runtime training option; no checkpoint parameter or inference change.
        # Recompute complete blocks in backward instead of retaining every
        # layer's attention activations. The token/audio context stays complete.
        self.gradient_checkpointing = False
        if history_dim not in (0,HISTORY_DIM):
            raise ValueError('unsupported causal history dimension')
        self.history_dim = history_dim
        if type(relation_auxiliary) is not bool or (relation_auxiliary and not history_dim):
            raise ValueError('relation auxiliary head requires explicit causal history configuration')
        self.head_dim, self.rope_theta, self.time_dim = hidden_dim // num_heads, rope_theta, time_dim
        self.cross_window, self.token_dropout = cross_window, token_dropout
        self.n_lanes, self.factorized = n_lanes, factorized
        self._note0, self._n_note = 50, 4 * n_lanes
        self.tok_emb   = nn.Embedding(vocab_size, hidden_dim)
        self.history_proj = nn.Linear(history_dim,hidden_dim,bias=False) if history_dim else None
        if self.history_proj is not None:
            nn.init.zeros_(self.history_proj.weight)
        if factorized:
            self.type_emb = nn.Embedding(4, hidden_dim)
            self.pos_emb  = nn.Embedding(n_lanes, hidden_dim)
        self.offset_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, 1)) if offset_head else None
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.mem_proj  = nn.Linear(mem_dim, hidden_dim)
        # Opt-in experimental residual; absent from legacy state dicts. No bias:
        # the same architecture with zero inputs is the exact audio-only control.
        self.soft_onset_proj = nn.Linear(soft_onset_dim, hidden_dim, bias=False) if soft_onset_dim else None
        if self.soft_onset_proj is not None:
            nn.init.zeros_(self.soft_onset_proj.weight)
        self.mem_time  = nn.Linear(time_dim, hidden_dim)
        self.mem_norm  = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.drop      = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.lm_head  = nn.Linear(hidden_dim, vocab_size)
        self.relation_head = nn.Linear(hidden_dim,4) if relation_auxiliary else None
        if self.relation_head is not None:
            nn.init.zeros_(self.relation_head.weight);nn.init.zeros_(self.relation_head.bias)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def prepare_memory(self, features: Tensor, soft_onsets: Tensor | None = None) -> Tensor:
        """(B, T, mem_dim) encoder features → decoder memory with time embedding."""
        B, T, _ = features.shape
        frames = torch.arange(T, device=features.device).float().expand(B, T)
        mem = self.mem_proj(features) + self.mem_time(time_embed(frames, self.time_dim))
        if self.soft_onset_proj is not None:
            if soft_onsets is None or soft_onsets.shape != (B, T, self.soft_onset_dim):
                raise ValueError('experimental decoder requires aligned dense soft onset inputs')
            if not torch.isfinite(soft_onsets).all():
                raise ValueError('soft onset inputs must be finite')
            mem = mem + self.soft_onset_proj(soft_onsets)
        elif soft_onsets is not None:
            raise ValueError('legacy decoder does not accept soft onset inputs')
        return self.mem_norm(mem)

    def forward(self, tokens: Tensor, tok_frames: Tensor, memory: Tensor,
                mem_valid: Tensor | None = None, return_hidden: bool = False,
                history: Tensor | None = None):
        """
        tokens     : (B, L) long
        tok_frames : (B, L) float — musical time of each token in chart frames
        memory     : (B, T, D) from prepare_memory()
        mem_valid  : (B, T) bool — True on real (non-padded) audio frames
        → logits (B, L, vocab)
        """
        B, L = tokens.shape
        tok = self.embed(tokens)
        if self.history_proj is not None:
            if history is None or history.shape != (B,L,self.history_dim) or not torch.isfinite(history).all():
                raise ValueError('history decoder requires finite prefix-only history features')
            tok = tok + self.history_proj(history)
        elif history is not None:
            raise ValueError('decoder does not accept causal history inputs')
        if self.training and self.token_dropout > 0:
            keep = (torch.rand(B, L, 1, device=tok.device) >= self.token_dropout).to(tok.dtype)
            tok = tok * keep
        x = tok + self.time_proj(time_embed(tok_frames, self.time_dim))
        x = self.drop(x)
        freqs = rope1d(torch.arange(L, device=x.device).float(), self.head_dim, self.rope_theta)
        mask = self.cross_mask(tok_frames, memory.shape[1], mem_valid)
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, freqs, memory, mask, use_reentrant=False)
            else:
                x = blk(x, freqs, memory, mask)
        h = self.norm_out(x)
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        return logits

    def init_cache(self, memory: Tensor) -> list[dict]:
        return [blk.init_cache(memory) for blk in self.blocks]

    def step(self, token: Tensor, frame: Tensor, pos: int, caches: list[dict],
             T_mem: int, mem_valid: Tensor | None, history: Tensor | None = None):
        """One incremental decoding step.
        token (B,) long · frame (B,) float · pos = sequence index of this token.
        Returns (logits (B,V), hidden (B,D))."""
        x = self.embed(token[:, None]) + self.time_proj(time_embed(frame[:, None], self.time_dim))
        if self.history_proj is not None:
            if history is None or history.shape != (len(token),self.history_dim) or not torch.isfinite(history).all():
                raise ValueError('history decoder step requires finite prefix-only history features')
            x = x + self.history_proj(history[:,None])
        elif history is not None:
            raise ValueError('decoder does not accept causal history inputs')
        freqs = rope1d(torch.tensor([pos], device=x.device).float(), self.head_dim, self.rope_theta)
        mask = self.cross_mask(frame[:, None], T_mem, mem_valid)
        for blk, cache in zip(self.blocks, caches):
            x = blk.step(x, freqs, cache, mask)
        h = self.norm_out(x)
        return self.lm_head(h)[:, 0], h[:, 0]

    def cross_mask(self, tok_frames: Tensor, T: int, mem_valid: Tensor | None):
        """Valid memory ∧ local window; empty windows use the nearest valid key.

        Equal-distance fallback keys resolve to the earlier frame. Invalid
        memory never becomes visible, including before/after the audio.
        """
        if T < 1:
            raise ValueError('cross-attention requires nonempty audio memory')
        if mem_valid is None and self.cross_window is None:
            return None
        B = tok_frames.shape[0]
        mask = torch.ones(B, 1, 1, T, dtype=torch.bool, device=tok_frames.device)
        if mem_valid is not None:
            if mem_valid.shape != (B, T) or mem_valid.dtype != torch.bool:
                raise ValueError('mem_valid must be a boolean (batch, memory_frames) mask')
            mask = mask & mem_valid[:, None, None, :]
        if self.cross_window is not None:
            fidx = torch.arange(T, device=tok_frames.device).float()
            distance = (tok_frames[:, None, :, None] - fidx[None, None, None, :]).abs()
            mask = mask & (distance <= self.cross_window)      # (B,1,L,T)
            empty = ~mask.any(dim=-1, keepdim=True)            # (B,1,L,1)
            if empty.any():
                if mem_valid is not None:
                    if not mem_valid.any(dim=-1).all():
                        raise ValueError('each sample needs at least one valid audio frame')
                    distance.masked_fill_(~mem_valid[:, None, None, :], float('inf'))
                nearest = distance.argmin(dim=-1, keepdim=True)
                fallback = torch.zeros_like(mask)
                fallback.scatter_(-1, nearest, True)
                mask = mask | (fallback & empty)
        elif mem_valid is not None and not mem_valid.any(dim=-1).all():
            raise ValueError('each sample needs at least one valid audio frame')
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

def nucleus_probs(probs: Tensor, top_p: float) -> Tensor:
    """Nucleus on the final axis, supporting zero-mass conditional groups."""
    if not 0.0 < top_p < 1.0:
        return probs
    values, indices = probs.sort(descending=True, dim=-1)
    values = values * (values.cumsum(-1) - values < top_p)
    result = torch.zeros_like(probs).scatter(-1, indices, values)
    return result / result.sum(-1, keepdim=True).clamp_min(1e-30)


def type_preserving_probs(probs: Tensor, top_p: float, note_start: int, n_lanes: int) -> Tensor:
    """Truncate positions within each type while retaining all type masses.

    Non-NOTE actions form a separate conditional nucleus. No type prior or
    boost is added; grammar-masked zeros remain zero. At p=1 this is identity.
    """
    if not 0.0 < top_p < 1.0:
        return probs
    end = note_start + 4 * n_lanes
    notes = probs[note_start:end].reshape(n_lanes, 4).T
    mass = notes.sum(-1, keepdim=True)
    notes = nucleus_probs(notes / mass.clamp_min(1e-30), top_p) * mass
    other = torch.cat([probs[:note_start], probs[end:]])
    other_mass = other.sum()
    other = nucleus_probs(other / other_mass.clamp_min(1e-30), top_p) * other_mass
    return torch.cat([other[:note_start], notes.T.reshape(-1), other[note_start:]])


@torch.no_grad()
def generate_tokens(
    decoder:     ChartDecoder,
    memory:      Tensor,          # (1, T, D) prepared memory
    mem_valid:   Tensor,          # (1, T) bool
    bpm:         float,
    frame_ms:    float,
    audio_frames: float,          # real audio length in chart frames (time cutoff)
    temperature: float = 1.0,
    top_p:       float = 0.95,
    max_tokens:  int   = 6000,
    generator:   torch.Generator | None = None,
    tokenizer    = None,          # ChartTokenizer instance (default: 4-lane)
    skeleton_ticks: list[int] | None = None,
    offset: float = 0.0,         # chart delay in audio seconds
    sampling_policy: str = 'flat',  # experimental: type_preserving
    max_chord_size: int | None = None,
    progress_callback = None,   # optional cancellation/resource checks; no sampling changes
    require_complete: bool = False,  # product jobs reject budget-forced finalization
) -> list[int]:
    """Sample a grammar-valid token sequence.

    Every step: grammar mask from GrammarState (chord lane exclusivity,
    mandatory hold durations, chain rules) ∧ time cutoff (once the musical
    clock passes the end of the audio only EOS — or the tokens needed to
    close an open duration — remain legal) → temperature / top-p sampling.

    skeleton_ticks (hybrid "listen-then-write" mode): sorted onset ticks from
    the transcriber.  Time advance is FORCED: DT tokens are never sampled —
    "advance" is one action whose probability is the sum over DT tokens, and
    when chosen the canonical DT chain to the next skeleton onset is injected.
    At every onset at least one NOTE must be placed; extra chord members,
    lanes, types and hold durations are the decoder's decisions.  Alignment
    is therefore exactly the skeleton's; the decoder supplies design semantics.
    """
    from src.data.chart_tokenizer import (
        BOS, EOS, N_DTB, N_DTT, TICKS_PER_BEAT, ChartTokenizer, GrammarState,
    )
    T = tokenizer if tokenizer is not None else ChartTokenizer()
    if sampling_policy not in ('flat', 'type_preserving'):
        raise ValueError(f'unknown sampling policy: {sampling_policy}')
    if max_chord_size is not None and (type(max_chord_size) is not int or
                                      not 1 <= max_chord_size <= T.n_lanes):
        raise ValueError('max_chord_size must be an integer in [1, n_lanes]')
    assert T.vocab_size == decoder.lm_head.out_features, "tokenizer/decoder vocab mismatch"
    device = memory.device
    frames_per_tick = 60.0 / (32.0 * bpm) * 1000.0 / frame_ms
    kinds = [T.kind(i) for i in range(decoder.lm_head.out_features)]
    values = [T.value(i) for i in range(decoder.lm_head.out_features)]

    offset_frames = offset * 1000.0 / frame_ms
    history_state = (CausalChartHistory(T.n_lanes,60000.0/bpm/frame_ms)
                     if getattr(decoder,'history_dim',0) else None)
    if history_state is not None and skeleton_ticks is not None:
        raise ValueError('causal history experiment requires free decoding, not forced skeleton')
    tokens, frames = [BOS], [offset_frames]
    offsets: dict[int, float] = {}       # token position → sub-bin offset (free-x)
    gs = GrammarState(T); gs.step(BOS)
    cur_tick = 0
    caches = decoder.init_cache(memory)
    T_mem = memory.shape[1]
    is_dt   = torch.tensor([k in ("DTB", "DTT") for k in kinds], device=device)
    is_note = torch.tensor([k == "NOTE" for k in kinds], device=device)
    is_eos  = torch.tensor([k == "EOS" for k in kinds], device=device)
    skel = (sorted({int(x) for x in skeleton_ticks
                    if x >= 0 and 0 <= int(x) * frames_per_tick + offset_frames < audio_frames})
            if skeleton_ticks is not None else None)
    skel_i, notes_here = 0, 0            # next skeleton onset index; notes placed at cur_tick
    if skel and skel[0] == 0:            # an onset at tick 0: we already stand on it
        skel_i = 1

    def feed(tok: int) -> None:
        """Append a token, advance grammar/clock, run one decoder step."""
        nonlocal cur_tick
        tokens.append(tok); gs.step(tok)
        if kinds[tok] == "DTB":
            cur_tick += values[tok] * TICKS_PER_BEAT
        elif kinds[tok] == "DTT":
            cur_tick += values[tok]
        frames.append(cur_tick * frames_per_tick + offset_frames)

    for step_index in range(max_tokens):
        if progress_callback is not None and step_index % 32 == 0:
            progress_callback(step_index, skel_i, len(skel) if skel is not None else None)
        pos = len(tokens) - 1
        history_args = ({'history':torch.tensor([history_state.features(frames[-1])],device=device)}
                        if history_state is not None else {})
        logits, hidden = decoder.step(
            torch.tensor([tokens[-1]], device=device),
            torch.tensor([frames[-1]], device=device, dtype=torch.float32),
            pos, caches, T_mem, mem_valid, **history_args)
        logits = logits[0].float()
        if decoder.offset_head is not None and kinds[tokens[-1]] == "NOTE":
            offsets[pos] = float(decoder.predict_offset(hidden[:, None])[0, 0])
        if history_state is not None:
            history_state.observe(tokens[-1],frames[-1],offsets.get(pos))

        allowed = torch.tensor([gs.is_allowed(i) for i in range(len(kinds))], device=device)
        # used_lanes resets on every DT, including forced skeleton chains. Mask
        # only NOTE; a capped chord must still close a pending Hold duration.
        if max_chord_size is not None and len(gs.used_lanes) >= max_chord_size:
            allowed = allowed & ~is_note
        past_end = cur_tick * frames_per_tick + offset_frames >= audio_frames
        if skel is not None and not gs.in_dur:
            at_onset = skel_i > 0 and cur_tick == skel[skel_i - 1]
            more = skel_i < len(skel)
            if not at_onset:                  # NOTE only at a skeleton onset
                allowed = allowed & ~is_note
            elif notes_here == 0:             # ≥1 note per onset before advancing
                allowed = allowed & is_note
            if not more:                      # no onsets left → no advance
                allowed = allowed & ~is_dt
            eos_ok = (not more) and (notes_here > 0 or not at_onset)
            if not eos_ok:
                allowed = allowed & ~is_eos
        elif past_end:
            # close any open duration, otherwise only EOS
            closing = torch.tensor(
                [k in ("DURB", "DURT") if gs.in_dur else k == "EOS" for k in kinds],
                device=device)
            if (allowed & closing).any():
                allowed = allowed & closing
        if not allowed.any():                      # safety: never dead-end
            allowed = torch.tensor([k == "EOS" for k in kinds], device=device)
        logits = logits.masked_fill(~allowed, float("-inf"))

        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if skel is not None and (allowed & is_dt).any():
            # collapse all DT tokens into one "advance" action
            p_adv = probs[is_dt].sum()
            probs = torch.where(is_dt, torch.zeros_like(probs), probs)
            adv_id = int(is_dt.nonzero()[0])
            probs[adv_id] = p_adv
        probs = (type_preserving_probs(probs, top_p, T.note(0, 1), T.n_lanes)
                 if sampling_policy == 'type_preserving' else nucleus_probs(probs, top_p))
        nxt = int(torch.multinomial(probs, 1, generator=generator).item())

        if skel is not None and kinds[nxt] in ("DTB", "DTT"):
            # forced advance: inject the canonical DT chain to the next onset
            gap = skel[skel_i] - cur_tick
            chain: list[int] = []
            T._emit_span(chain, gap, T.dtb, T.dtt, N_DTB, N_DTT)
            for tok in chain[:-1]:
                feed(tok)
                pos = len(tokens) - 1
                decoder.step(torch.tensor([tokens[-1]], device=device),
                             torch.tensor([frames[-1]], device=device, dtype=torch.float32),
                             pos, caches, T_mem, mem_valid)
            nxt = chain[-1]
            skel_i += 1; notes_here = 0
        elif kinds[nxt] == "NOTE":
            notes_here += 1
        feed(nxt)
        if nxt == EOS:
            break
    if tokens[-1] != EOS:
        if require_complete:
            raise RuntimeError('decoder exhausted step budget before natural EOS; chart is incomplete')
        # The budget may end immediately after a Hold. Its mandatory duration
        # must precede EOS even when this takes up to two extra closing tokens.
        if gs.in_dur:
            feed(T.durt(1))
        feed(EOS)
    if decoder.offset_head is not None:
        return tokens, offsets
    return tokens

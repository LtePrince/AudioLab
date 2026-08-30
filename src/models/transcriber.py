"""
src/models/transcriber.py
─────────────────────────
Transcription baseline: audio mel → per-chart-frame note predictions.

A DISCRIMINATIVE counterpart to the diffusion pipeline: the chart is treated
as a dense labelling of the audio (like automatic music transcription), so
audio→note alignment is rewarded DIRECTLY by the loss instead of arriving as
a side effect of denoising.  No VAE, no noise schedule, no sampler — one
forward pass produces the whole chart, at any input length (conv + RoPE
attention are length-agnostic; train on random crops, infer full songs).

Architecture
~~~~~~~~~~~~
    mel (B, 128, T_mel = 2·T_chart)
      → Conv1d(128→D) → k × DilatedResnetBlock1D (mel rate, rhythm RF)
      → Conv1d(D→D, k4 s2)                  # mel rate → chart-frame rate
      → depth × EncoderBlock (bidirectional self-attn + RoPE, pre-LN)
      → LayerNorm → Linear(D → 32)          # per-frame heads

Head layout (32 channels; lane k = 0..3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [ 0+k]  onset logit        BCE          (is there a note start?)
    [ 4+k]  holding logit      BCE          (inside a Hold body?)
    [ 8+k]  start_off (raw)    sigmoid+MSE  (sub-frame onset offset)
    [12+k]  end_off  (raw)     sigmoid+MSE  (Hold-end sub-frame offset)
    [16+4k … 16+4k+3]  type logits (4-way)  CE over Tap/Drag/Hold/Flick

note_type is a genuine 4-class CE — no ordinal 0.25/0.5/0.75/1.0 regression,
so there is no regression-to-the-mean collapse of Flick/Hold counts.

``to_note_array()`` converts head outputs into the standard 20-channel
note_array (logit-valued binary channels), so the existing
``Phigros4kConvertor(from_logits=True)`` decodes and serialises unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.condition.wave import DilatedResnetBlock1D
from src.data.chart2array import (
    CH_END_OFFSET,
    CH_IS_HOLDING,
    CH_IS_START,
    CH_NOTE_TYPE,
    CH_START_OFF,
    NUM_LANES,
)
from src.models.attention import RMSNorm, apply_rope, rope1d

__all__ = [
    "DEFAULT_TRANSCRIBER_CONFIG",
    "TranscriberNet",
    "TranscriptionLoss",
    "to_note_array",
    "onset_f1",
]


DEFAULT_TRANSCRIBER_CONFIG = dict(
    n_mels        = 128,
    hidden_dim    = 256,
    conv_blocks   = 4,       # dilated conv blocks at mel rate (rhythm receptive field)
    depth         = 6,       # transformer encoder blocks at chart-frame rate
    num_heads     = 8,
    mlp_ratio     = 4.0,
    dropout       = 0.1,
    rope_theta    = 10000.0,
    mel_per_frame = 2,       # mel frames per chart frame (hop 512 vs frame_ms 46.44)
)

# head channel offsets in the 32-channel output
H_ONSET   = 0
H_HOLDING = 4
H_SOFF    = 8
H_EOFF    = 12
H_TYPE    = 16   # 16 + 4*lane .. +3
OUT_CHANNELS = 16 + 4 * NUM_LANES   # 32


# ─────────────────────────────────────────────────────────────────────────────
# Encoder block (bidirectional, pre-LN, RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Pre-LN bidirectional self-attention block with RoPE and QK-RMSNorm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        mlp_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv   = nn.Linear(dim, 3 * dim, bias=True)
        self.qnorm = RMSNorm(self.head_dim)
        self.knorm = RMSNorm(self.head_dim)
        self.proj  = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            B, T, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)                      # each (B, H, T, Dh)
        q, k = self.qnorm(q), self.knorm(k)
        q, k = apply_rope(q, k, freqs)
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, H, T, Dh)
        attn = attn.permute(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# TranscriberNet
# ─────────────────────────────────────────────────────────────────────────────

class TranscriberNet(nn.Module):
    """mel (B, n_mels, 2·T) → per-chart-frame predictions (B, 32, T).

    Length-agnostic: any even T_mel works; output length is T_mel // 2.
    """

    def __init__(
        self,
        n_mels:        int   = 128,
        hidden_dim:    int   = 256,
        conv_blocks:   int   = 4,
        depth:         int   = 6,
        num_heads:     int   = 8,
        mlp_ratio:     float = 4.0,
        dropout:       float = 0.1,
        rope_theta:    float = 10000.0,
        mel_per_frame: int   = 2,
    ):
        super().__init__()
        self.mel_per_frame = mel_per_frame
        self.num_heads     = num_heads
        self.head_dim      = hidden_dim // num_heads
        self.rope_theta    = rope_theta

        self.conv_in = nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1)
        self.conv_blocks = nn.ModuleList([
            DilatedResnetBlock1D(
                hidden_dim, hidden_dim,
                dilations=(1, 2) if i % 2 == 0 else (4, 8),
                dropout=dropout,
            )
            for i in range(conv_blocks)
        ])
        # mel rate → chart-frame rate (exactly T_mel / mel_per_frame)
        self.pool = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=2 * mel_per_frame, stride=mel_per_frame,
            padding=mel_per_frame // 2,
        )
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.head     = nn.Linear(hidden_dim, OUT_CHANNELS)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def features(self, mel: Tensor) -> Tensor:
        """Chart-frame-rate hidden states (B, T_chart, D) — the audio memory
        consumed by the AR chart decoder (src/models/chart_decoder.py)."""
        h = self.conv_in(mel)
        for blk in self.conv_blocks:
            h = blk(h)
        h = self.pool(h)
        T_chart = mel.shape[-1] // self.mel_per_frame
        h = h[..., :T_chart]
        x = h.transpose(1, 2)
        pos   = torch.arange(T_chart, device=x.device).float()
        freqs = rope1d(pos, self.head_dim, self.rope_theta)
        for blk in self.blocks:
            x = blk(x, freqs)
        return self.norm_out(x)

    def forward(self, mel: Tensor) -> Tensor:
        """
        mel : (B, n_mels, T_mel)  with T_mel divisible by mel_per_frame
        →     (B, OUT_CHANNELS, T_mel // mel_per_frame)
        """
        return self.head(self.features(mel)).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionLoss(nn.Module):
    """Per-frame supervised loss against the 20-channel ground truth.

    onset / holding : BCE-with-logits (onset uses pos_weight — onsets occupy
                      only ~5% of lane-frames)
    start/end off   : MSE on sigmoid(pred), masked to gt onset / hold-end
    note_type       : 4-way cross-entropy, masked to gt onset frames
    Everything masked by valid_flag.
    """

    def __init__(
        self,
        onset_pos_weight:   float = 3.0,
        holding_pos_weight: float = 5.0,
        w_holding: float = 0.5,
        w_soff:    float = 0.5,
        w_eoff:    float = 0.2,
        w_type:    float = 0.3,
        type_class_weights: tuple[float, float, float, float] | None = (1.7, 4.3, 8.3, 15.4),
    ):
        """type_class_weights: per-class CE weights (Tap/Drag/Hold/Flick).

        Default = inverse of the training-set marginals (58.3/23.2/12.0/6.5 %),
        countering the type head's collapse to the majority class under argmax
        decoding (measured: 99.8% Tap without weighting).  holding_pos_weight
        counters the ~5%-positive is_holding channel, whose silence was the
        second gate killing Holds (holdTime=0 → downgraded to Tap on save).
        None disables class weighting.
        """
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(onset_pos_weight)))
        self.register_buffer("hold_pos_weight", torch.tensor(float(holding_pos_weight)))
        if type_class_weights is not None:
            self.register_buffer("type_weights",
                                 torch.tensor([float(x) for x in type_class_weights]))
        else:
            self.type_weights = None
        self.wh, self.ws, self.we, self.wt = w_holding, w_soff, w_eoff, w_type

    def forward(
        self,
        pred:       Tensor,   # (B, 32, T)
        target:     Tensor,   # (B, 20, T)  ground-truth note_array
        valid_flag: Tensor,   # (B, T)
    ) -> tuple[Tensor, dict]:
        K  = NUM_LANES
        vf = valid_flag.unsqueeze(1)                       # (B, 1, T)

        gt_on   = target[:, CH_IS_START   : CH_IS_START   + K]
        gt_hold = target[:, CH_IS_HOLDING : CH_IS_HOLDING + K]
        gt_soff = target[:, CH_START_OFF  : CH_START_OFF  + K]
        gt_eoff = target[:, CH_END_OFFSET : CH_END_OFFSET + K]
        gt_type = target[:, CH_NOTE_TYPE  : CH_NOTE_TYPE  + K]

        pr_on   = pred[:, H_ONSET   : H_ONSET   + K]
        pr_hold = pred[:, H_HOLDING : H_HOLDING + K]
        pr_soff = pred[:, H_SOFF    : H_SOFF    + K]
        pr_eoff = pred[:, H_EOFF    : H_EOFF    + K]

        m_on = (gt_on > 0.5).float() * vf                  # onset frames
        pad  = F.pad(gt_hold, (0, 1))
        m_end = ((gt_hold > 0.5) & (pad[:, :, 1:] <= 0.5)).float() * vf

        loss_on = (F.binary_cross_entropy_with_logits(
            pr_on, gt_on, pos_weight=self.pos_weight, reduction="none",
        ) * vf).sum() / vf.expand_as(pr_on).sum().clamp_min(1.0)

        loss_hold = (F.binary_cross_entropy_with_logits(
            pr_hold, gt_hold, pos_weight=self.hold_pos_weight, reduction="none",
        ) * vf).sum() / vf.expand_as(pr_hold).sum().clamp_min(1.0)

        loss_soff = ((torch.sigmoid(pr_soff) - gt_soff) ** 2
                     * m_on).sum() / m_on.sum().clamp_min(1.0)
        loss_eoff = ((torch.sigmoid(pr_eoff) - gt_eoff) ** 2
                     * m_end).sum() / m_end.sum().clamp_min(1.0)

        # type: 4-way CE on onset frames.  gt stores label/4 (0.25..1.0).
        type_logits = pred[:, H_TYPE : H_TYPE + 4 * K]     # (B, 16, T)
        B, _, T = type_logits.shape
        type_logits = type_logits.reshape(B, K, 4, T)      # (B, lane, class, T)
        gt_cls = (gt_type * 4.0).round().long().clamp(1, 4) - 1   # (B, K, T) in 0..3
        ce = F.cross_entropy(
            type_logits.permute(0, 2, 1, 3),               # (B, class, lane, T)
            gt_cls, weight=self.type_weights, reduction="none",
        )                                                  # (B, lane, T)
        loss_type = (ce * m_on).sum() / m_on.sum().clamp_min(1.0)

        total = (loss_on + self.wh * loss_hold + self.ws * loss_soff
                 + self.we * loss_eoff + self.wt * loss_type)
        log = {
            "loss_onset":   loss_on.detach().item(),
            "loss_holding": loss_hold.detach().item(),
            "loss_soff":    loss_soff.detach().item(),
            "loss_eoff":    loss_eoff.detach().item(),
            "loss_type":    loss_type.detach().item(),
        }
        return total, log


# ─────────────────────────────────────────────────────────────────────────────
# Decode adapter + metric
# ─────────────────────────────────────────────────────────────────────────────

def to_note_array(pred: Tensor) -> Tensor:
    """Convert head outputs (B, 32, T) into the standard 20-channel note_array
    layout with LOGIT-valued binary channels, ready for
    ``Phigros4kConvertor(from_logits=True)``.

    note_type channel gets the argmax class mapped back to 0.25/0.5/0.75/1.0 —
    exactly the stored values the convertor's thresholds decode.
    """
    B, _, T = pred.shape
    K = NUM_LANES
    note = pred.new_zeros(B, 20, T)
    note[:, CH_IS_START   : CH_IS_START   + K] = pred[:, H_ONSET   : H_ONSET   + K]
    note[:, CH_IS_HOLDING : CH_IS_HOLDING + K] = pred[:, H_HOLDING : H_HOLDING + K]
    note[:, CH_START_OFF  : CH_START_OFF  + K] = torch.sigmoid(pred[:, H_SOFF : H_SOFF + K])
    note[:, CH_END_OFFSET : CH_END_OFFSET + K] = torch.sigmoid(pred[:, H_EOFF : H_EOFF + K])
    cls = pred[:, H_TYPE : H_TYPE + 4 * K].reshape(B, K, 4, T).argmax(dim=2)  # (B,K,T)
    note[:, CH_NOTE_TYPE : CH_NOTE_TYPE + K] = (cls.float() + 1.0) / 4.0
    return note


@torch.no_grad()
def onset_f1(
    pred:       Tensor,   # (B, 32, T) head outputs
    target:     Tensor,   # (B, 20, T)
    valid_flag: Tensor,   # (B, T)
    tol_frames: int   = 2,
    threshold:  float = 0.0,   # logit threshold (0 = p 0.5)
) -> tuple[float, float, float]:
    """Lane-aware onset F1 with ±tol_frames tolerance (greedy matching).

    Returns (f1, precision, recall) aggregated over the batch.
    """
    import bisect

    K = NUM_LANES
    tp = fp = fn = 0
    for b in range(pred.shape[0]):
        valid = valid_flag[b] > 0.5
        for k in range(K):
            p_idx = torch.nonzero(
                (pred[b, H_ONSET + k] > threshold) & valid
            ).flatten().tolist()
            g_idx = torch.nonzero(
                (target[b, CH_IS_START + k] > 0.5) & valid
            ).flatten().tolist()
            used = [False] * len(g_idx)
            # both lists are sorted — restrict candidates to the ±tol window
            # via bisect, so matching is O((P+G)·tol) not O(P·G)
            for p in p_idx:
                lo = bisect.bisect_left(g_idx, p - tol_frames)
                hi = bisect.bisect_right(g_idx, p + tol_frames)
                best, best_d = -1, tol_frames + 1
                for i in range(lo, hi):
                    if used[i] or abs(g_idx[i] - p) >= best_d:
                        continue
                    best, best_d = i, abs(g_idx[i] - p)
                if best >= 0:
                    used[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += used.count(False)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, prec, rec

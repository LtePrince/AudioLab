"""encoder.py — P2
-----------------
Chart VAE: compress note_array (B, NUM_CHANNELS=20, T) ↔ latent (B, z_channels, T//stride).

Architecture
------------
  ChartEncoder  : Conv1d(x→mid) → [ResBlock×r (+Downsample?)]×N levels
                  → GroupNorm+SiLU+Conv1d(mid→2z)
  ChartDecoder  : Conv1d(z→mid) → mid blocks
                  → [ResBlock×(r+1) (+Upsample?)]×N levels
                  → GroupNorm+SiLU+Conv1d(mid→x)
  ChartVAE      : encode() → DiagonalGaussianDistribution → sample/mode → decode()
  ChartReconLoss: per-semantic-group typed loss with valid_flag masking

Channel layout (x_channels = NUM_CHANNELS = 20, NUM_LANES = 4)
--------------------------------------------------------------
  k = 0..3  (lane index)
  x[:, k +  0, :]  is_start   ∈ {0, 1}               → BCE (logits)
  x[:, k +  4, :]  start_off  ∈ [0, 1]                → MSE × is_start mask
  x[:, k +  8, :]  is_holding ∈ {0, 1}                → BCE (logits)
  x[:, k + 12, :]  end_offset ∈ [0, 1]                → MSE × is_end mask
  x[:, k + 16, :]  note_type  ∈ {0, .25, .50, .75, 1} → MSE × is_start mask

Default config (for 4096-frame charts)
---------------------------------------
  x_channels     = 20          (NUM_CHANNELS)
  z_channels     = 16
  middle_channels = 64
  channel_mult   = (1, 1, 2, 4, 4)  → stride = 2⁴ = 16  → latent_T = 4096 // 16 = 256
  num_res_blocks  = 2
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.chart2array import (
    NUM_CHANNELS,
    NUM_LANES,
    CH_IS_START,
    CH_START_OFF,
    CH_IS_HOLDING,
    CH_END_OFFSET,
    CH_NOTE_TYPE,
)

# ===========================================================================
# Default configuration
# ===========================================================================

DEFAULT_VAE_CONFIG: dict = {
    "x_channels":      NUM_CHANNELS,   # 20
    "z_channels":      16,
    "middle_channels": 64,
    "channel_mult":    (1, 1, 2, 4, 4),  # 4 downsamplings → stride 16 → latent_T = 256
    "num_res_blocks":  2,
    "num_groups":      32,
}


# ===========================================================================
# Section 1 – Building blocks
# ===========================================================================

def _norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True
    )


class Upsample1D(nn.Module):
    """2× nearest-neighbour upsample followed by Conv1d(same channels)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample1D(nn.Module):
    """Stride-2 Conv1d with right-pad to maintain exact length halving."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1)))   # right-pad by 1 before stride-2


class ResnetBlock1D(nn.Module):
    """1-D residual block: Norm+SiLU+Conv → Norm+SiLU+Dropout+Conv + shortcut.

    No time-embedding (VAE does not condition on diffusion timestep).
    """

    def __init__(
        self,
        in_ch:      int,
        out_ch:     int,
        dropout:    float = 0.0,
        num_groups: int   = 32,
    ) -> None:
        super().__init__()
        self.norm1    = _norm(in_ch,  num_groups)
        self.conv1    = nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=1)
        self.norm2    = _norm(out_ch, num_groups)
        self.drop     = nn.Dropout(dropout)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


# ===========================================================================
# Section 2 – Diagonal Gaussian distribution
# ===========================================================================

class DiagonalGaussianDistribution:
    """Wraps encoder output (mean‖logvar on channel axis).

    Parameters
    ----------
    parameters : Tensor (B, 2*z_channels, T//stride)
        Concatenated mean and log-variance output from the encoder.
    scale : float
        Scaling factor applied to sampled / mode latents.
    """

    def __init__(self, parameters: torch.Tensor, scale: float = 1.0) -> None:
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -10.0, 20.0)
        self.std    = torch.exp(0.5 * self.logvar)
        self.var    = torch.exp(self.logvar)
        self.scale  = scale

    def sample(self) -> torch.Tensor:
        """Reparameterised sample: z = (mean + std * ε) * scale."""
        return (self.mean + self.std * torch.randn_like(self.mean)) * self.scale

    def mode(self) -> torch.Tensor:
        """Deterministic latent: mean * scale."""
        return self.mean * self.scale

    def kl(self) -> torch.Tensor:
        """KL(q ‖ N(0,1)) averaged over all batch×channel×time elements."""
        return 0.5 * torch.mean(self.mean ** 2 + self.var - 1.0 - self.logvar)


# ===========================================================================
# Section 3 – Encoder
# ===========================================================================

class ChartEncoder(nn.Module):
    """1-D convolutional encoder.

    Maps ``(B, x_channels, T)`` → ``(B, z_channels*2, T // stride)``
    where ``stride = 2 ** (len(channel_mult) - 1)``.

    Parameters
    ----------
    x_channels, z_channels, middle_channels, channel_mult,
    num_res_blocks, num_groups, dropout
        See ``DEFAULT_VAE_CONFIG`` for defaults.
    """

    def __init__(
        self,
        x_channels:      int,
        z_channels:      int,
        middle_channels: int,
        channel_mult:    Sequence[int],
        num_res_blocks:  int,
        num_groups:      int   = 32,
        dropout:         float = 0.0,
        **_,
    ) -> None:
        super().__init__()

        self.num_levels = len(channel_mult)
        inc_mult = (1,) + tuple(channel_mult)

        self.conv_in = nn.Conv1d(x_channels, middle_channels, kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        for lvl in range(self.num_levels):
            ch_in  = middle_channels * inc_mult[lvl]
            ch_out = middle_channels * channel_mult[lvl]
            level  = nn.Module()
            level.blocks = nn.ModuleList([
                ResnetBlock1D(ch_in if i == 0 else ch_out, ch_out, dropout, num_groups)
                for i in range(num_res_blocks)
            ])
            if lvl != self.num_levels - 1:
                level.downsample = Downsample1D(ch_out)
            self.down.append(level)

        ch = middle_channels * channel_mult[-1]
        self.mid1     = ResnetBlock1D(ch, ch, dropout, num_groups)
        self.mid2     = ResnetBlock1D(ch, ch, dropout, num_groups)
        self.norm_out = _norm(ch, num_groups)
        self.conv_out = nn.Conv1d(ch, z_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for lvl, level in enumerate(self.down):
            for blk in level.blocks:
                h = blk(h)
            if lvl != self.num_levels - 1:
                h = level.downsample(h)
        h = self.mid1(h)
        h = self.mid2(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ===========================================================================
# Section 4 – Decoder
# ===========================================================================

class ChartDecoder(nn.Module):
    """1-D convolutional decoder (mirror of ChartEncoder).

    Maps ``(B, z_channels, T // stride)`` → ``(B, x_channels, T)``.
    """

    def __init__(
        self,
        x_channels:      int,
        z_channels:      int,
        middle_channels: int,
        channel_mult:    Sequence[int],
        num_res_blocks:  int,
        num_groups:      int   = 32,
        dropout:         float = 0.0,
        **_,
    ) -> None:
        super().__init__()

        self.num_levels = len(channel_mult)
        ch = middle_channels * channel_mult[-1]

        self.conv_in = nn.Conv1d(z_channels, ch, kernel_size=3, padding=1)
        self.mid1    = ResnetBlock1D(ch, ch, dropout, num_groups)
        self.mid2    = ResnetBlock1D(ch, ch, dropout, num_groups)

        # Build levels from highest (most-compressed) to lowest, then
        # insert(0, ...) so that self.up[lvl] ≡ level lvl.
        self.up = nn.ModuleList()
        for lvl in reversed(range(self.num_levels)):
            ch_out = middle_channels * channel_mult[lvl]
            level  = nn.Module()
            level.blocks = nn.ModuleList([
                ResnetBlock1D(ch if i == 0 else ch_out, ch_out, dropout, num_groups)
                for i in range(num_res_blocks + 1)
            ])
            if lvl != 0:
                level.upsample = Upsample1D(ch_out)
            self.up.insert(0, level)   # self.up[lvl] ≡ level lvl
            ch = ch_out

        self.norm_out = _norm(ch, num_groups)
        self.conv_out = nn.Conv1d(ch, x_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid1(h)
        h = self.mid2(h)
        for lvl in reversed(range(self.num_levels)):
            for blk in self.up[lvl].blocks:
                h = blk(h)
            if lvl != 0:
                h = self.up[lvl].upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ===========================================================================
# Section 5 – ChartVAE
# ===========================================================================

class ChartVAE(nn.Module):
    """Variational Autoencoder for Phigros 4-lane chart arrays.

    Parameters
    ----------
    config : dict
        Passed to both ``ChartEncoder`` and ``ChartDecoder``.
        Required keys: ``x_channels``, ``z_channels``, ``middle_channels``,
        ``channel_mult``, ``num_res_blocks``.
        Optional: ``num_groups`` (default 32), ``dropout`` (default 0.0).
    kl_weight : float
        KL-divergence weight in ``compute_loss()``.  Default ``1e-6``.
    scale : float
        Scaling factor applied to latent sample / mode.  Default ``1.0``.

    Usage
    -----
    >>> vae   = ChartVAE(DEFAULT_VAE_CONFIG)
    >>> recon, post = vae(x)              # forward  (x: B,20,T)
    >>> z     = vae.encode(x).sample()   # latent   (B,16,T//16)
    >>> recon = vae.decode(z)            # decode   (B,20,T)
    """

    def __init__(
        self,
        config:    dict,
        kl_weight: float = 1e-6,
        scale:     float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder   = ChartEncoder(**config)
        self.decoder   = ChartDecoder(**config)
        self.kl_weight = kl_weight
        self.scale     = scale

    # ---------------------------------------------------------------------- #
    # Core API                                                                #
    # ---------------------------------------------------------------------- #

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """Encode ``x`` → posterior distribution."""
        return DiagonalGaussianDistribution(self.encoder(x), scale=self.scale)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent ``z`` (already scaled) → raw channel logits/values."""
        return self.decoder(z / self.scale)

    def forward(
        self,
        x:      torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """Encode → sample/mode → decode.

        Returns
        -------
        recon     : Tensor, same shape as ``x`` — raw logits/values
        posterior : :class:`DiagonalGaussianDistribution`
        """
        posterior = self.encode(x)
        z         = posterior.sample() if sample else posterior.mode()
        return self.decode(z), posterior

    def compute_loss(
        self,
        x:          torch.Tensor,
        valid_flag: torch.Tensor,
        loss_fn:    "ChartReconLoss",
    ) -> tuple[torch.Tensor, dict]:
        """Full training step: forward → recon loss + KL.

        Parameters
        ----------
        x          : Tensor (B, NUM_CHANNELS, T)
        valid_flag : Tensor (B, T)   — 1 inside the valid region, 0 after
        loss_fn    : :class:`ChartReconLoss`

        Returns
        -------
        loss : scalar Tensor  (recon + kl_weight * kl)
        log  : dict with per-group losses and kl/total
        """
        recon, posterior = self(x)
        recon_loss, log  = loss_fn(x, recon, valid_flag)
        kl_loss          = posterior.kl()
        loss             = recon_loss + self.kl_weight * kl_loss
        log["kl_loss"]    = kl_loss.detach().item()
        log["kl_var"]     = posterior.std.mean().detach().item()
        log["total_loss"] = loss.detach().item()
        return loss, log


# ===========================================================================
# Section 6 – Reconstruction loss
# ===========================================================================

class ChartReconLoss(nn.Module):
    """Per-semantic-group typed reconstruction loss for the 20-channel array.

    Group → loss type → secondary mask:

    =========== ========= ==============================
    Group        Loss      Mask
    =========== ========= ==============================
    is_start     BCE       valid_flag
    start_off    MSE       valid_flag × is_start > 0.5
    is_holding   BCE       valid_flag
    end_offset   MSE       valid_flag × is_end > 0.5
    note_type    MSE       valid_flag × is_start > 0.5
    =========== ========= ==============================

    ``is_end[k, t] = 1``  iff  ``is_holding[k,t]=1`` and ``is_holding[k,t+1]=0``.

    Parameters
    ----------
    weight_start_off  : weight for start_off MSE term.   Default ``0.5``.
    weight_holding    : weight for is_holding BCE term.   Default ``0.5``.
    weight_end_offset : weight for end_offset MSE term.   Default ``0.2``.
    weight_note_type  : weight for note_type  MSE term.   Default ``0.2``.
    label_smoothing   : ε for BCE label smoothing.        Default ``0.001``.
    """

    def __init__(
        self,
        weight_start_off:  float = 0.5,
        weight_holding:    float = 0.5,
        weight_end_offset: float = 0.2,
        weight_note_type:  float = 0.2,
        label_smoothing:   float = 0.001,
    ) -> None:
        super().__init__()
        self.ws  = weight_start_off
        self.wh  = weight_holding
        self.we  = weight_end_offset
        self.wn  = weight_note_type
        self.ls  = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _smooth(self, t: torch.Tensor) -> torch.Tensor:
        return t * (1.0 - 2.0 * self.ls) + self.ls

    def _bce_masked(
        self,
        pred: torch.Tensor,   # (B, K, T)
        tgt:  torch.Tensor,
        mask: torch.Tensor,   # (B, 1, T) or (B, K, T)
    ) -> torch.Tensor:
        return (self.bce(pred, self._smooth(tgt)) * mask).sum() / (mask.sum() + 1e-6)

    def _mse_masked(
        self,
        pred: torch.Tensor,
        tgt:  torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return (self.mse(pred, tgt) * mask).sum() / (mask.sum() + 1e-6)

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x:          torch.Tensor,   # (B, 20, T)  ground truth
        recon:      torch.Tensor,   # (B, 20, T)  reconstruction
        valid_flag: torch.Tensor,   # (B, T)
    ) -> tuple[torch.Tensor, dict]:
        K  = NUM_LANES          # 4
        vf = valid_flag.unsqueeze(1)   # (B, 1, T) for broadcast

        # ---- ground-truth channel slices ----
        gt_start   = x[:, CH_IS_START   : CH_IS_START   + K, :]
        gt_soff    = x[:, CH_START_OFF  : CH_START_OFF  + K, :]
        gt_holding = x[:, CH_IS_HOLDING : CH_IS_HOLDING + K, :]
        gt_eoff    = x[:, CH_END_OFFSET : CH_END_OFFSET + K, :]
        gt_ntype   = x[:, CH_NOTE_TYPE  : CH_NOTE_TYPE  + K, :]

        # ---- reconstruction channel slices ----
        pr_start   = recon[:, CH_IS_START   : CH_IS_START   + K, :]
        pr_soff    = recon[:, CH_START_OFF  : CH_START_OFF  + K, :]
        pr_holding = recon[:, CH_IS_HOLDING : CH_IS_HOLDING + K, :]
        pr_eoff    = recon[:, CH_END_OFFSET : CH_END_OFFSET + K, :]
        pr_ntype   = recon[:, CH_NOTE_TYPE  : CH_NOTE_TYPE  + K, :]

        # ---- secondary masks ----
        m_start = (gt_start > 0.5).float() * vf            # note onset exists
        h_pad   = F.pad(gt_holding, (0, 1))                 # right-pad by 1
        is_end  = ((gt_holding > 0.5) & (h_pad[:, :, 1:] <= 0.5)).float()
        m_end   = is_end * vf                               # hold tail frame

        # ---- per-group losses ----
        loss_start   = self._bce_masked(pr_start,   gt_start,   vf)
        loss_soff    = self._mse_masked(pr_soff,    gt_soff,    m_start)
        loss_holding = self._bce_masked(pr_holding, gt_holding, vf)
        loss_eoff    = self._mse_masked(pr_eoff,    gt_eoff,    m_end)
        loss_ntype   = self._mse_masked(pr_ntype,   gt_ntype,   m_start)

        total = (
            loss_start
            + self.ws * loss_soff
            + self.wh * loss_holding
            + self.we * loss_eoff
            + self.wn * loss_ntype
        )
        log = {
            "loss_start":   loss_start.detach().item(),
            "loss_soff":    loss_soff.detach().item(),
            "loss_holding": loss_holding.detach().item(),
            "loss_eoff":    loss_eoff.detach().item(),
            "loss_ntype":   loss_ntype.detach().item(),
        }
        return total, log

"""wave.py — P3
--------------
Audio feature extractor: mel spectrogram → per-frame condition tensor.

Architecture
------------
  AudioWaveEncoder (1-D):
    Conv1d(n_mels → mid)
    → [Downsample1D + ResBlock×r (+DilatResBlock?) ]×N levels
    → mid ResBlocks
    → GroupNorm + SiLU + Conv1d(mid → out_channels)

  Output:  (B, out_channels, T_chart)
           T_chart = T_mel // audio_stride
           audio_stride must equal chart VAE stride so the two time axes align.

Default config
--------------
  n_mels         = 128
  middle_channels = 128
  out_channels   = 256          latent condition dim fed into DiT
  channel_mult   = (1,1,2,2,4)  → 4 Downsample → audio_stride = 16
  num_res_blocks  = 2
  num_groups      = 32

  chart VAE stride = 16  (DEFAULT_VAE_CONFIG channel_mult has 4 downsamplings)
  → T_chart = T_mel // 16   ✓  same temporal resolution as VAE latent

Frame-rate alignment
--------------------
  mel hop_length = 512,  sr = 22050  → one mel frame ≈ 23.2 ms
  chart FRAME_MS ≈ 46.44 ms          → one chart frame ≈ 2 mel frames
  audio_stride   = 16                → one latent frame = 16 mel frames ≈ 371 ms

  In practice the DiT learns to align audio and chart tokens through
  attention; exact ms alignment is not required at this stage.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# Default configuration
# ===========================================================================

DEFAULT_WAVE_CONFIG: dict = {
    "n_mels":          128,
    "middle_channels": 128,
    "out_channels":    256,
    "channel_mult":    (1, 1, 2, 2, 4),   # 4 downsamples → stride 16
    "num_res_blocks":  2,
    "num_groups":      32,
    "dropout":         0.0,
}


# ===========================================================================
# Section 1 – Building blocks  (self-contained, no dependency on encoder.py)
# ===========================================================================

def _norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)


class Upsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample1D(nn.Module):
    """Stride-2 Conv1d with right-pad."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1)))


class ResnetBlock1D(nn.Module):
    """Standard residual block: Norm+SiLU+Conv → Norm+SiLU+Drop+Conv + shortcut."""

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


class DilatedResnetBlock1D(nn.Module):
    """Dilated residual block: alternates dilation (1,2) / (4,8) to widen receptive field.

    Mirrors ``MelspectrogramScaleEncoder1D`` in Mug-Diffusion/mug/cond/wave.py.
    """

    def __init__(
        self,
        in_ch:      int,
        out_ch:     int,
        dilations:  tuple[int, int] = (1, 2),
        dropout:    float = 0.0,
        num_groups: int   = 32,
    ) -> None:
        super().__init__()
        d0, d1 = dilations
        self.norm1    = _norm(in_ch,  num_groups)
        self.conv1    = nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=d0, dilation=d0)
        self.norm2    = _norm(out_ch, num_groups)
        self.drop     = nn.Dropout(dropout)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=d1, dilation=d1)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


# ===========================================================================
# Section 2 – AudioWaveEncoder
# ===========================================================================

class AudioWaveEncoder(nn.Module):
    """1-D convolutional audio feature extractor.

    Maps ``(B, n_mels, T_mel)`` → ``(B, out_channels, T_mel // audio_stride)``
    where ``audio_stride = 2 ** (len(channel_mult) - 1)``.

    The architecture mirrors ``MelspectrogramEncoder1D`` from Mug-Diffusion but:
      - uses dilated ResBlocks for wider temporal receptive field
      - alternates dilations (1,2) and (4,8) per block (same as STFTEncoder)
      - no position embedding (RoPE is handled in the DiT)

    Parameters
    ----------
    n_mels, middle_channels, out_channels, channel_mult,
    num_res_blocks, num_groups, dropout
        See ``DEFAULT_WAVE_CONFIG``.
    """

    def __init__(
        self,
        n_mels:          int,
        middle_channels: int,
        out_channels:    int,
        channel_mult:    Sequence[int],
        num_res_blocks:  int,
        num_groups:      int   = 32,
        dropout:         float = 0.0,
        **_,
    ) -> None:
        super().__init__()

        self.num_levels  = len(channel_mult)
        self.audio_stride = 2 ** (self.num_levels - 1)
        inc_mult = (1,) + tuple(channel_mult)

        # project mel bins → middle_channels
        self.conv_in = nn.Conv1d(n_mels, middle_channels, kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        for lvl in range(self.num_levels):
            ch_in  = middle_channels * inc_mult[lvl]
            ch_out = middle_channels * channel_mult[lvl]
            level  = nn.Module()
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                # alternate dilation pairs: (1,2) even blocks, (4,8) odd blocks
                dilations = (1, 2) if i % 2 == 0 else (4, 8)
                blocks.append(
                    DilatedResnetBlock1D(
                        ch_in if i == 0 else ch_out,
                        ch_out,
                        dilations=dilations,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
            level.blocks = blocks
            if lvl != self.num_levels - 1:
                level.downsample = Downsample1D(ch_out)
            self.down.append(level)

        ch = middle_channels * channel_mult[-1]
        self.mid1     = ResnetBlock1D(ch, ch, dropout, num_groups)
        self.mid2     = ResnetBlock1D(ch, ch, dropout, num_groups)
        self.norm_out = _norm(ch, num_groups)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, n_mels, T_mel)

        Returns
        -------
        Tensor (B, out_channels, T_mel // audio_stride)
        """
        h = self.conv_in(x)
        for lvl, level in enumerate(self.down):
            for blk in level.blocks:
                h = blk(h)
            if lvl != self.num_levels - 1:
                h = level.downsample(h)
        h = self.mid1(h)
        h = self.mid2(h)
        return self.conv_out(F.silu(self.norm_out(h)))

    @property
    def stride(self) -> int:
        """Temporal downsampling factor (mel frames → condition frames)."""
        return self.audio_stride

    def summary(self, n_mels: int = 128, T: int = 16384) -> None:
        total  = sum(p.numel() for p in self.parameters())
        device = next(self.parameters()).device
        x      = torch.zeros(1, n_mels, T, device=device)
        with torch.no_grad():
            y = self.forward(x)
        print(
            f"AudioWaveEncoder  input ({n_mels}, {T}) → output {tuple(y.shape[1:])}  "
            f"[stride={self.stride}]  params={total:,}"
        )

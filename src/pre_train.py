"""
src/pre_train.py
─────────────────────────────────────────────────────────────────────────────
Pre-training script for ChartVAE and AudioWaveEncoder.

The VAE must be pre-trained before running train.py (wave encoder is optional).

Modes (--mode)
~~~~~~~~~~~~~~
  vae   Pre-train ChartVAE (note_array ↔ latent), saves vae_final.pt
  wave  Pre-train AudioWaveEncoder (mel autoencoder), saves wave_final.pt
  both  Run vae first, then wave

ChartVAE training objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  L = ChartReconLoss(x, vae.decode(z), valid_flag)
      + β × KL(posterior ‖ N(0,1))

  β is linearly annealed from kl_weight_start to kl_weight to prevent KL collapse.

AudioWaveEncoder training objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  A temporary WaveDecoder (symmetric to the encoder) is built during training:
  mel → WaveEncoder → latent → WaveDecoder → mel̂
  L = MSE(mel̂, mel)
  Only the WaveEncoder weights are saved at the end; the decoder is discarded.

Checkpoint format
~~~~~~~~~~~~~~~~~
  - Intermediate: {"vae"/"wave": state_dict, "optimizer": ..., "epoch": ..., "step": ...}
  - Final vae_final.pt / wave_final.pt: plain state_dict (compatible with train.py)

Usage
~~~~~
  # Step 1: pre-train VAE (required)
  uv run python src/pre_train.py --mode vae \\
      --data-list data/data.txt \\
      --ckpt-dir  checkpoints/    \\
      --epochs    100

  # Step 2: pre-train Wave encoder (optional)
  uv run python src/pre_train.py --mode wave \\
      --data-list data/data.txt \\
      --ckpt-dir  checkpoints/    \\
      --epochs    50

  # Run both in one go
  uv run python src/pre_train.py --mode both \\
      --data-list data/data.txt \\
      --ckpt-dir  checkpoints/

  # Resume from a checkpoint
  uv run python src/pre_train.py --mode vae \\
      --data-list data/data.txt \\
      --vae-ckpt  checkpoints/vae/vae_e0010_s000500.pt
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.condition.wave import (
    AudioWaveEncoder,
    DEFAULT_WAVE_CONFIG,
    ResnetBlock1D,
    Upsample1D,
)
from src.data.dataset import PhigrosDataset
from src.encoder.encoder import (
    ChartReconLoss,
    ChartVAE,
    DEFAULT_VAE_CONFIG,
)


# ─────────────────────────────────────────────────────────────────────────────
# WaveDecoder ── used only during Wave pre-training, discarded afterwards
# ─────────────────────────────────────────────────────────────────────────────

class WaveDecoder(nn.Module):
    """Symmetric decoder for AudioWaveEncoder (used only during pre-training).

    Maps ``(B, out_channels, T_z)`` → ``(B, n_mels, T_z * audio_stride)``.

    Mirrors the AudioWaveEncoder architecture using standard ResnetBlock1D (no dilation).
    """

    def __init__(
        self,
        n_mels:          int,
        middle_channels: int,
        out_channels:    int,
        channel_mult:    tuple[int, ...] | list[int],
        num_res_blocks:  int,
        num_groups:      int   = 32,
        dropout:         float = 0.0,
        **_,
    ) -> None:
        super().__init__()

        self.num_levels = len(channel_mult)
        ch = middle_channels * channel_mult[-1]

        # input projection: out_channels → middle * channel_mult[-1]
        self.conv_in = nn.Conv1d(out_channels, ch, kernel_size=3, padding=1)
        self.mid1    = ResnetBlock1D(ch, ch, dropout, num_groups)
        self.mid2    = ResnetBlock1D(ch, ch, dropout, num_groups)

        # build upsampling levels in reverse order (same pattern as ChartDecoder)
        self.up = nn.ModuleList()
        for lvl in reversed(range(self.num_levels)):
            ch_out = middle_channels * channel_mult[lvl]
            level  = nn.Module()
            level.blocks = nn.ModuleList([
                ResnetBlock1D(
                    ch if i == 0 else ch_out,
                    ch_out,
                    dropout,
                    num_groups,
                )
                for i in range(num_res_blocks + 1)
            ])
            if lvl != 0:
                level.upsample = Upsample1D(ch_out)
            self.up.insert(0, level)   # self.up[lvl] ≡ level lvl
            ch = ch_out

        self.norm_out = nn.GroupNorm(num_groups, ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv1d(ch, n_mels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor (B, out_channels, T_z)

        Returns
        -------
        Tensor (B, n_mels, T_z * audio_stride)
        """
        h = self.conv_in(z)
        h = self.mid1(h)
        h = self.mid2(h)
        for lvl in reversed(range(self.num_levels)):
            for blk in self.up[lvl].blocks:
                h = blk(h)
            if lvl != 0:
                h = self.up[lvl].upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pad_or_trim(t: torch.Tensor, target: int) -> torch.Tensor:
    """Right-trim or zero-pad tensor to exactly ``target`` frames (last dimension)."""
    T = t.shape[-1]
    if T >= target:
        return t[..., :target]
    return F.pad(t, (0, target - T))


def _collate_vae(batch: list[dict], max_frame: int) -> dict:
    """Collate function for VAE training: returns note and valid_flag only."""
    notes = torch.stack([_pad_or_trim(s["note"],       max_frame) for s in batch])
    flags = torch.stack([_pad_or_trim(s["valid_flag"], max_frame) for s in batch])
    return {"note": notes, "valid_flag": flags}


def _collate_wave(batch: list[dict], max_mel_frames: int) -> dict:
    """Collate function for Wave training: returns mel spectrogram only."""
    mels = torch.stack([_pad_or_trim(s["audio"], max_mel_frames) for s in batch])
    return {"audio": mels}


# ─────────────────────────────────────────────────────────────────────────────
# ChartVAE pre-training
# ─────────────────────────────────────────────────────────────────────────────

def _pretrain_vae(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    print(f"[vae] device={device}  seed={args.seed}")

    ckpt_dir = Path(args.ckpt_dir) / "vae"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset = PhigrosDataset(
        data_list_path   = args.data_list,
        convertor_params = {"frame_ms": args.frame_ms, "max_frame": args.max_frame},
        cache_dir        = args.cache_dir,
        chart_cache_dir  = args.chart_cache_dir,
        augment          = True,
        hop_length       = args.hop_length,
        n_mels           = args.n_mels,
        sr               = args.sr,
        device           = str(device),
    )
    print(f"[vae] dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = partial(_collate_vae, max_frame=args.max_frame),
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    vae     = ChartVAE(DEFAULT_VAE_CONFIG, kl_weight=args.kl_weight).to(device)
    loss_fn = ChartReconLoss().to(device)
    print(f"[vae] ChartVAE  params: {sum(p.numel() for p in vae.parameters()):,}")

    # ── Optimizer & LR scheduler ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )
    total_steps = args.epochs * max(1, len(loader) // args.grad_accum)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.vae_ckpt:
        raw = torch.load(args.vae_ckpt, map_location=device, weights_only=True)
        if isinstance(raw, dict) and "vae" in raw:
            vae.load_state_dict(raw["vae"])
            if "optimizer" in raw:
                optimizer.load_state_dict(raw["optimizer"])
            start_epoch = raw.get("epoch", 0)
            global_step = raw.get("step", 0)
        else:
            vae.load_state_dict(raw)
        print(f"[vae] resumed ← {args.vae_ckpt}  epoch={start_epoch}  step={global_step}")

    # KL weight annealing: linearly ramp from kl_start to kl_end
    kl_start          = args.kl_weight_start
    kl_end            = args.kl_weight
    kl_anneal_epochs  = max(1, int(args.epochs * args.kl_anneal_frac))

    loss_sum  = 0.0
    loss_count = 0
    t0        = time.time()
    optimizer.zero_grad()

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        vae.train()

        # current β (linear anneal)
        frac          = min(1.0, epoch / kl_anneal_epochs)
        vae.kl_weight = kl_start + frac * (kl_end - kl_start)

        for batch_idx, batch in enumerate(loader):
            note = batch["note"].to(device)         # (B, 20, max_frame)
            flag = batch["valid_flag"].to(device)   # (B, max_frame)

            loss, log = vae.compute_loss(note, flag, loss_fn)
            (loss / args.grad_accum).backward()

            loss_sum   += loss.item()
            loss_count += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = loss_sum / loss_count
                    lr_now   = optimizer.param_groups[0]["lr"]
                    elapsed  = time.time() - t0
                    print(
                        f"[vae] epoch {epoch + 1:>4d}/{args.epochs}  "
                        f"step {global_step:>6d}  "
                        f"loss {avg_loss:.5f}  "
                        f"recon {log['loss_start']:.4f}+{log['loss_soff']:.4f}"
                        f"+{log['loss_holding']:.4f}  "
                        f"kl {log['kl_loss']:.5f}  "
                        f"β {vae.kl_weight:.1e}  "
                        f"lr {lr_now:.2e}  "
                        f"t {elapsed:.0f}s"
                    )
                    loss_sum   = 0.0
                    loss_count = 0

        # save intermediate checkpoint every save_every epochs (with optimizer state for resuming)
        if (epoch + 1) % args.save_every == 0:
            mid_path = ckpt_dir / f"vae_e{epoch + 1:04d}_s{global_step:07d}.pt"
            torch.save(
                {
                    "vae":       vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch + 1,
                    "step":      global_step,
                },
                mid_path,
            )
            print(f"[vae] saved → {mid_path}")

    # final checkpoint: plain state_dict, directly compatible with train.py --vae-ckpt
    final_path = ckpt_dir / "vae_final.pt"
    torch.save(vae.state_dict(), final_path)
    print(f"[vae] done → {final_path}  elapsed={time.time() - t0:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# AudioWaveEncoder pre-training (mel autoencoder)
# ─────────────────────────────────────────────────────────────────────────────

def _pretrain_wave(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    print(f"[wave] device={device}  seed={args.seed}")

    ckpt_dir = Path(args.ckpt_dir) / "wave"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    # Wave pre-training does not need chart data, but reusing PhigrosDataset is simplest
    dataset = PhigrosDataset(
        data_list_path   = args.data_list,
        convertor_params = {"frame_ms": args.frame_ms, "max_frame": args.max_frame},
        cache_dir        = args.cache_dir,
        chart_cache_dir  = args.chart_cache_dir,
        augment          = False,   # mel autoencoder does not need chart mirror augmentation
        hop_length       = args.hop_length,
        n_mels           = args.n_mels,
        sr               = args.sr,
        device           = str(device),
    )
    print(f"[wave] dataset: {len(dataset)} samples")

    max_mel_frames = args.max_frame   # must stay consistent with train.py
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = partial(_collate_wave, max_mel_frames=max_mel_frames),
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    encoder = AudioWaveEncoder(**DEFAULT_WAVE_CONFIG).to(device)
    decoder = WaveDecoder(**DEFAULT_WAVE_CONFIG).to(device)

    params_enc = sum(p.numel() for p in encoder.parameters())
    params_dec = sum(p.numel() for p in decoder.parameters())
    print(f"[wave] WaveEncoder params: {params_enc:,}  WaveDecoder params: {params_dec:,}")

    # ── Optimizer & LR scheduler ───────────────────────────────────────────
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )
    total_steps = args.epochs * max(1, len(loader) // args.grad_accum)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.wave_ckpt:
        raw = torch.load(args.wave_ckpt, map_location=device, weights_only=True)
        if isinstance(raw, dict) and "wave" in raw:
            encoder.load_state_dict(raw["wave"])
            if "decoder" in raw:
                decoder.load_state_dict(raw["decoder"])
            if "optimizer" in raw:
                optimizer.load_state_dict(raw["optimizer"])
            start_epoch = raw.get("epoch", 0)
            global_step = raw.get("step", 0)
        else:
            encoder.load_state_dict(raw)
        print(f"[wave] resumed ← {args.wave_ckpt}  epoch={start_epoch}  step={global_step}")

    # encoder stride — used to trim mel length to an exact multiple of stride
    stride = encoder.stride  # = 16

    loss_sum   = 0.0
    loss_count = 0
    t0         = time.time()
    optimizer.zero_grad()

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        decoder.train()

        for batch_idx, batch in enumerate(loader):
            mel = batch["audio"].to(device)   # (B, n_mels, T_mel)

            # trim to a multiple of stride so encode→decode length is exact
            T_trim = (mel.shape[-1] // stride) * stride
            mel    = mel[..., :T_trim]

            # autoencoding
            z       = encoder(mel)              # (B, 256, T_mel // stride)
            mel_hat = decoder(z)                # (B, n_mels, T_mel)

            # length alignment (upsample may occasionally produce one extra frame)
            if mel_hat.shape[-1] != mel.shape[-1]:
                mel_hat = mel_hat[..., :mel.shape[-1]]

            loss = F.mse_loss(mel_hat, mel)
            (loss / args.grad_accum).backward()

            loss_sum   += loss.item()
            loss_count += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(all_params, args.clip_grad)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = loss_sum / loss_count
                    lr_now   = optimizer.param_groups[0]["lr"]
                    elapsed  = time.time() - t0
                    print(
                        f"[wave] epoch {epoch + 1:>4d}/{args.epochs}  "
                        f"step {global_step:>6d}  "
                        f"mse_loss {avg_loss:.5f}  "
                        f"lr {lr_now:.2e}  "
                        f"t {elapsed:.0f}s"
                    )
                    loss_sum   = 0.0
                    loss_count = 0

        # save intermediate checkpoint every save_every epochs (encoder + decoder, resumable)
        if (epoch + 1) % args.save_every == 0:
            mid_path = ckpt_dir / f"wave_e{epoch + 1:04d}_s{global_step:07d}.pt"
            torch.save(
                {
                    "wave":      encoder.state_dict(),
                    "decoder":   decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch + 1,
                    "step":      global_step,
                },
                mid_path,
            )
            print(f"[wave] saved → {mid_path}")

    # final checkpoint: encoder state_dict only, directly compatible with train.py --wave-ckpt
    final_path = ckpt_dir / "wave_final.pt"
    torch.save(encoder.state_dict(), final_path)
    print(f"[wave] done → {final_path}  elapsed={time.time() - t0:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-train ChartVAE and/or AudioWaveEncoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mode",
        choices=["vae", "wave", "both"],
        default="vae",
        help="pre-training target: vae / wave / both (vae first, then wave)",
    )

    # ── data ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("data")
    g.add_argument("--data-list",   required=True,
                   help="path to data_list.txt")
    g.add_argument("--cache-dir",   default=None,
                   help="directory for cached mel .npy files (optional)")
    g.add_argument("--chart-cache-dir", default=None,
                   help="directory for cached chart .npz files (skips JSON parse on cache hit; disables rate augmentation)")
    g.add_argument("--frame-ms",    type=float,
                   default=512 / 22050 / 4 * 8 * 1000,
                   help="milliseconds per chart frame (≈ 46.44 ms)")
    g.add_argument("--max-frame",   type=int,   default=4096,
                   help="maximum number of chart frames")
    g.add_argument("--hop-length",  type=int,   default=512)
    g.add_argument("--n-mels",      type=int,   default=128)
    g.add_argument("--sr",          type=int,   default=22050)
    g.add_argument("--num-workers", type=int,   default=4)

    # ── checkpoint ─────────────────────────────────────────────────────────
    g = p.add_argument_group("checkpoint")
    g.add_argument("--ckpt-dir",   default="checkpoints",
                   help="root checkpoint directory (subdirs vae/ wave/ are created automatically)")
    g.add_argument("--vae-ckpt",   default=None,
                   help="VAE checkpoint path to resume training from")
    g.add_argument("--wave-ckpt",  default=None,
                   help="Wave checkpoint path to resume training from")
    g.add_argument("--save-every", type=int, default=10,
                   help="save intermediate checkpoint every N epochs")

    # ── training hyperparameters ────────────────────────────────────────────
    g = p.add_argument_group("training")
    g.add_argument("--epochs",       type=int,   default=100)
    g.add_argument("--batch-size",   type=int,   default=8)
    g.add_argument("--lr",           type=float, default=1e-4)
    g.add_argument("--min-lr",       type=float, default=1e-6)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--grad-accum",   type=int,   default=1,
                   help="gradient accumulation steps")
    g.add_argument("--clip-grad",    type=float, default=1.0,
                   help="max gradient norm for clipping")

    # ── VAE-specific ────────────────────────────────────────────────────────
    g = p.add_argument_group("vae")
    g.add_argument("--kl-weight",       type=float, default=1e-6,
                   help="final KL divergence weight β_end")
    g.add_argument("--kl-weight-start", type=float, default=1e-8,
                   help="initial KL divergence weight β_start (prevents KL collapse)")
    g.add_argument("--kl-anneal-frac",  type=float, default=0.2,
                   help="fraction of epochs over which β is linearly annealed from β_start to β_end")

    # ── misc ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("misc")
    g.add_argument("--seed",      type=int, default=42)
    g.add_argument("--device",    default=None,
                   help="cuda / cpu (auto-detected if omitted)")
    g.add_argument("--log-every", type=int, default=50,
                   help="print log every N optimizer steps")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    if args.mode == "vae":
        _pretrain_vae(args)
    elif args.mode == "wave":
        _pretrain_wave(args)
    else:   # both: vae first, then wave
        _pretrain_vae(args)
        _pretrain_wave(args)


if __name__ == "__main__":
    main()

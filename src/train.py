"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
RhythmDiT training script (plain PyTorch, no framework dependency).

Training pipeline
~~~~~~~~~~~~~~~~~
  1. Sample a batch (note, mel, valid_flag) from PhigrosDataset
  2. Frozen ChartVAE encoder:    z₀ = VAE.encode(note).sample()  → (B, 16, T_z)
  3. Frozen AudioWaveEncoder:    audio_c = wave(mel)              → (B, 256, T_a)
  4. Temporal alignment:         audio_c → (B, 256, T_z)  via interpolate
  5. CFG Dropout (prob cfg_drop): randomly zero out audio_c for a subset of the batch
  6. Forward diffusion:          z_t, ε = schedule.q_sample(z₀, t)
  7. DiT noise prediction:       ε̂ = dit(z_t, audio_c, t.float())
  8. MSE loss → backward → AdamW + CosineAnnealingLR

Frame-rate alignment
~~~~~~~~~~~~~~~~~~~~~
  With default settings (hop_length=512, sr=22050, frame_ms≈23.22) the mel and
  chart time axes have the same length (1 mel frame ≈ 1 chart frame), so after
  stride-16 downsampling both VAE and Wave encoder produce T ≈ max_frame // 16 = 256.
  If different hop_length / frame_ms values are used, train.py automatically
  aligns the time axis with F.interpolate.

Usage
~~~~~
  uv run python src/train.py \\
      --data-list  data/data.txt \\
      --ckpt-dir   checkpoints/  \\
      --epochs     200 \\
      --batch-size 8 \\
      --lr         1e-4 \\
      --cfg-drop   0.1
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.condition.wave import AudioWaveEncoder, DEFAULT_WAVE_CONFIG
from src.data.dataset import PhigrosDataset
from src.diffusion.schedule import DEFAULT_SCHEDULE_CONFIG, NoiseSchedule
from src.encoder.encoder import ChartVAE, DEFAULT_VAE_CONFIG
from src.models.dit import DEFAULT_DIT_CONFIG, RhythmDiT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align_time(audio_c: torch.Tensor, T_target: int) -> torch.Tensor:
    """Linearly interpolate audio_c (B, C, T_a) along the time axis to T_target.
    Returns the tensor unchanged when the length already matches."""
    if audio_c.shape[-1] == T_target:
        return audio_c
    return F.interpolate(audio_c, size=T_target, mode="linear", align_corners=False)


def _pad_or_trim_mel(mel: torch.Tensor, target: int) -> torch.Tensor:
    """Zero-pad or right-trim a single mel (n_mels, T) to exactly `target` frames."""
    T = mel.shape[-1]
    if T >= target:
        return mel[..., :target]
    return F.pad(mel, (0, target - T))


def _collate(batch: list[dict], max_mel_frames: int) -> dict:
    """DataLoader collate function: align mel time axes and stack all tensors."""
    notes  = torch.stack([s["note"]       for s in batch])
    flags  = torch.stack([s["valid_flag"] for s in batch])
    mels   = torch.stack([_pad_or_trim_mel(s["audio"], max_mel_frames) for s in batch])
    return {"note": notes, "audio": mels, "valid_flag": flags}


def _save_ckpt(
    path: Path,
    dit:  RhythmDiT,
    opt:  torch.optim.Optimizer,
    epoch: int,
    step:  int,
) -> None:
    torch.save(
        {"epoch": epoch, "step": step,
         "dit": dit.state_dict(), "optimizer": opt.state_dict()},
        path,
    )
    print(f"  [ckpt] saved → {path}")


def _load_ckpt(
    path: Path,
    dit:  RhythmDiT,
    opt:  Optional[torch.optim.Optimizer] = None,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    dit.load_state_dict(ckpt["dit"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train RhythmDiT (plain PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("data")
    g.add_argument("--data-list",   required=True,          help="path to data_list.txt")
    g.add_argument("--cache-dir",   default=None,           help="directory for cached mel .npy files")
    g.add_argument("--frame-ms",    type=float, default=512 / 22050 / 4 * 8 * 1000,
                   help="milliseconds per chart frame (hop/sr/4*8*1000 ≈ 46.44 ms)")
    g.add_argument("--max-frame",   type=int,   default=4096, help="maximum number of chart frames")
    g.add_argument("--hop-length",  type=int,   default=512,  help="mel STFT hop length")
    g.add_argument("--n-mels",      type=int,   default=128,  help="number of mel filter banks")
    g.add_argument("--sr",          type=int,   default=22050, help="audio sample rate")
    g.add_argument("--num-workers", type=int,   default=4)

    # ── checkpoint ─────────────────────────────────────────────────────────
    g = p.add_argument_group("checkpoint")
    g.add_argument("--ckpt-dir",   default="checkpoints",   help="root checkpoint directory (subdir dit/ is created automatically)")
    g.add_argument("--vae-ckpt",   default=None,            help="path to pretrained VAE .pt")
    g.add_argument("--wave-ckpt",  default=None,            help="path to pretrained Wave encoder .pt")
    g.add_argument("--dit-ckpt",   default=None,            help="path to DiT checkpoint to resume from")
    g.add_argument("--save-every", type=int,   default=10,  help="save checkpoint every N epochs")

    # ── training hyperparameters ────────────────────────────────────────────
    g = p.add_argument_group("training")
    g.add_argument("--epochs",     type=int,   default=200)
    g.add_argument("--batch-size", type=int,   default=8)
    g.add_argument("--lr",         type=float, default=1e-4,  help="AdamW initial learning rate")
    g.add_argument("--min-lr",     type=float, default=1e-6,  help="cosine annealing minimum lr")
    g.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    g.add_argument("--cfg-drop",   type=float, default=0.1,   help="CFG unconditional dropout probability")
    g.add_argument("--grad-accum", type=int,   default=1,     help="gradient accumulation steps")
    g.add_argument("--clip-grad",  type=float, default=1.0,   help="max gradient norm for clipping")

    # ── misc ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("misc")
    g.add_argument("--seed",       type=int,   default=42)
    g.add_argument("--device",     default=None, help="cuda / cpu (auto-detected if omitted)")
    g.add_argument("--log-every",  type=int,   default=50,  help="print log every N optimizer steps")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    # ── device & random seed ───────────────────────────────────────────────
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    print(f"[train] device={device}  seed={args.seed}")

    # ── output directory ───────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir) / "dit"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    # mel padding target equals max_frame (hop_length/sr gives 1 mel ≈ 1 chart frame)
    max_mel_frames = args.max_frame

    dataset = PhigrosDataset(
        data_list_path   = args.data_list,
        convertor_params = {"frame_ms": args.frame_ms, "max_frame": args.max_frame},
        cache_dir        = args.cache_dir,
        augment          = True,
        hop_length       = args.hop_length,
        n_mels           = args.n_mels,
        sr               = args.sr,
        device           = str(device),
    )
    print(f"[train] dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = partial(_collate, max_mel_frames=max_mel_frames),
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )

    # ── model initialisation ───────────────────────────────────────────────
    vae  = ChartVAE(DEFAULT_VAE_CONFIG).to(device)
    wave = AudioWaveEncoder(**DEFAULT_WAVE_CONFIG).to(device)
    dit  = RhythmDiT(**DEFAULT_DIT_CONFIG).to(device)

    # load optional pretrained weights
    if args.vae_ckpt:
        vae.load_state_dict(
            torch.load(args.vae_ckpt, map_location=device, weights_only=True)
        )
        print(f"[train] loaded VAE  ← {args.vae_ckpt}")
    if args.wave_ckpt:
        wave.load_state_dict(
            torch.load(args.wave_ckpt, map_location=device, weights_only=True)
        )
        print(f"[train] loaded Wave ← {args.wave_ckpt}")

    # freeze VAE and Wave encoder — only DiT is trained
    vae.eval()
    wave.eval()
    for param in (*vae.parameters(), *wave.parameters()):
        param.requires_grad_(False)

    dit.train()
    print(f"[train] RhythmDiT  params: {dit.num_params:,}")
    print(f"[train] ChartVAE   params: {sum(p.numel() for p in vae.parameters()):,}  (frozen)")
    print(f"[train] WaveEncoder params: {sum(p.numel() for p in wave.parameters()):,}  (frozen)")

    # ── noise schedule ─────────────────────────────────────────────────────
    schedule = NoiseSchedule(**DEFAULT_SCHEDULE_CONFIG).to(device)

    # ── optimizer & lr scheduler ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )
    total_opt_steps = args.epochs * max(1, len(loader) // args.grad_accum)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = total_opt_steps,
        eta_min = args.min_lr,
    )

    # ── resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.dit_ckpt:
        start_epoch, global_step = _load_ckpt(Path(args.dit_ckpt), dit, optimizer)
        print(f"[train] resumed  epoch={start_epoch}  step={global_step}")

    # ── training loop ──────────────────────────────────────────────────────
    T_diffusion = schedule.T
    loss_sum    = 0.0
    loss_count  = 0
    t0          = time.time()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(loader):

            note = batch["note"].to(device)          # (B, 20, max_frame)
            mel  = batch["audio"].to(device)         # (B, n_mels, max_mel_frames)

            # ── encode (VAE + Wave, no grad) ──────────────────────────
            with torch.no_grad():
                z0      = vae.encode(note).sample()  # (B, 16, T_z)
                audio_c = wave(mel)                  # (B, 256, T_a)
                audio_c = _align_time(audio_c, z0.shape[-1])  # → (B, 256, T_z)

            # ── CFG Dropout ───────────────────────────────────────────
            if args.cfg_drop > 0.0:
                drop_mask          = torch.rand(z0.shape[0], device=device) < args.cfg_drop
                audio_c            = audio_c.clone()
                audio_c[drop_mask] = 0.0

            # ── sample diffusion timestep t and add noise ─────────────
            B = z0.shape[0]
            t = torch.randint(0, T_diffusion, (B,), device=device)
            z_t, noise = schedule.q_sample(z0, t)

            # ── DiT noise prediction ──────────────────────────────────
            eps_pred = dit(z_t, audio_c, t.float())

            # ── MSE loss (predict clean noise ε) ─────────────────────
            loss = F.mse_loss(eps_pred, noise)
            (loss / args.grad_accum).backward()

            loss_sum   += loss.item()
            loss_count += 1

            # ── gradient accumulation: update parameters ──────────────
            if (batch_idx + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(dit.parameters(), args.clip_grad)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # logging
                if global_step % args.log_every == 0:
                    avg_loss = loss_sum / loss_count
                    lr_now   = optimizer.param_groups[0]["lr"]
                    elapsed  = time.time() - t0
                    print(
                        f"epoch {epoch + 1:>4d}/{args.epochs}  "
                        f"step {global_step:>7d}  "
                        f"loss {avg_loss:.5f}  "
                        f"lr {lr_now:.2e}  "
                        f"t {elapsed:.0f}s"
                    )
                    loss_sum   = 0.0
                    loss_count = 0

        # ── save checkpoint every save_every epochs ───────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"dit_e{epoch + 1:04d}_s{global_step:07d}.pt"
            _save_ckpt(ckpt_path, dit, optimizer, epoch + 1, global_step)

    # ── training complete: save final weights ──────────────────────────────
    final_path = ckpt_dir / "dit_final.pt"
    _save_ckpt(final_path, dit, optimizer, args.epochs, global_step)
    print(f"[train] done.  total_steps={global_step}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
RhythmDiT training script (plain PyTorch, no framework dependency).

Training pipeline
~~~~~~~~~~~~~~~~~
  1. Sample a batch (note, mel, valid_flag) from PhigrosDataset
  2. Frozen ChartVAE encoder:    z₀ = VAE.encode(note).sample()  → (B, 16, T_z)
     (the VAE checkpoint carries a calibrated latent_scale buffer, so z₀ is
     approximately unit-variance — required for a well-calibrated schedule)
  3. Frozen AudioWaveEncoder:    audio_c = wave(mel)              → (B, 256, T_a)
  4. Temporal alignment:         audio_c → (B, 256, T_z)  via interpolate
  5. CFG Dropout (prob cfg_drop): randomly zero out audio_c for a subset of the batch
  6. Forward diffusion:          z_t, ε = schedule.q_sample(z₀, t)
  7. DiT noise prediction:       ε̂ = dit(z_t, audio_c, t.float())
  8. Masked, min-SNR-weighted MSE → backward → AdamW + CosineAnnealingLR
     - valid_flag (downsampled to the latent grid) masks the padded tail so
       ~27% of the gradient is not spent denoising encoded emptiness
     - min-SNR-γ weighting (Hang et al. 2023) balances timestep difficulty
  9. EMA shadow weights updated after every optimizer step; validation and
     best-checkpoint selection run on the EMA weights

Validation / checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~~
  --val-list enables a deterministic per-epoch validation loss: fixed
  stratified timesteps × fixed-seed noise on held-out songs, evaluated with
  the EMA weights.  dit_best.pt keeps the best-on-val EMA weights;
  intermediate checkpoints are rotated (newest 2).  Metrics go to
  <ckpt_dir>/dit/metrics.jsonl (plot with script/plot_metrics.py).

Frame-rate alignment
~~~~~~~~~~~~~~~~~~~~~
  One chart frame (frame_ms ≈ 46.44 ms) spans TWO mel frames (hop/sr ≈ 23.22 ms),
  so the mel batch is padded/trimmed to max_frame × frame_ms / (hop/sr) ≈ 8192
  frames — the same *duration* as the 4096-frame chart axis.  After stride-16
  downsampling the wave encoder yields T_a = 512 audio tokens, which
  _align_time() linearly interpolates onto the VAE's T_z = 256 latent grid, so
  chart token i and audio token i cover the same span of time.

Usage
~~~~~
  uv run python src/train.py \\
      --data-list  data/train.txt \\
      --val-list   data/val.txt   \\
      --vae-ckpt   checkpoints/vae/vae_final.pt   \\
      --wave-ckpt  checkpoints/wave/wave_final.pt \\
      --ckpt-dir   checkpoints/  \\
      --epochs     1000 \\
      --batch-size 8 \\
      --lr         1e-4 \\
      --cfg-drop   0.1

  --vae-ckpt is required (the VAE is frozen during DiT training); pre-train it
  with src/pre_train.py --mode vae.  --wave-ckpt is optional but strongly
  recommended — see the warning printed when it is omitted.
  DiT capacity is configurable: --dit-depth 8 --dit-hidden-dim 256
  --dit-dropout 0.1 gives a ~20M model better matched to 300 songs.
"""

from __future__ import annotations

import argparse
import math
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.condition.wave import AudioWaveEncoder, DEFAULT_WAVE_CONFIG
from src.data.audio2mel import chart_frames_to_mel_frames
from src.data.dataset import PhigrosDataset
from src.diffusion.schedule import DEFAULT_SCHEDULE_CONFIG, NoiseSchedule
from src.encoder.encoder import ChartVAE, DEFAULT_VAE_CONFIG
from src.models.dit import DEFAULT_DIT_CONFIG, RhythmDiT
from src.train_utils import (
    EMA,
    EarlyStopper,
    MetricLogger,
    pad_or_trim,
    rotate_checkpoints,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align_time(audio_c: torch.Tensor, T_target: int) -> torch.Tensor:
    """Linearly interpolate audio_c (B, C, T_a) along the time axis to T_target.
    Returns the tensor unchanged when the length already matches."""
    if audio_c.shape[-1] == T_target:
        return audio_c
    return F.interpolate(audio_c, size=T_target, mode="linear", align_corners=False)


def _collate(batch: list[dict], max_mel_frames: int) -> dict:
    """DataLoader collate: pad mel with its dB floor, stack all tensors."""
    notes  = torch.stack([s["note"]       for s in batch])
    flags  = torch.stack([s["valid_flag"] for s in batch])
    mels   = torch.stack([pad_or_trim(s["audio"], max_mel_frames) for s in batch])
    return {"note": notes, "audio": mels, "valid_flag": flags}


def _latent_valid_mask(valid_flag: torch.Tensor, T_z: int) -> torch.Tensor:
    """Downsample a frame-level valid_flag (B, T) to the latent grid (B, 1, T_z).

    A latent token is valid if ANY of the chart frames it covers is valid
    (max-pooling), so boundary tokens keep full gradient.
    """
    return F.adaptive_max_pool1d(valid_flag.unsqueeze(1), T_z)


def _diffusion_loss(
    eps_pred:   torch.Tensor,   # (B, C, T_z)
    noise:      torch.Tensor,   # (B, C, T_z)
    valid_flag: torch.Tensor,   # (B, T_frames)
    snr:        torch.Tensor,   # (B,)
    gamma:      float,
) -> torch.Tensor:
    """Masked, min-SNR-weighted epsilon-MSE.

    - The padded tail beyond each song's last note is excluded via valid_flag
      (downsampled to the latent grid) — otherwise ~27% of the gradient is
      spent denoising VAE-encoded emptiness.
    - min-SNR-γ (Hang et al. 2023): per-sample weight min(SNR, γ)/SNR for
      ε-prediction, damping the over-weighted low-noise timesteps.  γ ≤ 0
      disables the weighting.
    """
    mask = _latent_valid_mask(valid_flag, eps_pred.shape[-1])       # (B, 1, T_z)
    err  = (eps_pred - noise) ** 2                                   # (B, C, T_z)
    per_sample = (err * mask).sum(dim=(1, 2)) / (
        mask.sum(dim=(1, 2)) * eps_pred.shape[1]
    ).clamp_min(1.0)                                                 # (B,)

    if gamma > 0:
        # min(SNR, γ)/SNR — written as where() so that SNR → 0 gives weight
        # exactly 1 (an epsilon guard in the denominator would instead crush
        # the weight at the terminal timestep, where SNR ≈ 2.4e-9).
        weight = torch.where(snr <= gamma, torch.ones_like(snr), gamma / snr)
    else:
        weight = torch.ones_like(snr)
    return (weight * per_sample).mean()


@torch.no_grad()
def _validate(
    dit:        RhythmDiT,
    vae:        ChartVAE,
    wave:       AudioWaveEncoder,
    schedule:   NoiseSchedule,
    val_loader: DataLoader,
    device:     torch.device,
    n_timesteps: int = 8,
) -> float:
    """Deterministic validation: fixed stratified timesteps × fixed-seed noise.

    Uses posterior.mode() (not sample) and a fresh fixed-seed generator each
    call, so the metric is exactly comparable across epochs.  Returns plain
    masked ε-MSE (unweighted — a monitoring metric, not the training loss).
    Call with EMA weights swapped in.
    """
    was_training = dit.training
    dit.eval()
    g = torch.Generator(device="cpu").manual_seed(12345)
    T = schedule.T
    t_values = [int((k + 0.5) / n_timesteps * T) for k in range(n_timesteps)]

    total, count = 0.0, 0
    for batch in val_loader:
        note = batch["note"].to(device)
        mel  = batch["audio"].to(device)
        flag = batch["valid_flag"].to(device)

        z0      = vae.encode(note).mode()               # deterministic
        audio_c = _align_time(wave(mel), z0.shape[-1])

        B = z0.shape[0]
        for t_val in t_values:
            t     = torch.full((B,), t_val, device=device, dtype=torch.long)
            noise = torch.randn(z0.shape, generator=g).to(device)
            z_t, _   = schedule.q_sample(z0, t, noise)
            eps_pred = dit(z_t, audio_c, t.float())

            mask = _latent_valid_mask(flag, eps_pred.shape[-1])
            err  = ((eps_pred - noise) ** 2 * mask).sum()
            total += (err / (mask.sum() * eps_pred.shape[1]).clamp_min(1.0)).item()
            count += 1

    if was_training:
        dit.train()
    return total / max(count, 1)


def _save_ckpt(
    path:         Path,
    dit:          RhythmDiT,
    ema:          EMA,
    opt:          torch.optim.Optimizer,
    lr_scheduler,
    dit_config:   dict,
    stopper:      EarlyStopper,
    epoch:        int,
    step:         int,
) -> None:
    torch.save(
        {
            "epoch": epoch, "step": step,
            "dit":          dit.state_dict(),
            "ema":          ema.state_dict(),
            "ema_step":     ema.step,
            "optimizer":    opt.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "dit_config":   dit_config,
            # persist best/patience so a resumed run cannot clobber the true
            # best checkpoint or restart the early-stop countdown
            "stopper": {"best": stopper.best, "best_epoch": stopper.best_epoch,
                        "bad_epochs": stopper.bad_epochs},
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


def _load_ckpt(
    path:         Path,
    dit:          RhythmDiT,
    ema:          Optional[EMA] = None,
    opt:          Optional[torch.optim.Optimizer] = None,
    lr_scheduler = None,
    stopper:      Optional[EarlyStopper] = None,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    dit.load_state_dict(ckpt["dit"])
    if ema is not None:
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"],
                                step=ckpt.get("ema_step", ckpt.get("step", 0)))
        else:
            # weights-only checkpoint (dit_best.pt): re-snapshot the EMA from
            # the loaded weights so the shadow does not keep the random init
            ema.load_state_dict(dit.state_dict(), step=ckpt.get("step", 0))
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    elif opt is not None:
        print(
            "[train] NOTE: checkpoint has no optimizer/scheduler state "
            "(weights-only file such as dit_best.pt) — resuming as a warm "
            "start with a fresh optimizer, lr schedule and step counter."
        )
    if lr_scheduler is not None and "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if stopper is not None and "stopper" in ckpt:
        s = ckpt["stopper"]
        stopper.best       = s.get("best", float("inf"))
        stopper.best_epoch = s.get("best_epoch", -1)
        stopper.bad_epochs = s.get("bad_epochs", 0)
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


def _load_frozen(path: str, model: torch.nn.Module, key: str,
                 device: torch.device) -> None:
    """Load a frozen VAE/wave checkpoint, unwrapping resume envelopes.

    Accepts both the plain state_dicts written by pre_train.py finals and the
    ``{"vae"/"wave": state_dict, "optimizer": ...}`` intermediate envelopes.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and key in ckpt:
        ckpt = ckpt[key]
    model.load_state_dict(ckpt)


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
    g.add_argument("--data-list",   required=True,
                   help="training list (use data/train.txt from script/split_datalist.py)")
    g.add_argument("--val-list",    default=None,
                   help="held-out validation list (data/val.txt) — enables val loss, "
                        "best-checkpoint selection and early stopping")
    g.add_argument("--cache-dir",   default=None,           help="directory for cached mel .npy files")
    g.add_argument("--chart-cache-dir", default=None,       help="directory for cached chart .npz files (skips JSON parse on cache hit; DISABLES rate augmentation)")
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
    g.add_argument("--vae-ckpt",   required=True,
                   help="path to pretrained VAE .pt (required: the VAE is frozen "
                        "during DiT training, so training against a randomly "
                        "initialised VAE is meaningless)")
    g.add_argument("--wave-ckpt",  default=None,
                   help="path to pretrained Wave encoder .pt (strongly recommended: "
                        "the wave encoder is frozen, so without a checkpoint the "
                        "audio condition comes from random weights)")
    g.add_argument("--dit-ckpt",   default=None,
                   help="DiT checkpoint to resume from — use an intermediate "
                        "dit_eNNNN_*.pt (full optimizer/EMA/scheduler state); "
                        "dit_best.pt / dit_final.pt work as weights-only warm "
                        "starts (optimizer and lr schedule restart)")
    g.add_argument("--save-every", type=int,   default=10,
                   help="save intermediate checkpoint every N epochs (only the "
                        "newest 2 are kept; dit_best.pt / dit_final.pt are separate)")

    # ── model (DiT capacity) ───────────────────────────────────────────────
    g = p.add_argument_group("model")
    g.add_argument("--dit-depth",      type=int,   default=DEFAULT_DIT_CONFIG["depth"],
                   help="number of DoubleStreamBlocks")
    g.add_argument("--dit-hidden-dim", type=int,   default=DEFAULT_DIT_CONFIG["hidden_dim"],
                   help="transformer width (must be divisible by --dit-heads)")
    g.add_argument("--dit-heads",      type=int,   default=DEFAULT_DIT_CONFIG["num_heads"],
                   help="attention heads")
    g.add_argument("--dit-dropout",    type=float, default=DEFAULT_DIT_CONFIG["dropout"],
                   help="FFN dropout (0.1 recommended for 300-song training)")

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
    g.add_argument("--min-snr-gamma", type=float, default=5.0,
                   help="min-SNR loss weighting γ (0 disables)")
    g.add_argument("--ema-decay",  type=float, default=0.999, help="EMA decay for shadow weights")
    g.add_argument("--patience",   type=int,   default=100,
                   help="early-stop after N epochs without val improvement "
                        "(0 disables; requires --val-list)")
    g.add_argument("--val-timesteps", type=int, default=8,
                   help="number of stratified timesteps per val sample")

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
    # Pad/trim mel to the same *duration* as the chart axis: one chart frame
    # (frame_ms ≈ 46.44 ms) spans two mel frames (hop/sr ≈ 23.22 ms), so
    # 4096 chart frames ≈ 8192 mel frames.  Padding mel to max_frame instead
    # would cover only half the song and desync audio/chart tokens by 2×.
    max_mel_frames = chart_frames_to_mel_frames(
        args.max_frame, args.frame_ms, args.hop_length, args.sr
    )
    print(
        f"[train] time axes: chart {args.max_frame} frames × {args.frame_ms:.2f} ms "
        f"≡ mel {max_mel_frames} frames × {args.hop_length / args.sr * 1000:.2f} ms"
    )

    def make_dataset(list_path: str, augment: bool) -> PhigrosDataset:
        return PhigrosDataset(
            data_list_path   = list_path,
            convertor_params = {"frame_ms": args.frame_ms, "max_frame": args.max_frame},
            cache_dir        = args.cache_dir,
            chart_cache_dir  = args.chart_cache_dir,
            augment          = augment,
            hop_length       = args.hop_length,
            n_mels           = args.n_mels,
            sr               = args.sr,
        )

    dataset = make_dataset(args.data_list, augment=True)
    print(f"[train] train dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = partial(_collate, max_mel_frames=max_mel_frames),
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )

    val_loader = None
    if args.val_list:
        val_set    = make_dataset(args.val_list, augment=False)
        val_loader = DataLoader(
            val_set,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            collate_fn  = partial(_collate, max_mel_frames=max_mel_frames),
            pin_memory  = (device.type == "cuda"),
            drop_last   = False,
        )
        print(f"[train] val dataset:   {len(val_set)} samples")
    else:
        print(
            "[train] WARNING: no --val-list — validation, best-checkpoint "
            "selection and early stopping are DISABLED.  Generate a split "
            "with script/split_datalist.py and pass --val-list data/val.txt."
        )

    # ── model initialisation ───────────────────────────────────────────────
    dit_config = dict(DEFAULT_DIT_CONFIG)
    dit_config.update(
        depth      = args.dit_depth,
        hidden_dim = args.dit_hidden_dim,
        num_heads  = args.dit_heads,
        dropout    = args.dit_dropout,
    )

    vae  = ChartVAE(DEFAULT_VAE_CONFIG).to(device)
    wave = AudioWaveEncoder(**DEFAULT_WAVE_CONFIG).to(device)
    dit  = RhythmDiT(**dit_config).to(device)

    # load pretrained weights (VAE is mandatory — both models stay frozen below)
    _load_frozen(args.vae_ckpt, vae, "vae", device)
    print(f"[train] loaded VAE  ← {args.vae_ckpt}  (latent_scale={vae.scale:.4f})")
    if vae.scale == 1.0:
        print(
            "[train] WARNING: VAE latent_scale is 1.0 — the checkpoint was not "
            "calibrated.  Re-run pre_train.py (it calibrates automatically) or "
            "the noise schedule's SNR will be miscalibrated."
        )
    if args.wave_ckpt:
        _load_frozen(args.wave_ckpt, wave, "wave", device)
        print(f"[train] loaded Wave ← {args.wave_ckpt}")
    else:
        print(
            "[train] " + "!" * 68 + "\n"
            "[train] WARNING: no --wave-ckpt given — the AudioWaveEncoder is\n"
            "[train] FROZEN with RANDOM weights, so the audio condition fed to\n"
            "[train] the DiT is essentially noise.  Pre-train it first:\n"
            "[train]   python src/pre_train.py --mode wave --data-list <list>\n"
            "[train] " + "!" * 68
        )

    # freeze VAE and Wave encoder — only DiT is trained
    vae.eval()
    wave.eval()
    for param in (*vae.parameters(), *wave.parameters()):
        param.requires_grad_(False)

    dit.train()
    print(f"[train] RhythmDiT  params: {dit.num_params:,}  "
          f"[depth={dit_config['depth']} D={dit_config['hidden_dim']} "
          f"H={dit_config['num_heads']} dropout={dit_config['dropout']}]")
    print(f"[train] ChartVAE   params: {sum(p.numel() for p in vae.parameters()):,}  (frozen)")
    print(f"[train] WaveEncoder params: {sum(p.numel() for p in wave.parameters()):,}  (frozen)")

    # ── noise schedule ─────────────────────────────────────────────────────
    schedule = NoiseSchedule(**DEFAULT_SCHEDULE_CONFIG).to(device)

    # ── optimizer, lr scheduler, EMA ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )
    # ceil: the trailing partial accumulation window is flushed at epoch end,
    # so it counts as a real optimizer step for the cosine horizon
    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_opt_steps = args.epochs * steps_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = total_opt_steps,
        eta_min = args.min_lr,
    )
    ema     = EMA(dit, decay=args.ema_decay)
    stopper = EarlyStopper(patience=args.patience)

    # ── resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.dit_ckpt:
        start_epoch, global_step = _load_ckpt(
            Path(args.dit_ckpt), dit, ema, optimizer, lr_scheduler, stopper
        )
        # The loaded scheduler state carries the OLD run's T_max; past it the
        # periodic cosine RISES back toward base lr (verified: 1e-4 → 3.6e-1).
        # Re-anchor the horizon to the current --epochs so "resume with more
        # epochs" extends the schedule instead of exploding it.
        if lr_scheduler.T_max != total_opt_steps:
            print(f"[train] resume: lr schedule horizon {lr_scheduler.T_max} → "
                  f"{total_opt_steps} steps (recomputed from --epochs {args.epochs})")
            lr_scheduler.T_max = total_opt_steps
        lr_scheduler.eta_min = args.min_lr
        print(f"[train] resumed  epoch={start_epoch}  step={global_step}  "
              f"best_val={stopper.best:.5f}@{stopper.best_epoch}")

    metrics = MetricLogger(ckpt_dir / "metrics.jsonl", stage="dit")

    # ── training loop ──────────────────────────────────────────────────────
    T_diffusion = schedule.T
    loss_sum    = 0.0
    loss_count  = 0
    t0          = time.time()
    optimizer.zero_grad()

    def _opt_step() -> None:
        """One optimizer step: clip → step → schedule → zero → EMA."""
        nonlocal global_step
        nn.utils.clip_grad_norm_(dit.parameters(), args.clip_grad)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        ema.update(dit)
        global_step += 1

    final_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        pending_grads = False

        for batch_idx, batch in enumerate(loader):

            note = batch["note"].to(device)          # (B, 20, max_frame)
            mel  = batch["audio"].to(device)         # (B, n_mels, max_mel_frames)
            flag = batch["valid_flag"].to(device)    # (B, max_frame)

            # ── encode (VAE + Wave, no grad) ──────────────────────────
            with torch.no_grad():
                z0      = vae.encode(note).sample()  # (B, 16, T_z), scaled
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

            # ── masked, min-SNR-weighted loss ─────────────────────────
            loss = _diffusion_loss(
                eps_pred, noise, flag,
                snr=schedule.snr(t), gamma=args.min_snr_gamma,
            )
            (loss / args.grad_accum).backward()
            pending_grads = True

            loss_sum   += loss.item()
            loss_count += 1
            epoch_loss_sum   += loss.item()
            epoch_loss_count += 1

            # ── gradient accumulation: update parameters ──────────────
            if (batch_idx + 1) % args.grad_accum == 0:
                _opt_step()
                pending_grads = False

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

        # ── flush the trailing partial accumulation window ────────────
        # Without this, leftover gradients from len(loader) % grad_accum
        # batches leak into the next epoch's first step (and with
        # grad_accum > len(loader) no step would ever run).
        if pending_grads:
            _opt_step()

        final_epoch = epoch + 1

        # ── per-epoch validation (EMA weights) + metric log ───────────
        epoch_train = epoch_loss_sum / max(epoch_loss_count, 1)
        record = {
            "epoch": epoch + 1, "step": global_step,
            "train_loss": epoch_train,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            with ema.swapped_into(dit):
                val_loss = _validate(
                    dit, vae, wave, schedule, val_loader, device,
                    n_timesteps=args.val_timesteps,
                )
            record["val_loss"] = val_loss
            is_best = stopper.update(val_loss, epoch + 1)
            if is_best:
                torch.save(
                    {"dit": ema.state_dict(), "dit_config": dit_config,
                     "epoch": epoch + 1, "val_loss": val_loss},
                    ckpt_dir / "dit_best.pt",
                )
            print(
                f"epoch {epoch + 1:>4d}  train {epoch_train:.5f}  "
                f"val {val_loss:.5f}{'  *best*' if is_best else ''}"
            )
        metrics.log(**record)

        # ── save intermediate checkpoint every save_every epochs ──────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"dit_e{epoch + 1:04d}_s{global_step:07d}.pt"
            _save_ckpt(ckpt_path, dit, ema, optimizer, lr_scheduler,
                       dit_config, stopper, epoch + 1, global_step)
            rotate_checkpoints(ckpt_dir, "dit_e*.pt", keep=2)

        if stopper.should_stop:
            print(f"[train] early stop at epoch {epoch + 1} "
                  f"(best val {stopper.best:.5f} @ epoch {stopper.best_epoch})")
            break

    # ── training complete: save final weights (EMA preferred at inference) ─
    final_path = ckpt_dir / "dit_final.pt"
    torch.save(
        {"dit": dit.state_dict(), "ema": ema.state_dict(), "ema_step": ema.step,
         "dit_config": dit_config, "epoch": final_epoch, "step": global_step},
        final_path,
    )
    print(f"[train] done.  total_steps={global_step}  elapsed={time.time() - t0:.0f}s")
    if val_loader is not None and stopper.best_epoch > 0:
        print(f"[train] best val {stopper.best:.5f} @ epoch {stopper.best_epoch} "
              f"→ use {ckpt_dir / 'dit_best.pt'} for inference")


if __name__ == "__main__":
    main()

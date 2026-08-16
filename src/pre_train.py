"""
src/pre_train.py
─────────────────────────────────────────────────────────────────────────────
Pre-training script for ChartVAE and AudioWaveEncoder.

The VAE must be pre-trained before running train.py (wave encoder is optional
but strongly recommended).

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

Validation / checkpointing (all stages)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - --val-list holds out songs never trained on; a deterministic val loss
    (posterior.mode(), no augmentation) runs every epoch.
  - The best-on-val weights are kept; --patience epochs without improvement
    stops the stage early (0 disables).
  - Intermediate (resumable) checkpoints are rotated: only the newest 2 kept.
  - One JSON line per epoch is appended to <ckpt_dir>/<stage>/metrics.jsonl
    (plot with script/plot_metrics.py).

Latent scale calibration (VAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  After training, the best weights are reloaded and the latent std is
  measured over the training set; ``latent_scale = 1 / std`` is stored as a
  buffer inside vae_final.pt (LDM scale_factor).  train.py / test.py pick it
  up automatically via load_state_dict — no extra flags.

Checkpoint format
~~~~~~~~~~~~~~~~~
  - Intermediate: {"vae"/"wave": state_dict, "optimizer": ..., "lr_scheduler": ...,
                   "epoch": ..., "step": ...}
  - Final vae_final.pt / wave_final.pt: plain state_dict (best-on-val weights;
    vae_final.pt carries the calibrated latent_scale buffer)

Usage
~~~~~
  uv run python src/pre_train.py --mode vae \\
      --data-list data/train.txt --val-list data/val.txt \\
      --ckpt-dir  checkpoints/ --epochs 100

  uv run python src/pre_train.py --mode wave \\
      --data-list data/train.txt --val-list data/val.txt \\
      --ckpt-dir  checkpoints/ --epochs 50
"""

from __future__ import annotations

import argparse
import copy
import math
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

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
from src.data.audio2mel import chart_frames_to_mel_frames
from src.data.dataset import PhigrosDataset
from src.encoder.encoder import (
    ChartReconLoss,
    ChartVAE,
    DEFAULT_VAE_CONFIG,
)
from src.train_utils import (
    EarlyStopper,
    MetricLogger,
    pad_or_trim,
    rotate_checkpoints,
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

def _collate_vae(batch: list[dict], max_frame: int) -> dict:
    """Collate for VAE training: note + valid_flag, zero-padded (0 = empty)."""
    notes = torch.stack([pad_or_trim(s["note"],       max_frame, 0.0) for s in batch])
    flags = torch.stack([pad_or_trim(s["valid_flag"], max_frame, 0.0) for s in batch])
    return {"note": notes, "valid_flag": flags}


def _collate_wave(batch: list[dict], max_mel_frames: int) -> dict:
    """Collate for Wave training: mel only, padded with the dB floor (mel.min())."""
    mels = torch.stack([pad_or_trim(s["audio"], max_mel_frames) for s in batch])
    return {"audio": mels}


def _make_dataset(args: argparse.Namespace, list_path: str, augment: bool) -> PhigrosDataset:
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


def _make_loader(dataset, args, collate_fn, *, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = train,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = train,
    )


def _warn_no_val() -> None:
    warnings.warn(
        "no --val-list given: validation loss, best-checkpoint selection and "
        "early stopping are all DISABLED.  With 300 songs this stage can "
        "overfit silently — generate a split with script/split_datalist.py "
        "and pass --val-list data/val.txt.",
        UserWarning,
        stacklevel=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ChartVAE pre-training
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_vae(
    vae:        ChartVAE,
    loss_fn:    ChartReconLoss,
    val_loader: DataLoader,
    device:     torch.device,
) -> dict:
    """Deterministic val pass: posterior.mode(), no augmentation, no sampling.

    Returns the masked reconstruction loss (the early-stop metric — comparable
    across epochs, unlike the β-annealed total) plus KL for logging.
    """
    vae.eval()
    recon_sum = kl_sum = 0.0
    n = 0
    for batch in val_loader:
        note = batch["note"].to(device)
        flag = batch["valid_flag"].to(device)
        recon, posterior = vae(note, sample=False)   # mode(): deterministic
        recon_loss, _    = loss_fn(note, recon, flag)
        recon_sum += recon_loss.item()
        kl_sum    += posterior.kl().item()
        n += 1
    vae.train()
    n = max(n, 1)
    return {"val_recon": recon_sum / n, "val_kl": kl_sum / n}


@torch.no_grad()
def _calibrate_latent_scale(
    vae:         ChartVAE,
    loader:      DataLoader,
    device:      torch.device,
    max_batches: int = 64,
) -> float:
    """Measure latent std over the (unaugmented) training set and store
    ``latent_scale = 1 / std`` in the VAE (LDM scale_factor).

    Must run with scale reset to 1.0 so the measurement is of the raw latent.
    """
    vae.eval()
    vae.set_scale(1.0)
    total, total_sq, count = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        note      = batch["note"].to(device)
        posterior = vae.encode(note)
        # DiT training consumes posterior.sample(), whose variance is
        # Var(mean) + E[var].  Measure that second moment analytically
        # (mean² + var) instead of sampling — deterministic and exact.
        total    += posterior.mean.sum().item()
        total_sq += (posterior.mean ** 2 + posterior.var).sum().item()
        count    += posterior.mean.numel()
    if count == 0:
        warnings.warn("latent-scale calibration saw no data; keeping scale=1.0",
                      UserWarning)
        return 1.0
    mean = total / count
    var  = max(total_sq / count - mean ** 2, 1e-12)
    std  = var ** 0.5
    scale = 1.0 / std
    vae.set_scale(scale)
    print(f"[vae] latent calibration: mean={mean:+.4f}  std={std:.4f}  "
          f"→ latent_scale={scale:.4f}")
    return scale


def _pretrain_vae(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    print(f"[vae] device={device}  seed={args.seed}")

    ckpt_dir = Path(args.ckpt_dir) / "vae"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & DataLoaders ─────────────────────────────────────────────
    train_set = _make_dataset(args, args.data_list, augment=True)
    print(f"[vae] train: {len(train_set)} samples")
    loader = _make_loader(train_set, args,
                          partial(_collate_vae, max_frame=args.max_frame), train=True)

    val_loader = None
    if args.val_list:
        val_set    = _make_dataset(args, args.val_list, augment=False)
        val_loader = _make_loader(val_set, args,
                                  partial(_collate_vae, max_frame=args.max_frame),
                                  train=False)
        print(f"[vae] val:   {len(val_set)} samples")
    else:
        _warn_no_val()

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
    # ceil: the trailing partial accumulation window is flushed at epoch end
    total_steps = args.epochs * max(1, math.ceil(len(loader) / args.grad_accum))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    metrics = MetricLogger(ckpt_dir / "metrics.jsonl", stage="vae")
    stopper = EarlyStopper(patience=args.patience)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.vae_ckpt:
        raw = torch.load(args.vae_ckpt, map_location=device, weights_only=True)
        if isinstance(raw, dict) and "vae" in raw:
            vae.load_state_dict(raw["vae"])
            if "optimizer" in raw:
                optimizer.load_state_dict(raw["optimizer"])
            if "lr_scheduler" in raw:
                lr_scheduler.load_state_dict(raw["lr_scheduler"])
                # re-anchor the horizon: past the OLD T_max the periodic
                # cosine rises back toward base lr instead of annealing
                if lr_scheduler.T_max != total_steps:
                    print(f"[vae] resume: lr horizon {lr_scheduler.T_max} → {total_steps}")
                    lr_scheduler.T_max = total_steps
                lr_scheduler.eta_min = args.min_lr
            if "stopper" in raw:
                stopper.best       = raw["stopper"].get("best", float("inf"))
                stopper.best_epoch = raw["stopper"].get("best_epoch", -1)
                stopper.bad_epochs = raw["stopper"].get("bad_epochs", 0)
            start_epoch = raw.get("epoch", 0)
            global_step = raw.get("step", 0)
        else:
            vae.load_state_dict(raw)
        print(f"[vae] resumed ← {args.vae_ckpt}  epoch={start_epoch}  step={global_step}")

    # KL weight annealing: linearly ramp from kl_start to kl_end
    kl_start          = args.kl_weight_start
    kl_end            = args.kl_weight
    kl_anneal_epochs  = max(1, int(args.epochs * args.kl_anneal_frac))

    best_state: Optional[dict] = None

    loss_sum   = 0.0
    loss_count = 0
    epoch_loss_sum   = 0.0
    epoch_loss_count = 0
    t0 = time.time()
    optimizer.zero_grad()

    def _opt_step() -> None:
        nonlocal global_step
        nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        vae.train()
        pending_grads = False

        # current β (linear anneal)
        frac          = min(1.0, epoch / kl_anneal_epochs)
        vae.kl_weight = kl_start + frac * (kl_end - kl_start)

        for batch_idx, batch in enumerate(loader):
            note = batch["note"].to(device)         # (B, 20, max_frame)
            flag = batch["valid_flag"].to(device)   # (B, max_frame)

            loss, log = vae.compute_loss(note, flag, loss_fn)
            (loss / args.grad_accum).backward()
            pending_grads = True

            loss_sum   += loss.item()
            loss_count += 1
            epoch_loss_sum   += loss.item()
            epoch_loss_count += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                _opt_step()
                pending_grads = False

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

        # flush the trailing partial accumulation window (grad_accum > 1)
        if pending_grads:
            _opt_step()

        # ── per-epoch validation + metric log ─────────────────────────────
        epoch_train = epoch_loss_sum / max(epoch_loss_count, 1)
        epoch_loss_sum, epoch_loss_count = 0.0, 0

        record = {
            "epoch": epoch + 1, "step": global_step,
            "train_loss": epoch_train,
            "beta": vae.kl_weight,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            val = _evaluate_vae(vae, loss_fn, val_loader, device)
            record.update(val)
            is_best = stopper.update(val["val_recon"], epoch + 1)
            if is_best:
                best_state = copy.deepcopy(vae.state_dict())
                torch.save(best_state, ckpt_dir / "vae_best.pt")
            print(
                f"[vae] epoch {epoch + 1:>4d}  train {epoch_train:.5f}  "
                f"val_recon {val['val_recon']:.5f}"
                f"{'  *best*' if is_best else ''}"
            )
        metrics.log(**record)

        # save intermediate checkpoint every save_every epochs, keep newest 2
        if (epoch + 1) % args.save_every == 0:
            mid_path = ckpt_dir / f"vae_e{epoch + 1:04d}_s{global_step:07d}.pt"
            torch.save(
                {
                    "vae":          vae.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "stopper":      {"best": stopper.best,
                                     "best_epoch": stopper.best_epoch,
                                     "bad_epochs": stopper.bad_epochs},
                    "epoch":        epoch + 1,
                    "step":         global_step,
                },
                mid_path,
            )
            rotate_checkpoints(ckpt_dir, "vae_e*.pt", keep=2)
            print(f"[vae] saved → {mid_path}")

        if stopper.should_stop:
            print(f"[vae] early stop at epoch {epoch + 1} "
                  f"(best val_recon {stopper.best:.5f} @ epoch {stopper.best_epoch})")
            break

    # ── Final: best weights + latent scale calibration ────────────────────
    best_path = ckpt_dir / "vae_best.pt"
    if best_state is not None:
        vae.load_state_dict(best_state)
        print(f"[vae] using best-on-val weights "
              f"(val_recon {stopper.best:.5f} @ epoch {stopper.best_epoch})")
    elif best_path.exists():
        # resumed run that never improved on the previous run's best:
        # the true best lives on disk from before the resume
        vae.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print(f"[vae] using best-on-val weights from disk ← {best_path}")

    calib_set    = _make_dataset(args, args.data_list, augment=False)
    calib_loader = _make_loader(calib_set, args,
                                partial(_collate_vae, max_frame=args.max_frame),
                                train=False)
    _calibrate_latent_scale(vae, calib_loader, device,
                            max_batches=args.calib_batches)

    final_path = ckpt_dir / "vae_final.pt"
    torch.save(vae.state_dict(), final_path)
    print(f"[vae] done → {final_path}  elapsed={time.time() - t0:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# AudioWaveEncoder pre-training (mel autoencoder)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_wave(
    encoder:    AudioWaveEncoder,
    decoder:    WaveDecoder,
    val_loader: DataLoader,
    device:     torch.device,
    stride:     int,
) -> float:
    encoder.eval()
    decoder.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        mel    = batch["audio"].to(device)
        T_trim = (mel.shape[-1] // stride) * stride
        mel    = mel[..., :T_trim]
        mel_hat = decoder(encoder(mel))
        if mel_hat.shape[-1] != mel.shape[-1]:
            mel_hat = mel_hat[..., :mel.shape[-1]]
        total += F.mse_loss(mel_hat, mel).item()
        n += 1
    encoder.train()
    decoder.train()
    return total / max(n, 1)


def _pretrain_wave(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    print(f"[wave] device={device}  seed={args.seed}")

    ckpt_dir = Path(args.ckpt_dir) / "wave"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Same duration as the chart axis — must stay consistent with train.py.
    max_mel_frames = chart_frames_to_mel_frames(
        args.max_frame, args.frame_ms, args.hop_length, args.sr
    )

    # ── Datasets & DataLoaders ─────────────────────────────────────────────
    train_set = _make_dataset(args, args.data_list, augment=False)
    print(f"[wave] train: {len(train_set)} samples")
    loader = _make_loader(train_set, args,
                          partial(_collate_wave, max_mel_frames=max_mel_frames),
                          train=True)

    val_loader = None
    if args.val_list:
        val_set    = _make_dataset(args, args.val_list, augment=False)
        val_loader = _make_loader(val_set, args,
                                  partial(_collate_wave, max_mel_frames=max_mel_frames),
                                  train=False)
        print(f"[wave] val:   {len(val_set)} samples")
    else:
        _warn_no_val()

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
    # ceil: the trailing partial accumulation window is flushed at epoch end
    total_steps = args.epochs * max(1, math.ceil(len(loader) / args.grad_accum))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    metrics = MetricLogger(ckpt_dir / "metrics.jsonl", stage="wave")
    stopper = EarlyStopper(patience=args.patience)

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
            if "lr_scheduler" in raw:
                lr_scheduler.load_state_dict(raw["lr_scheduler"])
                if lr_scheduler.T_max != total_steps:
                    print(f"[wave] resume: lr horizon {lr_scheduler.T_max} → {total_steps}")
                    lr_scheduler.T_max = total_steps
                lr_scheduler.eta_min = args.min_lr
            if "stopper" in raw:
                stopper.best       = raw["stopper"].get("best", float("inf"))
                stopper.best_epoch = raw["stopper"].get("best_epoch", -1)
                stopper.bad_epochs = raw["stopper"].get("bad_epochs", 0)
            start_epoch = raw.get("epoch", 0)
            global_step = raw.get("step", 0)
        else:
            encoder.load_state_dict(raw)
        print(f"[wave] resumed ← {args.wave_ckpt}  epoch={start_epoch}  step={global_step}")

    # encoder stride — used to trim mel length to an exact multiple of stride
    stride = encoder.stride  # = 16

    loss_sum   = 0.0
    loss_count = 0
    epoch_loss_sum   = 0.0
    epoch_loss_count = 0
    t0 = time.time()
    optimizer.zero_grad()

    def _opt_step() -> None:
        nonlocal global_step
        nn.utils.clip_grad_norm_(all_params, args.clip_grad)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        decoder.train()
        pending_grads = False

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
            pending_grads = True

            loss_sum   += loss.item()
            loss_count += 1
            epoch_loss_sum   += loss.item()
            epoch_loss_count += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                _opt_step()
                pending_grads = False

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

        # flush the trailing partial accumulation window (grad_accum > 1)
        if pending_grads:
            _opt_step()

        # ── per-epoch validation + metric log ─────────────────────────────
        epoch_train = epoch_loss_sum / max(epoch_loss_count, 1)
        epoch_loss_sum, epoch_loss_count = 0.0, 0

        record = {
            "epoch": epoch + 1, "step": global_step,
            "train_loss": epoch_train,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            val_mse = _evaluate_wave(encoder, decoder, val_loader, device, stride)
            record["val_loss"] = val_mse
            is_best = stopper.update(val_mse, epoch + 1)
            if is_best:
                torch.save(encoder.state_dict(), ckpt_dir / "wave_best.pt")
            print(
                f"[wave] epoch {epoch + 1:>4d}  train {epoch_train:.5f}  "
                f"val {val_mse:.5f}{'  *best*' if is_best else ''}"
            )
        metrics.log(**record)

        # save intermediate checkpoint every save_every epochs, keep newest 2
        if (epoch + 1) % args.save_every == 0:
            mid_path = ckpt_dir / f"wave_e{epoch + 1:04d}_s{global_step:07d}.pt"
            torch.save(
                {
                    "wave":         encoder.state_dict(),
                    "decoder":      decoder.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "stopper":      {"best": stopper.best,
                                     "best_epoch": stopper.best_epoch,
                                     "bad_epochs": stopper.bad_epochs},
                    "epoch":        epoch + 1,
                    "step":         global_step,
                },
                mid_path,
            )
            rotate_checkpoints(ckpt_dir, "wave_e*.pt", keep=2)
            print(f"[wave] saved → {mid_path}")

        if stopper.should_stop:
            print(f"[wave] early stop at epoch {epoch + 1} "
                  f"(best val {stopper.best:.5f} @ epoch {stopper.best_epoch})")
            break

    # ── Final: best-on-val encoder weights (decoder is discarded) ─────────
    best_path = ckpt_dir / "wave_best.pt"
    final_path = ckpt_dir / "wave_final.pt"
    if val_loader is not None and best_path.exists():
        final_state = torch.load(best_path, map_location=device, weights_only=True)
        print(f"[wave] using best-on-val weights "
              f"(val {stopper.best:.5f} @ epoch {stopper.best_epoch})")
    else:
        final_state = encoder.state_dict()
    torch.save(final_state, final_path)
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
                   help="training list (use data/train.txt from script/split_datalist.py)")
    g.add_argument("--val-list",    default=None,
                   help="held-out validation list (data/val.txt) — enables val loss, "
                        "best-checkpoint selection and early stopping")
    g.add_argument("--cache-dir",   default=None,
                   help="directory for cached mel .npy files (optional)")
    g.add_argument("--chart-cache-dir", default=None,
                   help="directory for cached chart .npz files (skips JSON parse on "
                        "cache hit; DISABLES rate augmentation)")
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
                   help="save intermediate checkpoint every N epochs (only the "
                        "newest 2 are kept; best/final are separate files)")

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
    g.add_argument("--patience",     type=int,   default=25,
                   help="early-stop after N epochs without val improvement "
                        "(0 disables; requires --val-list)")

    # ── VAE-specific ────────────────────────────────────────────────────────
    g = p.add_argument_group("vae")
    g.add_argument("--kl-weight",       type=float, default=1e-6,
                   help="final KL divergence weight β_end")
    g.add_argument("--kl-weight-start", type=float, default=1e-8,
                   help="initial KL divergence weight β_start (prevents KL collapse)")
    g.add_argument("--kl-anneal-frac",  type=float, default=0.2,
                   help="fraction of epochs over which β is linearly annealed from β_start to β_end")
    g.add_argument("--calib-batches",   type=int,   default=64,
                   help="number of batches used for latent-scale calibration after VAE training")

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

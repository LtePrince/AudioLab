"""
src/train_transcription.py
──────────────────────────
Train the transcription baseline (audio → per-frame note labels).

Compared with the diffusion pipeline this is plain supervised learning:
one forward pass, per-frame BCE/CE losses, and the validation metric is the
REAL target — lane-aware onset F1 on full-length held-out songs — computed
every epoch with a single forward (no sampling needed).  Early stopping and
best-checkpoint selection run on val F1 (EMA weights).

Training uses random crops (default 1024 chart frames ≈ 47.6 s): the model
is length-agnostic (conv + RoPE), so cropping is pure augmentation — every
epoch sees each song from a different window — and full-length inference
needs no windowing.

Usage
~~~~~
  uv run python src/train_transcription.py \\
      --data-list data/train.txt --val-list data/val.txt \\
      --cache-dir data/cache_mel --ckpt-dir checkpoints/ \\
      --epochs 500
"""

from __future__ import annotations

import argparse
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.audio2mel import chart_frames_to_mel_frames
from src.data.dataset import PhigrosDataset
from src.models.transcriber import (
    DEFAULT_TRANSCRIBER_CONFIG,
    TranscriberNet,
    TranscriptionLoss,
    onset_f1,
)
from src.train_utils import (
    EMA,
    EarlyStopper,
    MetricLogger,
    pad_or_trim,
    rotate_checkpoints,
)


# ─────────────────────────────────────────────────────────────────────────────
# Collates
# ─────────────────────────────────────────────────────────────────────────────

def _collate_crop(batch: list[dict], crop: int, mel_per_frame: int) -> dict:
    """Random aligned crop of (mel, note, valid_flag).

    The crop start is drawn inside the song's valid content where possible,
    so batches are almost padding-free; short songs are floor-padded (mel)
    / zero-padded (labels) up to the crop length.
    """
    notes, flags, mels = [], [], []
    for s in batch:
        note, flag, mel = s["note"], s["valid_flag"], s["audio"]
        content   = int(flag.sum().item())
        mel_chart = mel.shape[-1] // mel_per_frame           # chart frames covered by mel
        usable    = min(content, mel_chart, note.shape[-1])
        start     = random.randint(0, max(usable - crop, 0))
        notes.append(pad_or_trim(note[:, start : start + crop], crop, 0.0))
        flags.append(pad_or_trim(flag[start : start + crop],    crop, 0.0))
        m = mel[:, start * mel_per_frame : (start + crop) * mel_per_frame]
        mels.append(pad_or_trim(m, crop * mel_per_frame))    # dB-floor padding
    return {
        "note":       torch.stack(notes),
        "valid_flag": torch.stack(flags),
        "audio":      torch.stack(mels),
    }


def _collate_full(batch: list[dict], max_frame: int, mel_per_frame: int) -> dict:
    """Full-length collate for validation (padded to max_frame)."""
    notes = torch.stack([pad_or_trim(s["note"],       max_frame, 0.0) for s in batch])
    flags = torch.stack([pad_or_trim(s["valid_flag"], max_frame, 0.0) for s in batch])
    mels  = torch.stack([pad_or_trim(s["audio"], max_frame * mel_per_frame)
                         for s in batch])
    return {"note": notes, "valid_flag": flags, "audio": mels}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    model:      TranscriberNet,
    loss_fn:    TranscriptionLoss,
    val_loader: DataLoader,
    device:     torch.device,
) -> tuple[float, float, float, float]:
    """Full-length deterministic validation.

    Returns (val_loss, f1, precision, recall) — F1 is the early-stop metric.
    """
    was_training = model.training
    model.eval()
    loss_sum, n = 0.0, 0
    tp_agg = []
    for batch in val_loader:
        note = batch["note"].to(device)
        flag = batch["valid_flag"].to(device)
        mel  = batch["audio"].to(device)
        pred = model(mel)
        loss, _ = loss_fn(pred, note, flag)
        loss_sum += loss.item(); n += 1
        tp_agg.append((pred.cpu(), note.cpu(), flag.cpu()))
    # aggregate F1 over the whole val set (not per-batch averaging)
    preds = torch.cat([a for a, _, _ in tp_agg])
    notes = torch.cat([b for _, b, _ in tp_agg])
    flags = torch.cat([c for _, _, c in tp_agg])
    f1, prec, rec = onset_f1(preds, notes, flags)
    if was_training:
        model.train()
    return loss_sum / max(n, 1), f1, prec, rec


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the transcription baseline (plain supervised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("data")
    g.add_argument("--data-list",   required=True)
    g.add_argument("--val-list",    default=None,
                   help="held-out list — enables val F1, best ckpt, early stop")
    g.add_argument("--cache-dir",   default=None)
    g.add_argument("--chart-cache-dir", default=None)
    g.add_argument("--frame-ms",    type=float, default=512 / 22050 / 4 * 8 * 1000)
    g.add_argument("--max-frame",   type=int,   default=4096)
    g.add_argument("--hop-length",  type=int,   default=512)
    g.add_argument("--n-mels",      type=int,   default=128)
    g.add_argument("--sr",          type=int,   default=22050)
    g.add_argument("--num-workers", type=int,   default=4)
    g.add_argument("--crop-frames", type=int,   default=1024,
                   help="random crop length in chart frames (≈47.6 s)")

    g = p.add_argument_group("checkpoint")
    g.add_argument("--ckpt-dir",   default="checkpoints")
    g.add_argument("--resume",     default=None, help="checkpoint to resume from")
    g.add_argument("--save-every", type=int, default=20)

    g = p.add_argument_group("model")
    g.add_argument("--depth",      type=int,   default=DEFAULT_TRANSCRIBER_CONFIG["depth"])
    g.add_argument("--hidden-dim", type=int,   default=DEFAULT_TRANSCRIBER_CONFIG["hidden_dim"])
    g.add_argument("--conv-blocks", type=int,  default=DEFAULT_TRANSCRIBER_CONFIG["conv_blocks"])
    g.add_argument("--model-dropout", type=float, default=DEFAULT_TRANSCRIBER_CONFIG["dropout"])

    g = p.add_argument_group("training")
    g.add_argument("--epochs",     type=int,   default=500)
    g.add_argument("--batch-size", type=int,   default=16)
    g.add_argument("--lr",         type=float, default=2e-4)
    g.add_argument("--min-lr",     type=float, default=1e-6)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--grad-accum", type=int,   default=1)
    g.add_argument("--clip-grad",  type=float, default=1.0)
    g.add_argument("--onset-pos-weight", type=float, default=3.0)
    g.add_argument("--holding-pos-weight", type=float, default=5.0,
                   help="BCE pos_weight for the sparse is_holding channel")
    g.add_argument("--type-weights", default="1.7,4.3,8.3,15.4",
                   help="CE class weights Tap,Drag,Hold,Flick — inverse train "
                        "marginals by default; 'none' disables")
    g.add_argument("--ema-decay",  type=float, default=0.999)
    g.add_argument("--patience",   type=int,   default=50,
                   help="early-stop after N epochs without val-F1 improvement")

    g = p.add_argument_group("misc")
    g.add_argument("--seed",      type=int, default=42)
    g.add_argument("--device",    default=None)
    g.add_argument("--log-every", type=int, default=50)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print(f"[tsc] device={device}  seed={args.seed}")

    ckpt_dir = Path(args.ckpt_dir) / "transcriber"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mel_per_frame = round(args.frame_ms / (args.hop_length / args.sr * 1000))
    assert mel_per_frame >= 1

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

    train_set = make_dataset(args.data_list, augment=True)
    print(f"[tsc] train: {len(train_set)} songs  (crop {args.crop_frames} frames)")
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial(_collate_crop, crop=args.crop_frames,
                           mel_per_frame=mel_per_frame),
        pin_memory=(device.type == "cuda"), drop_last=True,
    )

    val_loader = None
    if args.val_list:
        val_set = make_dataset(args.val_list, augment=False)
        val_loader = DataLoader(
            val_set, batch_size=4, shuffle=False, num_workers=args.num_workers,
            collate_fn=partial(_collate_full, max_frame=args.max_frame,
                               mel_per_frame=mel_per_frame),
            pin_memory=(device.type == "cuda"), drop_last=False,
        )
        print(f"[tsc] val:   {len(val_set)} songs (full length)")
    else:
        print("[tsc] WARNING: no --val-list — F1 tracking, best ckpt and early "
              "stopping DISABLED")

    config = dict(DEFAULT_TRANSCRIBER_CONFIG)
    config.update(
        n_mels=args.n_mels, hidden_dim=args.hidden_dim, depth=args.depth,
        conv_blocks=args.conv_blocks, dropout=args.model_dropout,
        mel_per_frame=mel_per_frame,
    )
    model = TranscriberNet(**config).to(device)
    model.train()
    type_w = (None if args.type_weights.strip().lower() == "none"
              else tuple(float(x) for x in args.type_weights.split(",")))
    loss_fn = TranscriptionLoss(
        onset_pos_weight   = args.onset_pos_weight,
        holding_pos_weight = args.holding_pos_weight,
        type_class_weights = type_w,
    ).to(device)
    print(f"[tsc] TranscriberNet params: {model.num_params:,}  "
          f"[conv={config['conv_blocks']} depth={config['depth']} "
          f"D={config['hidden_dim']} dropout={config['dropout']}]")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr,
    )
    ema     = EMA(model, decay=args.ema_decay)
    stopper = EarlyStopper(patience=args.patience)   # minimises 1 - F1

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"], step=ckpt.get("ema_step", 0))
        else:
            ema.load_state_dict(model.state_dict(), step=0)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if lr_scheduler.T_max != total_steps:
                print(f"[tsc] resume: lr horizon {lr_scheduler.T_max} → {total_steps}")
                lr_scheduler.T_max = total_steps
            lr_scheduler.eta_min = args.min_lr
        if "stopper" in ckpt:
            s = ckpt["stopper"]
            stopper.best, stopper.best_epoch = s["best"], s["best_epoch"]
            stopper.bad_epochs = s["bad_epochs"]
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[tsc] resumed ← {args.resume}  epoch={start_epoch}  step={global_step}")

    metrics = MetricLogger(ckpt_dir / "metrics.jsonl", stage="transcriber")

    def _opt_step() -> None:
        nonlocal global_step
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        ema.update(model)
        global_step += 1

    loss_sum, loss_count = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad()
    final_epoch = start_epoch

    for epoch in range(start_epoch, args.epochs):
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        pending_grads = False

        for batch_idx, batch in enumerate(loader):
            note = batch["note"].to(device)
            flag = batch["valid_flag"].to(device)
            mel  = batch["audio"].to(device)

            pred = model(mel)
            loss, log = loss_fn(pred, note, flag)
            (loss / args.grad_accum).backward()
            pending_grads = True

            loss_sum += loss.item(); loss_count += 1
            epoch_loss_sum += loss.item(); epoch_loss_count += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                _opt_step()
                pending_grads = False
                if global_step % args.log_every == 0:
                    print(f"[tsc] epoch {epoch + 1:>4d}/{args.epochs}  "
                          f"step {global_step:>6d}  "
                          f"loss {loss_sum / loss_count:.5f}  "
                          f"onset {log['loss_onset']:.4f} type {log['loss_type']:.4f}  "
                          f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                          f"t {time.time() - t0:.0f}s")
                    loss_sum, loss_count = 0.0, 0

        if pending_grads:
            _opt_step()
        final_epoch = epoch + 1

        record = {
            "epoch": epoch + 1, "step": global_step,
            "train_loss": epoch_loss_sum / max(epoch_loss_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            with ema.swapped_into(model):
                val_loss, f1, prec, rec = _validate(model, loss_fn, val_loader, device)
            record.update(val_loss=val_loss, val_f1=f1,
                          val_precision=prec, val_recall=rec)
            is_best = stopper.update(1.0 - f1, epoch + 1)
            if is_best:
                torch.save({"model": ema.state_dict(), "config": config,
                            "epoch": epoch + 1, "val_f1": f1},
                           ckpt_dir / "transcriber_best.pt")
            print(f"[tsc] epoch {epoch + 1:>4d}  "
                  f"train {record['train_loss']:.5f}  val {val_loss:.5f}  "
                  f"F1 {f1:.4f} (P {prec:.3f} R {rec:.3f})"
                  f"{'  *best*' if is_best else ''}")
        metrics.log(**record)

        if (epoch + 1) % args.save_every == 0:
            path = ckpt_dir / f"tsc_e{epoch + 1:04d}_s{global_step:07d}.pt"
            torch.save({
                "model": model.state_dict(), "ema": ema.state_dict(),
                "ema_step": ema.step, "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(), "config": config,
                "stopper": {"best": stopper.best, "best_epoch": stopper.best_epoch,
                            "bad_epochs": stopper.bad_epochs},
                "epoch": epoch + 1, "step": global_step,
            }, path)
            rotate_checkpoints(ckpt_dir, "tsc_e*.pt", keep=2)
            print(f"[tsc] saved → {path}")

        if stopper.should_stop:
            print(f"[tsc] early stop at epoch {epoch + 1} "
                  f"(best F1 {1.0 - stopper.best:.4f} @ epoch {stopper.best_epoch})")
            break

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "ema_step": ema.step, "config": config,
                "epoch": final_epoch, "step": global_step},
               ckpt_dir / "transcriber_final.pt")
    print(f"[tsc] done.  steps={global_step}  elapsed={time.time() - t0:.0f}s")
    if val_loader is not None and stopper.best_epoch > 0:
        print(f"[tsc] best F1 {1.0 - stopper.best:.4f} @ epoch {stopper.best_epoch} "
              f"→ {ckpt_dir / 'transcriber_best.pt'}")


if __name__ == "__main__":
    main()

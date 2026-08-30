"""
src/train_decoder.py
────────────────────
Train the AR chart decoder (audio encoder + token decoder) with teacher forcing.

- Encoder = TranscriberNet trunk, warm-started from a transcription checkpoint
  (it already knows WHEN notes happen) and finetuned at --encoder-lr.
- Decoder = ChartDecoder over the 129-token chart vocabulary; loss = next-token
  CE (PAD ignored).  Full songs (no cropping) so training matches inference.
- Mirror augmentation in token space (lane k ↔ 3-k, re-canonicalised).
- Val metric: teacher-forced NLL (bits/token) + token accuracy, on EMA weights;
  early stop / best checkpoint on val NLL.  Sampling-based onset F1 is a
  separate step (src/test_decoder.py --list + script/eval_onset_f1.py).

Usage
~~~~~
  uv run python src/train_decoder.py \\
      --data-list data/train.txt --val-list data/val.txt \\
      --cache-dir data/cache_mel --ckpt-dir checkpoints/ \\
      --encoder-ckpt checkpoints/transcriber/transcriber_best.pt --epochs 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.audio2mel import chart_frames_to_mel_frames
from src.data.chart_tokenizer import (
    EOS, FREE_BINS, PAD, ChartTokenizer, TokenNote, encode_gameplay,
)
from src.data.gameplay_miner import GameplayNote, load_jsonl
from src.data.dataset import PhigrosDataset
from src.models.chart_decoder import DEFAULT_DECODER_CONFIG, ChartDecoder
from src.models.transcriber import TranscriberNet, TranscriptionLoss
from src.train_utils import EMA, EarlyStopper, MetricLogger, pad_or_trim, rotate_checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TokenChartDataset(Dataset):
    """(mel, token sequence, per-token frame times) per song."""

    def __init__(self, list_path: str, cache_dir: str | None, *, frame_ms: float,
                 max_frame: int, hop_length: int, sr: int, n_mels: int,
                 mirror_prob: float = 0.0, gameplay_dir: str | None = None):
        """gameplay_dir: free-x mode — notes come from data/gameplay/<song>.jsonl
        (continuous hit_x → 64 bins + sub-bin offsets) instead of the 4k JSON."""
        self.base = PhigrosDataset(
            data_list_path=list_path,
            convertor_params={"frame_ms": frame_ms, "max_frame": max_frame},
            cache_dir=cache_dir, augment=False,
            hop_length=hop_length, n_mels=n_mels, sr=sr,
        )
        self.freeform = gameplay_dir is not None
        self.gameplay_dir = Path(gameplay_dir) if gameplay_dir else None
        self.tk = ChartTokenizer(n_lanes=FREE_BINS if self.freeform else 4)
        self.frame_ms, self.max_frame, self.mirror_prob = frame_ms, max_frame, mirror_prob
        self._cache: dict[int, tuple[list, float]] = {}

    def __len__(self) -> int:
        return len(self.base)

    def _notes(self, i: int) -> tuple[list, float]:
        if i not in self._cache:
            json_path = self.base._entries[i][0]
            d = json.load(open(json_path, encoding="utf-8"))
            bpm = float(d["judgeLineList"][0]["bpm"])
            if self.freeform:
                song = Path(json_path).parent.name
                notes = load_jsonl(self.gameplay_dir / f"{song}.jsonl")   # GameplayNote list
            else:
                notes = self.tk.decode_tokens(self.tk.encode_chart(json_path), strict=False)
            self._cache[i] = (notes, bpm)
        return self._cache[i]

    def __getitem__(self, i: int) -> dict:
        json_path, audio_path, _ = self.base._entries[i]
        base_item = self.base[i]                                      # mel + 4k note array
        mel = base_item["audio"]                                      # (n_mels, T)
        notes, bpm = self._notes(i)
        mirror = self.mirror_prob > 0 and random.random() < self.mirror_prob
        frames_per_tick = 60.0 / (32.0 * bpm) * 1000.0 / self.frame_ms
        horizon = lambda tick: tick * frames_per_tick < self.max_frame   # noqa: E731
        if self.freeform:
            if mirror:
                notes = [GameplayNote(**{**n.__dict__, "hit_x": 1.0 - n.hit_x}) for n in notes]
            notes = [n for n in notes if horizon(n.tick)]
            tokens, offs, omask = encode_gameplay(self.tk, notes)
        else:
            if mirror:
                notes = [TokenNote(n.tick, 3 - n.lane, n.type, n.dur) for n in notes]
            notes = [n for n in notes if horizon(n.tick)]
            tokens = self.tk.encode_notes(notes)
            offs, omask = [0.0] * len(tokens), [False] * len(tokens)
        frames = [t * frames_per_tick for t in self.tk.ticks_per_position(tokens)]
        note_arr, vflag = base_item["note"], base_item["valid_flag"]
        if mirror:                                                    # keep aux targets consistent
            from src.data.dataset import _MIRROR_PERM
            note_arr = note_arr[_MIRROR_PERM, :]
        return {"mel": mel, "tokens": torch.tensor(tokens), "frames": torch.tensor(frames),
                "offsets": torch.tensor(offs), "offset_mask": torch.tensor(omask),
                "note": note_arr, "valid_flag": vflag,
                "bpm": bpm, "mel_frames": mel.shape[-1]}


def _collate(batch: list[dict], max_mel: int, mel_per_frame: int) -> dict:
    L = max(len(b["tokens"]) for b in batch)
    tokens = torch.full((len(batch), L), PAD, dtype=torch.long)
    frames = torch.zeros(len(batch), L)
    offsets = torch.zeros(len(batch), L); omask = torch.zeros(len(batch), L, dtype=torch.bool)
    mels, valid = [], []
    for i, b in enumerate(batch):
        n = len(b["tokens"])
        tokens[i, :n] = b["tokens"]; frames[i, :n] = b["frames"]
        offsets[i, :n] = b["offsets"]; omask[i, :n] = b["offset_mask"]
        mels.append(pad_or_trim(b["mel"], max_mel))
        v = torch.zeros(max_mel // mel_per_frame, dtype=torch.bool)
        v[: min(b["mel_frames"] // mel_per_frame, len(v))] = True
        valid.append(v)
    return {"mel": torch.stack(mels), "tokens": tokens, "frames": frames,
            "offsets": offsets, "offset_mask": omask, "mem_valid": torch.stack(valid),
            "note": torch.stack([b["note"] for b in batch]),
            "valid_flag": torch.stack([b["valid_flag"] for b in batch])}


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ARChartModel(nn.Module):
    """encoder.features → decoder; the encoder's transcription head stays
    attached so an auxiliary per-frame loss can keep the memory onset-aligned."""

    def __init__(self, encoder: TranscriberNet, decoder: ChartDecoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, mel, tokens, frames, mem_valid, return_hidden=False, return_feats=False):
        feats  = self.encoder.features(mel)
        memory = self.decoder.prepare_memory(feats)
        out = self.decoder(tokens, frames, memory, mem_valid, return_hidden=return_hidden)
        return (out, feats) if return_feats else out


OFFSET_LOSS_WEIGHT = 1.0
AUX_LOSS_WEIGHT    = 1.0     # auxiliary transcription loss on encoder features
_aux_loss_fn: TranscriptionLoss | None = None


def _step_loss(model, batch, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    global _aux_loss_fn
    mel    = batch["mel"].to(device)
    tokens = batch["tokens"].to(device)
    frames = batch["frames"].to(device)
    valid  = batch["mem_valid"].to(device)
    has_offset = model.decoder.offset_head is not None
    out, feats = model(mel, tokens[:, :-1], frames[:, :-1], valid,
                       return_hidden=has_offset, return_feats=True)
    logits, hidden = (out if has_offset else (out, None))
    if AUX_LOSS_WEIGHT > 0:
        if _aux_loss_fn is None:
            _aux_loss_fn = TranscriptionLoss().to(device)
        pred = model.encoder.head(feats).transpose(1, 2)              # (B, 32, T)
        note = batch["note"].to(device); vflag = batch["valid_flag"].to(device)
        T = min(pred.shape[-1], note.shape[-1])
        aux, _ = _aux_loss_fn(pred[..., :T], note[..., :T], vflag[..., :T])
    else:
        aux = torch.zeros((), device=device)
    target = tokens[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                           ignore_index=PAD) + AUX_LOSS_WEIGHT * aux
    if has_offset:
        # offset of the NOTE token at input position t is predicted from h_t
        omask = batch["offset_mask"][:, :-1].to(device)
        if omask.any():
            pred = model.decoder.predict_offset(hidden)
            tgt  = batch["offsets"][:, :-1].to(device)
            loss = loss + OFFSET_LOSS_WEIGHT * (((pred - tgt) ** 2) * omask).sum() / omask.sum()
    with torch.no_grad():
        m = target != PAD
        correct = ((logits.argmax(-1) == target) & m).sum()
    return loss, correct, int(m.sum())


@torch.no_grad()
def _validate(model, loader, device) -> tuple[float, float]:
    was = model.training; model.eval()
    nll, correct, count = 0.0, 0, 0
    for batch in loader:
        loss, c, n = _step_loss(model, batch, device)
        nll += loss.item() * n; correct += int(c); count += n
    if was: model.train()
    return nll / max(count, 1) / math.log(2), correct / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(description="Train the AR chart decoder",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data-list", required=True)
    g.add_argument("--val-list",  default=None)
    g.add_argument("--cache-dir", default=None)
    g.add_argument("--frame-ms",  type=float, default=512 / 22050 / 4 * 8 * 1000)
    g.add_argument("--max-frame", type=int, default=4096)
    g.add_argument("--hop-length", type=int, default=512)
    g.add_argument("--n-mels",    type=int, default=128)
    g.add_argument("--sr",        type=int, default=22050)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--mirror-prob", type=float, default=0.5)
    g.add_argument("--gameplay-dir", default=None,
                   help="FREE-X mode: data/gameplay (mined hit_x); enables 64-bin fused "
                        "NOTE tokens, factorized embeddings and the offset head")
    g = p.add_argument_group("model")
    g.add_argument("--encoder-ckpt", required=True, help="transcriber_best.pt for warm start")
    g.add_argument("--freeze-encoder", action="store_true")
    g.add_argument("--depth",      type=int, default=DEFAULT_DECODER_CONFIG["depth"])
    g.add_argument("--hidden-dim", type=int, default=DEFAULT_DECODER_CONFIG["hidden_dim"])
    g.add_argument("--dec-dropout", type=float, default=DEFAULT_DECODER_CONFIG["dropout"])
    g.add_argument("--cross-window", type=int, default=64,
                   help="local cross-attention window in chart frames (≈3 s); 0 = global")
    g.add_argument("--token-dropout", type=float, default=0.15,
                   help="word-dropout on token embeddings during training")
    g.add_argument("--aux-weight", type=float, default=1.0,
                   help="auxiliary transcription-loss weight on encoder features (0 = off)")
    g = p.add_argument_group("checkpoint")
    g.add_argument("--ckpt-dir",  default="checkpoints")
    g.add_argument("--resume",    default=None)
    g.add_argument("--save-every", type=int, default=10)
    g = p.add_argument_group("training")
    g.add_argument("--epochs",     type=int, default=300)
    g.add_argument("--batch-size", type=int, default=6)
    g.add_argument("--lr",         type=float, default=3e-4)
    g.add_argument("--encoder-lr", type=float, default=5e-5)
    g.add_argument("--min-lr",     type=float, default=1e-6)
    g.add_argument("--weight-decay", type=float, default=1e-2)
    g.add_argument("--clip-grad",  type=float, default=1.0)
    g.add_argument("--ema-decay",  type=float, default=0.999)
    g.add_argument("--patience",   type=int, default=40)
    g.add_argument("--seed",       type=int, default=42)
    g.add_argument("--device",     default=None)
    g.add_argument("--log-every",  type=int, default=20)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed); random.seed(args.seed)
    ckpt_dir = Path(args.ckpt_dir) / ("decoder_freex" if args.gameplay_dir else "decoder")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dec] device={device} seed={args.seed}")

    mel_per_frame = round(args.frame_ms / (args.hop_length / args.sr * 1000))
    max_mel = chart_frames_to_mel_frames(args.max_frame, args.frame_ms, args.hop_length, args.sr)
    ds_kw = dict(frame_ms=args.frame_ms, max_frame=args.max_frame, hop_length=args.hop_length,
                 sr=args.sr, n_mels=args.n_mels)
    ds_kw["gameplay_dir"] = args.gameplay_dir
    train_set = TokenChartDataset(args.data_list, args.cache_dir, mirror_prob=args.mirror_prob, **ds_kw)
    collate = partial(_collate, max_mel=max_mel, mel_per_frame=mel_per_frame)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=device.type == "cuda", drop_last=True)
    val_loader = None
    if args.val_list:
        val_set = TokenChartDataset(args.val_list, args.cache_dir, mirror_prob=0.0, **ds_kw)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate)
        print(f"[dec] train {len(train_set)} songs, val {len(val_set)} songs")
    else:
        print(f"[dec] train {len(train_set)} songs — WARNING: no val list")

    # ── models ───────────────────────────────────────────────────────────
    enc_ckpt = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=True)
    encoder = TranscriberNet(**enc_ckpt["config"])
    encoder.load_state_dict(enc_ckpt.get("ema", enc_ckpt["model"]))
    dec_config = dict(DEFAULT_DECODER_CONFIG)
    dec_config.update(depth=args.depth, hidden_dim=args.hidden_dim, dropout=args.dec_dropout,
                      mem_dim=enc_ckpt["config"]["hidden_dim"],
                      cross_window=(args.cross_window or None), token_dropout=args.token_dropout)
    global AUX_LOSS_WEIGHT
    AUX_LOSS_WEIGHT = args.aux_weight
    print(f"[dec] conditioning aids: cross_window={dec_config['cross_window']} "
          f"token_dropout={args.token_dropout} aux_weight={args.aux_weight}")
    if args.gameplay_dir:
        dec_config.update(vocab_size=train_set.tk.vocab_size, n_lanes=FREE_BINS,
                          factorized=True, offset_head=True)
        print(f"[dec] FREE-X mode: vocab {dec_config['vocab_size']}, 64 position bins, "
              f"factorized NOTE embeddings + offset head")
    decoder = ChartDecoder(**dec_config)
    model = ARChartModel(encoder, decoder).to(device)
    if args.freeze_encoder:
        for p_ in encoder.parameters(): p_.requires_grad_(False)
    print(f"[dec] encoder {sum(p.numel() for p in encoder.parameters()):,} "
          f"({'frozen' if args.freeze_encoder else f'lr {args.encoder_lr:.0e}'})  "
          f"decoder {decoder.num_params:,} [depth={dec_config['depth']} D={dec_config['hidden_dim']}]")

    groups = [{"params": list(decoder.parameters()), "lr": args.lr}]
    if not args.freeze_encoder:
        groups.append({"params": list(encoder.parameters()), "lr": args.encoder_lr})
    trainable = [p_ for g_ in groups for p_ in g_["params"]]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(loader))
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    ema = EMA(model, decay=args.ema_decay)
    stopper = EarlyStopper(patience=args.patience)
    metrics = MetricLogger(ckpt_dir / "metrics.jsonl", stage="decoder")

    start_epoch, step = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck.get("ema", model.state_dict()), step=ck.get("ema_step", 0))
        if "optimizer" in ck: optimizer.load_state_dict(ck["optimizer"])
        if "lr_scheduler" in ck:
            lr_sched.load_state_dict(ck["lr_scheduler"]); lr_sched.T_max = total_steps
        if "stopper" in ck:
            s = ck["stopper"]; stopper.best, stopper.best_epoch, stopper.bad_epochs = s["best"], s["best_epoch"], s["bad_epochs"]
        start_epoch, step = ck.get("epoch", 0), ck.get("step", 0)
        print(f"[dec] resumed ← {args.resume} epoch={start_epoch}")

    t0 = time.time(); final_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_encoder: encoder.eval()
        e_loss, e_n = 0.0, 0
        for batch in loader:
            loss, _, n = _step_loss(model, batch, device)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            optimizer.step(); lr_sched.step(); ema.update(model); step += 1
            e_loss += loss.item() * n; e_n += n
            if step % args.log_every == 0:
                print(f"[dec] ep {epoch+1:>4d} step {step:>6d} loss {loss.item():.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} t {time.time()-t0:.0f}s")
        final_epoch = epoch + 1
        rec = {"epoch": epoch + 1, "step": step, "train_nll": e_loss / max(e_n, 1) / math.log(2),
               "lr": optimizer.param_groups[0]["lr"]}
        if val_loader is not None:
            with ema.swapped_into(model):
                v_nll, v_acc = _validate(model, val_loader, device)
            rec.update(val_nll=v_nll, val_acc=v_acc)
            best = stopper.update(v_nll, epoch + 1)
            if best:
                torch.save({"model": ema.state_dict(), "dec_config": dec_config,
                            "enc_config": enc_ckpt["config"], "epoch": epoch + 1, "val_nll": v_nll},
                           ckpt_dir / "decoder_best.pt")
            print(f"[dec] epoch {epoch+1:>4d}  train {rec['train_nll']:.4f} bits  "
                  f"val {v_nll:.4f} bits  acc {v_acc:.3f}{'  *best*' if best else ''}")
        metrics.log(**rec)
        if (epoch + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "ema_step": ema.step,
                        "optimizer": optimizer.state_dict(), "lr_scheduler": lr_sched.state_dict(),
                        "stopper": {"best": stopper.best, "best_epoch": stopper.best_epoch,
                                    "bad_epochs": stopper.bad_epochs},
                        "dec_config": dec_config, "enc_config": enc_ckpt["config"],
                        "epoch": epoch + 1, "step": step},
                       ckpt_dir / f"dec_e{epoch+1:04d}.pt")
            rotate_checkpoints(ckpt_dir, "dec_e*.pt", keep=2)
        if stopper.should_stop:
            print(f"[dec] early stop at epoch {epoch+1} (best val {stopper.best:.4f} @ {stopper.best_epoch})")
            break
    torch.save({"model": ema.state_dict(), "dec_config": dec_config, "enc_config": enc_ckpt["config"],
                "epoch": final_epoch}, ckpt_dir / "decoder_final.pt")
    print(f"[dec] done. best val {stopper.best:.4f} bits @ {stopper.best_epoch} → {ckpt_dir/'decoder_best.pt'}")


if __name__ == "__main__":
    main()

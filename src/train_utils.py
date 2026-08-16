"""
src/train_utils.py
──────────────────
Shared training infrastructure for pre_train.py (VAE / wave) and train.py (DiT):

  pad_or_trim()        right-trim or pad the last dim; pads with the tensor's
                       own minimum by default (correct for dB-scale log-mels,
                       where 0.0 would read as a LOUD signal, not silence)
  MetricLogger         one JSON line per epoch → <ckpt_dir>/metrics.jsonl
  rotate_checkpoints() keep only the newest K intermediate checkpoints
  EMA                  exponential moving average of model weights
  EarlyStopper         patience-based stop on a validation metric
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Padding
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_trim(
    t: torch.Tensor,
    target: int,
    pad_value: Optional[float] = None,
) -> torch.Tensor:
    """Right-trim or pad the last dimension of *t* to exactly *target*.

    pad_value:
        ``None`` (default) → pad with ``t.min()``, the tensor's own floor.
        This is the correct choice for dB-scale log-mel spectrograms, where
        silence sits near the minimum (≈ -100 dB) and a 0.0 pad would be
        indistinguishable from a loud broadband signal.
        Pass an explicit value (e.g. ``0.0``) for tensors where zero really
        means "empty", such as note arrays and valid flags.
    """
    T = t.shape[-1]
    if T >= target:
        return t[..., :target]
    if pad_value is None:
        pad_value = float(t.min()) if t.numel() > 0 else 0.0
    return F.pad(t, (0, target - T), value=pad_value)


# ─────────────────────────────────────────────────────────────────────────────
# Metric logging
# ─────────────────────────────────────────────────────────────────────────────

class MetricLogger:
    """Append-only JSONL metric log: one line per epoch, survives everything.

    Usage::

        logger = MetricLogger(ckpt_dir / "metrics.jsonl", stage="vae")
        logger.log(epoch=3, step=111, train_loss=0.12, val_loss=0.15, lr=1e-4)
    """

    def __init__(self, path: str | Path, stage: str) -> None:
        self.path  = Path(path)
        self.stage = stage
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log(self, **metrics) -> None:
        record = {"stage": self.stage, "elapsed_s": round(time.time() - self._t0, 1)}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, float):
                v = round(v, 8)
            record[k] = v
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint rotation
# ─────────────────────────────────────────────────────────────────────────────

def rotate_checkpoints(ckpt_dir: str | Path, pattern: str, keep: int = 2) -> None:
    """Delete all but the newest *keep* files matching *pattern* in *ckpt_dir*.

    Intermediate checkpoints carry full optimizer state (~3× weight size —
    a DiT checkpoint is ~774 MB) and the default save cadence would fill the
    disk mid-run.  Call after every periodic save.  Best/final checkpoints
    use different filenames and are never touched.
    """
    files = sorted(Path(ckpt_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in files[:-keep] if keep > 0 else files:
        old.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of a model's parameters.

    With ~37 optimizer steps per epoch on a 300-song dataset the raw weights
    oscillate with every re-shuffled epoch; the EMA copy is what should be
    validated and sampled from.  Decay ramps up as ``(1+step)/(10+step)``
    capped at *decay*, so early training is not dominated by the random init.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay  = decay
        self.step   = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._non_float = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self._non_float[k] = v.detach().clone()

    def state_dict(self) -> dict:
        """EMA weights in a load_state_dict-compatible dict (original dtypes)."""
        out = {k: v.clone() for k, v in self.shadow.items()}
        out.update({k: v.clone() for k, v in self._non_float.items()})
        return out

    def load_state_dict(self, state: dict, step: int = 0) -> None:
        """Restore shadow weights, keeping them on the model's device.

        Checkpoints are torch.load-ed with map_location="cpu"; the incoming
        tensors are moved onto the device of the existing shadow (captured
        from the model at construction), otherwise the next update() would
        mix cpu and cuda tensors.
        """
        dev = (next(iter(self.shadow.values())).device
               if self.shadow else torch.device("cpu"))
        for k, v in state.items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone().float().to(dev)
            else:
                self._non_float[k] = v.detach().clone().to(dev)
        self.step = step

    class _Swapped:
        """Context manager: temporarily load EMA weights into the model."""

        def __init__(self, ema: "EMA", model: torch.nn.Module) -> None:
            self.ema, self.model = ema, model

        def __enter__(self):
            self.backup = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            self.model.load_state_dict(self.ema.state_dict())
            return self.model

        def __exit__(self, *exc):
            self.model.load_state_dict(self.backup)
            return False

    def swapped_into(self, model: torch.nn.Module) -> "EMA._Swapped":
        """``with ema.swapped_into(model): validate(model)`` — restores after."""
        return EMA._Swapped(self, model)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopper:
    """Stop when the validation metric has not improved for *patience* epochs.

    ``patience <= 0`` disables stopping (``update()`` still tracks the best
    value so best-checkpoint logic can share this object).
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record an epoch's val metric.  Returns True if it is a new best."""
        if value < self.best - self.min_delta:
            self.best       = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience

"""
src/test.py
─────────────────────────────────────────────────────────────────────────────
RhythmDiT inference script — audio → Phigros chart JSON.

Inference pipeline
~~~~~~~~~~~~~~~~~~
  1. Load audio file → log-mel spectrogram        (1, n_mels, T_audio)
  2. Pad / trim mel to max_frame                  (1, n_mels, max_frame)
  3. AudioWaveEncoder (frozen):  mel → audio_c    (1, 256, T_a)
  4. Align audio_c time axis →                    (1, 256, T_z)
  5. DDIM sampling:  z_T ~ N(0,I) → z_0           (1, z_ch, T_z)
  6. ChartVAE decode (frozen):   z_0 → note_array  (1, 20, max_frame)
  7. Phigros4kConvertor.save_phigros_file()       → output .json

Key constants (must match training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  frame_ms  = 512 / 22050 / 4 * 8 * 1000  ≈ 46.44 ms
  max_frame = 4096
  T_z       = max_frame // 16 = 256  (VAE / Wave encoder stride = 16)

Usage
~~~~~
  # Minimal (random model weights — pipeline smoke test):
  uv run python src/test.py --audio data/audio/Eltaw.ogg --output out/test.json

  # With trained checkpoints:
  uv run python src/test.py \\
      --audio     data/audio/Eltaw.ogg \\
      --output    out/Eltaw_IN.json \\
      --dit-ckpt  checkpoints/dit_final.pt \\
      --vae-ckpt  checkpoints/vae_final.pt \\
      --wave-ckpt checkpoints/wave_final.pt \\
      --bpm       193.0 \\
      --cfg-scale 3.0 \\
      --ddim-steps 50

  # Use BPM from a reference chart JSON:
  uv run python src/test.py \\
      --audio        data/audio/Eltaw.ogg \\
      --output       out/Eltaw_IN.json \\
      --ref-chart    data/json/Eltaw.json \\
      --dit-ckpt     checkpoints/dit_final.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.condition.wave import AudioWaveEncoder, DEFAULT_WAVE_CONFIG
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart2array import Phigros4kConvertor
from src.diffusion.sampler import DDIMSampler
from src.diffusion.schedule import DEFAULT_SCHEDULE_CONFIG, NoiseSchedule
from src.encoder.encoder import ChartVAE, DEFAULT_VAE_CONFIG
from src.models.dit import DEFAULT_DIT_CONFIG, RhythmDiT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align_time(audio_c: torch.Tensor, T_target: int) -> torch.Tensor:
    """Linearly interpolate audio_c (B, C, T_a) along the time axis to T_target."""
    if audio_c.shape[-1] == T_target:
        return audio_c
    return F.interpolate(audio_c, size=T_target, mode="linear", align_corners=False)


def _pad_or_trim_mel(mel: torch.Tensor, target: int) -> torch.Tensor:
    """Zero-pad or right-trim mel (n_mels, T) to exactly `target` frames."""
    T = mel.shape[-1]
    if T >= target:
        return mel[..., :target]
    return F.pad(mel, (0, target - T))


def _load_bpm_from_chart(json_path: str) -> float:
    """Read the BPM of the first judge line from a Phigros JSON file."""
    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return float(raw["judgeLineList"][0]["bpm"])


def _load_ckpt_weights(path: str, model: torch.nn.Module, device: torch.device) -> None:
    """Load a checkpoint produced by train.py into *model* (in-place)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # Support both raw state_dict and train.py envelope {"dit": ..., "epoch": ...}
    if isinstance(ckpt, dict):
        for key in ("dit", "vae", "wave", "state_dict", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_chart(
    audio_path:  str,
    output_path: str,
    *,
    dit:         RhythmDiT,
    vae:         ChartVAE,
    wave:        AudioWaveEncoder,
    schedule:    NoiseSchedule,
    device:      torch.device,
    bpm:         float        = 120.0,
    offset:      float        = 0.0,
    ddim_steps:  int          = 50,
    eta:         float        = 0.0,
    cfg_scale:   float        = 3.0,
    max_frame:   int          = 4096,
    hop_length:  int          = 512,
    n_mels:      int          = 128,
    sr:          int          = 22050,
    seed:        Optional[int] = None,
) -> np.ndarray:
    """Full audio → chart inference pipeline.

    Parameters
    ----------
    audio_path  : path to the input audio file (.ogg / .wav / .mp3 …)
    output_path : destination Phigros JSON file path (parent dirs auto-created)
    dit         : RhythmDiT model (eval, on device)
    vae         : ChartVAE       (eval, on device)
    wave        : AudioWaveEncoder (eval, on device)
    schedule    : NoiseSchedule  (on device)
    device      : torch device
    bpm         : BPM written into the output chart
    offset      : audio offset in seconds written into the output chart
    ddim_steps  : number of DDIM denoising steps
    eta         : DDIM stochasticity (0=deterministic, 1=DDPM-like)
    cfg_scale   : CFG guidance scale (1.0 = no guidance)
    max_frame   : chart time axis length (must match training)
    hop_length  : STFT hop length for mel (must match training)
    n_mels      : mel filter banks (must match training)
    sr          : audio sample rate (must match training)
    seed        : optional RNG seed for reproducibility

    Returns
    -------
    note_array : np.ndarray, shape (20, max_frame), raw VAE decoder output
    """
    if seed is not None:
        torch.manual_seed(seed)

    t0 = time.time()

    # ── Step 1: audio → log-mel ──────────────────────────────────────────
    print("[gen] step 1 / 4 : loading audio and computing mel …")
    audio_proc = AudioGPUprocessor(
        sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, device=str(device),
    )
    waveform = audio_proc.load_from_path(audio_path)   # (1, C, T_wav)
    mel = audio_proc.forward(waveform)                 # (1, n_mels, T_audio)
    mel = mel.squeeze(0)                               # (n_mels, T_audio)
    print(f"         audio  : {audio_path}")
    print(f"         mel    : {tuple(mel.shape)}")

    # ── Step 2: pad / trim mel to max_frame ──────────────────────────────
    mel = _pad_or_trim_mel(mel, max_frame)             # (n_mels, max_frame)
    mel = mel.unsqueeze(0)                             # (1, n_mels, max_frame)

    # ── Step 3: AudioWaveEncoder → audio_c ───────────────────────────────
    print("[gen] step 2 / 4 : encoding audio condition …")
    audio_c = wave(mel)                                # (1, 256, T_a)

    # Align time axis to T_z = max_frame // 16
    T_z = max_frame // 16                              # 256 with default settings
    audio_c = _align_time(audio_c, T_z)               # (1, 256, T_z)
    print(f"         audio_c: {tuple(audio_c.shape)}")

    # ── Step 4: DDIM sampling z_T → z_0 ─────────────────────────────────
    print(f"[gen] step 3 / 4 : DDIM sampling  steps={ddim_steps}  cfg={cfg_scale}  eta={eta} …")
    z_channels = DEFAULT_VAE_CONFIG["z_channels"]      # 16
    shape = (1, z_channels, T_z)

    sampler = DDIMSampler(schedule)
    z0 = sampler.sample(
        dit            = dit,
        audio_c        = audio_c,
        shape          = shape,
        steps          = ddim_steps,
        eta            = eta,
        cfg_scale      = cfg_scale,
        audio_c_uncond = None,          # zeros vector used as unconditional
        show_progress  = True,
    )                                                  # (1, 16, T_z)
    print(f"         z_0    : {tuple(z0.shape)}")

    # ── Step 5: VAE decode z_0 → note_array ─────────────────────────────
    print("[gen] step 4 / 4 : decoding chart …")
    note_raw = vae.decode(z0)                          # (1, 20, max_frame)
    note_array = note_raw.squeeze(0).cpu().numpy()     # (20, max_frame)
    print(f"         note   : {note_array.shape}  min={note_array.min():.3f}  max={note_array.max():.3f}")

    # ── Step 6: save Phigros JSON ─────────────────────────────────────────
    frame_ms = hop_length / sr / 4 * 8 * 1000         # ≈ 46.44 ms  (must match training)
    conv = Phigros4kConvertor(frame_ms=frame_ms, max_frame=max_frame)
    conv.save_phigros_file(
        note_array  = note_array,
        bpm         = bpm,
        output_path = output_path,
        offset      = offset,
    )

    elapsed = time.time() - t0
    print(f"[gen] done  →  {output_path}  ({elapsed:.1f}s)")
    return note_array


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Phigros chart from audio (RhythmDiT inference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("I/O")
    g.add_argument("--audio",      required=True,  help="path to input audio file")
    g.add_argument("--output",     required=True,  help="path to output Phigros JSON")
    g.add_argument("--ref-chart",  default=None,
                   help="reference chart JSON — BPM and offset are read from this file "
                        "(overrides --bpm and --offset when provided)")

    # ── chart metadata ──────────────────────────────────────────────────────
    g = p.add_argument_group("chart metadata")
    g.add_argument("--bpm",    type=float, default=120.0, help="chart BPM written to output")
    g.add_argument("--offset", type=float, default=0.0,   help="audio offset in seconds")

    # ── model checkpoints ───────────────────────────────────────────────────
    g = p.add_argument_group("model checkpoints")
    g.add_argument("--dit-ckpt",  default=None,
                   help="DiT checkpoint (.pt) from train.py  (omit → random weights)")
    g.add_argument("--vae-ckpt",  default=None,
                   help="VAE checkpoint (.pt)  (omit → random weights)")
    g.add_argument("--wave-ckpt", default=None,
                   help="Wave encoder checkpoint (.pt)  (omit → random weights)")

    # ── audio / frame settings ───────────────────────────────────────────────
    g = p.add_argument_group("audio / frame settings (must match training)")
    g.add_argument("--max-frame",   type=int,   default=4096,  help="chart time axis frames")
    g.add_argument("--hop-length",  type=int,   default=512,   help="mel STFT hop length")
    g.add_argument("--n-mels",      type=int,   default=128,   help="mel filter banks")
    g.add_argument("--sr",          type=int,   default=22050, help="audio sample rate")

    # ── diffusion / CFG ─────────────────────────────────────────────────────
    g = p.add_argument_group("diffusion / CFG")
    g.add_argument("--ddim-steps", type=int,   default=50,  help="DDIM denoising steps")
    g.add_argument("--eta",        type=float, default=0.0, help="DDIM η (0=deterministic)")
    g.add_argument("--cfg-scale",  type=float, default=3.0, help="CFG guidance scale")

    # ── misc ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("misc")
    g.add_argument("--seed",   type=int,   default=42)
    g.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    # ── device & seed ──────────────────────────────────────────────────────
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    print(f"[gen] device={device}  seed={args.seed}")

    # ── BPM / offset: prefer --ref-chart ──────────────────────────────────
    bpm    = args.bpm
    offset = args.offset
    if args.ref_chart is not None:
        bpm    = _load_bpm_from_chart(args.ref_chart)
        with open(args.ref_chart, encoding="utf-8") as fh:
            offset = float(json.load(fh)["offset"])
        print(f"[gen] BPM={bpm}  offset={offset}  (read from {args.ref_chart})")
    else:
        print(f"[gen] BPM={bpm}  offset={offset}")

    # ── output directory ───────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── build models ───────────────────────────────────────────────────────
    vae  = ChartVAE(DEFAULT_VAE_CONFIG).to(device)
    wave = AudioWaveEncoder(**DEFAULT_WAVE_CONFIG).to(device)
    dit  = RhythmDiT(**DEFAULT_DIT_CONFIG).to(device)

    # ── load checkpoints ───────────────────────────────────────────────────
    if args.vae_ckpt:
        _load_ckpt_weights(args.vae_ckpt, vae, device)
        print(f"[gen] loaded VAE  ← {args.vae_ckpt}")
    else:
        print("[gen] WARNING: VAE  using random weights (no --vae-ckpt)")

    if args.wave_ckpt:
        _load_ckpt_weights(args.wave_ckpt, wave, device)
        print(f"[gen] loaded Wave ← {args.wave_ckpt}")
    else:
        print("[gen] WARNING: Wave using random weights (no --wave-ckpt)")

    if args.dit_ckpt:
        _load_ckpt_weights(args.dit_ckpt, dit, device)
        print(f"[gen] loaded DiT  ← {args.dit_ckpt}")
    else:
        print("[gen] WARNING: DiT  using random weights (no --dit-ckpt)")

    # ── set all models to eval mode ────────────────────────────────────────
    vae.eval()
    wave.eval()
    dit.eval()

    # ── noise schedule ─────────────────────────────────────────────────────
    schedule = NoiseSchedule(**DEFAULT_SCHEDULE_CONFIG).to(device)

    print(f"[gen] RhythmDiT  params: {dit.num_params:,}")
    print(f"[gen] ChartVAE   params: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"[gen] WaveEncoder params: {sum(p.numel() for p in wave.parameters()):,}")

    # ── inference ─────────────────────────────────────────────────────────
    generate_chart(
        audio_path  = args.audio,
        output_path = args.output,
        dit         = dit,
        vae         = vae,
        wave        = wave,
        schedule    = schedule,
        device      = device,
        bpm         = bpm,
        offset      = offset,
        ddim_steps  = args.ddim_steps,
        eta         = args.eta,
        cfg_scale   = args.cfg_scale,
        max_frame   = args.max_frame,
        hop_length  = args.hop_length,
        n_mels      = args.n_mels,
        sr          = args.sr,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()

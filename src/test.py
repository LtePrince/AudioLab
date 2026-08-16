"""
src/test.py
─────────────────────────────────────────────────────────────────────────────
RhythmDiT inference script — audio → Phigros chart JSON.

Inference pipeline (variable-length)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Load audio file → log-mel spectrogram        (1, n_mels, T_audio)
  2. Size the chart axis from the ACTUAL audio duration:
       n_chart = ceil(T_audio / 2) rounded up to the latent grid (×16)
       T_z     = n_chart // 16 latent tokens
  3. Pad / trim mel to the chart axis duration (dB-floor padding)
  4. AudioWaveEncoder (frozen) → align to the token grid  (1, 256, T_z)
  5. DDIM sampling  z_T ~ N(0,I) → z_0            (1, z_ch, T_z)
       - native   : T_z ≤ 256 → one pass at the song's own length
       - windowed : T_z > 256 → overlapping 256-token windows chained with
         RePaint-style prefix conditioning (arbitrary song length; each
         window sees only trained RoPE positions)
       - --fixed-length forces the legacy padded-to-4096 axis (A/B baseline)
  6. ChartVAE decode (frozen):   z_0 → note_array  (1, 20, n_chart)
  7. Clip to the audio duration, save Phigros JSON

Key constants (must match training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  frame_ms  = 512 / 22050 / 4 * 8 * 1000  ≈ 46.44 ms per chart frame
  1 chart frame ≈ 2 mel frames;  VAE / Wave stride = 16
  trained window = max_frame // 16 = 256 tokens ≈ 190 s

Usage
~~~~~
  # Minimal (random model weights — pipeline smoke test):
  uv run python src/test.py --audio data/audio/Eltaw.ogg --output out/test.json

  # With trained checkpoints (dit_best.pt = best-on-val EMA weights):
  uv run python src/test.py \\
      --audio     data/audio/Eltaw.ogg \\
      --output    out/Eltaw_IN.json \\
      --dit-ckpt  checkpoints/dit/dit_best.pt \\
      --vae-ckpt  checkpoints/vae/vae_final.pt \\
      --wave-ckpt checkpoints/wave/wave_final.pt \\
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
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.condition.wave import AudioWaveEncoder, DEFAULT_WAVE_CONFIG
from src.data.audio2mel import AudioGPUprocessor, chart_frames_to_mel_frames
from src.data.chart2array import Phigros4kConvertor
from src.diffusion.sampler import DDIMSampler
from src.diffusion.schedule import DEFAULT_SCHEDULE_CONFIG, NoiseSchedule
from src.encoder.encoder import ChartVAE, DEFAULT_VAE_CONFIG
from src.models.dit import DEFAULT_DIT_CONFIG, RhythmDiT
from src.train_utils import pad_or_trim


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align_time(audio_c: torch.Tensor, T_target: int) -> torch.Tensor:
    """Linearly interpolate audio_c (B, C, T_a) along the time axis to T_target."""
    if audio_c.shape[-1] == T_target:
        return audio_c
    return F.interpolate(audio_c, size=T_target, mode="linear", align_corners=False)


def _load_bpm_from_chart(json_path: str) -> float:
    """Read the BPM of the first judge line from a Phigros JSON file."""
    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return float(raw["judgeLineList"][0]["bpm"])


def _load_ckpt_weights(path: str, model: torch.nn.Module, device: torch.device) -> None:
    """Load a checkpoint produced by train.py / pre_train.py into *model*.

    For DiT checkpoints the EMA weights are preferred when present — they are
    what validation selected on and sample noticeably better than the raw
    weights on a 300-song dataset.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # Support raw state_dict and train.py envelopes ({"ema"/"dit"/...: state_dict})
    if isinstance(ckpt, dict):
        for key in ("ema", "dit", "vae", "wave", "state_dict", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt)


def _build_dit(ckpt_path: Optional[str], device: torch.device) -> RhythmDiT:
    """Build a RhythmDiT sized from the checkpoint's stored dit_config.

    train.py saves ``dit_config`` inside every checkpoint so inference always
    reconstructs the exact trained architecture (e.g. --dit-depth 8
    --dit-hidden-dim 256 runs) without needing matching CLI flags here.
    """
    config = dict(DEFAULT_DIT_CONFIG)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "dit_config" in ckpt:
            config.update(ckpt["dit_config"])
            print(f"[gen] DiT config from checkpoint: "
                  f"depth={config['depth']} D={config['hidden_dim']} "
                  f"H={config['num_heads']}")
    return RhythmDiT(**config).to(device)


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
    fixed_length:      bool          = False,
    max_window_tokens: Optional[int] = None,
    window_overlap:    int           = 64,
) -> np.ndarray:
    """Full audio → chart inference pipeline (variable-length).

    Length modes
    ------------
    The latent axis is sized from the ACTUAL audio duration:

    - native   (T_z ≤ max_window_tokens): one DDIM pass at the song's own
      length.  RoPE positions stay inside the trained range, and there is no
      padded tail to hallucinate notes into.
    - windowed (T_z > max_window_tokens): overlapping windows of exactly
      max_window_tokens tokens, chained with RePaint-style prefix
      conditioning (DDIMSampler.sample_windowed) — supports songs of any
      length beyond the 190 s training cap.
    - fixed    (fixed_length=True): legacy behaviour — pad everything to
      max_frame chart frames; kept for A/B comparison.

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
    max_frame   : trained chart axis length (fixed mode; also the default
                  window size source: max_frame // vae_stride tokens)
    hop_length  : STFT hop length for mel (must match training)
    n_mels      : mel filter banks (must match training)
    sr          : audio sample rate (must match training)
    seed        : optional RNG seed for reproducibility
    fixed_length      : force the legacy fixed-length axis
    max_window_tokens : single-pass token limit / window size
                        (None → max_frame // 16 = the trained length)
    window_overlap    : context tokens carried between windows (windowed mode)

    Returns
    -------
    note_array : np.ndarray, shape (20, n_chart), raw VAE decoder output
    """
    if seed is not None:
        torch.manual_seed(seed)

    t0 = time.time()

    # ── timing constants (must match training) ───────────────────────────
    frame_ms      = hop_length / sr / 4 * 8 * 1000   # ≈ 46.44 ms per chart frame
    mel_frame_ms  = hop_length / sr * 1000           # ≈ 23.22 ms per mel frame
    mel_per_chart = round(frame_ms / mel_frame_ms)   # = 2: 1 chart frame ≈ 2 mel frames
    vae_stride    = 2 ** (len(DEFAULT_VAE_CONFIG["channel_mult"]) - 1)   # 16
    if max_window_tokens is None:
        max_window_tokens = max_frame // vae_stride                      # 256
    if not 0 <= window_overlap < max_window_tokens:
        raise ValueError(
            f"--window-overlap {window_overlap} must be in [0, "
            f"max_window_tokens={max_window_tokens})"
        )

    # ── Step 1: audio → log-mel ──────────────────────────────────────────
    print("[gen] step 1 / 4 : loading audio and computing mel …")
    audio_proc = AudioGPUprocessor(
        sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, device=str(device),
    )
    waveform = audio_proc.load_from_path(audio_path)   # (1, C, T_wav)
    mel = audio_proc.forward(waveform)                 # (1, n_mels, T_audio)
    mel = mel.squeeze(0)                               # (n_mels, T_audio)
    actual_audio_frames = mel.shape[-1]                # capture before padding
    print(f"         audio  : {audio_path}")
    print(f"         mel    : {tuple(mel.shape)}  (actual {actual_audio_frames} frames)")

    # ── Step 2: size the chart axis from the audio duration ──────────────
    # n_chart = audio duration in chart frames, rounded UP to the latent grid
    # (multiple of vae_stride) so encode/decode lengths are exact.
    n_chart_content = math.ceil(actual_audio_frames / mel_per_chart)
    if fixed_length:
        n_chart = max_frame
    else:
        n_chart = max(vae_stride,
                      math.ceil(n_chart_content / vae_stride) * vae_stride)
    T_z = n_chart // vae_stride
    windowed = (not fixed_length) and T_z > max_window_tokens
    mode = "fixed" if fixed_length else ("windowed" if windowed else "native")
    print(f"         length : {n_chart_content} chart frames "
          f"({n_chart_content * frame_ms / 1000:.1f}s) → axis {n_chart} "
          f"→ {T_z} tokens  [mode={mode}]")

    # pad/trim mel to the chart axis duration (pad value = the mel's own dB
    # floor — 0.0 would read as a LOUD signal, not silence)
    mel_target = chart_frames_to_mel_frames(n_chart, frame_ms, hop_length, sr)
    mel = pad_or_trim(mel, mel_target)                 # (n_mels, mel_target)
    mel = mel.unsqueeze(0)                             # (1, n_mels, mel_target)

    # ── Step 3: AudioWaveEncoder → audio_c on the latent token grid ──────
    print("[gen] step 2 / 4 : encoding audio condition …")
    audio_c = wave(mel)                                # (1, 256, ~2*T_z)
    audio_c = _align_time(audio_c, T_z)               # (1, 256, T_z)
    print(f"         audio_c: {tuple(audio_c.shape)}")

    # ── Step 4: DDIM sampling z_T → z_0 ─────────────────────────────────
    print(f"[gen] step 3 / 4 : DDIM sampling  steps={ddim_steps}  cfg={cfg_scale}  eta={eta} …")
    z_channels = DEFAULT_VAE_CONFIG["z_channels"]      # 16

    sampler = DDIMSampler(schedule)
    if windowed:
        print(f"         windowed: {T_z} tokens > window {max_window_tokens}, "
              f"overlap {window_overlap}")
        z0 = sampler.sample_windowed(
            dit         = dit,
            audio_c     = audio_c,
            z_channels  = z_channels,
            window      = max_window_tokens,
            overlap     = window_overlap,
            steps       = ddim_steps,
            eta         = eta,
            cfg_scale   = cfg_scale,
            show_progress = True,
        )                                              # (1, 16, T_z)
    else:
        z0 = sampler.sample(
            dit            = dit,
            audio_c        = audio_c,
            shape          = (1, z_channels, T_z),
            steps          = ddim_steps,
            eta            = eta,
            cfg_scale      = cfg_scale,
            audio_c_uncond = None,      # zeros vector used as unconditional
            show_progress  = True,
        )                                              # (1, 16, T_z)
    print(f"         z_0    : {tuple(z0.shape)}")

    # ── Step 5: VAE decode z_0 → note_array ─────────────────────────────
    print("[gen] step 4 / 4 : decoding chart …")
    note_raw = vae.decode(z0)                          # (1, 20, n_chart)
    note_array = note_raw.squeeze(0).cpu().numpy()     # (20, n_chart)
    print(f"         note   : {note_array.shape}  min={note_array.min():.3f}  max={note_array.max():.3f}")

    # ── Step 6: save Phigros JSON ─────────────────────────────────────────
    # Clip to the actual audio duration: removes notes in the rounded-up tail
    # (≤ 15 frames in native mode) or the padded region (fixed mode).
    actual_chart_frames = min(n_chart_content, n_chart)
    if actual_chart_frames < n_chart:
        note_array[:, actual_chart_frames:] = 0.0
        print(f"         clipped: notes beyond chart frame {actual_chart_frames} "
              f"({actual_chart_frames * frame_ms / 1000:.1f}s) removed")

    # from_logits=True: the VAE decoder outputs raw logits for the binary
    # channels (trained with BCEWithLogitsLoss), so onset decoding must
    # threshold at 0 (p=0.5), not at 0.5 (p≈0.62) — the 0.5 threshold would
    # systematically drop notes with confidence in (0.5, 0.62).
    conv = Phigros4kConvertor(frame_ms=frame_ms, max_frame=n_chart,
                              from_logits=True)
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
    g.add_argument("--max-frame",   type=int,   default=4096,
                   help="trained chart axis length (fixed-length mode; also the "
                        "default window size = max_frame/16 tokens)")
    g.add_argument("--hop-length",  type=int,   default=512,   help="mel STFT hop length")
    g.add_argument("--n-mels",      type=int,   default=128,   help="mel filter banks")
    g.add_argument("--sr",          type=int,   default=22050, help="audio sample rate")

    # ── length mode ─────────────────────────────────────────────────────────
    g = p.add_argument_group("length mode")
    g.add_argument("--fixed-length", action="store_true",
                   help="legacy behaviour: pad the chart axis to --max-frame "
                        "regardless of the audio duration (A/B baseline)")
    g.add_argument("--max-window-tokens", type=int, default=None,
                   help="single-pass token limit and window size for long songs "
                        "(default: max_frame/16 = 256, the trained length)")
    g.add_argument("--window-overlap", type=int, default=64,
                   help="latent tokens of generated context carried into each "
                        "next window (~0.74 s per token; windowed mode only)")

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

    # ── fail fast on invalid window config (before loading any model) ──────
    _mwt = args.max_window_tokens or (args.max_frame // 16)
    if not 0 <= args.window_overlap < _mwt:
        raise SystemExit(
            f"[ERROR] --window-overlap {args.window_overlap} must be in "
            f"[0, max_window_tokens={_mwt})"
        )

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
    dit  = _build_dit(args.dit_ckpt, device)   # sized from ckpt's dit_config

    # ── load checkpoints ───────────────────────────────────────────────────
    if args.vae_ckpt:
        _load_ckpt_weights(args.vae_ckpt, vae, device)
        print(f"[gen] loaded VAE  ← {args.vae_ckpt}  (latent_scale={vae.scale:.4f})")
        if vae.scale == 1.0:
            print("[gen] WARNING: VAE latent_scale is 1.0 (uncalibrated checkpoint)")
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
        fixed_length      = args.fixed_length,
        max_window_tokens = args.max_window_tokens,
        window_overlap    = args.window_overlap,
    )


if __name__ == "__main__":
    main()

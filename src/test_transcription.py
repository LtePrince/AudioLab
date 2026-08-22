"""
src/test_transcription.py
─────────────────────────
Transcription-baseline inference: audio → Phigros chart JSON.

One deterministic forward pass at the song's own length — no sampler, no
windowing, no length cap.  Shares the mel pipeline, note_array decoding and
official-format serialisation with the diffusion pipeline.

Usage
~~~~~
  uv run python src/test_transcription.py \\
      --audio song.ogg --output out/chart.json \\
      --ckpt checkpoints/transcriber/transcriber_best.pt \\
      --bpm 180        # or --ref-chart chart.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.data.audio2mel import AudioGPUprocessor
from src.data.chart2array import Phigros4kConvertor
from src.models.transcriber import TranscriberNet, to_note_array
from src.train_utils import pad_or_trim


@torch.no_grad()
def transcribe_chart(
    audio_path:  str,
    output_path: str,
    *,
    model:       TranscriberNet,
    device:      torch.device,
    bpm:         float = 120.0,
    offset:      float = 0.0,
    onset_bias:  float = 0.0,
    hop_length:  int   = 512,
    n_mels:      int   = 128,
    sr:          int   = 22050,
) -> None:
    t0 = time.time()
    frame_ms      = hop_length / sr / 4 * 8 * 1000
    mel_per_frame = model.mel_per_frame

    print("[tsc] step 1/3 : audio → mel …")
    proc = AudioGPUprocessor(sr=sr, n_fft=2048, hop_length=hop_length,
                             n_mels=n_mels, device=str(device))
    mel = proc.forward(proc.load_from_path(audio_path)).squeeze(0)  # (n_mels, T)
    actual = mel.shape[-1]
    n_chart = max(1, math.ceil(actual / mel_per_frame))
    mel = pad_or_trim(mel, n_chart * mel_per_frame).unsqueeze(0)
    print(f"        mel {tuple(mel.shape)} → {n_chart} chart frames "
          f"({n_chart * frame_ms / 1000:.1f}s)  [single pass, no window]")

    print("[tsc] step 2/3 : transcribing …")
    pred = model(mel.to(device))                       # (1, 32, n_chart)
    if onset_bias != 0.0:
        pred[:, 0:4] += onset_bias                     # shift onset logits (density knob)
    note_array = to_note_array(pred).squeeze(0).cpu().numpy()

    n_content = min(actual // mel_per_frame, n_chart)
    if n_content < n_chart:
        note_array[:, n_content:] = -100.0             # logits: certainly no note

    print("[tsc] step 3/3 : saving …")
    conv = Phigros4kConvertor(frame_ms=frame_ms, max_frame=n_chart,
                              from_logits=True)
    conv.save_phigros_file(note_array=note_array, bpm=bpm,
                           output_path=output_path, offset=offset)
    print(f"[tsc] done → {output_path}  ({time.time() - t0:.1f}s)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a Phigros chart with the transcription baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ckpt",   required=True,
                   help="transcriber_best.pt (EMA weights + config)")
    p.add_argument("--ref-chart", default=None,
                   help="read bpm/offset from this chart JSON")
    p.add_argument("--bpm",    type=float, default=120.0)
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--onset-bias", type=float, default=0.0,
                   help="added to onset logits: >0 denser charts, <0 sparser")
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--n-mels",     type=int, default=128)
    p.add_argument("--sr",         type=int, default=22050)
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    bpm, offset = args.bpm, args.offset
    if args.ref_chart:
        d = json.load(open(args.ref_chart, encoding="utf-8"))
        bpm    = float(d["judgeLineList"][0]["bpm"])
        offset = float(d["offset"])
        print(f"[tsc] bpm={bpm} offset={offset} (from {args.ref_chart})")

    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = TranscriberNet(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt.get("ema", ckpt["model"])
                          if isinstance(ckpt, dict) and ("ema" in ckpt or "model" in ckpt)
                          else ckpt)
    print(f"[tsc] loaded {args.ckpt}  "
          f"(val_f1={ckpt.get('val_f1', float('nan')):.4f} "
          f"@ epoch {ckpt.get('epoch', '?')})  params={model.num_params:,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    transcribe_chart(
        audio_path=args.audio, output_path=args.output,
        model=model, device=device, bpm=bpm, offset=offset,
        onset_bias=args.onset_bias,
        hop_length=args.hop_length, n_mels=args.n_mels, sr=args.sr,
    )


if __name__ == "__main__":
    main()

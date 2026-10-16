"""
src/test_decoder.py
───────────────────
AR chart decoder inference: audio → grammar-constrained token sampling →
official Phigros JSON.  Variable length natively (memory = the song's own
frames; the musical clock forces EOS at the end of the audio).

Usage
~~~~~
  # single song
  uv run python src/test_decoder.py --audio song.ogg --output out/c.json \\
      --ckpt checkpoints/decoder/decoder_best.pt --bpm 180
  # batch over a data list (for script/eval_onset_f1.py)
  uv run python src/test_decoder.py --list data/val.txt --out-dir out/eval_dec \\
      --ckpt checkpoints/decoder/decoder_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.data.timing import check_checkpoint_timing
from src.data.audio2mel import AudioGPUprocessor
from src.data.chart2array import save_phigros_notes
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.data.chart_tokenizer import ChartTokenizer, free_notes_to_phigros
from src.models.chart_decoder import ChartDecoder, generate_tokens
from src.models.transcriber import TranscriberNet
from src.train_utils import pad_or_trim
from src.data.timing import tick_seconds


def skeleton_from_chart(path: str, bpm: float | None = None, offset: float | None = None) -> list[int]:
    """Onset ticks of every note in a chart JSON (e.g. a transcriber output) —
    the rhythm skeleton for listen-then-write decoding."""
    d = json.load(open(path, encoding="utf-8"))
    bpm = float(d["judgeLineList"][0]["bpm"]) if bpm is None else bpm
    source_offset = float(d.get("offset", 0.0))
    offset = source_offset if offset is None else offset
    seconds = [tick_seconds(n["time"], float(l["bpm"])) + source_offset - offset
               for l in d["judgeLineList"] for n in l["notesAbove"] + l["notesBelow"]]
    return sorted({round(t / tick_seconds(1, bpm)) for t in seconds if t >= 0})


@torch.no_grad()
def skeleton_from_transcriber(mel: torch.Tensor, transcriber, bpm: float, frame_ms: float,
                              threshold: float = 0.5, offset: float = 0.0,
                              min_interval_ms: float = 0.0) -> list[int]:
    """Rhythm skeleton straight from transcriber logits (no lane collapse):
    a frame is an onset if ANY lane fires, p_any = 1 - Π(1 - p_lane) ≥ threshold;
    sub-frame start offsets refine the tick.  Returns sorted unique ticks."""
    from src.models.rhythm_skeleton import onset_candidates, candidates_to_ticks
    times, confidence = onset_candidates(transcriber(mel), frame_ms)
    return candidates_to_ticks(times, confidence, bpm, offset, threshold, min_interval_ms)


def load_models(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    check_checkpoint_timing(ck)
    enc = TranscriberNet(**ck["enc_config"]); dec = ChartDecoder(**ck["dec_config"])
    state = ck["model"]
    enc.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
    dec.load_state_dict({k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")})
    return enc.to(device).eval(), dec.to(device).eval()


@torch.no_grad()
def generate_chart(audio_path: str, output_path: str, *, enc, dec, device, bpm: float,
                   offset: float = 0.0, temperature: float = 1.0, top_p: float = 0.95,
                   seed: int | None = None, hop_length: int = 512, n_mels: int = 128,
                   sr: int = 22050, quiet: bool = False,
                   skeleton_ticks: list[int] | None = None,
                   transcriber=None, onset_threshold: float = 0.5,
                   raw_output: bool = False, free_min_gap: float = 0.04,
                   sampling_policy: str = 'flat', onset_min_interval_ms: float = 0.0) -> int:
    t0 = time.time()
    frame_ms = hop_length / sr / 4 * 8 * 1000
    mpf = enc.mel_per_frame
    proc = AudioGPUprocessor(sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, device=str(device))
    waveform = proc.load_from_path(audio_path)
    audio_seconds = waveform.shape[-1] / sr
    mel = proc.forward(waveform).squeeze(0)
    del waveform
    n_frames = max(1, math.ceil(mel.shape[-1] / mpf))
    mel = pad_or_trim(mel, n_frames * mpf).unsqueeze(0).to(device)
    memory = dec.prepare_memory(enc.features(mel))
    valid = torch.ones(1, n_frames, dtype=torch.bool, device=device)
    if transcriber is not None and skeleton_ticks is None:
        skeleton_ticks = skeleton_from_transcriber(mel, transcriber, bpm, frame_ms,
                                                    onset_threshold, offset=offset,
                                                    min_interval_ms=onset_min_interval_ms)
    gen = torch.Generator(device=device)
    if seed is not None: gen.manual_seed(seed)
    tk = ChartTokenizer(n_lanes=dec.n_lanes)
    out = generate_tokens(dec, memory, valid, bpm, frame_ms, audio_seconds * 1000 / frame_ms,
                          temperature=temperature, top_p=top_p, generator=gen, tokenizer=tk,
                          skeleton_ticks=skeleton_ticks, offset=offset, sampling_policy=sampling_policy)
    if dec.offset_head is not None:                      # free-x: (tokens, offsets)
        tokens, pos_offsets = out
        notes = tk.decode_tokens(tokens, strict=True)
        ticks = tk.ticks_per_position(tokens)
        offsets = {(ticks[i], tk.value(tokens[i]) // 4): o for i, o in pos_offsets.items()}
        pnotes = free_notes_to_phigros(tk, notes, offsets, bpm)
    else:
        tokens = out
        notes = tk.decode_tokens(tokens, strict=True)
        pnotes = tk.to_phigros_notes(notes, bpm)
    repairs = {}
    if not raw_output:
        end_tick = (audio_seconds - offset) / tick_seconds(1, bpm)
        if dec.n_lanes == 4:
            pnotes, repairs = constrain_4k_notes(pnotes, end_tick)
        else:
            pnotes, repairs = constrain_free_notes(pnotes, end_tick, free_min_gap)
    save_phigros_notes(pnotes, bpm, output_path, offset=offset)
    if not quiet:
        print(f"[dec] {Path(audio_path).name}: {n_frames} frames, {len(tokens)} tokens, "
              f"{len(notes)} notes → {output_path} ({time.time()-t0:.1f}s), repairs={repairs}")
    return len(notes)


def main() -> None:
    p = argparse.ArgumentParser(description="AR chart decoder inference")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--audio"); p.add_argument("--output")
    p.add_argument("--list"); p.add_argument("--out-dir")
    p.add_argument("--ref-chart", default=None)
    p.add_argument("--bpm", type=float, default=120.0); p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--sampling-policy", choices=['flat', 'type_preserving'], default='flat',
                   help="experimental type_preserving truncates positions within each note type")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-output", action="store_true",
                   help="export model output without static-line playability repairs")
    p.add_argument("--free-min-gap", type=float, default=0.04,
                   help="free-x static-line chord/Hold clearance as screen width (default: 0.04)")
    p.add_argument("--device", default=None)
    p.add_argument("--skeleton-json", default=None,
                   help="hybrid mode: chart JSON whose onset times force the time structure")
    p.add_argument("--skeleton-dir", default=None,
                   help="hybrid batch mode: <song_dir>.json skeletons (e.g. out/eval_tsc_r2)")
    p.add_argument("--skeleton-ckpt", default=None,
                   help="hybrid mode: transcriber checkpoint — skeleton from any-lane onset "
                        "probability (no lane collapse) at --onset-threshold")
    p.add_argument("--onset-threshold", type=float, default=0.5)
    p.add_argument("--onset-min-interval-ms", type=float, default=0.0,
                   help="optional confidence NMS on sub-frame onsets; 0 keeps all threshold crossings")
    args = p.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc, dec = load_models(args.ckpt, device)
    print(f"[dec] loaded {args.ckpt}")
    transcriber = None
    if args.skeleton_ckpt:
        tck = torch.load(args.skeleton_ckpt, map_location="cpu", weights_only=True)
        check_checkpoint_timing(tck)
        transcriber = TranscriberNet(**tck["config"]).to(device).eval()
        transcriber.load_state_dict(tck.get("ema", tck["model"]))
        print(f"[dec] skeleton from transcriber {args.skeleton_ckpt} @ p_any ≥ {args.onset_threshold}")

    if args.list:
        base = Path(args.list).resolve().parent
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        entries = [l.strip().split(",") for l in open(args.list) if l.strip() and not l.startswith("#")]
        for j, a, *_ in entries:
            d = json.load(open(base / j, encoding="utf-8"))
            song = Path(j).parent.name
            if (Path(args.out_dir) / f"{song}.json").exists():
                continue                                   # resumable batch
            skel = None
            if args.skeleton_dir:
                sp = Path(args.skeleton_dir) / f"{song}.json"
                if not sp.exists():
                    print(f"[dec] skeleton missing for {song}, skipped"); continue
                skel = skeleton_from_chart(str(sp), float(d["judgeLineList"][0]["bpm"]),
                                           float(d["offset"]))
            generate_chart(str(base / a), str(Path(args.out_dir) / f"{song}.json"),
                           enc=enc, dec=dec, device=device,
                           bpm=float(d["judgeLineList"][0]["bpm"]), offset=float(d["offset"]),
                           temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                           skeleton_ticks=skel, transcriber=transcriber,
                           onset_threshold=args.onset_threshold, raw_output=args.raw_output,
                           free_min_gap=args.free_min_gap, sampling_policy=args.sampling_policy,
                           onset_min_interval_ms=args.onset_min_interval_ms)
        return
    bpm, offset = args.bpm, args.offset
    if args.ref_chart:
        d = json.load(open(args.ref_chart, encoding="utf-8"))
        bpm, offset = float(d["judgeLineList"][0]["bpm"]), float(d["offset"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_chart(args.audio, args.output, enc=enc, dec=dec, device=device, bpm=bpm,
                   offset=offset, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                   skeleton_ticks=skeleton_from_chart(args.skeleton_json, bpm, offset) if args.skeleton_json else None,
                   transcriber=transcriber, onset_threshold=args.onset_threshold,
                   raw_output=args.raw_output, free_min_gap=args.free_min_gap,
                   sampling_policy=args.sampling_policy, onset_min_interval_ms=args.onset_min_interval_ms)


if __name__ == "__main__":
    main()

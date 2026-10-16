"""Shared chart/audio clock contract (Phigros ticks = 1/32 beat).

Chart-local ticks use the first judgment line's BPM after normalization.
Audio seconds = chart seconds + chart offset. Offsets never affect durations
or line-event interpolation, which stays in the source line's native ticks.
Decoder input positions use the clock AFTER consuming that input token.
"""
import math
import warnings

TIMING_SCHEMA = "audio-offset-post-token-v1"


def check_checkpoint_timing(checkpoint: dict, *, resume: bool = False) -> None:
    """Old weights remain loadable for comparison/transfer, but not silent resume."""
    if checkpoint.get("timing_schema") == TIMING_SCHEMA:
        return
    message = ("Checkpoint uses legacy/unknown timing. Audio offset and decoder DT "
               "alignment have changed; retrain on the rebuilt data before comparing quality.")
    if resume:
        raise ValueError(message + " Start a new training run instead of --resume.")
    warnings.warn(message, UserWarning, stacklevel=2)


def tick_seconds(tick: float, bpm: float) -> float:
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError(f"BPM must be finite and positive: {bpm}")
    return tick * 60.0 / (32.0 * bpm)


def normalized_span(tick: float, duration: float, source_bpm: float,
                    target_bpm: float) -> tuple[int, int]:
    """Quantize endpoints once; preserve hold endpoints to within half a tick."""
    scale = tick_seconds(1, source_bpm) / tick_seconds(1, target_bpm)
    start = round(tick * scale)
    end = round((tick + duration) * scale)
    return start, max(0, end - start)

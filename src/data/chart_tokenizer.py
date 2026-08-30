"""
src/data/chart_tokenizer.py
───────────────────────────
Chart ↔ token-sequence codec for the AR decoder (docs/ar_tokenizer_design.md).

Vocabulary (129 ids)
~~~~~~~~~~~~~~~~~~~~
    0            PAD
    1            BOS
    2            EOS
    3..18        DTB_1..DTB_16    time advance, n × 32 ticks (beats)
    19..49       DTT_1..DTT_31    time advance, n ticks
    50..65       NOTE_L{0-3}_{TAP,DRAG,HOLD,FLICK}   (lane*4 + type-1)
    66..81       DURB_1..DURB_16  hold duration, n × 32 ticks
    82..112      DURT_1..DURT_31  hold duration, n ticks
    113..128     COND_0..COND_15  reserved conditioning slots (unused in v1)

Canonical form (unique encoding, verified by tests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- notes sorted by (tick, lane); chord members (same tick) share one time
  position — no time token between them
- a time gap of ``g = 32·b + r`` ticks emits maximal-chunk DTB chain
  (DTB_16 × ⌊b/16⌋, then DTB_(b mod 16) if >0), then DTT_r if r > 0
- NOTE_*_HOLD is immediately followed by its duration in the same scheme
  (DURB chain + DURT remainder); duration ≥ 1 tick
- sequence = BOS {time-advance | note[+duration]}* EOS

All timing is native integer ticks (1 tick = 1/32 beat) — zero loss on the
dataset (gaps and hold durations are 100% integral, measured over 80 songs).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

from src.data.chart2array import LANE_CENTERS, Phigros4kConvertor

__all__ = ["ChartTokenizer", "GrammarState", "TokenNote", "VOCAB_SIZE"]

# ── id layout ───────────────────────────────────────────────────────────────
PAD, BOS, EOS = 0, 1, 2
N_DTB, N_DTT, N_DURB, N_DURT, N_COND = 16, 31, 16, 31, 16
_DTB0, _DTT0, _NOTE0 = 2, 18, 50       # fixed prefix; NOTE block size = 4 × n_lanes


@dataclass(frozen=True)
class VocabLayout:
    """Token-id layout for a given number of position slots ("lanes").

    n_lanes=4  → the 4k chart vocabulary (129 ids, docs/ar_tokenizer_design.md)
    n_lanes=64 → the free-x vocabulary (369 ids): lane == horizontal position
                 bin, NOTE_{bin}_{type} = fused position×type token
                 (docs/freeform_design.md P1).  Everything else is shared.
    """
    n_lanes: int = 4

    @property
    def n_note(self) -> int: return 4 * self.n_lanes
    @property
    def durb0(self) -> int: return _NOTE0 + self.n_note - 1
    @property
    def durt0(self) -> int: return self.durb0 + N_DURB
    @property
    def cond0(self) -> int: return self.durt0 + N_DURT + 1
    @property
    def vocab_size(self) -> int: return self.cond0 + N_COND


VOCAB_SIZE = VocabLayout(4).vocab_size      # 129 — the 4k default

TICKS_PER_BEAT = 32
NOTE_TYPES     = ("TAP", "DRAG", "HOLD", "FLICK")   # type ids 1..4


@dataclass(frozen=True)
class TokenNote:
    """Lossless note record used on both sides of the codec."""
    tick: int      # onset, 1/32-beat ticks
    lane: int      # 0..3
    type: int      # 1=Tap 2=Drag 3=Hold 4=Flick
    dur:  int = 0  # hold duration in ticks (0 unless type==3)


class ChartTokenizer:
    """Encode/decode between note lists (or chart JSON) and token ids.

    ``n_lanes`` selects the vocabulary: 4 (4k lanes) or 64 (free-x bins).
    """

    def __init__(self, n_lanes: int = 4) -> None:
        self.layout = VocabLayout(n_lanes)
        self.n_lanes = n_lanes
        self.vocab_size = self.layout.vocab_size

    # ── id constructors / predicates ────────────────────────────────────
    def dtb(self, n: int) -> int:
        assert 1 <= n <= N_DTB
        return _DTB0 + n

    def dtt(self, n: int) -> int:
        assert 1 <= n <= N_DTT
        return _DTT0 + n

    def note(self, lane: int, type_: int) -> int:
        assert 0 <= lane < self.n_lanes and 1 <= type_ <= 4
        return _NOTE0 + lane * 4 + (type_ - 1)

    def durb(self, n: int) -> int:
        assert 1 <= n <= N_DURB
        return self.layout.durb0 + n

    def durt(self, n: int) -> int:
        assert 1 <= n <= N_DURT
        return self.layout.durt0 + n

    def cond(self, i: int) -> int:
        assert 0 <= i < N_COND
        return self.layout.cond0 + i

    def kind(self, tok: int) -> str:
        """Coarse token class: PAD/BOS/EOS/DTB/DTT/NOTE/DURB/DURT/COND."""
        L = self.layout
        if tok == PAD: return "PAD"
        if tok == BOS: return "BOS"
        if tok == EOS: return "EOS"
        if _DTB0 < tok <= _DTB0 + N_DTB:      return "DTB"
        if _DTT0 < tok <= _DTT0 + N_DTT:      return "DTT"
        if _NOTE0 <= tok < _NOTE0 + L.n_note: return "NOTE"
        if L.durb0 < tok <= L.durb0 + N_DURB: return "DURB"
        if L.durt0 < tok <= L.durt0 + N_DURT: return "DURT"
        if L.cond0 <= tok < L.cond0 + N_COND: return "COND"
        raise ValueError(f"invalid token id {tok} for n_lanes={self.n_lanes}")

    def value(self, tok: int) -> int:
        """Numeric payload: n for DTB/DTT/DURB/DURT, lane*4+type-1 for NOTE."""
        k = self.kind(tok)
        base = {"DTB": _DTB0, "DTT": _DTT0, "NOTE": _NOTE0,
                "DURB": self.layout.durb0, "DURT": self.layout.durt0,
                "COND": self.layout.cond0}.get(k)
        return 0 if base is None else tok - base

    def token_name(self, tok: int) -> str:
        k = self.kind(tok)
        if k == "NOTE":
            v = tok - _NOTE0
            return f"NOTE_L{v // 4}_{NOTE_TYPES[v % 4]}"
        if k in ("DTB", "DTT", "DURB", "DURT"):
            return f"{k}_{self.value(tok)}"
        if k == "COND":
            return f"COND_{self.value(tok)}"
        return k

    # ── time/duration emission (canonical maximal chunks) ───────────────
    @staticmethod
    def _emit_span(out: list[int], ticks: int, beat_fn, tick_fn,
                   max_beat: int, max_tick: int) -> None:
        b, r = divmod(ticks, TICKS_PER_BEAT)
        while b > 0:
            chunk = min(b, max_beat)
            out.append(beat_fn(chunk))
            b -= chunk
        if r > 0:
            assert r <= max_tick
            out.append(tick_fn(r))

    # ── encode ──────────────────────────────────────────────────────────
    def encode_notes(self, notes: list[TokenNote]) -> list[int]:
        """Canonical token encoding of a note list (any order; deduped)."""
        canon: dict[tuple[int, int], TokenNote] = {}
        n_dup = 0
        for n in notes:
            key = (n.tick, n.lane)
            if key in canon:
                n_dup += 1
                continue                       # first note wins (matches array codec)
            canon[key] = n
        if n_dup:
            warnings.warn(f"encode_notes: {n_dup} duplicate (tick,lane) notes dropped",
                          RuntimeWarning, stacklevel=2)

        out: list[int] = [BOS]
        prev_tick = 0
        for (tick, lane), n in sorted(canon.items()):
            gap = tick - prev_tick
            if gap > 0:
                self._emit_span(out, gap, self.dtb, self.dtt, N_DTB, N_DTT)
                prev_tick = tick
            type_ = n.type
            dur   = n.dur
            if type_ == 3 and dur < 1:
                warnings.warn("encode_notes: zero-duration Hold encoded as Tap",
                              RuntimeWarning, stacklevel=2)
                type_, dur = 1, 0
            out.append(self.note(lane, type_))
            if type_ == 3:
                self._emit_span(out, dur, self.durb, self.durt, N_DURB, N_DURT)
        out.append(EOS)
        return out

    def encode_chart(self, json_path: str) -> list[int]:
        """Encode a Phigros chart JSON (4k-converted or raw; lanes quantised)."""
        d = json.load(open(json_path, encoding="utf-8"))
        notes: list[TokenNote] = []
        for line in d["judgeLineList"]:
            for n in line["notesAbove"] + line["notesBelow"]:
                notes.append(TokenNote(
                    tick=int(round(n["time"])),
                    lane=Phigros4kConvertor._lane_from_x(n["positionX"]),
                    type=int(n["type"]),
                    dur=int(round(n["holdTime"])) if int(n["type"]) == 3 else 0,
                ))
        return self.encode_notes(notes)

    # ── decode ──────────────────────────────────────────────────────────
    def decode_tokens(self, tokens: list[int], strict: bool = True) -> list[TokenNote]:
        """Token ids → note list.  strict=True validates the grammar."""
        gs = GrammarState(self)
        notes: list[TokenNote] = []
        cur_tick = 0
        pending: TokenNote | None = None       # HOLD awaiting duration
        pending_dur = 0

        def flush() -> None:
            nonlocal pending, pending_dur
            if pending is not None:
                notes.append(TokenNote(pending.tick, pending.lane, 3,
                                       max(pending_dur, 1)))
                pending, pending_dur = None, 0

        for tok in tokens:
            if strict and not gs.is_allowed(tok):
                raise ValueError(
                    f"grammar violation at {self.token_name(tok)} "
                    f"(context: {gs.describe()})")
            gs.step(tok)
            k = self.kind(tok)
            if k in ("PAD", "BOS", "COND"):
                continue
            if k == "EOS":
                flush()
                break
            if k == "DTB":
                flush(); cur_tick += self.value(tok) * TICKS_PER_BEAT
            elif k == "DTT":
                flush(); cur_tick += self.value(tok)
            elif k == "NOTE":
                flush()
                v = tok - _NOTE0
                lane, type_ = v // 4, v % 4 + 1
                if type_ == 3:
                    pending = TokenNote(cur_tick, lane, 3, 0)
                else:
                    notes.append(TokenNote(cur_tick, lane, type_, 0))
            elif k == "DURB":
                pending_dur += self.value(tok) * TICKS_PER_BEAT
            elif k == "DURT":
                pending_dur += self.value(tok)
        flush()
        return notes

    # ── per-position musical time (for decoder time-embedding injection) ─
    def ticks_per_position(self, tokens: list[int]) -> list[int]:
        """Tick reached BEFORE emitting each token (time of the event the
        token belongs to).  DTB/DTT advance the clock for later positions."""
        out, cur = [], 0
        for tok in tokens:
            out.append(cur)
            k = self.kind(tok)
            if k == "DTB":
                cur += self.value(tok) * TICKS_PER_BEAT
            elif k == "DTT":
                cur += self.value(tok)
        return out

    # ── convenience: decoded notes → Phigros Note dicts ─────────────────
    @staticmethod
    def to_phigros_notes(notes: list[TokenNote], bpm: float) -> list[dict]:
        out = []
        for n in sorted(notes, key=lambda x: (x.tick, x.lane)):
            out.append({
                "type":          n.type,
                "time":          n.tick,
                "positionX":     float(LANE_CENTERS[n.lane]),
                "holdTime":      float(n.dur),
                "speed":         1.0,
                "floorPosition": n.tick * 60.0 / (32.0 * bpm),
            })
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Grammar state machine (constrained decoding + validation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GrammarState:
    """Incremental validity tracker for the canonical token grammar.

    Rules enforced (docs/ar_tokenizer_design.md §2-3):
    - sequence starts with BOS; COND only directly after BOS
    - within a chord (no time advance between notes) each lane at most once
    - NOTE_*_HOLD must be followed immediately by its duration; the DURB
      chain may continue only after a maximal DURB_16 chunk; at most one
      DURT remainder per duration
    - the DTB chain likewise continues only after DTB_16; at most one DTT
      remainder per gap; a gap group closes at the first NOTE
    - EOS is legal only when no duration is pending

    Use ``is_allowed`` as the constrained-decoding mask and ``step`` to
    advance after emitting a token.
    """
    tok:          ChartTokenizer = field(default_factory=ChartTokenizer)
    started:      bool = False
    after_bos:    bool = False   # COND window (directly after BOS)
    in_dur:       bool = False   # HOLD emitted, zero duration tokens so far
    dur_can_beat: bool = False   # duration group open, DURB may continue
    dur_had_tick: bool = False   # duration group already has its DURT
    dur_open:     bool = False   # ≥1 duration token, group not yet closed
    dtb_ok:       bool = True    # DTB permitted (chain state of current gap)
    dtt_had:      bool = False   # current gap already has its DTT
    used_lanes:   set  = field(default_factory=set)
    finished:     bool = False

    def is_allowed(self, tok: int) -> bool:
        T = self.tok
        k = T.kind(tok)
        if self.finished:
            return k == "PAD"
        if not self.started:
            return k == "BOS"
        if k in ("BOS", "PAD"):
            return False
        if k == "COND":
            return self.after_bos
        if self.in_dur:                        # duration must start now
            return k in ("DURB", "DURT")
        if k == "DURB":
            return self.dur_open and self.dur_can_beat
        if k == "DURT":
            return self.dur_open and not self.dur_had_tick
        if k == "DTB":
            return self.dtb_ok
        if k == "DTT":
            return not self.dtt_had
        if k == "NOTE":
            return T.value(tok) // 4 not in self.used_lanes
        if k == "EOS":
            return True                        # in_dur already excluded above
        return False

    def step(self, tok: int) -> None:
        T = self.tok
        k = T.kind(tok)
        if k == "BOS":
            self.started, self.after_bos = True, True
            return
        if k != "COND":
            self.after_bos = False
        if k == "EOS":
            self.finished = True
        elif k == "DTB":
            self.used_lanes.clear()
            self._close_dur()
            self.dtb_ok  = (T.value(tok) == N_DTB)   # chain only after max chunk
            self.dtt_had = False
        elif k == "DTT":
            self.used_lanes.clear()
            self._close_dur()
            self.dtb_ok, self.dtt_had = False, True
        elif k == "NOTE":
            v = T.value(tok)
            self._close_dur()
            self.used_lanes.add(v // 4)
            self.dtb_ok, self.dtt_had = True, False  # a fresh gap may follow
            if v % 4 + 1 == 3:                       # HOLD → duration required
                self.in_dur = True
        elif k == "DURB":
            self.in_dur, self.dur_open = False, True
            self.dur_can_beat = (T.value(tok) == N_DURB)
        elif k == "DURT":
            self.in_dur, self.dur_open = False, True
            self.dur_can_beat, self.dur_had_tick = False, True

    def _close_dur(self) -> None:
        self.dur_open = self.dur_can_beat = self.dur_had_tick = False

    def describe(self) -> str:
        return (f"started={self.started} in_dur={self.in_dur} "
                f"dur_open={self.dur_open} can_beat={self.dur_can_beat} "
                f"lanes={sorted(self.used_lanes)} finished={self.finished}")


# ─────────────────────────────────────────────────────────────────────────────
# Free-x (trackless) support: continuous hit_x ↔ 64 position bins + offset
# ─────────────────────────────────────────────────────────────────────────────

FREE_BINS         = 64
FREE_LO, FREE_HI  = 0.05, 0.95          # playfield span in screen-normalised x
_FREE_W           = (FREE_HI - FREE_LO) / FREE_BINS


def hit_to_bin(hit_x: float) -> tuple[int, float]:
    """screen x → (bin, sub-bin offset ∈ [0,1))."""
    u = (min(max(hit_x, FREE_LO), FREE_HI - 1e-9) - FREE_LO) / _FREE_W
    b = int(u)
    return b, u - b


def bin_to_hit(b: int, offset: float = 0.5) -> float:
    return FREE_LO + (b + offset) * _FREE_W


def encode_gameplay(tk: ChartTokenizer, gnotes) -> tuple[list[int], list[float], list[bool]]:
    """Mined gameplay notes (GameplayNote-like: tick/type/dur/hit_x) → tokens
    for a 64-lane tokenizer, plus the per-position sub-bin offset targets
    (offset_mask marks NOTE positions).  Same dedupe rule as encode_notes:
    first note per (tick, bin) wins."""
    assert tk.n_lanes == FREE_BINS, "encode_gameplay needs ChartTokenizer(n_lanes=64)"
    offsets: dict[tuple[int, int], float] = {}
    notes: list[TokenNote] = []
    for g in gnotes:
        b, off = hit_to_bin(g.hit_x)
        key = (int(g.tick), b)
        if key in offsets:
            continue
        offsets[key] = off
        type_ = g.type if not (g.type == 3 and g.dur < 1) else 1
        notes.append(TokenNote(int(g.tick), b, type_, int(g.dur) if type_ == 3 else 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tokens = tk.encode_notes(notes)
    ticks = tk.ticks_per_position(tokens)
    offs, mask = [], []
    for tok, tick in zip(tokens, ticks):
        if tk.kind(tok) == "NOTE":
            b = tk.value(tok) // 4
            offs.append(offsets.get((tick, b), 0.5)); mask.append(True)
        else:
            offs.append(0.0); mask.append(False)
    return tokens, offs, mask


def free_notes_to_phigros(tk: ChartTokenizer, notes: list[TokenNote],
                          offsets: dict[tuple[int, int], float], bpm: float) -> list[dict]:
    """Decoded 64-lane TokenNotes (+ per-(tick,bin) offsets) → Phigros notes
    on a single static line at x=0.5 (positionX = (hit_x - 0.5) / 0.05625)."""
    from src.data.gameplay_miner import X_UNIT_W
    out = []
    for n in sorted(notes, key=lambda x: (x.tick, x.lane)):
        hx = bin_to_hit(n.lane, offsets.get((n.tick, n.lane), 0.5))
        out.append({
            "type": n.type, "time": n.tick,
            "positionX": (hx - 0.5) / X_UNIT_W,
            "holdTime": float(n.dur), "speed": 1.0,
            "floorPosition": n.tick * 60.0 / (32.0 * bpm),
        })
    return out

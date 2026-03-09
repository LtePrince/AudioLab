"""
chart2array.py
--------------
Phigros official chart format (formatVersion=3) type definitions,
read-only chart data store, flattened 4-lane representation, and convertor.

Sections
--------
1. Constants
2. Phigros base TypedDict definitions
3. FlatNote
4. PhigrosChart         -- read-only store; notes nested inside JudgeLines
5. PhigrosFlatChart     -- flattened 4-lane snapshot with note_array
6. Phigros4kConvertor   -- flatten / chart_to_array / array_to_notes / save
7. parse_phigros_file   -- convenience loader, returns PhigrosChart
8. main                 -- smoke test

Data flow
---------
    Phigros JSON
        └─ parse_phigros_file()  ──►  PhigrosChart
                                            │
                              Phigros4kConvertor.flatten()
                                            │
                                      PhigrosFlatChart
                                      ├─ .flat_notes  list[FlatNote]
                                      ├─ .note_array  ndarray (20, T)
                                      └─ .valid_flag  ndarray (T,)

    note_array  ──►  Phigros4kConvertor.array_to_notes()  ──►  list[Note]
    note_array  ──►  Phigros4kConvertor.save_phigros_file()  ──►  JSON file
"""

from __future__ import annotations

import json
import os
import bisect
from dataclasses import dataclass
from typing import Optional

import numpy as np
from typing import TypedDict


# ===========================================================================
# Section 1 – Constants
# ===========================================================================

# --- Note type labels (as stored in Phigros JSON) ---
NOTE_TAP   = 1
NOTE_DRAG  = 2
NOTE_HOLD  = 3
NOTE_FLICK = 4

# Normalised storage values for the note_type channel  (raw label / 4).
#
#   stored   ── decision boundary ──  raw label
#   0.25             ≤ 0.375           1  Tap
#   0.50       0.375 ~ 0.625           2  Drag
#   0.75       0.625 ~ 0.875           3  Hold
#   1.00             > 0.875           4  Flick
NOTE_TYPE_STORED: dict[int, float] = {
    NOTE_TAP:   0.25,
    NOTE_DRAG:  0.50,
    NOTE_HOLD:  0.75,
    NOTE_FLICK: 1.00,
}
NOTE_TYPE_THRESHOLDS: list[float] = [0.375, 0.625, 0.875]  # ascending boundaries

# --- 4-lane positionX quantisation ---
#
#   positionX <  -2.5  →  lane 0   (rebuild centre: -3.75)
#   -2.5 ≤ posX <  0   →  lane 1   (rebuild centre: -1.25)
#    0   ≤ posX < 2.5  →  lane 2   (rebuild centre:  1.25)
#   positionX ≥  2.5   →  lane 3   (rebuild centre:  3.75)
LANE_BOUNDARIES: list[float] = [-2.5, 0.0, 2.5]   # len == NUM_LANES - 1
LANE_CENTERS:    list[float] = [-3.75, -1.25, 1.25, 3.75]
NUM_LANES = 4

# --- note_array channel offsets  (for lane k add the base offset) ---
#
#   k +  0  is_start   ∈ {0, 1}
#   k +  4  start_off  ∈ [0, 1]    sub-frame onset offset
#   k +  8  is_holding ∈ {0, 1}    1 on every Hold body / tail frame
#   k + 12  end_offset ∈ [0, 1]    sub-frame Hold-end offset (tail frame only)
#   k + 16  note_type  ∈ {0, 0.25, 0.50, 0.75, 1.00}   0 when is_start == 0
NUM_CHANNELS  = NUM_LANES * 5   # 20
CH_IS_START   = 0
CH_START_OFF  = 4
CH_IS_HOLDING = 8
CH_END_OFFSET = 12
CH_NOTE_TYPE  = 16


# ===========================================================================
# Section 2 – Phigros base TypedDict definitions  (formatVersion = 3)
# ===========================================================================

class SpeedEvent(TypedDict):
    """Controls how fast notes fall past the judgment line."""
    startTime: float   # 1/32-beat ticks
    endTime:   float
    value:     float   # speed multiplier


class MoveEvent(TypedDict):
    """Moves the judgment line to a new (X, Y) position.

    Coordinate system: bottom-left = (0, 0), top-right = (1, 1).
    """
    startTime: float
    endTime:   float
    start:     float   # X at startTime
    end:       float   # X at endTime
    start2:    float   # Y at startTime
    end2:      float   # Y at endTime


class RotateEvent(TypedDict):
    """Rotates the judgment line in degrees."""
    startTime: float
    endTime:   float
    start:     float   # rotation angle at startTime
    end:       float   # rotation angle at endTime


class DisappearEvent(TypedDict):
    """Controls judgment-line opacity (0 = invisible, 1 = fully opaque)."""
    startTime: float
    endTime:   float
    start:     float
    end:       float


class Note(TypedDict):
    """A single note attached to a judgment line.

    type          1=Tap  2=Drag  3=Hold  4=Flick
    time          Onset in 1/32-beat ticks relative to the line's BPM.
    positionX     Horizontal position on the line (typical range: -6 to +6).
    holdTime      Hold duration in 1/32-beat ticks; 0 for non-Hold types.
    speed         Per-note visual speed multiplier.
    floorPosition Pre-computed render distance; not used for ML training.
    """
    type:          int
    time:          int
    positionX:     float
    holdTime:      float
    speed:         float
    floorPosition: float


class JudgeLine(TypedDict):
    """One judgment line with its attached events and notes."""
    bpm:                      float
    notesAbove:               list[Note]
    notesBelow:               list[Note]
    speedEvents:              list[SpeedEvent]
    judgeLineMoveEvents:      list[MoveEvent]
    judgeLineRotateEvents:    list[RotateEvent]
    judgeLineDisappearEvents: list[DisappearEvent]


class Chart(TypedDict):
    """Top-level Phigros chart object (formatVersion = 3)."""
    formatVersion: int     # always 3 for official charts
    offset:        float   # audio playback delay in seconds
    judgeLineList: list[JudgeLine]


# ===========================================================================
# Section 3 – FlatNote
# ===========================================================================

@dataclass(frozen=True)
class FlatNote:
    """Immutable view of one note with its parent judgment line's BPM.

    Attributes
    ----------
    note:       Original :class:`Note` dict from the JSON.
    bpm:        BPM of the owning judgment line.
    above:      True if from ``notesAbove``, False for ``notesBelow``.
    line_idx:   Index of the parent judgment line in ``judgeLineList``.
    """
    note:     Note
    bpm:      float
    above:    bool
    line_idx: int

    @property
    def time_ms(self) -> float:
        """Onset time in milliseconds (unadjusted for rate or offset)."""
        return self.note["time"] / 32.0 * (60_000.0 / self.bpm)


# ===========================================================================
# Section 4 – PhigrosChart  (read-only data store)
# ===========================================================================

class PhigrosChart:
    """Read-only container for a complete Phigros chart (formatVersion = 3).

    Notes are stored in their original nested structure inside each
    :class:`JudgeLine`; no flattening is performed here.

    Use :meth:`from_json` or :func:`parse_phigros_file` to load from disk.
    Attribute assignment raises :class:`AttributeError` to prevent mutation.

    Examples
    --------
    >>> chart = PhigrosChart.from_json("chart.json", version="IN")
    >>> chart.bpm
    193.0
    >>> chart.judge_lines[0]["notesAbove"][0]
    {'type': 1, 'time': 128, ...}
    """

    __slots__ = ("_raw", "_json_path", "_audio_path", "_version")

    def __init__(
        self,
        raw:        Chart,
        json_path:  str,
        audio_path: str = "",
        version:    str = "",
    ) -> None:
        object.__setattr__(self, "_raw",        raw)
        object.__setattr__(self, "_json_path",  os.path.abspath(json_path))
        object.__setattr__(self, "_audio_path", audio_path)
        object.__setattr__(self, "_version",    version)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"PhigrosChart is read-only; cannot set attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        json_path:  str,
        audio_path: str = "",
        version:    str = "",
    ) -> "PhigrosChart":
        """Load a Phigros JSON file and return a :class:`PhigrosChart`.

        Parameters
        ----------
        json_path:   Path to the ``.json`` chart file.
        audio_path:  Optional path to the accompanying audio file.
        version:     Optional difficulty label (``"EZ"``, ``"HD"``, ``"IN"``, ``"AT"``).
        """
        with open(json_path, encoding="utf-8") as fh:
            raw: Chart = json.load(fh)
        return cls(raw, json_path, audio_path, version)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def format_version(self) -> int:
        """Chart format version (always 3 for official charts)."""
        return self._raw["formatVersion"]

    @property
    def offset(self) -> float:
        """Audio playback delay in seconds."""
        return self._raw["offset"]

    @property
    def bpm(self) -> float:
        """BPM of the first judgment line (used as the representative BPM)."""
        return self._raw["judgeLineList"][0]["bpm"]

    @property
    def judge_lines(self) -> list[JudgeLine]:
        """All judgment lines in their original JSON order (notes nested inside)."""
        return self._raw["judgeLineList"]

    @property
    def json_path(self) -> str:
        """Absolute path to the source JSON file."""
        return self._json_path

    @property
    def audio_path(self) -> str:
        """Path to the accompanying audio file."""
        return self._audio_path

    @property
    def version(self) -> str:
        """Difficulty label string."""
        return self._version

    @property
    def name(self) -> str:
        """Stem of the JSON filename, e.g. ``'9752727302241212'``."""
        return os.path.splitext(os.path.basename(self._json_path))[0]

    @property
    def chart_dir(self) -> str:
        """Directory that contains the JSON file."""
        return os.path.dirname(self._json_path)

    @property
    def total_note_count(self) -> int:
        """Total number of notes across all judgment lines."""
        return sum(
            len(line["notesAbove"]) + len(line["notesBelow"])
            for line in self._raw["judgeLineList"]
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def for_batch(self) -> dict:
        """JSON-serialisable dict safe for DataLoader collation.

        Excludes raw judge-line data (not tensor-ready).
        """
        return {
            "json_path":  self._json_path,
            "audio_path": self._audio_path,
            "bpm":        self.bpm,
            "offset":     self.offset,
            "version":    self._version,
        }

    def __repr__(self) -> str:
        return (
            f"PhigrosChart(name={self.name!r}, bpm={self.bpm}, "
            f"lines={len(self.judge_lines)}, notes={self.total_note_count})"
        )


# ===========================================================================
# Section 5 – PhigrosFlatChart  (flattened 4-lane representation)
# ===========================================================================

@dataclass
class PhigrosFlatChart:
    """Flattened 4-lane snapshot produced by :meth:`Phigros4kConvertor.flatten`.

    Holds both the time-sorted flat note list and the encoded ``note_array``
    ready for model training.

    Attributes
    ----------
    flat_notes:  All notes sorted by onset time, each carrying its line BPM.
    note_array:  Float32 ndarray of shape ``(NUM_CHANNELS, max_frame)``.
    valid_flag:  Float32 ndarray of shape ``(max_frame,)``.
                 ``1`` for frames up to the last note onset; ``0`` after.
    bpm:         Representative BPM used for tick <-> frame conversion.
    offset:      Chart-level audio offset in seconds.
    json_path:   Source JSON file path.
    audio_path:  Accompanying audio file path.
    version:     Difficulty label string.
    """

    flat_notes:  list[FlatNote]
    note_array:  np.ndarray      # shape: (NUM_CHANNELS, max_frame)
    valid_flag:  np.ndarray      # shape: (max_frame,)
    bpm:         float
    offset:      float = 0.0
    json_path:   str   = ""
    audio_path:  str   = ""
    version:     str   = ""

    @property
    def note_count(self) -> int:
        """Number of notes in the flat list (before max_frame clipping)."""
        return len(self.flat_notes)

    @property
    def encoded_onset_count(self) -> int:
        """Number of onset frames actually encoded in ``note_array``."""
        return int(
            np.sum([self.note_array[k + CH_IS_START] for k in range(NUM_LANES)])
        )

    def for_batch(self) -> dict:
        """JSON-serialisable dict safe for DataLoader collation."""
        return {
            "json_path":  self.json_path,
            "audio_path": self.audio_path,
            "bpm":        self.bpm,
            "offset":     self.offset,
            "version":    self.version,
        }

    def __repr__(self) -> str:
        return (
            f"PhigrosFlatChart("
            f"notes={self.note_count}, "
            f"encoded={self.encoded_onset_count}, "
            f"array={self.note_array.shape})"
        )


# ===========================================================================
# Section 6 – Phigros4kConvertor
# ===========================================================================

class Phigros4kConvertor:
    """Converts between :class:`PhigrosChart`, :class:`PhigrosFlatChart`, and
    the fixed-schema ``note_array``, and serialises arrays to Phigros JSON.

    Parameters
    ----------
    frame_ms:
        Duration of one note-array frame in milliseconds.
        Canonical derivation: ``n_fft / sr / 4 * audio_note_window_ratio * 1000``.
        Default pipeline: ``512 / 22050 / 4 * 8 * 1000 approx 46.44 ms``.
    max_frame:
        Maximum number of frames along the time axis.
    mirror:
        If ``True``, flip lanes left <-> right (lane 0 <-> 3, lane 1 <-> 2).
    rate:
        Playback speed multiplier for data augmentation.
        Applied as: ``adjusted_ms = original_ms / rate + offset_ms``.
    offset_ms:
        Additional time shift in milliseconds added after rate adjustment.
    from_logits:
        If ``True``, decode binary channels with threshold ``0`` instead of
        ``0.5`` (for models that output raw logits before sigmoid).
    """

    def __init__(
        self,
        frame_ms:    float,
        max_frame:   int,
        mirror:      bool  = False,
        rate:        float = 1.0,
        offset_ms:   float = 0.0,
        from_logits: bool  = False,
    ) -> None:
        self.frame_ms    = frame_ms
        self.max_frame   = max_frame
        self.mirror      = mirror
        self.rate        = rate
        self.offset_ms   = offset_ms
        self.from_logits = from_logits

        self._lane_map: list[int] = list(range(NUM_LANES))
        if mirror:
            self._lane_map = [NUM_LANES - 1 - k for k in range(NUM_LANES)]

    # ------------------------------------------------------------------
    # Internal: timing
    # ------------------------------------------------------------------

    def _tick_to_frame(
        self, tick: float, bpm: float
    ) -> tuple[float, int, float]:
        """Convert a Phigros tick to a note-array frame index.

        Returns
        -------
        adjusted_ms:  Rate/offset-adjusted onset time in ms.
        frame_idx:    Integer frame index.
        sub_offset:   Fractional position within the frame, in [0, 1).
        """
        raw_ms = tick / 32.0 * (60_000.0 / bpm)
        adj_ms = raw_ms / self.rate + self.offset_ms
        frame  = int(adj_ms / self.frame_ms)
        sub    = adj_ms / self.frame_ms - frame
        return adj_ms, frame, sub

    def _frame_to_tick(
        self, frame_idx: int, sub_offset: float, bpm: float
    ) -> int:
        """Inverse of ``_tick_to_frame``: frame index -> Phigros tick."""
        adj_ms = (frame_idx + sub_offset) * self.frame_ms
        raw_ms = (adj_ms - self.offset_ms) * self.rate
        return int(round(raw_ms * 32.0 * bpm / 60_000.0))

    # ------------------------------------------------------------------
    # Internal: decode helpers
    # ------------------------------------------------------------------

    def _is_positive(self, v: float) -> bool:
        """Return True if *v* passes the binary decoding threshold."""
        return v > 0.0 if self.from_logits else v > 0.5

    @staticmethod
    def _lane_from_x(position_x: float) -> int:
        """Map a continuous positionX to a lane index 0-3."""
        return bisect.bisect_right(LANE_BOUNDARIES, position_x)

    @staticmethod
    def _decode_type(stored: float) -> int:
        """Map a stored normalised note_type value to a raw label 1-4."""
        for i, thr in enumerate(NOTE_TYPE_THRESHOLDS):
            if stored <= thr:
                return i + 1
        return NOTE_FLICK

    # ------------------------------------------------------------------
    # Internal: encode flat note list -> note_array
    # ------------------------------------------------------------------

    def _encode_flat_notes(
        self, flat: list[FlatNote]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a time-sorted flat note list into ``(note_array, valid_flag)``.

        Notes that exceed ``max_frame`` are skipped with a warning.
        When two notes collide in the same (lane, frame) slot, the later note
        in the sorted list wins and a warning is emitted; this is an inherent
        limitation of the 4-lane fixed-grid representation.
        """
        import warnings

        array      = np.zeros((NUM_CHANNELS, self.max_frame), dtype=np.float32)
        max_idx    = 0
        n_overflow  = 0
        n_collision = 0

        for fn in flat:
            note = fn.note
            bpm  = fn.bpm

            _, start_idx, start_off = self._tick_to_frame(note["time"], bpm)
            if start_idx >= self.max_frame:
                n_overflow += 1
                continue

            lane = self._lane_map[self._lane_from_x(note["positionX"])]

            # Detect same-lane same-frame collision
            if array[lane + CH_IS_START, start_idx] > 0.5:
                n_collision += 1

            # Onset channels
            array[lane + CH_IS_START,  start_idx] = 1.0
            array[lane + CH_START_OFF, start_idx] = float(start_off)
            array[lane + CH_NOTE_TYPE, start_idx] = NOTE_TYPE_STORED[note["type"]]
            max_idx = max(max_idx, start_idx)

            # Hold body / tail channels (vectorised slice, avoids Python loop)
            if note["type"] == NOTE_HOLD and note["holdTime"] > 0:
                end_tick            = note["time"] + note["holdTime"]
                _, end_idx, end_off = self._tick_to_frame(end_tick, bpm)
                end_idx             = min(end_idx, self.max_frame - 1)

                array[lane + CH_IS_HOLDING, start_idx + 1 : end_idx + 1] = 1.0
                array[lane + CH_END_OFFSET, end_idx] = float(end_off)
                max_idx = max(max_idx, end_idx)

        if n_overflow > 0:
            warnings.warn(
                f"{n_overflow} note(s) skipped: onset frame >= max_frame "
                f"({self.max_frame}).",
                RuntimeWarning, stacklevel=3,
            )
        if n_collision > 0:
            warnings.warn(
                f"{n_collision} note(s) lost to lane-frame collision "
                f"(multiple notes quantised to the same 4-lane slot). "
                f"This is expected when converting a multi-line Phigros chart "
                f"to a fixed 4-lane grid.",
                RuntimeWarning, stacklevel=3,
            )

        valid_flag           = np.zeros(self.max_frame, dtype=np.float32)
        valid_flag[:max_idx] = 1.0
        return array, valid_flag

    # ------------------------------------------------------------------
    # Public: flatten
    # ------------------------------------------------------------------

    def flatten(self, chart: PhigrosChart) -> PhigrosFlatChart:
        """Collapse a multi-line :class:`PhigrosChart` into a :class:`PhigrosFlatChart`.

        All notes from every judgment line are merged into a time-sorted flat
        list; ``note_array`` and ``valid_flag`` are computed in one pass.

        Parameters
        ----------
        chart:  Source :class:`PhigrosChart`.

        Returns
        -------
        :class:`PhigrosFlatChart` with ``note_array`` shape
        ``(NUM_CHANNELS, max_frame)``.
        """
        flat: list[FlatNote] = []
        for line_idx, line in enumerate(chart.judge_lines):
            bpm = line["bpm"]
            for note in line["notesAbove"]:
                flat.append(FlatNote(note=note, bpm=bpm, above=True,  line_idx=line_idx))
            for note in line["notesBelow"]:
                flat.append(FlatNote(note=note, bpm=bpm, above=False, line_idx=line_idx))
        flat.sort(key=lambda fn: fn.time_ms)

        note_array, valid_flag = self._encode_flat_notes(flat)

        return PhigrosFlatChart(
            flat_notes = flat,
            note_array = note_array,
            valid_flag = valid_flag,
            bpm        = chart.bpm,
            offset     = chart.offset,
            json_path  = chart.json_path,
            audio_path = chart.audio_path,
            version    = chart.version,
        )

    # ------------------------------------------------------------------
    # Public: chart_to_array
    # ------------------------------------------------------------------

    def chart_to_array(
        self, chart: PhigrosChart
    ) -> tuple[np.ndarray, np.ndarray]:
        """Directly convert a :class:`PhigrosChart` to ``(note_array, valid_flag)``.

        Equivalent to ``(fc.note_array, fc.valid_flag)`` from :meth:`flatten`
        but avoids materialising the :class:`PhigrosFlatChart` wrapper.

        Returns
        -------
        note_array:  Float32 ndarray of shape ``(NUM_CHANNELS, max_frame)``.
        valid_flag:  Float32 ndarray of shape ``(max_frame,)``.
        """
        flat: list[FlatNote] = []
        for line_idx, line in enumerate(chart.judge_lines):
            bpm = line["bpm"]
            for note in line["notesAbove"]:
                flat.append(FlatNote(note=note, bpm=bpm, above=True,  line_idx=line_idx))
            for note in line["notesBelow"]:
                flat.append(FlatNote(note=note, bpm=bpm, above=False, line_idx=line_idx))
        flat.sort(key=lambda fn: fn.time_ms)
        return self._encode_flat_notes(flat)

    # ------------------------------------------------------------------
    # Public: array_to_notes
    # ------------------------------------------------------------------

    def array_to_notes(
        self,
        note_array: np.ndarray,
        bpm:        float,
    ) -> list[Note]:
        """Decode a ``note_array`` back into a list of Phigros :class:`Note` dicts.

        Parameters
        ----------
        note_array:  Float32 ndarray of shape ``(NUM_CHANNELS, T)``.
        bpm:         BPM used for frame -> tick conversion.

        Returns
        -------
        Notes sorted by onset tick, ready for placement in a
        ``JudgeLine``'s ``notesAbove`` list.
        """
        notes: list[tuple[int, Note]] = []
        thr   = 0.0 if self.from_logits else 0.5

        for lane in range(NUM_LANES):
            mapped = self._lane_map[lane]

            is_start_ch   = note_array[mapped + CH_IS_START]
            start_off_ch  = note_array[mapped + CH_START_OFF]
            is_holding_ch = note_array[mapped + CH_IS_HOLDING]
            end_offset_ch = note_array[mapped + CH_END_OFFSET]
            note_type_ch  = note_array[mapped + CH_NOTE_TYPE]

            onset_frames = np.where(is_start_ch > thr)[0]

            for sidx in onset_frames:
                s_off      = float(np.clip(start_off_ch[sidx], 0.0, 1.0))
                start_tick = self._frame_to_tick(int(sidx), s_off, bpm)
                raw_type   = self._decode_type(float(note_type_ch[sidx]))
                hold_time  = 0

                if raw_type == NOTE_HOLD:
                    # Vectorised body scan: find contiguous is_holding region
                    # after the onset with no new onset.
                    tail      = is_holding_ch[int(sidx) + 1:]
                    no_onset  = is_start_ch[int(sidx) + 1:] <= thr
                    body_mask = (tail > thr) & no_onset
                    # argmin gives first False; concatenate a sentinel
                    body_len  = int(np.argmin(
                        np.concatenate([body_mask, [False]])
                    ))
                    eidx = int(sidx) + body_len  # inclusive tail frame

                    if eidx > int(sidx):
                        e_off     = float(np.clip(end_offset_ch[eidx], 0.0, 1.0))
                        end_tick  = self._frame_to_tick(eidx, e_off, bpm)
                        hold_time = max(0, end_tick - start_tick)

                # floorPosition base formula (speed_value = 1.0):
                #   floorPosition = tick * 60 / (32 * bpm) = time_seconds
                # save_phigros_file multiplies this by its speed_value parameter
                # so that currentFloor(t) = speed_value * t_seconds reaches
                # floorPosition exactly at judgment time.
                note: Note = {
                    "type":          raw_type,
                    "time":          start_tick,
                    "positionX":     float(LANE_CENTERS[lane]),
                    "holdTime":      float(hold_time),
                    "speed":         1.0,
                    "floorPosition": start_tick * 60.0 / (32.0 * bpm),
                }
                notes.append((start_tick, note))

        notes.sort(key=lambda x: x[0])
        return [n for _, n in notes]

    # ------------------------------------------------------------------
    # Public: save_phigros_file  (P1-6)
    # ------------------------------------------------------------------

    def save_phigros_file(
        self,
        note_array:     np.ndarray,
        bpm:            float,
        output_path:    str,
        offset:         float = 0.0,
        format_version: int   = 3,
        line_y:         float = 0.1,
        speed_value:    float = 3.0,
    ) -> None:
        """Serialise a ``note_array`` to a valid Phigros JSON file.

        The output contains a single fixed horizontal judgment line with all
        decoded notes placed in ``notesAbove`` (falling top-to-bottom).

        Parameters
        ----------
        note_array:      Float32 ndarray of shape ``(NUM_CHANNELS, T)``.
        bpm:             BPM written to the judgment line.
        output_path:     Destination ``.json`` file path (parent dirs created).
        offset:          Chart-level audio offset in seconds.
        format_version:  Phigros format version tag; always 3 for official format.
        line_y:          Vertical position of the judgment line in Phigros
                         screen coordinates (0=bottom, 1=top).  Default 0.1
                         places the line near the bottom with a small margin,
                         matching standard fixed-drop 4k layout.
        speed_value:     Visual scroll speed (``speedEvent.value``).  Controls
                         how far above the judgment line notes appear; higher
                         values give more spread-out note layouts.  Default 6.0.
                         ``floorPosition`` is scaled by the same factor to keep
                         timing exact:
                         ``floorPosition = speed_value * tick * 60 / (32 * bpm)``
        """
        notes     = self.array_to_notes(note_array, bpm)
        # Scale floorPosition and hold note.speed to match speed_value.
        #
        # For all notes:
        #   floorPosition = speed_value * time_seconds
        #   → currentFloor(t_onset) = speed_value * t_onset = floorPosition  ✓
        #
        # For hold notes additionally:
        #   The renderer computes tail floor as:
        #     tailFloor = headFloor + note.speed * holdTicks * 60 / (32 * bpm)
        #   We need tailFloor = speed_value * tailTimeSec, so:
        #     note.speed must equal speed_value (not 1.0)
        #   Non-hold notes have no body, so their note.speed has no visual effect.
        for n in notes:
            n["floorPosition"] = n["floorPosition"] * speed_value
            if n["holdTime"] > 0:
                n["speed"] = speed_value

        last_tick = max(
            (int(n["time"]) + int(n["holdTime"]) for n in notes), default=0
        )
        # All event times are in 1/32-beat tick units.
        # Use a large sentinel value so events cover the full chart.
        end_time = float(last_tick + 128)   # 4-beat tail margin
        inf_time = 1_000_000_000.0          # standard Phigros terminal sentinel

        judge_line: JudgeLine = {
            "bpm":        bpm,
            "notesAbove": notes,
            "notesBelow": [],
            # speedEvents: currentFloor(t) = speed_value * t_seconds.
            # floorPosition = speed_value * time_seconds, so the note reaches
            # the judgment line exactly at its onset time.
            "speedEvents": [
                {"startTime": 0.0,      "endTime": end_time, "value": speed_value},
                {"startTime": end_time, "endTime": inf_time,  "value": 0.0},
            ],
            # move/rotate/disappear events must start at -999999 so the renderer
            # has a defined initial state before the chart begins.
            "judgeLineMoveEvents": [
                {
                    "startTime": -999999.0, "endTime": 0.0,
                    "start":  0.5,  "end":  0.5,
                    "start2": line_y, "end2": line_y,
                },
                {
                    "startTime": 0.0,     "endTime": inf_time,
                    "start":  0.5,  "end":  0.5,
                    "start2": line_y, "end2": line_y,
                },
            ],
            "judgeLineRotateEvents": [
                {"startTime": -999999.0, "endTime": 0.0,      "start": 0.0, "end": 0.0},
                {"startTime": 0.0,       "endTime": inf_time, "start": 0.0, "end": 0.0},
            ],
            "judgeLineDisappearEvents": [
                {"startTime": -999999.0, "endTime": 0.0,      "start": 1.0, "end": 1.0},
                {"startTime": 0.0,       "endTime": inf_time, "start": 1.0, "end": 1.0},
            ],
        }

        chart: Chart = {
            "formatVersion": format_version,
            "offset":        offset,
            "judgeLineList": [judge_line],
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(chart, fh, ensure_ascii=False, indent=2)


# ===========================================================================
# Section 7 – parse_phigros_file
# ===========================================================================

def parse_phigros_file(
    json_path:  str,
    audio_path: str = "",
    version:    str = "",
) -> PhigrosChart:
    """Load a Phigros JSON file and return a read-only :class:`PhigrosChart`.

    Parameters
    ----------
    json_path:   Path to the ``.json`` chart file.
    audio_path:  Optional path to the accompanying audio file.
    version:     Optional difficulty label string.

    Returns
    -------
    :class:`PhigrosChart`
    """
    return PhigrosChart.from_json(json_path, audio_path=audio_path, version=version)


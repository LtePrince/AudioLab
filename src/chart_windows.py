"""Physical output windows. These bounds never mask the audio memory."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ChartWindow:
    index: int
    start: float
    end: float

    def contains(self, time: float) -> bool:
        return self.start <= time < self.end


def chart_windows(duration: float, width: float = 10.) -> list[ChartWindow]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('audio duration must be finite and positive')
    if not math.isfinite(width) or width <= 0:
        raise ValueError('window width must be finite and positive')
    return [ChartWindow(i, i*width, min((i+1)*width, duration))
            for i in range(math.ceil(duration/width))]


def partition_notes(notes: list[dict], duration: float, width: float = 10.) -> list[list[dict]]:
    """Assign by onset, preserving order, duplicates, coordinates and full Holds.

    Returned dictionaries are the original objects: no quantization, snapping,
    deduplication or change of note duration is performed by this protocol.
    """
    windows=chart_windows(duration,width)
    output=[[] for _ in windows]
    for note in notes:
        time=float(note['time'])
        if not math.isfinite(time) or not 0 <= time < duration:
            raise ValueError('note onset outside the physical audio duration')
        index=min(int(time//width),len(windows)-1)
        # Division may round up at an exact floating-point predecessor of an edge.
        while index and time < windows[index].start:index-=1
        while index+1<len(windows) and time >= windows[index].end:index+=1
        assert windows[index].contains(time)
        output[index].append(note)
    return output


def active_holds(past_notes: list[dict], window_start: float) -> list[dict]:
    """Past Hold events still active at a boundary, not new output events."""
    return [note for note in past_notes if note['type']==3
            and note['time'] < window_start < note['time']+note['duration']]

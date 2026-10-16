"""Optional keyboard contact projection, preserving music onset times."""
from itertools import combinations, permutations, groupby
import math

from src.data.contact_audit import (LANES, MIN_INTERVAL_S, RELEASE_S,
                                    shorten_to, changes_report)


def constrain_4k_contacts(notes, end_tick, bpm):
    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(end_tick):
        raise ValueError('finite endpoint and positive bpm required')
    if any(min(abs(n['positionX']-x) for x in LANES) > 1e-7 for n in notes):
        raise ValueError('expected four fixed lane anchors')
    sec = 60/(32*bpm)
    result, indices, contacts = [], [], []
    last = [None] * 4
    for tick, group in groupby(sorted(enumerate(notes), key=lambda p: p[1]['time']),
                               key=lambda p: p[1]['time']):
        chord = list(group)
        available = [h for h in range(4) if last[h] is None or
                     (tick-last[h]['time'])*sec >= MIN_INTERVAL_S-1e-9]
        size = min(len(chord), len(available))
        if not size:
            continue
        candidates = []
        for selected in combinations(chord, size):
            for hands in permutations(available, size):
                cost = sum(abs(LANES[h]-n['positionX']) for (_, n), h in zip(selected, hands))
                for h in hands:
                    prev = last[h]
                    if prev and prev['type'] == 3:
                        cost += .01*max(0, (prev['time']+prev['holdTime']-tick)*sec+RELEASE_S)
                candidates.append((cost, tuple(i for i, _ in selected), hands))
        _, selected, hands = min(candidates)
        for index, h in zip(selected, hands):
            if last[h] is not None:
                shorten_to(last[h], tick-RELEASE_S/sec)
            n = dict(notes[index]); n['positionX'] = LANES[h]
            shorten_to(n, end_tick)
            result.append(n); indices.append(index); contacts.append(h)
            last[h] = n
    return result, changes_report(notes, result, indices, contacts, bpm, '4k')

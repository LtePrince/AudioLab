"""Bounded contact assignment for an explicitly selected free-x configuration.

No universal two-finger limit: callers must declare the contact budget. A beam
search supplies a feasible witness, not an optimal layout or a hand biomechanics
model. Failure to keep a note does not prove that the source is unplayable.
"""
from itertools import groupby
import math

from src.data.chart_constraints import constrain_free_notes
from src.data.contact_audit import (MIN_INTERVAL_S, RELEASE_S, FREE_SPEED, FREE_GAP,
                                    shorten_to, changes_report)


def constrain_free_contacts(notes, end_tick, bpm, *, max_contacts):
    if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(end_tick):
        raise ValueError('finite endpoint and positive bpm required')
    if type(max_contacts) is not int or not 1 <= max_contacts <= 10:
        raise ValueError('max_contacts must be an integer in [1, 10]')
    sec = 60 / (32 * bpm)
    groups = [(t, list(g)) for t, g in groupby(sorted(enumerate(notes),
              key=lambda p: (p[1]['time'], p[1]['positionX'], p[0])), key=lambda p: p[1]['time'])]
    result, indices, contacts = [], [], []
    last = [None] * max_contacts
    for gi, (tick, chord) in enumerate(groups):
        t = tick * sec
        available = [h for h in range(max_contacts) if last[h] is None or
                     (tick-last[h]['time'])*sec >= MIN_INTERVAL_S-1e-9]
        # State: [(source index, contact, position, previous Hold limit)], cost.
        # Retain up to 32 alternatives per cardinality, bounding large chords.
        states = [([], 0.0)]
        for index, source in chord:
            desired = .5 + source['positionX']*.05625
            expanded = list(states)
            for chosen, cost in states:
                used = {p[1] for p in chosen}
                for h in available:
                    if h in used:
                        continue
                    prev = last[h]
                    lo, hi = .05, .95
                    if prev:
                        px = .5 + prev['positionX']*.05625
                        radius = max(0, (tick-prev['time'])*sec-RELEASE_S)*FREE_SPEED
                        lo, hi = max(lo, px-radius), min(hi, px+radius)
                    if chosen:
                        lo = max(lo, chosen[-1][2]+FREE_GAP)
                    if lo > hi+1e-9:
                        continue
                    x = min(hi, max(lo, desired))
                    limit, lost_hold = tick, 0.0
                    if prev:
                        travel = abs(x-(.5+prev['positionX']*.05625))/FREE_SPEED
                        limit = tick-(RELEASE_S+travel)/sec
                        if prev['type'] == 3:
                            lost_hold = max(0, prev['time']+prev['holdTime']-limit)*sec
                    expanded.append((chosen+[(index, h, x, limit)],
                                     cost+(x-desired)**2+.005*lost_hold))
            buckets = {}
            for state in expanded:
                buckets.setdefault(len(state[0]), []).append(state)
            states = [s for bucket in buckets.values()
                      for s in sorted(bucket, key=lambda s: (s[1], s[0]))[:32]]

        def rank(state):
            chosen, cost = state
            ready = [(n['time']*sec+MIN_INTERVAL_S if n else -math.inf) for n in last]
            for _, h, _, _ in chosen:
                ready[h] = t+MIN_INTERVAL_S
            missed = 0
            for future, _ in groups[gi+1:gi+max_contacts+1]:
                ft = future*sec
                h = min(range(max_contacts), key=lambda h: ready[h])
                if ready[h] <= ft+1e-9:
                    ready[h] = ft+MIN_INTERVAL_S
                else:
                    missed += 1
            return missed, -len(chosen), cost, chosen

        candidates = [s for s in states if s[0]]
        if not candidates:
            continue
        chosen, _ = min(candidates, key=rank)
        for index, h, x, limit in chosen:
            if last[h] is not None:
                shorten_to(last[h], limit)
            n = dict(notes[index])
            n['positionX'] = (x-.5)/.05625
            shorten_to(n, end_tick)
            result.append(n); indices.append(index); contacts.append(h)
            last[h] = n
    # Extra spatial Hold clearance only shortens durations. The beam already
    # enforces chord spacing, so this position projection must be an identity.
    result, _ = constrain_free_notes(result, end_tick, FREE_GAP)
    return result, changes_report(notes, result, indices, contacts, bpm, 'freex',
                                  max_contacts=max_contacts)

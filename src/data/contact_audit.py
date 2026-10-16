"""Independent verification of a proposed static-chart contact assignment.

An assignment is an executable witness under a declared engineering profile,
not proof of human comfort. All note types conservatively require one contact.
"""
from __future__ import annotations

import math

MIN_INTERVAL_S = 0.080
RELEASE_S = 0.020
FREE_SPEED = 4.0
FREE_GAP = 0.04
LANES = (-3.75, -1.25, 1.25, 3.75)


def audit_contacts(notes, contacts, bpm, mode, *, max_contacts=None):
    if mode not in ('4k', 'freex') or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('invalid mode or bpm')
    if len(notes) != len(contacts):
        raise ValueError('one contact required for every note')
    count = 4 if mode == '4k' else max_contacts
    if type(count) is not int or not 1 <= count <= 10:
        raise ValueError('free-x requires an explicit max_contacts in [1, 10]')
    sec = 60 / (32 * bpm)
    result = dict(invalid_contacts=0, repeat_violations=0, release_violations=0,
                  movement_violations=0, chord_gap_violations=0)
    previous = {}
    chords = {}
    for n, hand in sorted(zip(notes, contacts), key=lambda pair: pair[0]['time']):
        if type(hand) is not int or not 0 <= hand < count:
            result['invalid_contacts'] += 1
            continue
        t, x = n['time'] * sec, .5 + n['positionX'] * .05625
        if mode == '4k' and abs(n['positionX'] - LANES[hand]) > 1e-7:
            result['invalid_contacts'] += 1
        if hand in previous:
            pt, px, end = previous[hand]
            result['repeat_violations'] += t - pt < MIN_INTERVAL_S - 1e-7
            room = t - end - RELEASE_S
            result['release_violations'] += room < -1e-7
            if mode == 'freex':
                result['movement_violations'] += abs(x - px) > max(0, room) * FREE_SPEED + 1e-7
        previous[hand] = (t, x, t + (n['holdTime'] * sec if n['type'] == 3 else 0))
        chords.setdefault(n['time'], []).append(x)
    if mode == 'freex':
        for xs in chords.values():
            xs.sort()
            result['chord_gap_violations'] += sum(b-a < FREE_GAP-1e-7 for a, b in zip(xs, xs[1:]))
    return result


def shorten_to(note, end_tick):
    """Release an occupied contact; never extend a Hold or change its onset."""
    if note['type'] != 3:
        return
    duration = max(0, min(note['holdTime'], math.floor(end_tick - note['time'] + 1e-7)))
    note['holdTime'] = duration
    if duration < 1:
        note['type'], note['speed'] = 1, 1.0


def changes_report(original, result, source_indices, contacts, bpm, mode, *, max_contacts=None):
    kept = set(source_indices)
    starts, remaining = {n['time'] for n in original}, {n['time'] for n in result}
    moved = shortened = converted = 0
    displacement = 0.0
    for note, index in zip(result, source_indices):
        source = original[index]
        delta = abs(note['positionX'] - source['positionX']) * .05625
        moved += delta > 1e-9
        displacement = max(displacement, delta)
        shortened += source['type'] == 3 and note['holdTime'] < source['holdTime']
        converted += source['type'] == 3 and note['type'] != 3
        if note['time'] != source['time']:
            raise AssertionError('contact projection must not shift onsets')
    audit = audit_contacts(result, contacts, bpm, mode, max_contacts=max_contacts)
    if any(audit.values()):
        raise AssertionError(f'contact projection produced invalid witness: {audit}')
    return dict(profile=mode + '-contacts-v1', max_contacts=4 if mode == '4k' else max_contacts,
                min_interval_s=MIN_INTERVAL_S, release_s=RELEASE_S,
                max_speed_screen_per_s=FREE_SPEED if mode == 'freex' else None,
                input_notes=len(original), output_notes=len(result),
                dropped_notes=len(original)-len(kept), dropped_source_indices=sorted(set(range(len(original)))-kept),
                source_indices=source_indices, contacts=contacts, moved_notes=moved,
                max_displacement=displacement, shortened_holds=shortened, holds_to_taps=converted,
                unique_onset_retention=len(remaining)/len(starts) if starts else 1.0,
                dropped_onsets=len(starts-remaining), audit=audit)

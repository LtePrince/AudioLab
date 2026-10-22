"""Behavioral counterexamples prevent benchmark gaming and clock regressions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.model_benchmark import evaluate, pattern_metrics, rhythm_bonus, score_case
from script.benchmark_models import decode_notes

PROTOCOL = json.loads((Path(__file__).parents[1]/'benchmarks/v1/protocol.json').read_text())
CLOCK = dict(bpm=120, beat_zero_seconds=0., source='construction', verified=True)


def case(ref=(1., 4., 7.)):
    return dict(id='piano_three', source_group='probe', label_source='constructed_acoustic_probe',
                split='constructed_development', events_seconds=list(ref), duration_seconds=10.,
                coverage=[[0., 10.]], contract='exact_onsets', clock=CLOCK)


def score(times, reference=(1., 4., 7.)):
    return score_case(case(reference), dict(status='completed', onsets_seconds=times), PROTOCOL)


def notes(positions, times=None, kinds=None):
    times = list(range(len(positions))) if times is None else times
    kinds = ['tap']*len(positions) if kinds is None else kinds
    return [dict(time_seconds=t, end_seconds=t+(1 if k == 'hold' else 0), type=k,
                 position=p, screen_x=.1+p*.1) for p, t, k in zip(positions, times, kinds)]


def test_phase_bonus_dedup_boundary_and_no_order_constraint():
    times = [0, 4/64, 8/64, 16/64, 24/64, 32/64, 32/64]
    r = rhythm_bonus(times, CLOCK, PROTOCOL['rhythm'])
    assert r['unique_onsets'] == r['matched_onsets'] == 6
    assert r['bonus_per_onset'] == 1
    assert r['matched_phase_counts']['0'] == 2  # 32 is the next beat's 0.
    assert rhythm_bonus([.5, 1., 2.], CLOCK, PROTOCOL['rhythm'])['bonus_per_onset'] == 1


def test_partial_bonus_never_negative_and_empty_not_perfect():
    r = rhythm_bonus([4.5/64, 12/64], CLOCK, PROTOCOL['rhythm'])
    assert r['bonus_points'] == pytest.approx(.5, abs=1e-4)
    assert r['bonus_per_onset'] == pytest.approx(.25, abs=1e-4)
    assert rhythm_bonus([], CLOCK, PROTOCOL['rhythm'])['bonus_per_onset'] is None
    assert rhythm_bonus([1], None, PROTOCOL['rhythm'])['status'] == 'unavailable'


def test_clock_offset_and_no_fitting_to_predictions():
    clock = dict(CLOCK, beat_zero_seconds=.123)
    assert rhythm_bonus([1.123], clock, PROTOCOL['rhythm'])['bonus_per_onset'] == 1
    assert rhythm_bonus([1.123+2/64], clock, PROTOCOL['rhythm'])['bonus_points'] == 0


def test_regular_spam_cannot_pass_positive_contract():
    r = score([x/2 for x in range(20)])
    assert r['rhythm']['bonus_per_onset'] == 1
    assert r['events']['at_50ms']['recall'] == 1
    assert r['events']['at_50ms']['precision'] == .15
    assert r['contract_status'] == 'fail'


def test_valid_off_profile_rhythm_is_not_a_failure():
    t = [12/64, 20/64, 28/64]
    r = score(t, t)
    assert r['rhythm']['bonus_points'] == 0
    assert r['contract_status'] == 'pass'


def test_empty_output_fails_positive_and_passes_silence():
    assert score([])['contract_status'] == 'fail'
    c = case([]); c['contract'] = 'silence'
    assert score_case(c, dict(status='completed', onsets_seconds=[]), PROTOCOL)['contract_status'] == 'pass'


def test_matching_is_one_to_one_and_outside_predictions_still_cost_precision():
    r = score([.99, 1.01, 4., 7., 11.])
    assert r['events']['at_50ms']['matched'] == 3
    assert r['events']['at_50ms']['precision'] == .6
    assert r['out_of_audio_onsets'] == [11.]


def test_unreviewed_is_not_silence_or_false_positive():
    c = case([1.]); c['coverage'] = [[0., 2.]]
    r = score_case(c, dict(status='completed', onsets_seconds=[1., 5.]), PROTOCOL)
    assert r['events']['ignored_predictions'] == 1
    assert r['contract_status'] == 'pass'


def test_weak_event_recall_and_tail_misses_are_explicit():
    c = case(); c['weak_events_seconds'] = [4.]
    r = score_case(c, dict(status='completed', onsets_seconds=[1.]), PROTOCOL)
    assert r['weak_recall_50ms'] == 0
    assert r['coverage']['last_quarter_reference_recall_50ms'] == 0


@pytest.mark.parametrize('positions,label', [([1,1,1,1], 'AAAA'), ([1,2,1,2], 'ABAB'),
                                            ([0,1,2,3], 'monotonic_run')])
def test_patterns_require_history(positions, label):
    assert pattern_metrics(notes(positions))['four_onset_patterns'] == {label: 1}


def test_chords_break_chains_and_four_types_have_no_quota():
    ns = notes([0,1,2,3,0], [0,0,1,2,3], ['tap','drag','hold','flick','tap'])
    r = pattern_metrics(ns)
    assert r['four_onset_patterns'] == {}
    assert r['chord_interrupted_windows'] == 1
    assert set(r['type_counts']) == {'tap','drag','hold','flick'}
    assert pattern_metrics(notes([0,1,2,3], [0,0,0,0]))['chord_sizes'] == {'4': 1}


def test_hold_touch_overlap_and_jack_runs():
    assert pattern_metrics(notes([0,0], [0,1], ['hold','hold']))['hold_touching_pairs'] == 1
    assert pattern_metrics(notes([0,0], [0,.5], ['hold','hold']))['hold_overlap_pairs'] == 1
    assert pattern_metrics(notes([0,0,0,0,0,0]))['longest_jack_run'] == 6


def test_missing_generation_is_in_denominator_and_input_is_not_mutated():
    c = case(); before = deepcopy(c)
    report = evaluate(dict(schema=PROTOCOL['schema'], runs=[dict(id='x', metadata={}, cases=[c], predictions={})]), PROTOCOL)
    r = report['runs'][0]
    assert r['summary'][0]['expected_cases'] == r['summary'][0]['missing_or_failed'] == 1
    assert r['contract_status'] == 'incomplete'
    assert c == before


def test_free_coordinate_residual_roundtrip_and_hold_duration():
    from src.data.chart_tokenizer import ChartTokenizer, TokenNote, bin_to_hit
    tk = ChartTokenizer(64)
    tokens = tk.encode_notes([TokenNote(32, 10, 3, 16)])
    idx = next(i for i, token in enumerate(tokens) if tk.kind(token) == 'NOTE')
    row = decode_notes(dict(tokens=tokens, residuals={str(idx): .2}), 'freex', 120, .1)[0]
    assert row['time_seconds'] == .6
    assert row['end_seconds'] == .85
    assert row['screen_x'] == bin_to_hit(10, .2)


def test_note_onset_inconsistency_is_not_silently_scored():
    with pytest.raises(ValueError, match='disagree'):
        score_case(case(), dict(status='completed', onsets_seconds=[], notes=notes([0])), PROTOCOL)


def test_modes_validate_geometry_without_imposing_two_fingers():
    from src.pattern_diagnostics import LANE_X
    ns = notes([0,1,2,3], [1,1,1,1])
    for n, x in zip(ns, LANE_X):
        n['screen_x'] = x
    pred = dict(status='completed', onsets_seconds=[1], notes=ns)
    assert score_case(case([1]), pred, PROTOCOL, '4k')['contract_status'] == 'pass'
    ns[0]['screen_x'] = .5
    assert score_case(case([1]), pred, PROTOCOL, '4k')['contract_status'] == 'fail'


def test_silence_false_positive_and_explicit_pattern_target():
    c = case([1,2,3,4]); c['silent_regions'] = [[5,10]]
    c['pattern_targets'] = [dict(start=1, end=5, label='ABAB')]
    ns = notes([0,1,0,1,2], [1,2,3,4,6])
    r = score_case(c, dict(status='completed', onsets_seconds=[1,2,3,4,6], notes=ns), PROTOCOL)
    assert not r['gates']['reviewed_silence_no_events']
    assert r['pattern_targets'][0]['matched_windows'] == 1


def test_overlapping_duplicate_source_cannot_inflate_macro():
    c = case(); other = dict(c, id='same-song-second-difficulty')
    with pytest.raises(ValueError, match='one case per source'):
        evaluate(dict(schema=PROTOCOL['schema'], runs=[dict(id='x', metadata={}, cases=[c, other], predictions={})]), PROTOCOL)

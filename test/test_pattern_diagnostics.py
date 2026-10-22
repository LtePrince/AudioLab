"""Hand-calculated descriptive fixtures, not human musical-quality labels."""
from copy import deepcopy
import json

import pytest

from src.pattern_diagnostics import (BIN_WIDTH, LANE_X, candidate_coverage, centered_residual,
                                     chart_events, compare, distribution, geometry,
                                     label_pattern, patterns, summarize)


def events(positions, times=None, offsets=None, mode='freex'):
    times = list(range(len(positions))) if times is None else times
    offsets = [.5]*len(positions) if offsets is None else offsets
    return [dict(id=f'n-{i}', source_index=i, tick=t*64, time_seconds=t, end_seconds=t,
                 screen_x=LANE_X[p] if mode == '4k' else .05+(p+o)*BIN_WIDTH,
                 position=p, type=('tap', 'drag', 'hold', 'flick')[i % 4])
            for i, (p, t, o) in enumerate(zip(positions, times, offsets))]


@pytest.mark.parametrize('positions,label', [([1,1,1,1], 'AAAA'), ([1,2,1,2], 'ABAB'),
    ([1,2,4,7], 'monotonic_run'), ([7,4,2,1], 'monotonic_run'), ([1,2,2,1], 'unclassified')])
def test_fixed_discrete_classes_and_nonuniform_rhythm(positions, label):
    result = patterns(events(positions, times=[0,.25,.75,1.5]))
    assert result['counts'] == {label: 1}
    row = result['instances'][0]
    assert row['ioi_seconds'] == [.25,.5,.75]
    assert row['ioi_cv'] == pytest.approx((1/24)**.5/.5)


def test_chord_interrupts_global_four_onsets_instead_of_skipping():
    values = events([1,1,2,1,1,1], times=[0,1,1,2,3,4])
    result = patterns(values)
    assert result['total_four_onset_windows'] == 2
    assert result['singleton_windows'] == 0
    assert result['chord_interrupted_windows'] == 2


def test_known_role_deviation_and_residual_counterfactual_preserves_discrete_structure():
    original = events([5,10,5,10], offsets=[.1,.2,.9,.4])
    before = deepcopy(original)
    actual = patterns(original)['instances'][0]
    centered = centered_residual(original)
    synthetic = patterns(centered)['instances'][0]
    assert actual['geometry']['same_role_deviation_screen_width'] == pytest.approx(.8*BIN_WIDTH)
    assert synthetic['geometry']['same_role_deviation_screen_width'] == 0
    assert actual['positions'] == synthetic['positions'] and actual['label'] == synthetic['label'] == 'ABAB'
    assert original == before
    for a, b in zip(original, centered):
        assert {k:v for k,v in a.items() if k!='screen_x'} == {k:v for k,v in b.items() if k!='screen_x'}
        assert abs(a['screen_x']-b['screen_x']) <= BIN_WIDTH/2


def test_chord_geometry_is_descriptive_with_four_fingers_and_four_types():
    result = summarize(events([0,1,2,3], times=[0,0,0,0], mode='4k'))
    assert result['chord_size_counts'] == {'4':1}
    chord = result['chords'][0]
    assert chord['gaps_screen_width'] == [.140625]*3
    assert chord['span_screen_width'] == .421875
    assert chord['gap_std_screen_width'] == 0
    assert set(result['type_counts']) == {'tap','drag','hold','flick'}


def test_raw_ids_survive_reordering_and_class_change_zero_denominator():
    raw = events([1,2,3,4])
    final = deepcopy(raw)
    for e in final: e.update(screen_x=.5, position=32)
    final.reverse()
    result = compare(raw, final)
    assert result['matched_ids'] == 4 and result['pattern_class_changes'] == 1
    row = result['raw_pattern_instances'][0]
    assert row['label_before'] == 'monotonic_run' and row['label_after'] == 'AAAA'
    assert row['geometry_after']['step_cv'] is None


def test_chord_delta_follows_ids_and_hold_duration_separately():
    raw = events([5,8,5], times=[0,0,.5])
    raw[0].update(type='hold', end_seconds=1.)
    final = deepcopy(raw)
    final[1]['screen_x'] += .01
    final[0]['end_seconds'] = .5
    result = compare(raw, final)
    assert result['moved_notes'] == result['shortened_notes'] == 1
    assert result['removed_duration_seconds'] == .5
    assert len(result['changed_chord_geometry']) == 1
    assert summarize(raw)['hold_summary']['nearby_later_pairs'] == 1
    assert summarize(final)['hold_summary']['nearby_later_pairs'] == 0


@pytest.mark.parametrize('fault', ['onset','id','count'])
def test_compare_rejects_undeclared_identity_or_time_changes(fault):
    raw=events([0,1,2,3]); final=deepcopy(raw)
    if fault=='onset':final[0]['time_seconds']+=.1
    elif fault=='id':final[0]['id']='wrong'
    else:final.pop()
    with pytest.raises(ValueError):compare(raw,final)


def test_offset_and_different_bpm_are_compared_in_audio_seconds():
    def chart(bpm, tick):
        return dict(formatVersion=3,offset=.25,judgeLineList=[dict(bpm=bpm,notesBelow=[],notesAbove=[
            dict(type=3,time=tick,holdTime=tick,positionX=-3.75)])])
    a=chart_events(chart(120,32),'4k')[0]
    b=chart_events(chart(240,64),'4k')[0]
    assert a['tick'] != b['tick']
    assert a['time_seconds'] == b['time_seconds'] == .75
    assert a['end_seconds'] == b['end_seconds'] == 1.25


def test_candidate_missing_extra_and_chords_have_explicit_denominators():
    values=events([0,1,2],times=[0,0,2])
    result=candidate_coverage(values,[0,64,64],[.7,.8,.9])
    assert result['unique_candidate_ticks']==2
    assert result['missing_ticks']==[64] and result['extra_ticks']==[128]
    assert result['notes_per_candidate']=={'2':1,'0':1}
    assert result['selected_confidence']['count']==3


def test_empty_denominators_are_null_and_serialize():
    result=summarize([])
    assert result['patterns']['counts']=={}
    assert distribution([])==dict(count=0,mean=None,median=None,min=None,max=None)
    assert result['pattern_metrics']['AAAA']['ioi_cv']['mean'] is None
    assert json.loads(json.dumps(result,allow_nan=False))==result


def report_fixture():
    from script import diagnose_patterns as script
    values=events([1,2,1,2]); stats=script.compact(summarize(values))
    coverage=candidate_coverage(values,[e['tick'] for e in values],[.8]*4)
    return dict(schema=script.SCHEMA,status='completed',expected_rows=1,completed_rows=1,failed_rows=0,
                results=[dict(status='completed',raw=stats,final=deepcopy(stats),
                              repair=script.compact_comparison(compare(values,values)),
                              coverage=dict(raw=coverage,final=deepcopy(coverage)))],resources=dict(output_bytes=0))


@pytest.mark.parametrize('fault',['pattern','identity','candidate','chord'])
def test_report_checks_denominators(fault):
    from script import diagnose_patterns as script
    report=report_fixture();row=report['results'][0]
    if fault=='pattern':row['raw']['patterns']['singleton_windows']+=1
    elif fault=='identity':row['repair']['matched_ids']-=1
    elif fault=='candidate':row['coverage']['raw']['unique_candidate_ticks']+=1
    else:row['raw']['chord_size_counts']['3']=2
    with pytest.raises(ValueError):script.validate_report(report)


def test_real_report_readback_counts_own_bytes(tmp_path,monkeypatch):
    from script import diagnose_patterns as script
    report=report_fixture()
    monkeypatch.setattr(script,'resources',lambda folder:dict(output_bytes=sum(p.stat().st_size for p in folder.iterdir())))
    script.write_report(tmp_path,report)
    path=tmp_path/'report.json'
    assert json.loads(path.read_text())==report
    assert report['resources']['output_bytes']==path.stat().st_size


@pytest.mark.parametrize('fault',['resource','readback'])
def test_final_failure_removes_completed_report(tmp_path,monkeypatch,fault):
    from script import diagnose_patterns as script
    def refuse(_):raise RuntimeError('fixture failure')
    monkeypatch.setattr(script,'resources' if fault=='resource' else 'read_json',refuse)
    with pytest.raises(RuntimeError,match='fixture failure'):script.write_report(tmp_path,report_fixture())
    assert not (tmp_path/'report.json').exists()


def posterior_fixture(tmp_path):
    import numpy as np
    from src.audio_understanding.inference import make_result
    arrays=dict(event_probability=np.array([.6,.8,.9,.75,.95],dtype=np.float32),
                event_times_seconds=np.array([0,.1,.12,.5,1.1]),
                valid_event_time=np.array([True,True,True,True,False]))
    path=tmp_path/'posterior.npz'
    np.savez_compressed(path,**arrays,features=np.zeros((100,256)))
    result=make_result(arrays,1.,dict(label_source='chart_onset'),threshold=.7,min_interval_ms=35)
    from script.diagnose_patterns import digest
    result.update(dense_file=path.name,dense_sha256=digest(path))
    return path,result


def test_saved_frame_selector_reproduces_threshold_and_nms_without_features(tmp_path,monkeypatch):
    from script import diagnose_patterns as script
    path,evidence=posterior_fixture(tmp_path)
    from numpy.lib.npyio import NpzFile
    original=NpzFile.__getitem__
    def small_members_only(obj,key):
        assert key!='features'
        return original(obj,key)
    monkeypatch.setattr(NpzFile,'__getitem__',small_members_only)
    result=script.selection_replay(tmp_path,evidence,evidence['dense_sha256'])
    assert result['valid_frames']==4 and result['invalid_tail_frames']==1
    assert result['below_threshold_frames']==1
    assert result['at_or_above_threshold_frames']==3
    assert result['selected_events']==2 and result['nms_or_duplicate_time_removed']==1


@pytest.mark.parametrize('fault',['hash','selection','traversal'])
def test_saved_posterior_integrity_and_selector_drift_refused(tmp_path,fault):
    from script import diagnose_patterns as script
    path,evidence=posterior_fixture(tmp_path)
    if fault=='hash':evidence['dense_sha256']='0'*64
    elif fault=='selection':evidence['events'][0]['confidence']=.123
    else:evidence['dense_file']='../posterior.npz'
    with pytest.raises(ValueError):script.selection_replay(tmp_path,evidence,script.digest(path))

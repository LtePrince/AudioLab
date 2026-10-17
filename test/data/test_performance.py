import copy
import math

import pytest

from src.data.performance import PerformanceChart, Track, unpack_v1
from src.data.performance_reference import PhiraSourceReference


def scalar(start=0, end=320, before=0, after=0):
    return dict(startTime=start, endTime=end, start=before, end=after)


def move(start=0, end=320, x=.5, y=.5, end_x=None, end_y=None):
    return dict(startTime=start, endTime=end, start=x, end=x if end_x is None else end_x,
                start2=y, end2=y if end_y is None else end_y)


def note(time=32, kind=1, duration=0, position=0, speed=1):
    return dict(type=kind, time=time, positionX=position, holdTime=duration, speed=speed, floorPosition=time/64)


def line(bpm=120, notes=None):
    return dict(bpm=bpm, notesAbove=[note()] if notes is None else notes, notesBelow=[],
                judgeLineMoveEvents=[move()], judgeLineRotateEvents=[scalar()],
                judgeLineDisappearEvents=[scalar(before=1, after=1)],
                speedEvents=[dict(startTime=0, endTime=320, value=1)])


def chart(lines=None, offset=0, version=3, duration=5):
    return PerformanceChart(dict(formatVersion=version, offset=offset, judgeLineList=lines or [line()]), duration)


def test_different_native_bpm_and_offset_are_never_integer_quantized():
    result=chart([line(120,[note(32.125)]),line(240,[note(64.25)])],offset=.25)
    assert result.notes[0]['time_seconds']==result.notes[1]['time_seconds']==.751953125
    assert result.notes[0]['native_time']==32.125
    assert chart([line(120,[note(32)]),line(240,[note(64)])],offset=.25).notes[1]['time_seconds']==.75
    for index in (0,1):
        assert result.line_state(f'line-{index}',.25)['native_tick']==0


def test_native_v1_signed_remainder_and_v3_equivalent_pose():
    assert unpack_v1(-1001)==pytest.approx((-1/880,-1/520))
    assert unpack_v1(-1000)==pytest.approx((-1/880,0))
    assert unpack_v1(-.5)==pytest.approx((0,-.5/520))
    original=line();original['judgeLineMoveEvents']=[scalar(before=440260,after=440260)]
    legacy=chart([original],version=1)
    assert legacy.line_state('line-0',1)['position']==chart().line_state('line-0',1)['position']


def test_full_turn_and_hold_quarter_point_are_not_lost():
    source=line(notes=[note(0,3,128,2)]);source['judgeLineMoveEvents']=[move(y=.9)]
    source['judgeLineRotateEvents']=[scalar(0,128,0,360)]
    result=chart([source]);identifier=result.notes[0]['id']
    assert result.line_state('line-0',1)['rotation_degrees']==180
    assert result.line_state('line-0',2)['rotation_degrees']==360
    for time in (0,1,2):assert result.note_geometry(identifier,time)['in_viewport'] is True
    quarter=result.note_geometry(identifier,.5)
    assert quarter['target_world']==pytest.approx([.5,1.1])
    assert quarter['in_viewport'] is False
    trajectory=result.sample_hold(identifier)
    assert any(sample['in_viewport'] is False for sample in trajectory['samples'])
    assert trajectory['continuous_safety_proof'] is False
    assert max(b['audio_seconds']-a['audio_seconds'] for a,b in zip(trajectory['samples'],trajectory['samples'][1:]))<=.25


def test_negative_speed_integral_and_raw_alpha_are_preserved():
    source=line();source['speedEvents']=[dict(startTime=0,endTime=64,value=2),dict(startTime=64,endTime=192,value=-1)]
    source['judgeLineDisappearEvents']=[scalar(before=-1,after=5)]
    result=chart([source])
    assert [result.line_state('line-0',time)['scroll_integral']['value'] for time in (0,1,2,3)]==[0,2,1,0]
    assert result.line_state('line-0',0)['alpha_source']==-1
    assert result.line_state('line-0',0)['alpha_nonnegative']==0
    assert result.line_state('line-0',5)['alpha_nonnegative']==5


def test_shared_boundary_right_span_and_zero_span_policy():
    track=Track('rotate',[scalar(0,10,0,10),scalar(10,20,100,120)],3,'r')
    assert track.query(10)['value']==[100]
    assert track.query(20)['value']==[120]
    assert track.query(21)['status']=='outer_endpoint_hold'
    instant=Track('rotate',[scalar(0,20,0,20),scalar(10,10,99,88)],3,'r')
    assert instant.query(10)['value']==[88]
    assert instant.query(10.1)['value']==pytest.approx([10.1])
    boundary=Track('rotate',[scalar(10,10,99,88),scalar(10,20,100,120)],3,'r')
    assert boundary.query(10)['value']==[100]


def test_float_audio_round_trip_recovers_native_boundary_ownership():
    source=line(bpm=137.3)
    source['judgeLineRotateEvents']=[scalar(0,123.456,0,0),scalar(123.456,320,90,90)]
    result=chart([source],offset=.123)
    seconds=.123+123.456*60/(32*137.3)
    assert result.line_state('line-0',seconds)['rotation_degrees']==90
    assert result.line_state('line-0',seconds-1e-7)['rotation_degrees']==0


@pytest.mark.parametrize('events,time,status',[
    ([scalar(0,10),scalar(20,30)],15,'unsupported_interior_gap'),
    ([scalar(0,20),scalar(10,30)],15,'unsupported_overlap'),
    ([scalar(20,10)],15,'unsupported_reversed_interval'),
    ([],0,'unsupported_empty_track'),
    ([scalar(0,10,math.inf,1)],1,'unsupported_malformed_track'),
])
def test_unsupported_intervals_are_explicit_not_silently_repaired(events,time,status):
    state=Track('rotate',events,3,'r').query(time)
    assert state['status']==status and state['value'] is None


def test_overlap_end_returns_to_earlier_long_segment():
    track=Track('rotate',[scalar(0,100,0,100),scalar(10,20,30,30)],3,'r')
    assert track.query(15)['value'] is None
    assert track.query(25)['value']==[25]


def test_speed_gap_does_not_create_a_known_floor_position():
    track=Track('speed',[dict(startTime=0,endTime=10,value=1),dict(startTime=20,endTime=30,value=2)],3,'s')
    assert track.integral(0,30,1)['value'] is None
    assert track.integral(0,5,1)['value']==5


def test_supported_state_includes_requested_integral_not_only_current_speed():
    source=line();source['speedEvents']=[dict(startTime=0,endTime=64,value=1),dict(startTime=128,endTime=320,value=2)]
    result=chart([source]);state=result.line_state('line-0',3)
    assert state['current_state_supported'] is True
    assert state['scroll_supported'] is False and state['supported'] is False
    assert result.line_state('line-0',3,include_scroll=False)['supported'] is True


def test_hold_sampling_contains_translation_boundaries_and_both_sides_of_jumps():
    source=line(notes=[note(0,3,192)])
    source['judgeLineMoveEvents']=[move(0,70,x=.2),move(70,192,x=.8)]
    result=chart([source]);samples=result.sample_hold(result.notes[0]['id'])['samples']
    assert any(sample['audio_seconds']==70/64 for sample in samples)
    left=min(samples,key=lambda sample:abs(sample['audio_seconds']-(70/64-1e-7)))
    right=min(samples,key=lambda sample:abs(sample['audio_seconds']-70/64))
    assert left['target_world'][0]==.2 and right['target_world'][0]==.8


def test_last_long_hold_keeps_semantics_after_last_note_head_plus_one():
    result=chart([line(notes=[note(0,3,256)])])
    assert result.notes[0]['end_seconds']==4
    assert result.line_state('line-0',4)['scroll_integral']['value']==4
    assert result.sample_hold(result.notes[0]['id'])['samples'][-1]['audio_seconds']==4


def test_raw_source_unknown_fields_and_native_events_survive_serialization():
    result=chart(offset=-.125);result.source['unknownMetadata']={'preserve':True}
    document=result.to_document();restored=PerformanceChart.from_document(document)
    assert restored.source==result.source
    assert restored.to_document()==document
    changed=copy.deepcopy(document);changed['lines'][0]['tracks']['move'][0]['after'][0]=.1
    with pytest.raises(ValueError,match='canonical events'):PerformanceChart.from_document(changed)


def test_empty_note_lines_are_retained_and_classification_has_no_false_visibility_proof():
    decoration=line(notes=[]);idle=line(notes=[]);idle['judgeLineDisappearEvents']=[scalar(before=-1,after=0)]
    outside=line(notes=[]);outside['judgeLineMoveEvents']=[move(x=.5,y=2)]
    unknown=line(notes=[]);unknown['judgeLineMoveEvents']=[]
    result=chart([decoration,idle,outside,unknown])
    assert len(result.lines)==4
    assert [result.classify_line(f'line-{i}')['classification'] for i in range(4)]==[
        'visible_decoration_candidate','idle','sampled_not_visible','unknown']
    assert result.classify_line('line-0')['actual_texture_visibility']=='unknown'


def test_alpha_positive_left_limit_cannot_be_misclassified_as_idle():
    source=line(bpm=60,notes=[])
    source['judgeLineDisappearEvents']=[scalar(0,32,-3,1),scalar(32,64,-1,-1)]
    result=chart([source],duration=2)
    assert result.line_state('line-0',.9)['alpha_source']==pytest.approx(.6)
    assert result.classify_line('line-0')['classification']=='visible_decoration_candidate'


def test_short_full_rotations_before_boundary_jump_obey_continuous_sample_limit():
    source=line(bpm=60,notes=[note(0,3,1,6)])
    source['judgeLineRotateEvents']=[scalar(0,1,0,720),scalar(1,64,0,0)]
    result=chart([source],duration=2)
    trajectory=result.sample_hold(result.notes[0]['id'])
    assert result.note_geometry(result.notes[0]['id'],.125/32)['target_world']==pytest.approx([.5,1.1])
    assert any(sample['in_viewport'] is False for sample in trajectory['samples'])
    angles=[sample['line_state']['rotation_degrees'] for sample in trajectory['samples'] if sample['audio_seconds']<1/32]
    assert max(b-a for a,b in zip(angles,angles[1:]))<=22.5+1e-8
    assert any(jump['track']=='rotate' and jump['left_limit']==[720] and jump['at_boundary']==[0] for jump in trajectory['jumps'])


def test_outside_audio_reversed_sentinels_remain_in_source_and_diagnostics():
    source=line();source['judgeLineRotateEvents'].append(scalar(999999,999998,9999,-9999))
    result=chart([source])
    assert result.line_state('line-0',1)['supported'] is True
    issue=next(item for item in result.diagnostics() if item['code']=='reversed_interval')
    assert issue['intersects_audio'] is False
    assert result.to_document()['source_document']['judgeLineList'][0]['judgeLineRotateEvents'][-1]['start']==9999


def test_reference_parser_last_key_pop_is_not_treated_as_mathematical_truth():
    source=line();source['judgeLineRotateEvents']=[scalar(0,128,0,360)]
    canonical=chart([source]);reference=PhiraSourceReference(canonical.source)
    assert canonical.line_state('line-0',1)['rotation_degrees']==180
    assert reference.line_state(0,1)['rotation_degrees']==0


def test_reference_truncates_last_hold_scroll_at_max_head_plus_one():
    canonical=chart([line(notes=[note(0,3,256,speed=3)])])
    reference=PhiraSourceReference(canonical.source)
    assert reference.max_time==1
    assert reference.line_state(0,4)['scroll_source']==1
    assert canonical.line_state('line-0',4)['scroll_integral']['value']==4
    assert reference.note_interpretation(canonical.source['judgeLineList'][0]['notesAbove'][0])==dict(
        effective_note_speed=1.,source_note_speed=3,source_floor_position=0.,original_floor_used=False)


def test_reference_max_head_empty_line_default_does_not_clamp_nonempty_negative_heads():
    negative=line(notes=[note(-128)])
    canonical=chart([negative],offset=3)
    assert canonical.notes[0]['time_seconds']==1
    assert PhiraSourceReference(canonical.source).max_time==-1
    with_empty=chart([negative,line(notes=[])],offset=3)
    assert PhiraSourceReference(with_empty.source).max_time==1


def test_reference_agrees_with_closed_curve_with_final_sentinel():
    source=line();source['judgeLineRotateEvents']=[scalar(0,128,0,360),scalar(128,999999,360,360)]
    source['judgeLineMoveEvents']=[move(0,128,.2,.3,.8,.7),move(128,999999,.8,.7)]
    canonical=chart([source]);reference=PhiraSourceReference(canonical.source)
    for time in (.25,.5,1,2,3):
        actual=canonical.line_state('line-0',time);other=reference.line_state(0,time)
        assert actual['rotation_degrees']==pytest.approx(other['rotation_degrees'])
        assert actual['position']==pytest.approx(other['position'])


def test_reference_v1_negative_coordinates_and_duplicate_cursor_boundary():
    source=line();source['judgeLineMoveEvents']=[scalar(0,128,-1001,-1001),scalar(128,999999,-1001,-1001)]
    canonical=chart([source],version=1);reference=PhiraSourceReference(canonical.source)
    assert reference.line_state(0,1)['position']==pytest.approx([-1/880,-1/520])
    source['judgeLineRotateEvents']=[scalar(-999999,0),scalar(0,128),scalar(128,999999)]
    reference=PhiraSourceReference(chart([source],version=1).source)
    assert reference.line_state(0,0)['tracks']['rotate'][0]['status']=='unsupported_reference_duplicate_time_cursor_history'


@pytest.mark.parametrize('start,end,intersects',[
    (-2,-1.5,False),(6.25,6.75,False),(-.5,.5,True),(4.5,5.5,True),
    (6.25,6.25,False),(-1,-1,False),(2,2,True),
])
def test_note_diagnostic_audio_intersection_uses_full_source_interval(start,end,intersects):
    result=chart([line(notes=[note(start*64,3,(end-start)*64)])])
    issues=[issue for issue in result.diagnostics() if 'note_id' in issue]
    assert issues and all(issue['intersects_audio'] is intersects for issue in issues)
    assert result.classify_line('line-0')['effective_note_count']==int(intersects)
    assert len(result.source['judgeLineList'][0]['notesAbove'])==1

import math

import pytest

from src.chart_windows import active_holds, chart_windows, partition_notes


def test_boundary_events_and_crossing_holds_are_lossless():
    notes=[dict(time=t,type=kind,duration=duration,screen_x=x)
           for t,kind,duration,x in [(0.,1,0.,-.7), (9.9,3,10.2,.3),
                                     (10.,2,0.,.25),(10.,4,0.,.25),
                                     (20.,1,0.,.9),(24.999,3,4.,-.2)]]
    groups=partition_notes(notes,25.)
    assert [len(g) for g in groups]==[2,2,2]
    assert [n for g in groups for n in g]==notes
    assert all(a is b for a,b in zip(notes,[n for g in groups for n in g]))
    assert active_holds(groups[0],10.)==[notes[1]]
    assert active_holds(groups[0]+groups[1],20.)==[notes[1]]
    assert notes[-1]['duration']==4. # End beyond audio is preserved, not clipped.


def test_half_open_ownership_at_adjacent_floating_point_values():
    for width in (.1,.3,10.):
        edge=3*width
        before=math.nextafter(edge,-math.inf)
        after=math.nextafter(edge,math.inf)
        notes=[dict(time=t,type=1,duration=0.) for t in (before,edge,after)]
        groups=partition_notes(notes,5*width,width)
        assert groups[2]==[notes[0]]
        assert groups[3]==notes[1:]


def test_empty_windows_and_equal_end_of_hold():
    windows=chart_windows(32.)
    assert [(w.start,w.end) for w in windows]==[(0.,10.),(10.,20.),(20.,30.),(30.,32.)]
    assert partition_notes([],32.)==[[],[],[],[]]
    assert not active_holds([dict(time=1.,duration=9.,type=3)],10.)
    assert not active_holds([dict(time=11.,duration=20.,type=3)],10.)


@pytest.mark.parametrize('duration,width',[(0.,10.),(-1.,10.),(math.inf,10.),(10.,0.),(10.,math.nan)])
def test_invalid_windows(duration,width):
    with pytest.raises(ValueError):chart_windows(duration,width)


@pytest.mark.parametrize('time',[-.001,10.,math.inf,math.nan])
def test_reject_outside_onsets(time):
    with pytest.raises(ValueError):partition_notes([dict(time=time)],10.)

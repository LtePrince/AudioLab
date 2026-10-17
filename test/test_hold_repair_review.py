"""One-factor replay contracts; synthetic fixtures are not musical evaluations."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from script import review_hold_repairs as review
from src.data.chart_constraints import constrain_4k_notes, constrain_free_notes
from src.product.artifacts import chart_document, validate_chart, write_json


def note(kind=3,time=0,duration=64,x=-3.75):
    return dict(type=kind,time=time,holdTime=duration if kind==3 else 0,
                positionX=x,speed=1.,floorPosition=time/64)


def fixture(tmp_path,notes,mode='4k',duration=2.,offset=.25):
    raw=chart_document(notes,120,offset,tmp_path/'source.json')
    final=deepcopy(raw)
    end_tick=(duration-offset)*64
    final['judgeLineList'][0]['notesAbove']=(
        constrain_4k_notes(raw['judgeLineList'][0]['notesAbove'],end_tick) if mode=='4k' else
        constrain_free_notes(raw['judgeLineList'][0]['notesAbove'],end_tick))[0]
    return raw,final


def test_same_lane_trigger_four_types_offset_and_no_double_floor_scaling(tmp_path):
    notes=[note(time=8),note(4,24,x=-3.75),note(2,40,x=-1.25),note(1,48,x=3.75)]
    raw,final=fixture(tmp_path,notes)
    before=deepcopy(raw)
    result=review.replay(raw,final,'4k',2.)
    assert raw==before and result['on']==final
    a=result['on']['judgeLineList'][0]['notesAbove']
    b=result['off']['judgeLineList'][0]['notesAbove']
    assert a[0]['holdTime']==16 and b[0]['holdTime']==64
    assert a[0]['floorPosition']==b[0]['floorPosition']==.375
    assert set(result['on_events']['events'][i]['type'] for i in range(4))=={'tap','drag','hold','flick'}
    change=result['changes'][0]
    assert change['time_seconds']==.375 and change['retained_duration_seconds']==.75
    assert change['attribution']['trigger']==dict(
        id='4k-note-000001',source_index=1,type='flick',source_type='flick',
        time_seconds=.625,screen_x=.2890625,positionX=-3.75,relation='same_lane',same_tick=False,distance_screen_x=0.)
    assert result['summary']['endpoint_removed_seconds']==0


@pytest.mark.parametrize('mode',['4k','freex'])
def test_unrelated_future_and_no_future_do_not_change_hold(tmp_path,mode):
    raw,final=fixture(tmp_path,[note(),note(1,16,x=3.75)],mode)
    result=review.replay(raw,final,mode,2.)
    assert result['on']==result['off'] and not result['changes']
    assert result['holds'][0]['reason']=='unchanged' and result['holds'][0]['trigger'] is None


def test_spatial_projection_before_nearby_trigger_is_shared(tmp_path):
    # At the future chord, .55 moves to .535 after spacing from its .56 neighbor.
    # It is only then within .04 of the earlier Hold at .50.
    notes=[note(x=0.),note(1,16,x=(.55-.5)/.05625),note(4,16,x=(.56-.5)/.05625)]
    raw,final=fixture(tmp_path,notes,'freex')
    result=review.replay(raw,final,'freex',2.)
    a=result['on']['judgeLineList'][0]['notesAbove'];b=result['off']['judgeLineList'][0]['notesAbove']
    assert result['summary']['enabled_repair_summary']['moved_notes']==2
    assert [n['positionX'] for n in a]==[n['positionX'] for n in b]
    trigger=result['holds'][0]['trigger']
    assert trigger['source_index']==1 and trigger['screen_x']==pytest.approx(.535)
    assert trigger['relation']=='nearby_after_spacing' and trigger['distance_screen_x']<.04
    assert result['changes'][0]['retained_duration_seconds']==.75


@pytest.mark.parametrize('with_future',[False,True])
def test_audio_endpoint_retained_and_components_add_up(tmp_path,with_future):
    notes=[note(time=48,duration=64)]
    if with_future:notes.append(note(1,56))
    raw,final=fixture(tmp_path,notes,duration=1.25)
    result=review.replay(raw,final,'4k',1.25)
    hold=result['holds'][0]
    assert result['off']['judgeLineList'][0]['notesAbove'][0]['holdTime']==16
    assert hold['endpoint_removed_seconds']==.75
    assert hold['future_rule_removed_seconds']==(.125 if with_future else 0.)
    assert hold['original_duration_seconds']-hold['on_duration_seconds']==hold['endpoint_removed_seconds']+hold['future_rule_removed_seconds']
    assert hold['reason']==('future_onset_and_audio_end' if with_future else 'audio_end')


def test_subtick_future_converts_to_tap_and_restores_derived_speed(tmp_path):
    raw,final=fixture(tmp_path,[note(time=16,duration=32),note(4,16.5)])
    result=review.replay(raw,final,'4k',2.)
    a=result['on']['judgeLineList'][0]['notesAbove'][0]
    b=result['off']['judgeLineList'][0]['notesAbove'][0]
    assert (a['type'],a['holdTime'],a['speed'])==(1,0,1.)
    assert (b['type'],b['holdTime'],b['speed'])==(3,32,3.)
    assert set(result['changes'][0]['fields'])=={'type','holdTime','speed'}
    assert result['on_events']['events'][0]['id']==result['off_events']['events'][0]['id']


def test_same_tick_trigger_follows_existing_stable_reverse_iteration(tmp_path):
    raw,final=fixture(tmp_path,[note(1,16),note(time=16,duration=32)])
    result=review.replay(raw,final,'4k',2.)
    assert result['changes'][0]['source_index']==1
    assert result['changes'][0]['attribution']['trigger']['source_index']==0
    assert result['changes'][0]['attribution']['trigger']['same_tick']


@pytest.mark.parametrize('fault',['frozen','speed_template'])
def test_mismatch_or_uncovered_template_refused(tmp_path,fault):
    raw,final=fixture(tmp_path,[note(),note(1,16)])
    if fault=='frozen':final['judgeLineList'][0]['notesAbove'][0]['holdTime']+=1
    else:
        raw['judgeLineList'][0]['speedEvents'][0]['endTime']=32
        final['judgeLineList'][0]['speedEvents'][0]['endTime']=32
    with pytest.raises(ValueError,match='frozen final|template does not cover'):
        review.replay(raw,final,'4k',2.)


@pytest.fixture
def frozen_jobs(tmp_path):
    roots=[tmp_path/'manual',tmp_path/'automatic']
    for root in roots:
        for song in range(3):
            directory=root/f'song-{song}';directory.mkdir(parents=True)
            sf.write(directory/'audio.wav',np.zeros(44100),22050)
            manifest=dict(status='completed',routes={},implementation=dict(git_commit='fixture'),
                          config=dict(bpm=120,offset=.25),model_sha256={},artifacts={},
                          audio=dict(file='audio.wav',sha256=review.digest(directory/'audio.wav'),duration_seconds=2.))
            write_json(directory/'audio_events.json',dict(status='completed'))
            for mode in ('4k','freex'):
                raw,final=fixture(directory,[note(time=8),note(4,24),note(2,40,x=1.25)],mode)
                write_json(directory/f'{mode}.raw.json',raw);write_json(directory/f'{mode}.json',final)
                events,_=validate_chart(final,mode,2.,{8,24,40})
                write_json(directory/f'{mode}.events.json',events)
                write_json(directory/f'{mode}.repairs.json',dict(profile='static-v1'))
                manifest['routes'][mode]=dict(raw_file=f'{mode}.raw.json',chart_file=f'{mode}.json',events_file=f'{mode}.events.json')
                manifest['model_sha256'][mode]='0'*64
            for path in directory.iterdir():
                if path.name!='manifest.json':manifest['artifacts'][path.name]=dict(sha256=review.digest(path))
            write_json(directory/'manifest.json',manifest)
    return review.parser().parse_args(['--manual-root',str(roots[0]),'--automatic-root',str(roots[1]),'--output',str(tmp_path/'output')])


def test_twelve_source_replay_serialized_exports_previews_and_unchanged_inputs(frozen_jobs):
    args=frozen_jobs
    before={str(p):review.digest(p) for root in (args.manual_root,args.automatic_root) for p in Path(root).rglob('*') if p.is_file()}
    report=review.run(args);output=Path(args.output)
    assert report['status']=='completed' and report['completed_rows']==12 and report['failed_rows']==0
    assert len(report['previews'])==2 and report['product_default_changed'] is False
    assert json.loads((output/'report.json').read_text())==report
    assert report['resources']['output_bytes']==review.resources(output)['output_bytes']
    for row in report['results']:
        assert row['summary']['enabled_exactly_matches_frozen_export']
        for export in row['exports'].values(): assert review.digest(output/export['path'])==export['sha256']
    for preview in report['previews']:
        page=output/preview['path'];text=page.read_text()
        assert (page.parent/'audio.wav').is_symlink()
        assert 'id="data"' in text and 'id="audio"' in text and 'id="chart"' in text
        assert 'audio.readyState>=1' in text and 'loadedmetadata' in text
        assert 'fetch(' not in text and '<script src=' not in text
    assert '#t=' in (output/'index.html').read_text()
    assert before=={p:review.digest(p) for p in before}


def test_changed_source_stays_in_failure_denominator(frozen_jobs):
    path=Path(frozen_jobs.manual_root)/'song-0/4k.raw.json';path.write_text('{}')
    report=review.run(frozen_jobs)
    assert report['status']=='completed_with_failures'
    assert report['completed_rows']==11 and report['failed_rows']==1 and len(report['results'])==12
    assert report['results'][0]['error']['message'].startswith('frozen source artifact changed')


@pytest.mark.parametrize('fault',['write','read','resource'])
def test_final_failure_removes_success_report(frozen_jobs,monkeypatch,fault):
    original_write=review.write_json;original_read=Path.read_text;original_resources=review.resources
    if fault=='write':
        def fail_write(path,value):
            original_write(path,value)
            if Path(path).name=='report.json':raise OSError('report write failed')
        monkeypatch.setattr(review,'write_json',fail_write)
    elif fault=='read':
        def fail_read(path,*a,**k):
            return '{}' if path.name=='report.json' else original_read(path,*a,**k)
        monkeypatch.setattr(Path,'read_text',fail_read)
    else:
        def fail_budget(path):
            if (path/'report.json').exists():raise RuntimeError('simulated output budget')
            return original_resources(path)
        monkeypatch.setattr(review,'resources',fail_budget)
    with pytest.raises((OSError,ValueError,RuntimeError)):
        review.run(frozen_jobs)
    output=Path(frozen_jobs.output)
    assert not (output/'report.json').exists()
    assert json.loads((output/'failure.json').read_text())['status']=='failed'

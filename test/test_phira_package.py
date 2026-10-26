"""Portable package and failure contracts, not native Phira execution."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile
from urllib.parse import unquote

import numpy as np
from PIL import Image
import pytest
import soundfile as sf
import yaml

from script import package_phira
from src.product import phira
from src.product.artifacts import chart_document, write_json
from src.product.server import LocalJobs, Handler


@pytest.fixture(autouse=True)
def controlled_package_budget(monkeypatch):
    # Functional tests must also run on small CI hosts. Failure cases below
    # explicitly inject refusals; real artifact runs use the unpatched guard.
    monkeypatch.setattr(phira,'check_resources',lambda *a,**k:None)
    monkeypatch.setattr(package_phira,'check_resources',lambda *a,**k:None)


def make_input(tmp_path,mode='4k',suffix='.wav',rate=22050,frames=None):
    positions=[-3.75,-1.25,1.25,3.75] if mode=='4k' else [-6.,-2.2,1.45,5.4]
    notes=[dict(type=kind,time=i*8,holdTime=32 if kind==3 else 0,positionX=x,speed=1.,floorPosition=i*8/64)
           for i,(kind,x) in enumerate(zip([1,2,3,4],positions))]
    chart=tmp_path/f'{mode}.json';chart_document(notes,120,.375,chart)
    audio=tmp_path/f'audio{suffix}'
    sf.write(audio,np.zeros(frames or 2*rate,dtype=np.float32),rate)
    duration=math.ceil((frames or 2*rate)*22050/rate)/22050
    return chart,audio,duration


def build(tmp_path,mode='4k',suffix='.wav',source=None):
    chart,audio,duration=make_input(tmp_path,mode,suffix)
    output=tmp_path/f'{mode}.phira.zip'
    result=phira.build_phira_package(chart,audio,output,mode=mode,title='测试曲 "<&>',
                                     duration_seconds=duration,source=source)
    return chart,audio,output,result


@pytest.mark.parametrize('mode,suffix',[('4k','.wav'),('freex','.ogg')])
def test_complete_bundle_preserves_bytes_clock_four_types_and_works_without_sources(tmp_path,mode,suffix):
    chart,audio,output,result=build(tmp_path,mode,suffix)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist())=={'chart.json','audio'+suffix,'illustration.png','info.yml','audiolab.json'}
        assert archive.read('chart.json')==chart.read_bytes()
        assert archive.read('audio'+suffix)==audio.read_bytes()
        info=yaml.safe_load(archive.read('info.yml'))
        assert info['offset']==0 and info['difficulty']==0 and info['level'].startswith('未定级')
        assert info['composer']=='未提供' and 'format' not in info
        assert info['chart']=='chart.json' and info['music']=='audio'+suffix
        packaged=json.loads(archive.read('chart.json'))
        notes=packaged['judgeLineList'][0]['notesAbove']
        assert {n['type'] for n in notes}=={1,2,3,4}
        assert notes[2]['time']/64+packaged['offset']+info['offset']==.625
        image=Image.open(io.BytesIO(archive.read('illustration.png')));image.verify()
        assert image.size==(480,270)
        source=json.loads(archive.read('audiolab.json'))
        assert str(tmp_path) not in json.dumps(source)
        assert source['difficulty_semantics']=='unrated_placeholder'
    chart.unlink();audio.unlink()
    verified=phira.verify_package(output)
    assert verified['status']=='passed' and verified['target_player_run']=='not_executed'
    assert verified['touch_playtest']=='not_executed' and result['sha256']==phira.sha256(output)


def test_synthetic_calibration_is_not_labelled_model_generated(tmp_path):
    _,_,output,_=build(tmp_path,source={'kind':'synthetic_calibration'})
    with zipfile.ZipFile(output) as archive:
        info=yaml.safe_load(archive.read('info.yml'));provenance=json.loads(archive.read('audiolab.json'))
        assert '合成校准' in info['charter'] and '不是模型生成' in info['intro']
        assert provenance['source_kind']=='synthetic_calibration'
    assert phira.verify_package(output)['status']=='passed'


@pytest.mark.parametrize('fault',['overlap','reorder','nested_meta'])
def test_extra_speed_segments_or_phira_rpe_marker_cannot_claim_static_pgr(tmp_path,fault):
    chart,_,duration=make_input(tmp_path);document=json.loads(chart.read_text())
    line=document['judgeLineList'][0]
    if fault=='overlap':line['speedEvents'].append(dict(startTime=8,endTime=16,value=-1))
    elif fault=='reorder':line['speedEvents'].reverse()
    else:document['extra']={'META':{}}
    with pytest.raises(ValueError,match='speed|META|visual onset'):
        phira.chart_contract(phira.encoded(document),'4k',duration)


@pytest.mark.parametrize('rate,frames',[(48000,192011),(96000,384001),(192000,768001)])
def test_one_analysis_sample_duration_rounding_at_high_sample_rates(tmp_path,rate,frames):
    chart,audio,duration=make_input(tmp_path,rate=rate,frames=frames)
    result=phira.build_phira_package(chart,audio,tmp_path/'clock.zip',mode='4k',title='clock',duration_seconds=duration)
    audio_check=result['verification']['audio_checks']
    assert 0<audio_check['analysis_minus_source_seconds']<=1/22050
    assert audio_check['duration_seconds']==frames/rate
    with pytest.raises(ValueError,match='duration does not match'):
        phira.audio_contract(audio,duration+.0001,lambda:None)


@pytest.mark.parametrize('field,value',[('channels',9),('samplerate',192001),('duration',601)])
def test_audio_header_limits_precede_any_frame_allocation(monkeypatch,field,value):
    class Header:
        channels=2;samplerate=48000;frames=96000
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args,**kwargs):raise AssertionError('read ran before header guard')
    header=Header()
    if field!='duration':setattr(header,field,value)
    monkeypatch.setattr(phira.sf,'SoundFile',lambda path:header)
    with pytest.raises(ValueError,match='bounded decoder|within 600'):
        phira.audio_contract('audio.wav',value if field=='duration' else 2.,lambda:None)


def mutate_zip(source,target,transform):
    with zipfile.ZipFile(source) as archive:
        entries=[(deepcopy(entry),archive.read(entry)) for entry in archive.infolist()]
    with zipfile.ZipFile(target,'w') as archive:
        for entry,data in transform(entries):archive.writestr(entry,data)


@pytest.mark.parametrize('fault',['path','duplicate','symlink','missing','fingerprint','audio','oversize'])
def test_invalid_archive_contents_are_rejected_without_extraction(tmp_path,monkeypatch,fault):
    _,_,original,_=build(tmp_path)
    target=tmp_path/'broken.zip'
    def transform(entries):
        if fault=='missing':return entries[:-1]
        if fault=='duplicate':return entries+[entries[0]]
        if fault=='path':entries[0][0].filename='../outside.yml'
        if fault=='symlink':entries[0][0].external_attr=(stat.S_IFLNK|0o777)<<16
        if fault=='fingerprint':
            entries=[(entry,data+b'\n') if entry.filename=='chart.json' else (entry,data) for entry,data in entries]
        if fault=='audio':
            # Even self-consistent stored hashes must not accept invalid media.
            for i,(entry,data) in enumerate(entries):
                if entry.filename=='audio.wav':entries[i]=(entry,b'not audio')
            for i,(entry,data) in enumerate(entries):
                if entry.filename=='audiolab.json':
                    document=json.loads(data)
                    document['members']['audio.wav']=dict(sha256=phira.hashlib.sha256(b'not audio').hexdigest(),bytes=9)
                    entries[i]=(entry,phira.encoded(document))
        return entries
    if fault=='duplicate':
        with pytest.warns(UserWarning,match='Duplicate name'):mutate_zip(original,target,transform)
    else:mutate_zip(original,target,transform)
    if fault=='oversize':monkeypatch.setattr(phira,'MAX_AUDIO',10)
    with pytest.raises((ValueError,OSError,sf.LibsndfileError)):
        phira.verify_package(target)
    assert not (tmp_path.parent/'outside.yml').exists()


@pytest.mark.parametrize('fault',['cancel','write','verify','source_change','resource'])
def test_build_failures_never_leave_a_downloadable_partial_package(tmp_path,monkeypatch,fault):
    chart,audio,duration=make_input(tmp_path)
    output=tmp_path/'result.zip';temporary=tmp_path/'result.zip.tmp'
    check=None
    if fault=='cancel':
        def check():
            if temporary.exists():raise InterruptedError('cancelled during package stream')
    elif fault=='write':
        original=zipfile.ZipFile.open
        def broken(self,name,*args,**kwargs):
            if isinstance(name,zipfile.ZipInfo) and name.filename=='audio.wav':raise OSError('simulated write failure')
            return original(self,name,*args,**kwargs)
        monkeypatch.setattr(zipfile.ZipFile,'open',broken)
    elif fault in ('verify','source_change'):
        original=phira.verify_package
        def verify(*args,**kwargs):
            if fault=='verify':raise ValueError('simulated readback failure')
            result=original(*args,**kwargs);audio.write_bytes(audio.read_bytes()+b'changed');return result
        monkeypatch.setattr(phira,'verify_package',verify)
    else:
        original=phira.check_resources
        def budget(directory,extra_bytes=0,**kwargs):
            if temporary.exists():raise RuntimeError('Resource budget: simulated refusal')
            return original(directory,extra_bytes,**kwargs)
        monkeypatch.setattr(phira,'check_resources',budget)
    with pytest.raises((InterruptedError,OSError,ValueError,RuntimeError)):
        phira.build_phira_package(chart,audio,output,mode='4k',title='sample',duration_seconds=duration,check=check)
    assert not output.exists() and not temporary.exists()


def test_preexisting_package_and_temporary_are_never_overwritten(tmp_path):
    chart,audio,duration=make_input(tmp_path);output=tmp_path/'result.zip'
    for existing in (output,output.with_suffix('.zip.tmp')):
        existing.write_bytes(b'keep')
        with pytest.raises(FileExistsError):
            phira.build_phira_package(chart,audio,output,mode='4k',title='test',duration_seconds=duration)
        assert existing.read_bytes()==b'keep'
        existing.unlink()


def test_existing_job_cli_failure_cleans_first_package_and_never_marks_complete(tmp_path,monkeypatch):
    job=tmp_path/'job';job.mkdir()
    charts={mode:make_input(job,mode)[0] for mode in ('4k','freex')}
    audio=job/'audio.wav'
    manifest=dict(status='completed',title='test',routes={mode:dict(chart_file=path.name) for mode,path in charts.items()},
                  audio=dict(file=audio.name,sha256=phira.sha256(audio),duration_seconds=2.),
                  implementation=dict(git_commit='synthetic-test'),model_sha256={'4k':'x','freex':'y'},
                  config=dict(timing_source='provided',seed=42),
                  artifacts={path.name:dict(sha256=phira.sha256(path)) for path in [audio,*charts.values()]})
    write_json(job/'manifest.json',manifest)
    result=package_phira.package_job(job,tmp_path/'success')
    assert set(result['packages'])=={'4k','freex'}
    original=package_phira.build_phira_package
    def fail_second(*args,**kwargs):
        if kwargs['mode']=='freex':raise RuntimeError('second package failed')
        return original(*args,**kwargs)
    monkeypatch.setattr(package_phira,'build_phira_package',fail_second)
    with pytest.raises(RuntimeError):package_phira.package_job(job,tmp_path/'failure')
    assert not list((tmp_path/'failure').glob('*.zip'))
    assert not (tmp_path/'failure/packages.json').exists()
    assert json.loads((tmp_path/'failure/failure.json').read_text())['status']=='failed'


def test_completed_api_exposes_only_finished_packages_and_attachment_header(tmp_path):
    jobs=LocalJobs(tmp_path/'server',tmp_path/'models','cpu')
    identifier,_=jobs.reserve('song.wav','4k',120,100)
    folder=jobs.root/'jobs'/identifier;folder.mkdir()
    package=folder/'4k.phira.zip';package.write_bytes(b'zip fixture')
    write_json(folder/'manifest.json',dict(packages={'4k':dict(file=package.name,sha256=phira.sha256(package),bytes=package.stat().st_size)},
                                          artifacts={package.name:{}}))
    assert 'packages' not in jobs.status(identifier)
    jobs.records[identifier]['status']='completed';jobs.active=None
    state=jobs.status(identifier)
    assert state['packages']['4k']['url']==f'/jobs/{identifier}/4k.phira.zip'
    assert jobs.artifact(identifier,package.name)==package
    handler=object.__new__(Handler);handler.headers={};handler.wfile=io.BytesIO();headers={}
    handler.send_response=lambda code:headers.update(status=code)
    handler.send_header=lambda key,value:headers.update({key:value});handler.end_headers=lambda:None
    handler.send_file(package)
    assert headers['Content-Type']=='application/zip'
    assert headers['Content-Disposition']=='attachment; filename="4k.phira.zip"'
    assert handler.wfile.getvalue()==b'zip fixture'
    manifest=json.loads((folder/'manifest.json').read_text())
    expected_name='测试曲_4k_20261026.phira.zip'
    manifest['download_names']={package.name:expected_name}
    write_json(folder/'manifest.json',manifest)
    assert jobs.status(identifier)['packages']['4k']['download_name']==expected_name
    handler.wfile=io.BytesIO()
    handler.send_file(package,jobs.download_name(identifier,package.name))
    disposition=headers['Content-Disposition']
    assert disposition.isascii()  # HTTP headers must not contain raw Chinese characters.
    assert unquote(disposition.split("filename*=UTF-8''",1)[1])==expected_name
    jobs.records[identifier]['status']='failed'
    assert 'packages' not in jobs.status(identifier)
    with pytest.raises(FileNotFoundError):jobs.artifact(identifier,package.name)
    jobs.close()

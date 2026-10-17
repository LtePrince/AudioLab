"""Automatic timebase, local job isolation, and transport boundaries."""
import json
import io
import copy
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf
import librosa
from scipy.signal import find_peaks

from src.audio_understanding import FRAME_SECONDS, SAMPLE_RATE
from src.audio_understanding.inference import load_audio
from src.product.server import LocalJobs, Handler, BusyError, parse_range, valid_name
from src.product.tempo import estimate_tempo, provided_tempo


def pulses(bpm=120,duration=24,lead=1,pickup=False,alternating=False):
    mel=np.zeros((128,round(duration/FRAME_SECONDS)),dtype=np.float32)
    for i,t in enumerate(np.arange(lead,duration-.5,60/bpm)):
        mel[:,round(t/FRAME_SECONDS)]=3 if alternating and i%2 else 10
    if pickup:mel[:,round((lead-.25)/FRAME_SECONDS)]=2
    return mel


@pytest.mark.parametrize('bpm,lead,pickup',[(120,1,False),(120,4,False),(120,1,True)])
def test_known_click_periods_keep_frame_resolution_and_do_not_claim_truth(bpm,lead,pickup):
    result=estimate_tempo(pulses(bpm,lead=lead,pickup=pickup),24)
    assert result['status']=='estimated_unconfirmed'
    assert abs(60/result['timebase_bpm']-60/bpm)<=FRAME_SECONDS
    assert result['musical_tempo_estimate']['verified'] is False
    assert result['config']['start_bpm_prior']==120
    assert {candidate['relative'] for candidate in result['candidates']}=={.5,1.,2.}


@pytest.mark.parametrize('lead',[1.,4.])
def test_wav_clicks_preserve_audio_origin_and_centered_feature_convention(tmp_path,lead):
    duration=16.;times=np.arange(lead,duration-.5,.5)
    waveform=np.zeros(round(duration*SAMPLE_RATE),dtype=np.float32)
    waveform[np.round(times*SAMPLE_RATE).astype(int)]=1.
    path=tmp_path/'clicks.wav';sf.write(path,waveform,SAMPLE_RATE,subtype='FLOAT')
    mel,actual_duration=load_audio(path)
    result=estimate_tempo(mel,actual_duration)
    assert actual_duration==duration and result['analysis_range_seconds']==[0,duration]
    assert result['config']['feature_origin_seconds']==0
    assert result['config']['stft_center'] is True
    assert abs(60/result['timebase_bpm']-.5)<=FRAME_SECONDS
    envelope=librosa.onset.onset_strength(S=mel.numpy(),sr=SAMPLE_RATE,hop_length=512,center=False)
    peaks,_=find_peaks(envelope,prominence=float(envelope.max())*.1,distance=round(.25/FRAME_SECONDS))
    assert len(peaks)==len(times)
    # Centered FFT windows respond before a sample impulse; the flux is used
    # only for periods, never to replace the model's absolute event times.
    errors=peaks*FRAME_SECONDS-times
    assert np.all(np.abs(errors)<=1024/SAMPLE_RATE+FRAME_SECONDS)
    assert peaks[0]*FRAME_SECONDS >= lead-1024/SAMPLE_RATE-FRAME_SECONDS
    strength=mel.mean(dim=0).numpy()
    for at in times:
        center=round(at/FRAME_SECONDS)
        observed=center-2+np.argmax(strength[center-2:center+3])
        assert abs(observed*FRAME_SECONDS-at)<=FRAME_SECONDS


def test_tempo_control_rejects_sampling_and_shared_inference_drift():
    from script.audit_product_tempo import verify_control
    config=dict(seed=42,offset=0,timing_schema='v1',difficulty='baseline',layout_encoder='full',
                max_seconds=600,timeout_seconds=900)
    route=dict(seed=42,temperature=1.,top_p=.95,sampling_policy='flat',max_decoder_steps=12000,
               encoder='full-attention-v1',chunk_frames=None,context_frames=None,chord_size_cap=None,
               termination_policy='natural_eos_required')
    manifest=dict(modes=['4k'],model_sha256={'4k':'abc'},audio={'sha256':'abc'},config=config,
                  runtime=dict(torch='same',device='cpu',threads=2,device_name='CPU'),routes={'4k':route})
    evidence=dict(schema='v1',time_basis='audio_seconds',duration_seconds=16,checkpoint_sha256='abc',
                  label_source='chart_onset',feature_schema='v1',selection={'threshold':.7},
                  inference={'chunk_frames':1024},feature_shape=[10,128],
                  implementation={'source_sha256':{'inference.py':'abc'}},events=[{'time_seconds':1}])
    assert verify_control(manifest,manifest,evidence,evidence)['status']=='matched'
    for category,key,value in [('config','seed',12),('config','offset',.2),
                               ('route','top_p',.9),('route','encoder','chunked'),
                               ('evidence','selection',{'threshold':.6}),('evidence','inference',{'chunk_frames':512})]:
        changed=copy.deepcopy(manifest);events=copy.deepcopy(evidence)
        target=events if category=='evidence' else changed['routes']['4k'] if category=='route' else changed['config']
        target[key]=value
        with pytest.raises(ValueError,match='timebase control mismatch'):
            verify_control(manifest,changed,evidence,events)


def test_octaves_and_changing_sections_are_uncertainty_not_verified_tempo_maps():
    alternative=estimate_tempo(pulses(180,alternating=True),24)
    assert 'half_double_interpretation_unresolved' in alternative['uncertainty']['diagnostics']
    varying=np.concatenate([pulses(120,duration=16),pulses(160,duration=16)],axis=-1)
    result=estimate_tempo(varying,32)
    assert result['uncertainty']['local_relative_spread']>.12
    assert 'local_tempo_variation_or_octave_switching' in result['uncertainty']['diagnostics']
    assert result['tempo_map'] is None


def test_180_bpm_clicks_expose_known_half_speed_limitation():
    # This baseline does NOT recover the fixture's annotated 180 BPM directly.
    # Keep the failure visible instead of relabelling it as tempo accuracy.
    result=estimate_tempo(pulses(180),24)
    assert abs(60/result['timebase_bpm']-60/90)<=FRAME_SECONDS
    assert abs(60/(result['timebase_bpm']*2)-60/180)<=FRAME_SECONDS
    assert result['musical_tempo_estimate']['verified'] is False
    assert 'half_double_interpretation_unresolved' in result['uncertainty']['diagnostics']


def test_octave_tempo_change_can_remain_invisible_to_local_estimates():
    result=estimate_tempo(np.concatenate([pulses(90,duration=16),pulses(180,duration=16)],axis=-1),32)
    assert result['uncertainty']['local_relative_spread']==0
    assert result['tempo_map'] is None
    assert 'half_double_interpretation_unresolved' in result['uncertainty']['diagnostics']


@pytest.mark.parametrize('kind',['silence','sparse','short'])
def test_no_tempo_never_silently_returns_prior(kind):
    duration=2 if kind=='short' else 12
    mel=np.zeros((128,round(duration/FRAME_SECONDS)),dtype=np.float32)
    if kind=='sparse':mel[:,100]=10;mel[:,200]=10
    result=estimate_tempo(mel,duration)
    assert result['status']=='needs_tempo' and result['timebase_bpm'] is None


def test_provided_timebase_is_not_claimed_as_observed_musical_tempo():
    result=provided_tempo(180,30)
    assert result['source']=='provided' and result['timebase_bpm']==180
    assert result['musical_tempo_estimate'] is None
    for invalid in [0,float('nan'),float('inf'),1001]:
        with pytest.raises(ValueError):provided_tempo(invalid,30)


@pytest.mark.parametrize('header,expected',[(None,(0,99,False)),('bytes=20-29',(20,29,True)),
                                         ('bytes=90-',(90,99,True)),('bytes=-10',(90,99,True))])
def test_audio_byte_ranges(header,expected):
    assert parse_range(header,100)==expected


def test_invalid_paths_and_ranges_are_rejected():
    for name in ['../song.wav','x/y.ogg','x\\y.wav','bad.mp3','bad\x00.wav']:
        with pytest.raises(ValueError):valid_name(name)
    for header in ['bytes=100-','bytes=40-20','bytes=0-1,3-4','bytes=-0','other=0-4']:
        with pytest.raises(ValueError):parse_range(header,100)


def test_concurrent_submissions_reserve_exactly_one_slot_until_cancelled_upload_exits(tmp_path):
    jobs=LocalJobs(tmp_path/'server',tmp_path/'models','cpu')
    def reserve():
        try:return jobs.reserve('song.wav','both',None,100)[0]
        except BusyError:return None
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:reserve(),range(2)))
    assert sum(item is not None for item in results)==1
    identifier=next(item for item in results if item)
    jobs.cancel(identifier)
    assert jobs.active==identifier  # cancellation request has not released worker/upload yet
    with pytest.raises(InterruptedError):jobs.upload_check(identifier)
    with pytest.raises(BusyError):jobs.reserve('song.wav','both',None,100)
    jobs.fail(identifier,'cancelled','cancelled')
    assert jobs.active is None
    jobs.close()


def test_completed_downloads_only_resolve_manifest_artifacts(tmp_path):
    jobs=LocalJobs(tmp_path/'server',tmp_path/'models','cpu')
    identifier,source=jobs.reserve('song.wav','4k',120,100)
    folder=jobs.root/'jobs'/identifier;folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps({'artifacts':{'audio.wav':{}}}))
    (folder/'audio.wav').write_bytes(b'example')
    jobs.records[identifier]['status']='completed';jobs.active=None
    assert jobs.artifact(identifier,'audio.wav')==folder/'audio.wav'
    for name in ['../records/a.json','unknown.json','/etc/passwd']:
        with pytest.raises(FileNotFoundError):jobs.artifact(identifier,name)
    jobs.close()


def test_upload_response_disconnect_cannot_release_live_worker(tmp_path,monkeypatch):
    jobs=LocalJobs(tmp_path/'server',tmp_path/'models','cpu')
    worker=SimpleNamespace(poll=lambda:None)
    def start(identifier,path):
        jobs.process=worker;jobs.records[identifier]['status']='running'
    monkeypatch.setattr(jobs,'start',start)
    handler=object.__new__(Handler)
    handler.server=SimpleNamespace(jobs=jobs,server_port=7860)
    handler.headers={'Host':'127.0.0.1:7860','Content-Length':'4'}
    handler.path='/api/jobs?name=song.wav&mode=both'
    handler.connection=SimpleNamespace(settimeout=lambda n:None)
    handler.rfile=io.BytesIO(b'wave')
    def disconnected(*a,**k):raise BrokenPipeError('client aborted after upload')
    handler.reply=disconnected
    handler.do_POST()
    identifier=jobs.active
    assert identifier is not None and jobs.process.poll() is None
    assert (jobs.root/'uploads'/identifier/'song.wav').read_bytes()==b'wave'
    with pytest.raises(BusyError):jobs.reserve('next.wav','both',None,100)
    jobs.fail(identifier,'another handler error')
    assert jobs.active==identifier and jobs.records[identifier]['status']=='cancelling'
    jobs.process=None;jobs.active=None;jobs.close()

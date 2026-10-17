from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from script import audit_audio_conditioning_v2 as runner
from src import conditioning_audit as audit
from src.audio_understanding import FRAME_SECONDS
from src.audio_understanding.inference import load_audio, sha256
from src.data.chart_tokenizer import ChartTokenizer, TokenNote, BOS, EOS, hit_to_bin
from src.data.timing import TIMING_SCHEMA
from src.models.chart_decoder import ChartDecoder
from src.models.transcriber import TranscriberNet


def chart(notes,offset=0.,bpm=120):
    return dict(formatVersion=3,offset=offset,judgeLineList=[dict(bpm=bpm,notesAbove=notes,notesBelow=[])])


def note(tick,kind=1,dur=0,x=-6):
    return dict(time=tick,type=kind,holdTime=dur,positionX=x)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def mined_files(root,name,document,hit_positions=None):
    original=write(root/'data/chart'/name/'original.json',document)
    ref=write(original.with_name('generate.json'),document)
    rows=[]
    for i,n in enumerate(document['judgeLineList'][0]['notesAbove']):
        rows.append(dict(line_id=0,above=True,tick=round(n['time']),type=n['type'],
                         dur=round(n['time']+n['holdTime'])-round(n['time']) if n['type']==3 else 0,
                         pos_x=n['positionX'],hit_x=hit_positions[i] if hit_positions else .15+i*.2))
    path=root/'data/gameplay'/f'{name}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    write(path.with_suffix('.meta.json'),dict(timing_schema=TIMING_SCHEMA,bpm=120,offset=document['offset'],
          source_path=str(original.relative_to(root)),source_sha256=sha256(original),
          jsonl_sha256=sha256(path),notes=len(rows)))
    return ref,path,original


def small_models(lanes=4):
    enc_config=dict(n_mels=128,hidden_dim=32,conv_blocks=1,depth=1,num_heads=2,dropout=0.)
    dec_config=dict(vocab_size=ChartTokenizer(lanes).vocab_size,hidden_dim=32,depth=1,num_heads=2,
                    mem_dim=32,time_dim=16,n_lanes=lanes,factorized=lanes==64,
                    offset_head=lanes==64,dropout=0.,cross_window=2)
    return TranscriberNet(**enc_config).eval(),ChartDecoder(**dec_config).eval(),enc_config,dec_config


def test_odd_tail_and_longer_than_old_cap_preserve_all_frames():
    class CheapEncoder:
        mel_per_frame=2
        def features(self,mel):
            self.input=mel
            return mel[:,:4,::2].transpose(1,2)
    for frames in (5,9001):
        enc=CheapEncoder();duration=(frames-.25)*FRAME_SECONDS
        features,valid,meta=audit.encode_audio(enc,torch.ones(128,frames),duration,torch.device('cpu'))
        assert enc.input.shape[-1]==frames+1
        assert features.shape[1]==(frames+1)//2
        assert valid.all() and meta['last_memory_time_seconds']<duration
        assert not enc.input[:,:,-1].any()
    assert duration>190.218


def test_real_audio_reader_odd_tail_and_origin(tmp_path):
    frames=5;duration=(frames-.3)*FRAME_SECONDS
    audio=tmp_path/'short.wav';sf.write(audio,np.zeros(round(duration*22050),np.float32),22050)
    mel,actual=load_audio(audio)
    valid,meta=audit.feature_geometry(mel.shape[-1],actual)
    assert mel.shape[-1]==5 and valid.shape==(1,3)
    assert meta['memory_time_origin_seconds']==0 and meta['last_memory_time_seconds']==4*FRAME_SECONDS


def test_reference_tracks_quantization_dedupe_and_short_hold(tmp_path):
    p=write(tmp_path/'chart.json',chart([note(16.2),note(16.1,kind=4),note(64.1,kind=3,dur=.1)],offset=.25))
    ref=audit.load_reference(p,3.)
    tk=ChartTokenizer(4)
    assert tk.decode_tokens(ref.tokens.tolist())==[TokenNote(16,0,1),TokenNote(64,0,1)]
    assert ref.metadata['source_notes']==3 and ref.metadata['retained_notes']==2
    assert ref.metadata['transformations']['duplicate_tick_position_dropped']==1
    assert ref.metadata['transformations']['short_hold_to_tap']==1
    assert float(ref.frames[0])==pytest.approx(.25/(FRAME_SECONDS*2))
    assert ref.metadata['retained_note_roundtrip'].startswith('exact')


@pytest.mark.parametrize('event,offset,duration,reason',[
    (note(-32),1.,2.,'negative_token_tick'),
    (note(32,3,64),0.,1.,'native_outside_audio'),
    (note(64),0.,1.,'native_outside_audio'),
    (note(20,3,-1),0.,1.,'negative_hold'),
    (note(63.9),0.,1.,'quantized_outside_audio'),
])
def test_complete_reference_rejects_unsupported_without_clamping(tmp_path,event,offset,duration,reason):
    p=write(tmp_path/'chart.json',chart([event],offset))
    with pytest.raises(audit.ReferenceError) as result: audit.load_reference(p,duration)
    assert result.value.details[0]['reason']==reason


def test_free_provenance_source_link_and_offsets(tmp_path):
    document=chart([note(16),note(16,kind=4,x=-2),note(32,kind=3,dur=8,x=2)],.125)
    p,g,original=mined_files(tmp_path,'s',document,[.2,.20001,.6])
    ref=audit.load_reference(p,2.,gameplay_path=g,source_root=tmp_path)
    assert ref.metadata['source_notes']==3 and ref.metadata['retained_notes']==2
    assert int(ref.offset_mask.sum())==2
    assert ref.offsets[ref.offset_mask].tolist()==pytest.approx([hit_to_bin(.2)[1],hit_to_bin(.6)[1]])
    assert ref.metadata['provenance']['original_chart']['sha256']==sha256(original)
    original.unlink()
    with pytest.raises(FileNotFoundError): audit.load_reference(p,2.,gameplay_path=g,source_root=tmp_path)


def test_free_missing_metadata_or_mutated_content_fails(tmp_path):
    p,g,_=mined_files(tmp_path,'s',chart([note(8)]))
    g.write_text(g.read_text()+'\n')
    with pytest.raises(ValueError,match='metadata mismatch'):
        audit.load_reference(p,2.,gameplay_path=g,source_root=tmp_path)
    g.with_suffix('.meta.json').unlink()
    with pytest.raises(FileNotFoundError): audit.load_reference(p,2.,gameplay_path=g,source_root=tmp_path)


def test_free_clamping_count_only_counts_changed_coordinates(tmp_path):
    p,g,_=mined_files(tmp_path,'s',chart([note(t) for t in (4,8,12,16)]),[.05,.95-1e-9,.049,.96])
    reference=audit.load_reference(p,2.,gameplay_path=g,source_root=tmp_path)
    assert reference.metadata['transformations']['position_clamped']==2


def test_known_logits_and_conditional_denominators():
    tk=ChartTokenizer(64)
    targets=torch.tensor([tk.dtt(1),tk.note(4,1),tk.note(6,3),tk.durb(2),tk.durt(1),EOS])
    scores=audit.token_scores(torch.zeros(len(targets),tk.vocab_size),targets,tk)
    assert {key:scores[key]['count'] for key in ('all','time','note','duration')}==dict(all=6,time=1,note=2,duration=2)
    assert scores['all']['bits']==pytest.approx(math.log2(tk.vocab_size))
    assert scores['note_type']['bits']==pytest.approx(2.)
    assert scores['position']['bits']==pytest.approx(6.)


def test_offset_uses_current_note_hidden_and_screen_units():
    tk=ChartTokenizer(64)
    tokens=torch.tensor([BOS,tk.dtt(1),tk.note(2,1),tk.note(4,3),tk.durt(1),EOS])
    mask=torch.tensor([False,False,True,True,False,False])
    reference=audit.Reference(tokens,torch.arange(6,dtype=torch.float32),
                              torch.tensor([0.,0.,.2,.8,0.,0.]),mask,{})
    class KnownDecoder:
        n_lanes=64;offset_head=True
        def prepare_memory(self,features): return features
        def __call__(self,tokens,frames,memory,valid,return_hidden):
            assert return_hidden and tokens.shape==(1,5)
            return torch.zeros(1,5,tk.vocab_size),torch.arange(5,dtype=torch.float32)[None,:,None]/10
        def predict_offset(self,hidden): return hidden[:,:,0]
    result=audit.score_condition(KnownDecoder(),torch.ones(1,4,1),torch.ones(1,4,dtype=torch.bool),reference)
    assert result['conditional_offset']['count']==2
    assert result['conditional_offset']['mae_sub_bin']==pytest.approx(.25)
    assert result['conditional_offset']['mae_screen_width']==pytest.approx(.25*.9/64)
    assert result['conditional_offset']['by_type']['3']['mae_sub_bin']==pytest.approx(.5)


def test_feature_controls_and_empty_memory():
    source=torch.arange(5,dtype=torch.float32)[None,:,None];valid=torch.tensor([[True,True,True,False,False]])
    reverse,_=audit.alter_features(source,valid,'encoded_feature_reverse')
    assert reverse.flatten().tolist()==[2,1,0,3,4]
    wrong,mapping=audit.alter_features(source,valid,'wrong_song_resampled_features',
                                      other=torch.tensor([[[10.],[20.]]]),other_valid=torch.ones(1,2,dtype=torch.bool))
    assert wrong.flatten().tolist()==pytest.approx([10,15,20,3,4])
    assert mapping['source_valid_frames']==2 and not mapping['align_corners']
    zero,_=audit.alter_features(source,valid,'zero_acoustic_features');assert not zero.any()
    with pytest.raises(ValueError): audit.alter_features(source,torch.zeros_like(valid),'matched')


def test_preflight_uses_real_small_models_and_refuses_before_allocation(tmp_path,monkeypatch):
    enc,dec,_,_=small_models(64)
    estimate=audit.estimate_working_set(enc,dec,9001,12001)
    assert estimate['memory_frames']==4501
    assert estimate['parts_bytes']['decoder_self_scores']==2*2*12000**2*4
    budget=SimpleNamespace(check=lambda **kwargs:None,directory=tmp_path,
            args=SimpleNamespace(max_rss_gib=4,min_available_ram_gib=8,gpu_memory_fraction=.65))
    monkeypatch.setattr(audit,'resource_snapshot',lambda _:dict(rss_gib=3.99,ram_available_gib=9,disk_free_gib=10))
    with pytest.raises(audit.ReferenceError,match='RAM budget') as failure:
        audit.enforce_preflight(estimate,budget,torch.device('cpu'))
    assert failure.value.details['teacher_forcing_tokens']==12000


@pytest.mark.parametrize('lanes',[4,64])
def test_scoring_rejects_all_invalid_memory(tmp_path,lanes):
    p,g,_=mined_files(tmp_path,'s',chart([note(8)]))
    reference=audit.load_reference(p,1.,gameplay_path=g if lanes==64 else None,source_root=tmp_path)
    _,dec,_,_=small_models(lanes)
    with pytest.raises(ValueError,match='valid audio frame'):
        audit.score_condition(dec,torch.ones(1,3,32),torch.zeros(1,3,dtype=torch.bool),reference)


@pytest.fixture(scope='module')
def toy_run(tmp_path_factory):
    root=tmp_path_factory.mktemp('conditioning-v2')
    for index in range(2):
        name=f'song-{index}'
        ref,_,_=mined_files(root,name,chart([note(4),note(12,3,8,x=2)],.025))
        t=np.arange(round((.6+index*.1)*22050))/22050
        sf.write(ref.with_name('audio.wav'),(.1*np.sin(2*np.pi*(220+index*100)*t)).astype(np.float32),22050)
    (root/'data/val.txt').write_text('\n'.join(f'chart/song-{i}/generate.json,chart/song-{i}/audio.wav,IN' for i in range(2)))
    for model,lanes in [('decoder',4),('decoder_freex',64)]:
        enc,dec,ec,dc=small_models(lanes)
        state={**{'encoder.'+k:v for k,v in enc.state_dict().items()},**{'decoder.'+k:v for k,v in dec.state_dict().items()}}
        path=root/'weights'/model/'decoder_best.pt';path.parent.mkdir(parents=True)
        torch.save(dict(model=state,enc_config=ec,dec_config=dc,timing_schema=TIMING_SCHEMA),path)
    args=runner.parser().parse_args(['--data-list',str(root/'data/val.txt'),'--indices','0','1',
        '--split-label','synthetic-contract-only','--selection-reason','controlled tests; random tiny weights',
        '--checkpoint-root',str(root/'weights'),'--gameplay-dir',str(root/'data/gameplay'),
        '--source-root',str(root),'--output',str(root/'output')])
    result=runner.run(args)
    return root,args,result


def test_complete_serial_pipeline_readback_and_denominators(toy_run):
    root,_,report=toy_run
    assert report['status']=='completed' and report['completed_rows']==4
    assert json.loads((root/'output/report.json').read_text())==report
    assert all(row['reference']['retained_notes']==2 for row in report['results'])
    assert not list(root.rglob('*.npy')) and not list(root.rglob('*.npz'))


@pytest.mark.parametrize('missing',['frames','features','provenance'])
def test_missing_fingerprints_are_not_valid_report(toy_run,missing):
    report=deepcopy(toy_run[2]);row=report['results'][0]
    if missing=='frames': del row['reference']['fingerprints']['frames']
    elif missing=='features': del row['features']['features']['sha256']
    else: del row['reference']['provenance']['reference_chart']
    with pytest.raises((KeyError,ValueError)): audit.validate_report(report)


def test_atomic_final_read_failure_removes_completion(toy_run,tmp_path,monkeypatch):
    target=tmp_path/'report.json';read=Path.read_text
    def fail(self,*a,**kw):
        if self==target: raise OSError('simulated final read failure')
        return read(self,*a,**kw)
    monkeypatch.setattr(Path,'read_text',fail)
    with pytest.raises(OSError): audit.atomic_report(target,toy_run[2])
    assert not target.exists() and not target.with_suffix('.tmp').exists()


def test_corrupt_final_report_is_removed(toy_run,tmp_path,monkeypatch):
    target=tmp_path/'report.json';replace=audit.os.replace
    def corrupt(source,dest):
        replace(source,dest);Path(dest).write_text('{}')
    monkeypatch.setattr(audit.os,'replace',corrupt)
    with pytest.raises(ValueError): audit.atomic_report(target,toy_run[2])
    assert not target.exists()


def test_final_replace_failure_cannot_leave_completed_file(toy_run,tmp_path,monkeypatch):
    target=tmp_path/'report.json'
    def fail(*args): raise OSError('simulated final replacement failure')
    monkeypatch.setattr(audit.os,'replace',fail)
    with pytest.raises(OSError): audit.atomic_report(target,toy_run[2])
    assert not target.exists() and not target.with_suffix('.tmp').exists()


def test_changed_condition_clock_fingerprint_is_rejected(toy_run):
    report=deepcopy(toy_run[2])
    report['results'][0]['conditions']['encoded_feature_reverse']['fixed_inputs']['frames']='f'*64
    with pytest.raises(ValueError,match='fingerprint changed'): audit.validate_report(report)


def test_duplicated_row_cannot_satisfy_complete_denominator(toy_run):
    report=deepcopy(toy_run[2]);report['results'][1]=deepcopy(report['results'][0])
    with pytest.raises(ValueError,match='duplicated'): audit.validate_report(report)


def test_aggregate_is_token_count_weighted():
    rows=[]
    for count,bits in [(2,1.),(6,3.)]:
        rows.append(dict(model='decoder',status='completed',conditions={condition:
            dict(scores=dict(all=dict(count=count,bits=bits))) for condition in audit.CONDITIONS}))
    value=runner.aggregate_rows(rows)['decoder']['conditions']['matched']['scores']['all']
    assert value==dict(count=8,bits=2.5)


@pytest.mark.parametrize('problem',['missing_audio','same_audio','resource'])
def test_row_failure_keeps_denominator_and_never_encodes(toy_run,tmp_path,monkeypatch,problem):
    root,args,_=toy_run;args=deepcopy(args);args.output=str(tmp_path/'failed')
    original_select=runner.select_entries
    def selection(listing,indices):
        rows=original_select(listing,indices)
        if problem=='missing_audio': rows[0]['audio']=str(tmp_path/'absent.wav')
        if problem=='same_audio':
            copy=tmp_path/'identical-bytes-different-path.wav'
            copy.write_bytes(Path(rows[0]['audio']).read_bytes())
            rows[1]['audio']=str(copy)
        return rows
    monkeypatch.setattr(runner,'select_entries',selection)
    if problem=='resource':
        def refuse(*args): raise audit.ReferenceError('budget refused before encoding',{'fake_budget':True})
        monkeypatch.setattr(runner,'enforce_preflight',refuse)
    def no_encode(*args): raise AssertionError('encoder must not be called')
    monkeypatch.setattr(runner,'encode_audio',no_encode)
    report=runner.run(args)
    assert report['status']=='completed_with_failures' and report['failed_rows']==4
    assert all(row['error']['type']!='AssertionError' for row in report['results'])


def test_worker_entry_is_whitelisted():
    from script.train_models import WORKERS
    assert 'script.audit_audio_conditioning_v2' in WORKERS

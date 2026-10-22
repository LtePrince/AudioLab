from types import SimpleNamespace
import json

import pytest
import torch

import script.train_soft_onset as runner
from script.train_soft_onset import (check_size, epoch_diagnostic, evaluation_scope, load_sample,
                                    sample_order, store_features, train, training_rows)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_cache_roundtrip_only_quantizes_audio_features(tmp_path, dtype):
    source = torch.tensor([[[.123456789, -.987654321]]])
    stored, contract = store_features(source, dtype)
    # Low scores, subframe offsets and clocks stay float32, even below fp16 range.
    soft = torch.tensor([[[1e-10, .123456789]]])
    frames = torch.tensor([.123456789, 13.987654321])
    offsets = torch.tensor([.123456789, .987654321])
    sample = dict(soft_onsets=soft, routes={'4k':dict(features=stored, frames=frames,
                                                   offsets=offsets, feature_storage=contract)})
    torch.save(sample, tmp_path/'sample.pt')
    restored = load_sample(tmp_path, {'file':'sample.pt'})
    route = restored['routes']['4k']
    assert stored.dtype == getattr(torch, dtype)
    assert route['features'].dtype == torch.float32
    torch.testing.assert_close(route['features'], source.to(getattr(torch, dtype)).float(), rtol=0, atol=0)
    for key, expected in [('frames', frames), ('offsets', offsets)]:
        torch.testing.assert_close(route[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(restored['soft_onsets'], soft, rtol=0, atol=0)
    assert contract['max_abs_roundtrip_error'] == float((source-route['features']).abs().max())


def test_cache_rejects_float16_overflow():
    with pytest.raises(ValueError, match='overflow'):
        store_features(torch.tensor([[[70000.]]]), 'float16')


def test_artifact_budget_is_explicit_and_legacy_default_is_700(tmp_path):
    path = tmp_path/'tiny';path.write_bytes(b'abcd')
    checks = []
    budget = SimpleNamespace(args=SimpleNamespace(max_artifact_mib=3/1024**2),
                             check=lambda **kw:checks.append(kw))
    with pytest.raises(RuntimeError, match='artifact budget'):
        check_size(tmp_path,budget)
    budget.args.max_artifact_mib=4/1024**2
    assert check_size(tmp_path,budget)['bytes']==4
    del budget.args.max_artifact_mib
    assert check_size(tmp_path,budget)['bytes']==4
    assert all(x=={'force':True} for x in checks)


def test_train_all_includes_original_val_in_every_complete_epoch():
    rows = [dict(id=str(i),split='train' if i<270 else 'val') for i in range(300)]
    args = SimpleNamespace(train_all=True,train_limit=0,val_limit=0,steps=3000)
    selected = training_rows({'rows':rows},args)
    order = sample_order(selected,args.steps,42)
    assert len(selected)==300
    assert all(set(order[i:i+300])==set(range(300)) for i in range(0,3000,300))
    args.train_all=False
    assert len(training_rows({'rows':rows},args))==270


@pytest.mark.parametrize('change', [dict(train_limit=2),dict(val_limit=1),dict(steps=2),dict(steps=0)])
def test_train_all_rejects_partial_epochs_or_sample_limits(change):
    args = SimpleNamespace(train_all=True,train_limit=0,val_limit=0,steps=6)
    for key,value in change.items():setattr(args,key,value)
    with pytest.raises(ValueError,match='train-all'):
        training_rows({'rows':[dict(id=str(i),split='val' if i==2 else 'train') for i in range(3)]},args)


@pytest.mark.parametrize('training,fail', [(True,False),(False,False),(True,True)])
def test_epoch_diagnostic_restores_rng_and_training_state(monkeypatch,training,fail):
    decoder=torch.nn.Linear(2,2).train(training)
    def forward(*args,**kwargs):
        assert not decoder.training and not torch.is_grad_enabled()
        torch.rand(4)
        if fail:raise RuntimeError('diagnostic failed')
        return None,dict(scores={'all':dict(count=2,bits=3.)},conditional_offset=None)
    monkeypatch.setattr(runner,'load_sample',lambda *a:{})
    monkeypatch.setattr(runner,'training_preflight',lambda *a:None)
    monkeypatch.setattr(runner,'forward_loss',forward)
    before=torch.get_rng_state().clone()
    args=(decoder,[dict(id='a')],'.','4k','soft',SimpleNamespace(check=lambda **kw:None),torch.device('cpu'))
    if fail:
        with pytest.raises(RuntimeError,match='diagnostic failed'):epoch_diagnostic(*args)
    else:
        result=epoch_diagnostic(*args)
        assert result['weighted']['scores']['all']==dict(count=2,bits=3.)
    assert decoder.training is training
    assert torch.equal(torch.get_rng_state(),before)


def test_evaluation_scope_comes_from_checkpoint_and_legacy_remains_holdout():
    assert evaluation_scope({})=='development_holdout'
    assert evaluation_scope(dict(training_scope='all'))=='training_diagnostic'
    with pytest.raises(ValueError,match='scope mismatch'):
        evaluation_scope(dict(training_scope='all',evaluation_scope='development_holdout'))


def test_small_complete_training_writes_step_zero_curve_and_only_latest_final(tmp_path,monkeypatch):
    from src.conditioning_audit import file_fingerprint
    from src.data.chart_tokenizer import ChartTokenizer,TokenNote
    from src.data.timing import TIMING_SCHEMA
    from src.models.chart_decoder import ChartDecoder
    from src.soft_onset import CACHE_SCHEMA,load_decoder

    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        torch.manual_seed(8)
        config=dict(hidden_dim=16,depth=1,num_heads=2,mlp_ratio=2,dropout=.1,mem_dim=8,time_dim=8)
        old=ChartDecoder(**config)
        source=tmp_path/'source.pt'
        torch.save(dict(timing_schema=TIMING_SCHEMA,dec_config=config,
                        model={'decoder.'+k:v for k,v in old.state_dict().items()}),source)
        cache=tmp_path/'cache';cache.mkdir();rows=[];tk=ChartTokenizer()
        for i in range(3):
            tokens=tk.encode_notes([TokenNote(1,i,1,0),TokenNote(3,(i+1)%4,1,0)])
            path=cache/f'{i}.pt'
            torch.save(dict(soft_onsets=torch.rand(1,5,2),valid=torch.ones(1,5,dtype=torch.bool),routes={'4k':dict(
                features=torch.randn(1,5,8).half(),tokens=torch.tensor(tokens),
                frames=torch.tensor(tk.ticks_per_position(tokens),dtype=torch.float32),
                offsets=torch.zeros(len(tokens)),offset_mask=torch.zeros(len(tokens),dtype=torch.bool))}),path)
            rows.append(dict(id=str(i),split='val' if i==2 else 'train',file=path.name,sha256=runner.sha256(path)))
        (cache/'cache.json').write_text(json.dumps(dict(schema=CACHE_SCHEMA,status='completed',rows=rows,
                                                       models={'4k':file_fingerprint(source)},contract={})))
        args=SimpleNamespace(cache=str(cache),mode='4k',condition='soft',seed=42,lr=.001,steps=6,
            train_all=True,gradient_checkpointing=True,train_limit=0,val_limit=0,generate_limit=0)
        output=tmp_path/'run';output.mkdir()
        budget=SimpleNamespace(args=SimpleNamespace(max_artifact_mib=700),check=lambda **kw:{},
                               save=lambda payload,path:torch.save(payload,path))
        monkeypatch.setattr(runner,'training_preflight',lambda *a:dict(mock_resource=True))
        report=train(args,output,budget,torch.device('cpu'))
        assert report['status']=='completed' and report['original_split_counts']==dict(train=2,val=1)
        assert [x['step'] for x in report['epoch_diagnostics']]==[0,3,6]
        assert all(x['samples']==1 for x in report['epoch_diagnostics'])
        assert set(report['training_order'][:3])=={'0','1','2'}
        assert sum(x['sample']=='2' for x in report['metrics'])==2  # original val gets optimizer updates
        assert report['evaluation']['evaluation_scope']=='training_diagnostic'
        assert sorted(p.name for p in output.glob('*.pt'))==['decoder_final.pt','decoder_latest.pt']
        model,latest=load_decoder(output/'decoder_latest.pt','cpu')
        assert latest['steps']==6 and latest['checkpoint_kind']=='latest' and 'optimizer' not in latest
        assert latest['gradient_checkpointing'] and 'gradient_checkpointing' not in latest['dec_config']
        assert model.gradient_checkpointing is False  # training implementation switch, not inference architecture
        assert latest['model'].keys()==torch.load(output/'decoder_final.pt',weights_only=True)['model'].keys()
    finally:
        torch.set_num_threads(threads)

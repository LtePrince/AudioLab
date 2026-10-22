"""Small real decoders check the intervention; fixture weights are not quality evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test.test_product_pipeline import inputs, controlled_package_budget, arguments
from src.product import pipeline
from src.models.chart_decoder import ChartDecoder
from src.conditioning_audit import tensor_fingerprint
from script import ablate_layout_features as review


def control(tmp_path):
    return SimpleNamespace(directory=tmp_path,
        budget=SimpleNamespace(args=SimpleNamespace(max_rss_gib=4,min_available_ram_gib=8,gpu_memory_fraction=.65)),
        check=lambda:None,progress=lambda *args:None)


@pytest.mark.parametrize('mode,folder',[('4k','decoder'),('freex','decoder_freex')])
def test_zero_before_prepare_memory_preserves_bias_mask_and_candidate_clock(tmp_path,inputs,monkeypatch,mode,folder):
    seen=[];prepare=ChartDecoder.prepare_memory
    def capture(self, features):
        memory=prepare(self,features)
        seen.append((features.clone(),memory.clone()))
        return memory
    monkeypatch.setattr(ChartDecoder,'prepare_memory',capture)
    mel=torch.ones(128,45); kwargs=dict(max_tokens=128,encoder_mode='full')
    results=[]
    for condition in (None,'matched','zero_acoustic_features'):
        results.append(pipeline.decode_notes(mel,1.04,[0,16,32,48],inputs[1]/folder/'decoder_best.pt',mode,
                     120,.1,42,torch.device('cpu'),control(tmp_path),feature_condition=condition,**kwargs))
    assert results[0][0]==results[1][0]  # Existing default is exactly the matched path.
    assert 'feature_intervention' not in results[0][1]
    assert torch.equal(seen[0][0],seen[1][0]) and torch.count_nonzero(seen[1][0])>0
    assert torch.count_nonzero(seen[2][0])==0 and torch.count_nonzero(seen[2][1])>0
    for notes,details in results:
        assert {n['time'] for n in notes}=={0,16,32,48}
        assert details['termination_policy']=='natural_eos_required'
    a=results[1][1]['feature_intervention']; b=results[2][1]['feature_intervention']
    assert a['source_features']==b['source_features'] and a['valid_mask']==b['valid_mask']
    assert a['source_features']==a['altered_features']
    assert b['source_features']!=b['altered_features']
    assert 'sampled token history' in b['mapping']['remaining_conditioning']
    assert 'true token history' not in b['mapping']['remaining_conditioning']
    assert a['valid_mask']['shape']==[1,23]  # Odd mel tail is retained with minimal padding.


def test_experiment_does_not_bypass_natural_eos_budget(tmp_path,inputs):
    with pytest.raises(RuntimeError,match='before natural EOS'):
        pipeline.decode_notes(torch.ones(128,44),1,[0],inputs[1]/'decoder/decoder_best.pt','4k',
                              120,0,42,torch.device('cpu'),control(tmp_path),max_tokens=1,
                              feature_condition='zero_acoustic_features')


def test_invalid_feature_condition_fails_before_loading(tmp_path,monkeypatch):
    def must_not_load(*a):raise AssertionError('loading before validation')
    monkeypatch.setattr(pipeline,'load_models',must_not_load)
    with pytest.raises(ValueError,match='feature condition'):
        pipeline.decode_notes(None,1,[0],tmp_path/'missing','4k',120,0,42,torch.device('cpu'),None,
                              feature_condition='unknown')


@pytest.fixture
def frozen_source(tmp_path,inputs):
    args=arguments(tmp_path,inputs)
    pipeline.run_job(args)
    return review.parser().parse_args(['--source-job',args.output,'--model-root',str(inputs[1]),
                                     '--output',str(tmp_path/'ablation'),'--device','cpu'])


def test_four_small_model_runs_keep_candidates_and_serialize_outputs(frozen_source):
    result=review.run(frozen_source);root=Path(frozen_source.output)
    assert result['status']=='completed' and result['completed_rows']==4 and result['failed_rows']==0
    assert result['output_bytes']==review.output_bytes(root)
    assert json.loads((root/'report.json').read_text())==result
    assert len(result['previews'])==2
    for row in result['results']:
        assert row['checks']['complete_candidate_coverage']
        assert row['raw']['onset_count']==len(result['fixed_inputs']['candidate_ticks'])
        for file in row['files'].values():assert review.digest(root/file['path'])==file['sha256']
        if row['condition']=='matched':assert all(row['frozen_baseline_bytes_equal'].values())
    for preview in result['previews']:
        folder=(root/preview['path']).parent
        assert (folder/'audio.wav').is_symlink() and (folder/'preview.html').is_file()
    assert result['comparisons']['freex']['raw']['note_identity'].startswith('independent')


def test_one_failed_condition_retains_failure_denominator_without_complete_job(frozen_source,monkeypatch):
    original=review.decode_notes
    def fail_one(*args,**kwargs):
        if args[4]=='freex' and kwargs['feature_condition']=='zero_acoustic_features':
            raise RuntimeError('fixture decode failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(review,'decode_notes',fail_one)
    result=review.run(frozen_source)
    assert result['status']=='completed_with_failures' and result['failed_rows']==1
    assert len(result['results'])==4 and 'freex' not in result['comparisons']
    assert json.loads((Path(frozen_source.output)/'job.json').read_text())['status']=='failed'


def test_final_budget_failure_removes_report_and_previews(frozen_source,monkeypatch):
    original=review.check_output
    def fail_final(output,budget):
        if (output/'report.json').exists():raise RuntimeError('fixture final output budget')
        return original(output,budget)
    monkeypatch.setattr(review,'check_output',fail_final)
    with pytest.raises(RuntimeError,match='final output budget'):review.run(frozen_source)
    root=Path(frozen_source.output)
    assert not (root/'report.json').exists() and not list(root.glob('*/preview.html'))
    assert json.loads((root/'job.json').read_text())['status']=='failed'

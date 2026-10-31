import pytest
import torch

from script.train_windowed_corpus import load_parent_model, natural_reference
from src.anchored_ar import AnchoredAR, encode_target
from src.soft_onset import state_hash
from src.windowed_ar import encode_window_target
from src.data.chart_tokenizer import TokenNote
from test.test_windowed_ar import CONFIG, route


def test_natural_parent_migration_preserves_all_shared_parameters():
    torch.manual_seed(10);parent=AnchoredAR(CONFIG)
    ck=dict(config=dict(variant='audio',decoder_config=CONFIG),model=parent.state_dict())
    a=load_parent_model(ck,'whole',42);b=load_parent_model(ck,'window',42)
    assert not b.bounds_proj.weight.any()
    assert state_hash(a.state_dict())==state_hash(parent.state_dict())
    assert all(torch.equal(t,b.state_dict()[k]) for k,t in parent.state_dict().items())
    bad=dict(ck,model=dict(ck['model']));bad['model'].pop('stop_head.weight')
    with pytest.raises(ValueError,match='migration'):load_parent_model(bad,'window',42)


def test_reference_and_both_targets_share_original_audio_offset():
    r=route([TokenNote(32,1,1),TokenNote(64,3,3,16)]);r['reference']['offset_seconds']=.125
    sample=dict(routes={'4k':r});notes=natural_reference(sample)
    expected=[.625,1.125]
    assert [n['time_seconds'] for n in notes]==expected
    assert notes[1]['end_seconds']==1.375
    assert encode_target(r,3.)['reference_times']==expected
    assert encode_window_target(r,3.)['reference_times']==expected

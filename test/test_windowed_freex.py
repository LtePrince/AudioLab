import math
import pytest
import torch

from script.train_windowed_freex import free_route, score
from src.anchored_ar import AnchoredAR, encode_target, normalized_loss
from src.data.chart_tokenizer import ChartTokenizer, bin_to_hit
from src.soft_onset import MEMORY_SECONDS
from src.windowed_ar import WindowedAR, encode_window_target, generate


CONFIG=dict(vocab_size=369,n_lanes=64,offset_head=True,hidden_dim=32,depth=2,
    num_heads=4,mem_dim=32,time_dim=16,dropout=0.,token_dropout=0.,history_dim=28,soft_onset_dim=2)


def test_same_wave_factors_keep_precise_coordinates_across_window_reencoding():
    events=[(608,4,.923),(640,0,.173),(640,1,.688),(1281,2,.437)]
    route,refs=free_route(events);old=encode_target(route,25.,n_lanes=64)
    new=encode_window_target(route,25.,n_lanes=64)
    eow=ChartTokenizer(64).cond(2)
    assert torch.equal(new['tokens'][new['tokens']!=eow],old['tokens'])
    assert torch.equal(old['offsets'][old['offset_mask']],new['offsets'][new['offset_mask']])
    assert new['coordinate_count']==4 and new['time_count']==3
    for ref in refs:
        assert bin_to_hit(ref['position'],ref['residual'])==pytest.approx(ref['screen_x'],abs=1e-14)
    assert refs[0]['time_seconds']==9.5 and refs[0]['end_seconds']==10.5
    moved,_=free_route([(640,0,.7)])
    assert not torch.equal(route['offsets'],moved['offsets'])


def test_all_64_bins_are_scored_and_continuous_coordinate_error_is_separate():
    route,refs=free_route([(64,4,.923),(128,2,.437)])
    assert all(r['position']>3 for r in refs)
    row=dict(split='train',reference=refs)
    result=dict(notes=[dict(r) for r in refs],onsets_seconds=[r['time_seconds'] for r in refs])
    score(result,row)
    assert result['joint_f1_25ms']==1. and result['matched_position_bin_type_count']==2
    assert result['matched_screen_x_errors']==[0.,0.]
    result['notes'][0]['screen_x']+=.001
    score(result,row)
    assert result['joint_f1_25ms']==1. and max(result['matched_screen_x_errors'])==pytest.approx(.001)


def test_free_x_window_training_reaches_coordinate_and_boundary_parameters():
    torch.set_num_threads(2);torch.manual_seed(42);old=AnchoredAR(CONFIG)
    torch.manual_seed(42);model=WindowedAR(CONFIG)
    for k,v in old.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    route,_=free_route([(608,4,.923),(640,0,.173),(640,1,.688)])
    target=encode_window_target(route,15.,n_lanes=64);length=math.ceil(15/MEMORY_SECONDS)
    condition=dict(features=torch.randn(1,length,32),soft_onsets=torch.rand(1,length,2),
        valid=torch.ones(1,length,dtype=torch.bool),duration=15.)
    normalized_loss(model.components(target,condition,use_soft=True),target).backward()
    for layer in (model.bounds_proj,model.decoder.offset_head[0],model.time_query):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum()>0
    with torch.no_grad():model.stop_head.weight.zero_();model.stop_head.bias.fill_(40)
    output=generate(model.eval(),condition,120.,use_soft=True,greedy=True)
    assert output['status']=='completed' and output['notes']==[]
    assert output['learned_window_end_fraction']==1. and output['learned_song_eos'] is False

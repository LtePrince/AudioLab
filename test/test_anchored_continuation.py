"""Continuation must preserve the experiment and actual optimizer/RNG state."""
from copy import deepcopy
import random

import pytest
import torch

from script.train_alignment_scratch import training_state, restore_training
from script.train_anchored_ar import continuation_config, epoch_objective
from src.anchored_ar import SCHEMA


def parent_and_config():
    old=dict(schema=SCHEMA,variant='soft',decoder_config={'hidden_dim':32},seed=42,
        parameters=3,full_cache_sha256='full',pair_cache_sha256='pairs',loss='same loss',
        policy='same policy',source_sha256={'source':'old'},epochs=10)
    parent=dict(schema=SCHEMA,config=old,epoch=10,steps=3000,
        curves=[dict(epoch=e,steps=e*300) for e in range(1,11)],
        model={'weight':torch.ones(3)},training_state=dict(
            optimizer={'param_groups':[{'lr':3e-5}]},sampler=(),torch_rng=torch.zeros(1),cuda_rng=[]))
    config=deepcopy(old);config.update(epochs=300,source_sha256={'source':'new'})
    return parent,config


def test_continuation_preserves_terminal_lr_and_parent_provenance():
    parent,config=parent_and_config()
    result=continuation_config(parent,config,parent_path='original.pt',parent_sha256='abc')
    assert result['epochs']==300 and parent['config']['epochs']==10
    assert result['source_sha256']=={'source':'new'}
    assert result['continuation']==dict(parent_path='original.pt',parent_sha256='abc',
        parent_epoch=10,parent_steps=3000,parent_source_sha256={'source':'old'},
        schedule='constant_parent_terminal_lr',learning_rates=[3e-5])


@pytest.mark.parametrize('field,value', [('variant','audio'),('full_cache_sha256','other'),
    ('pair_cache_sha256','other'),('loss','other'),('policy','other'),('decoder_config',{})])
def test_continuation_rejects_task_changes(field,value):
    parent,config=parent_and_config();config[field]=value
    with pytest.raises(ValueError,match=field):
        continuation_config(parent,config,parent_path='a',parent_sha256='x')


def test_continuation_rejects_incomplete_epoch_or_missing_rng():
    parent,config=parent_and_config();parent['steps']=2999
    with pytest.raises(ValueError,match='epoch boundary'):
        continuation_config(parent,config,parent_path='a',parent_sha256='x')
    parent,config=parent_and_config();del parent['training_state']['torch_rng']
    with pytest.raises(ValueError,match='RNG'):
        continuation_config(parent,config,parent_path='a',parent_sha256='x')


def test_restoring_adamw_rng_and_sampler_reproduces_next_update():
    torch.set_num_threads(2);torch.manual_seed(3)
    net=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1))
    optimizer=torch.optim.AdamW(net.parameters(),lr=3e-5);sampler=random.Random(17)
    x=torch.randn(5,3)
    def update():
        optimizer.zero_grad();net(x).square().mean().backward();optimizer.step()
        return sampler.sample(range(9),5)
    update()
    weights=deepcopy(net.state_dict());state=deepcopy(training_state(optimizer,sampler))
    expected_order=update();expected_weights=deepcopy(net.state_dict())
    net.load_state_dict(weights);restore_training(state,optimizer,sampler)
    assert update()==expected_order
    for key,value in net.state_dict().items():
        assert torch.equal(value,expected_weights[key])


def test_evaluation_rng_fork_does_not_change_next_dropout():
    torch.manual_seed(13);state=torch.get_rng_state()
    expected=torch.nn.functional.dropout(torch.ones(50),.3,True)
    torch.set_rng_state(state)
    with torch.random.fork_rng(devices=[]):
        torch.nn.Linear(16,16);torch.rand(100)
    actual=torch.nn.functional.dropout(torch.ones(50),.3,True)
    assert torch.equal(expected,actual)


def test_pooled_objective_counts_stop_examples_separately():
    record=dict(frame_ce=1.,subframe_mse=.1,layout_ce=2.,stop_bce=3.,continue_bce=.2,
        counts={'stop_count':2,'continue_count':8})
    assert epoch_objective(record)==pytest.approx(1.+.1+.25*2.+.5*(3.*2+.2*8)/10)

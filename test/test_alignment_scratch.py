import torch

from script.audit_decoder_alignment import configure, oracle_condition
from script.train_alignment_scratch import silent_samples
from src.models.chart_decoder import ChartDecoder
from src.models.transcriber import TranscriberNet
from src.soft_onset import MEMORY_SECONDS, state_hash
from src.data.chart_tokenizer import BOS, EOS


def test_attention_variants_have_identical_initial_parameters():
    hashes=[]
    for variant in ('local','global','time_bias'):
        torch.manual_seed(42)
        decoder=ChartDecoder(hidden_dim=32,depth=1,num_heads=4,mem_dim=32,time_dim=16,
                             history_dim=28,soft_onset_dim=2)
        configure(decoder,variant)
        hashes.append(state_hash(decoder.state_dict()))
    assert len(set(hashes))==1


def test_silent_negatives_use_encoded_audio_and_empty_chart():
    torch.set_num_threads(2)
    model=TranscriberNet(hidden_dim=32,depth=1,num_heads=4,conv_blocks=1).eval()
    samples=silent_samples(model,model,'4k',torch.device('cpu'))
    assert 10. not in [s['duration'] for s in samples]
    for sample in samples:
        row=sample['routes']['4k']
        assert row['tokens'].tolist()==[BOS,EOS]
        assert not row['offset_mask'].any()
        assert torch.isfinite(row['features']).all() and row['features'].abs().max()>0
        assert row['features'].shape[1]==sample['valid'].shape[1]==sample['soft_onsets'].shape[1]
        assert sample['valid'].all()


def test_oracle_condition_preserves_subframe_clock():
    times=[1.,4.,7.]
    condition=oracle_condition(times,220)[0]
    indices=condition[:,0].nonzero().flatten()
    recovered=(indices+condition[indices,1])*MEMORY_SECONDS
    assert torch.allclose(recovered,torch.tensor(times),atol=1e-6)
    assert len(indices)==3

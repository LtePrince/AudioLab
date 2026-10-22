import pytest
import torch

from src.data.chart_tokenizer import ChartTokenizer
from src.models.chart_decoder import ChartDecoder, generate_tokens


def decoder(lanes=4):
    tk = ChartTokenizer(lanes)
    return ChartDecoder(vocab_size=tk.vocab_size, n_lanes=lanes, hidden_dim=32, depth=2,
                        num_heads=4, mem_dim=16, time_dim=16, dropout=0,
                        cross_time_scale=2.).eval()


def test_global_bias_prefers_current_time_without_hiding_distant_audio():
    d = decoder()
    valid = torch.tensor([[True]*99+[False]])
    bias = d.cross_mask(torch.tensor([[5.,50.]]),100,valid)[0,0]
    assert bias[0,5] == bias[1,50] == 0
    assert bias[0,4] > bias[0,0] > bias[0,98]
    assert torch.isfinite(bias[:,:99]).all()
    assert torch.isneginf(bias[:,99]).all()
    assert bias[0,4] == bias[0,6]


@pytest.mark.parametrize('lanes',[4,64])
def test_global_forward_equals_cached_steps_and_ignores_padding(lanes):
    torch.manual_seed(8)
    d = decoder(lanes)
    memory = d.prepare_memory(torch.randn(1,100,16))
    valid = torch.tensor([[True]*99+[False]])
    tokens = torch.tensor([[1,3,50,20,54]])
    frames = torch.tensor([[0.,32.,32.,33.,33.]])
    full = d(tokens,frames,memory,valid)
    caches = d.init_cache(memory)
    steps = torch.stack([d.step(tokens[:,i],frames[:,i],i,caches,100,valid)[0]
                         for i in range(5)],dim=1)
    assert torch.allclose(full,steps,atol=1e-5)
    padded = memory.clone(); padded[:,99] += 100
    assert torch.equal(full,d(tokens,frames,padded,valid))
    distant = memory.clone(); distant[:,98] += 100
    assert not torch.allclose(full,d(tokens,frames,distant,valid),atol=1e-5)
    changed = tokens.clone(); changed[:,3:] = 10
    assert torch.allclose(full[:,:3],d(changed,frames,memory,valid)[:,:3],atol=1e-5)


def test_greedy_is_seed_independent_and_old_config_loads():
    d=decoder(); memory=d.prepare_memory(torch.randn(1,20,16))
    args=dict(decoder=d,memory=memory,mem_valid=torch.ones(1,20,dtype=torch.bool),
              bpm=120,frame_ms=46.44,audio_frames=20,max_tokens=64,greedy=True)
    assert generate_tokens(**args,generator=torch.Generator().manual_seed(1)) == \
           generate_tokens(**args,generator=torch.Generator().manual_seed(9))
    with pytest.raises(ValueError,match='global time bias'):
        ChartDecoder(cross_window=64,cross_time_scale=64)

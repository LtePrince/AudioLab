"""test_chart_decoder.py — AR decoder 单元测试

  [T1] 前向形状 + 变长 memory
  [T2] ticks_per_position 与时间嵌入一致
  [T3] 随机权重下的约束采样:输出必须通过严格语法解码,且到达音频末尾后收尾
  [T4] 因果性:改动未来 token 不影响当前位置 logits
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from src.data.chart_tokenizer import BOS, EOS, ChartTokenizer, TokenNote
from src.models.chart_decoder import ChartDecoder, generate_tokens
from src.models.transcriber import TranscriberNet

CFG = dict(vocab_size=129, hidden_dim=64, depth=2, num_heads=4, mem_dim=48, time_dim=32)


@pytest.fixture(scope="module")
def dec():
    torch.manual_seed(0)
    return ChartDecoder(**CFG).eval()


def test_forward_shape(dec):
    feats = torch.randn(2, 100, 48)
    mem = dec.prepare_memory(feats)
    valid = torch.ones(2, 100, dtype=torch.bool); valid[1, 60:] = False
    toks = torch.randint(0, 129, (2, 17)); frames = torch.rand(2, 17) * 100
    out = dec(toks, frames, mem, valid)
    assert out.shape == (2, 17, 129) and torch.isfinite(out).all()
    print("[T1] 前向形状 ✓")


def test_ticks_per_position():
    tk = ChartTokenizer()
    toks = tk.encode_notes([TokenNote(128, 0, 1), TokenNote(128, 3, 3, 25), TokenNote(160, 0, 4)])
    ticks = tk.ticks_per_position(toks)
    names = [tk.token_name(t) for t in toks]
    # BOS@0, DTB_4@0, NOTE@128, NOTE@128, DURT@128, DTB_1@128, NOTE@160, EOS@160
    assert ticks == [0, 0, 128, 128, 128, 128, 160, 160], list(zip(names, ticks))
    print("[T2] ticks_per_position ✓")


def test_constrained_generation_is_valid(dec):
    tk = ChartTokenizer()
    feats = torch.randn(1, 40, 48)
    mem = dec.prepare_memory(feats); valid = torch.ones(1, 40, dtype=torch.bool)
    g = torch.Generator().manual_seed(1)
    for temp in (1.0, 2.5):        # high temperature stresses the mask
        toks = generate_tokens(dec, mem, valid, bpm=120.0, frame_ms=46.44,
                               audio_frames=40, temperature=temp, top_p=1.0,
                               max_tokens=400, generator=g)
        assert toks[0] == BOS and toks[-1] == EOS
        notes = tk.decode_tokens(toks, strict=True)      # 语法违规会抛异常
        # 音乐时钟不应远超音频末尾(允许最后一个 DT 跨过一点)
        last_tick = tk.ticks_per_position(toks)[-1]
        assert last_tick * (60/(32*120.0)) * 1000/46.44 < 40 + 32*16*(60/(32*120))*1000/46.44
    print("[T3] 约束采样合法 ✓")


def test_causality(dec):
    feats = torch.randn(1, 30, 48); mem = dec.prepare_memory(feats)
    valid = torch.ones(1, 30, dtype=torch.bool)
    toks = torch.randint(3, 66, (1, 12)); frames = torch.zeros(1, 12)
    a = dec(toks, frames, mem, valid)[0, :6]
    toks2 = toks.clone(); toks2[0, 8:] = 5
    b = dec(toks2, frames, mem, valid)[0, :6]
    assert torch.allclose(a, b, atol=1e-5), "未来 token 泄漏到了当前位置"
    print("[T4] 因果性 ✓")


def test_encoder_features_shape():
    enc = TranscriberNet(n_mels=128, hidden_dim=64, conv_blocks=1, depth=1, num_heads=4)
    f = enc.features(torch.randn(1, 128, 200))
    assert f.shape == (1, 100, 64)
    print("[T5] encoder.features 形状 ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

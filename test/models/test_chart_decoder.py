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


def test_cross_window_mask():
    """局部窗:token 只看 |frame-f| ≤ window 的 memory 帧;窗全落在无效区时回退到最近帧。"""
    d = ChartDecoder(**{**CFG, "cross_window": 2})
    frames = torch.tensor([[0.0, 5.0, 40.0]])          # 第 3 个 token 超出 memory(T=10)
    valid = torch.ones(1, 10, dtype=torch.bool); valid[0, 8:] = False
    m = d.cross_mask(frames, 10, valid)[0, 0]          # (L, T)
    assert m[0].nonzero().flatten().tolist() == [0, 1, 2]
    assert m[1].nonzero().flatten().tolist() == [3, 4, 5, 6, 7]
    assert m[2].sum() == 1 and m[2, 9]                  # 回退:最近帧(钳位到 T-1)
    print("[T6] 局部交叉注意力窗 ✓")


def test_token_dropout_train_only():
    torch.manual_seed(0)
    d = ChartDecoder(**{**CFG, "token_dropout": 0.5}).eval()
    mem = d.prepare_memory(torch.randn(1, 20, 48)); valid = torch.ones(1, 20, dtype=torch.bool)
    toks = torch.randint(3, 66, (1, 8)); fr = torch.zeros(1, 8)
    a = d(toks, fr, mem, valid); b = d(toks, fr, mem, valid)
    assert torch.allclose(a, b), "eval 模式不应有随机 dropout"
    d.train()
    c = d(toks, fr, mem, valid); e = d(toks, fr, mem, valid)
    assert not torch.allclose(c, e), "train 模式 token dropout 应引入随机性"
    print("[T7] token dropout 仅训练时 ✓")


def test_kv_cache_matches_full_forward():
    """增量 step(KV cache)的 logits 必须与全量前向逐位置一致(局部窗 + 全局两种)。"""
    for cw in (None, 3):
        torch.manual_seed(2)
        d = ChartDecoder(**{**CFG, "cross_window": cw}).eval()
        mem = d.prepare_memory(torch.randn(1, 25, 48)); valid = torch.ones(1, 25, dtype=torch.bool)
        toks = torch.randint(3, 66, (1, 9)); frames = torch.cumsum(torch.rand(1, 9) * 4, dim=1)
        full = d(toks, frames, mem, valid)[0]                          # (9, V)
        caches = d.init_cache(mem)
        for i in range(9):
            lg, _ = d.step(toks[:, i], frames[:, i], i, caches, 25, valid)
            assert torch.allclose(lg[0], full[i], atol=1e-4), f"cw={cw} pos {i} 不一致"
    print("[T8] KV cache ≡ 全量前向 ✓")


def test_skeleton_forced_generation():
    """骨架模式:生成的 onset tick 集合必须精确等于骨架;每个骨架时刻至少 1 个音符;语法合法。"""
    from src.data.chart_tokenizer import ChartTokenizer
    tk = ChartTokenizer()
    torch.manual_seed(5)
    d = ChartDecoder(**CFG).eval()
    mem = d.prepare_memory(torch.randn(1, 200, 48)); valid = torch.ones(1, 200, dtype=torch.bool)
    for skel in ([0, 16, 48, 96, 160, 400], [32, 33, 64, 1000], [5]):
        toks = generate_tokens(d, mem, valid, bpm=120.0, frame_ms=46.44, audio_frames=200,
                               temperature=1.5, top_p=1.0, max_tokens=400,
                               generator=torch.Generator().manual_seed(9), skeleton_ticks=skel)
        notes = tk.decode_tokens(toks, strict=True)
        got = sorted({n.tick for n in notes})
        assert got == sorted(skel), f"skeleton {skel} → onsets {got}"
        assert toks[-1] == EOS
    print("[T9] 骨架强制生成 ✓")


def test_skeleton_from_transcriber_monotone():
    """logits 骨架:返回排序去重 tick;阈值越高 onset 越少;tick 与帧数一致。"""
    import sys
    sys.path.insert(0, ".")
    from src.test_decoder import skeleton_from_transcriber
    torch.manual_seed(0)
    tr = TranscriberNet(n_mels=128, hidden_dim=64, conv_blocks=1, depth=1, num_heads=4).eval()
    mel = torch.randn(1, 128, 400)                     # 200 chart frames
    counts = []
    for thr in (0.05, 0.3, 0.6, 0.9):
        sk = skeleton_from_transcriber(mel, tr, bpm=120.0, frame_ms=46.44, threshold=thr)
        assert sk == sorted(set(sk)) and all(isinstance(x, int) for x in sk)
        assert all(0 <= x <= 200 * 46.44 / 1000 * 120 * 32 / 60 + 32 for x in sk)
        counts.append(len(sk))
    assert counts == sorted(counts, reverse=True), f"阈值升高 onset 应单调减少: {counts}"
    print("[T10] 转录骨架提取 ✓")

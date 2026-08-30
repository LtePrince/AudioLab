"""test_freeform_tokens.py — free-x 词表/编码/模型部件测试

  [T1] hit_to_bin / bin_to_hit 往返(bin 内误差 < 1 bin 宽)
  [T2] 64 轨词表布局(369 token)与 4 轨语法复用:encode_gameplay 输出可严格解码
  [T3] offset 目标与 NOTE 位置对齐
  [T4] 因子化嵌入:NOTE 嵌入 == E_type + E_pos;非 NOTE 走普通表
  [T5] offset head + 生成返回 NOTE 位置的 offsets
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from src.data.chart_tokenizer import (
    FREE_BINS, BOS, EOS, ChartTokenizer, VocabLayout,
    bin_to_hit, encode_gameplay, free_notes_to_phigros, hit_to_bin,
)
from src.data.gameplay_miner import GameplayNote
from src.models.chart_decoder import ChartDecoder, generate_tokens


def _g(tick, type_, hit_x, dur=0):
    return GameplayNote(tick=tick, type=type_, dur=dur, pos_x=0.0, hit_x=hit_x, hit_y=0.2,
                        theta=0.0, line_id=0, above=True, moving=False, rotating=False,
                        tail_x=hit_x, tail_y=0.2)


def test_bin_roundtrip():
    w = (0.95 - 0.05) / FREE_BINS
    for x in (0.05, 0.3, 0.5, 0.7777, 0.9499):
        b, off = hit_to_bin(x)
        assert 0 <= b < FREE_BINS and 0 <= off < 1
        assert abs(bin_to_hit(b, off) - x) < 1e-9
    assert hit_to_bin(-1.0)[0] == 0 and hit_to_bin(2.0)[0] == FREE_BINS - 1   # 钳位
    print("[T1] bin 往返 ✓")


def test_layout_and_grammar_reuse():
    assert VocabLayout(4).vocab_size == 129 and VocabLayout(64).vocab_size == 369
    tk = ChartTokenizer(n_lanes=64)
    notes = [_g(128, 1, 0.20), _g(128, 3, 0.80, dur=25), _g(160, 4, 0.5), _g(160, 2, 0.55)]
    toks, offs, mask = encode_gameplay(tk, notes)
    back = tk.decode_tokens(toks, strict=True)        # 语法/链规则在 64 轨下同样成立
    assert len(back) == 4 and back[1].type == 3 and back[1].dur == 25
    assert toks[0] == BOS and toks[-1] == EOS
    assert tk.token_name(toks[2]).startswith("NOTE_L")
    print("[T2] 64 轨词表 + 语法复用 ✓")


def test_offsets_align_with_notes():
    tk = ChartTokenizer(n_lanes=64)
    notes = [_g(0, 1, 0.31), _g(32, 1, 0.62)]
    toks, offs, mask = encode_gameplay(tk, notes)
    note_pos = [i for i, t in enumerate(toks) if tk.kind(t) == "NOTE"]
    assert [i for i, m in enumerate(mask) if m] == note_pos
    for i, x in zip(note_pos, (0.31, 0.62)):
        b = tk.value(toks[i]) // 4
        assert abs(bin_to_hit(b, offs[i]) - x) < 1e-9
    assert all(offs[i] == 0.0 for i, m in enumerate(mask) if not m)
    # 回写 Phigros:positionX 还原 hit_x
    back = tk.decode_tokens(toks)
    ticks = tk.ticks_per_position(toks)
    od = {(ticks[i], tk.value(toks[i]) // 4): offs[i] for i in note_pos}
    pn = free_notes_to_phigros(tk, back, od, bpm=120.0)
    assert abs(0.5 + pn[0]["positionX"] * 0.05625 - 0.31) < 1e-6
    print("[T3] offset 对齐 + positionX 还原 ✓")


def test_factorized_embedding():
    torch.manual_seed(0)
    dec = ChartDecoder(vocab_size=369, hidden_dim=32, depth=1, num_heads=4, mem_dim=16,
                       time_dim=16, n_lanes=64, factorized=True, offset_head=True)
    tk = ChartTokenizer(n_lanes=64)
    tok = tk.note(lane=17, type_=3)
    e = dec.embed(torch.tensor([[tok, BOS]]))[0]
    expect = dec.pos_emb.weight[17] + dec.type_emb.weight[2]
    assert torch.allclose(e[0], expect) and torch.allclose(e[1], dec.tok_emb.weight[BOS])
    print("[T4] 因子化嵌入 ✓")


def test_generation_returns_offsets():
    torch.manual_seed(1)
    dec = ChartDecoder(vocab_size=369, hidden_dim=32, depth=1, num_heads=4, mem_dim=16,
                       time_dim=16, n_lanes=64, factorized=True, offset_head=True).eval()
    tk = ChartTokenizer(n_lanes=64)
    mem = dec.prepare_memory(torch.randn(1, 30, 16)); valid = torch.ones(1, 30, dtype=torch.bool)
    toks, offs = generate_tokens(dec, mem, valid, bpm=120.0, frame_ms=46.44, audio_frames=30,
                                 temperature=2.0, top_p=1.0, max_tokens=200,
                                 generator=torch.Generator().manual_seed(3), tokenizer=tk)
    tk.decode_tokens(toks, strict=True)
    note_pos = {i for i, t in enumerate(toks) if tk.kind(t) == "NOTE"}
    assert set(offs.keys()) == note_pos, "每个 NOTE 位置都应有 offset,且仅 NOTE 有"
    assert all(0.0 < o < 1.0 for o in offs.values())
    print("[T5] 生成返回 offsets ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

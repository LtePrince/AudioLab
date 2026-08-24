"""test_chart_tokenizer.py — AR 词表编解码测试

  [T1] 129 词表双射:id↔kind/name/构造器一致
  [T2] 文档示例:アンビバレンス开头 → 预期 token 序列
  [T3] 合成往返:和弦/长静默链/长 Hold 链/整拍边界全部无损
  [T4] 规范唯一性:encode(decode(encode(x))) == encode(x)
  [T5] 语法状态机:合法序列全过,违规序列逐类拒绝
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest

from src.data.chart_tokenizer import (
    BOS, EOS, PAD, VOCAB_SIZE,
    ChartTokenizer, GrammarState, TokenNote,
)

TK = ChartTokenizer()


def test_vocab_bijection():
    names = set()
    for tok in range(VOCAB_SIZE):
        k = TK.kind(tok)
        name = TK.token_name(tok)
        assert name not in names, f"重名: {name}"
        names.add(name)
    assert TK.dtb(4) != TK.dtt(4) != TK.durb(4) != TK.durt(4)
    assert TK.note(0, 1) == 50 and TK.note(3, 4) == 65
    assert TK.token_name(TK.note(2, 3)) == "NOTE_L2_HOLD"
    assert TK.token_name(TK.dtb(16)) == "DTB_16"
    with pytest.raises(ValueError):
        TK.kind(VOCAB_SIZE)
    print("[T1] 词表双射 ✓")


def test_doc_example():
    """アンビバレンス开头(真实 tick):128:Tap@L0+Hold@L3(25) 160:Tap@L0+Flick@L3 192:Tap@L0+Tap@L2"""
    notes = [
        TokenNote(128, 0, 1), TokenNote(128, 3, 3, 25),
        TokenNote(160, 0, 1), TokenNote(160, 3, 4),
        TokenNote(192, 0, 1), TokenNote(192, 2, 1),
    ]
    toks = TK.encode_notes(notes)
    expected = ["BOS", "DTB_4", "NOTE_L0_TAP", "NOTE_L3_HOLD", "DURT_25",
                "DTB_1", "NOTE_L0_TAP", "NOTE_L3_FLICK",
                "DTB_1", "NOTE_L0_TAP", "NOTE_L2_TAP", "EOS"]
    assert [TK.token_name(t) for t in toks] == expected
    assert TK.decode_tokens(toks) == sorted(notes, key=lambda n: (n.tick, n.lane))
    print("[T2] 文档示例 ✓")


def test_synthetic_roundtrip():
    cases = [
        # 和弦(四押)
        [TokenNote(0, k, 1) for k in range(4)],
        # 超长静默 1632 ticks = DTB_16×3 + DTB_3
        [TokenNote(0, 0, 1), TokenNote(1632, 1, 2)],
        # 超长 Hold 1015 ticks = DURB_16 DURB_15 DURT_23
        [TokenNote(5, 2, 3, 1015)],
        # 整拍边界:gap 32 → DTB_1;gap 31 → DTT_31;dur 512 → DURB_16 单枚
        [TokenNote(0, 0, 1), TokenNote(32, 0, 1), TokenNote(63, 0, 3, 512)],
        # 双押 Hold+Tap + 紧接 1 tick 后的 Drag
        [TokenNote(10, 1, 3, 7), TokenNote(10, 2, 1), TokenNote(11, 3, 2)],
        # 空谱面
        [],
    ]
    for notes in cases:
        toks = TK.encode_notes(notes)
        back = TK.decode_tokens(toks, strict=True)
        assert back == sorted(notes, key=lambda n: (n.tick, n.lane)), \
            f"往返失败: {notes} → {[TK.token_name(t) for t in toks]} → {back}"
    # 长静默/长 Hold 的链式结构断言
    toks = TK.encode_notes(cases[1])
    assert [TK.token_name(t) for t in toks[2:6]] == ["DTB_16", "DTB_16", "DTB_16", "DTB_3"]
    toks = TK.encode_notes(cases[2])
    assert [TK.token_name(t) for t in toks[-4:-1]] == ["DURB_16", "DURB_15", "DURT_23"]
    print("[T3] 合成往返 ✓")


def test_canonical_uniqueness():
    notes = [TokenNote(0, 0, 1), TokenNote(0, 3, 3, 40),
             TokenNote(100, 2, 4), TokenNote(700, 1, 2)]
    t1 = TK.encode_notes(notes)
    t2 = TK.encode_notes(TK.decode_tokens(t1))
    assert t1 == t2, "规范编码应当幂等"
    # 乱序输入 → 同一编码
    import random
    shuffled = notes[:]
    random.Random(0).shuffle(shuffled)
    assert TK.encode_notes(shuffled) == t1
    print("[T4] 规范唯一性 ✓")


def test_grammar():
    ok_cases = [
        [BOS, EOS],
        [BOS, TK.cond(3), TK.dtb(4), TK.note(0, 1), EOS],
        [BOS, TK.dtb(16), TK.dtb(2), TK.dtt(5), TK.note(1, 3), TK.durb(16), TK.durb(3), EOS],
        [BOS, TK.note(0, 3), TK.durb(1), TK.durt(8), EOS],   # 非满块 DURB 后接余数 DURT
    ]
    for seq in ok_cases:
        TK.decode_tokens(seq, strict=True)   # 不应抛异常

    bad_cases = [
        ([BOS, TK.note(0, 3), EOS],                 "Hold 无时长即 EOS"),
        ([BOS, TK.dtt(5), TK.dtt(5)],               "双 DTT 余数"),
        ([BOS, TK.dtb(3), TK.dtb(2)],               "非满块后续 DTB 链"),
        ([BOS, TK.note(0, 1), TK.note(0, 2)],       "同和弦同轨道"),
        ([BOS, TK.durb(1)],                          "无 Hold 的 DURB"),
        ([BOS, TK.note(0, 3), TK.durb(1), TK.durb(1)], "非满块后续 DURB 链"),
        ([BOS, TK.note(0, 1), TK.cond(0)],          "COND 不在 BOS 后"),
        ([BOS, BOS],                                 "重复 BOS"),
        ([TK.note(0, 1)],                            "缺 BOS"),
        ([BOS, TK.note(0, 3), TK.durt(5), TK.durt(5)], "双 DURT 余数"),
    ]
    for seq, why in bad_cases:
        with pytest.raises(ValueError):
            TK.decode_tokens(seq, strict=True)
            pytest.fail(f"应当拒绝: {why}")
    # 合法序列:满块 DTB_16 后允许继续,DTB_16 DTB_16 DTB_16 链
    gs = GrammarState()
    for t in [BOS, TK.dtb(16), TK.dtb(16), TK.dtb(16), TK.dtb(3)]:
        assert gs.is_allowed(t)
        gs.step(t)
    assert not gs.is_allowed(TK.dtb(1)), "非满块 DTB_3 后不得续链"
    assert gs.is_allowed(TK.dtt(7))
    print("[T5] 语法状态机 ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

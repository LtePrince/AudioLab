"""test_transcriber.py — 转录基线单元测试

  [T1] 变长前向:任意偶数 T_mel → 输出 (B, 32, T_mel/2)
  [T2] 损失有限、可反传、各分量合理
  [T3] to_note_array 通道映射 + Phigros4kConvertor 解码闭环
  [T4] onset_f1:完美预测 F1=1,空预测 F1=0,容差匹配生效
  [T5] CE 键型头:无序数收缩(能自信输出 Flick)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch

from src.data.chart2array import (
    CH_IS_START, CH_NOTE_TYPE, NUM_LANES, Phigros4kConvertor,
)
from src.models.transcriber import (
    DEFAULT_TRANSCRIBER_CONFIG,
    H_ONSET, H_TYPE,
    TranscriberNet, TranscriptionLoss,
    onset_f1, to_note_array,
)

CFG = {**DEFAULT_TRANSCRIBER_CONFIG,
       "hidden_dim": 64, "depth": 2, "conv_blocks": 2, "num_heads": 4}
FRAME_MS = 512 / 22050 / 4 * 8 * 1000


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return TranscriberNet(**CFG)


def test_variable_length_forward(model):
    for T_chart in (64, 200, 1024):
        mel = torch.randn(2, 128, T_chart * 2)
        out = model(mel)
        assert out.shape == (2, 32, T_chart), f"T={T_chart}: {out.shape}"
    print("[T1] 变长前向 ✓")


def test_loss_backward(model):
    T = 128
    mel  = torch.randn(2, 128, T * 2)
    note = torch.zeros(2, 20, T)
    # 造几个真实音符:lane0 Tap@10, lane2 Flick@50 带偏移
    note[:, CH_IS_START + 0, 10] = 1.0
    note[:, CH_NOTE_TYPE + 0, 10] = 0.25
    note[:, CH_IS_START + 2, 50] = 1.0
    note[:, CH_NOTE_TYPE + 2, 50] = 1.0
    flag = torch.ones(2, T)

    pred = model(mel)
    loss_fn = TranscriptionLoss()
    loss, log = loss_fn(pred, note, flag)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.zero_grad()
    assert set(log) == {"loss_onset", "loss_holding", "loss_soff",
                        "loss_eoff", "loss_type"}
    print(f"[T2] loss={loss.item():.4f} 反传 ✓")


def test_note_array_roundtrip():
    """手工构造 head 输出 → to_note_array → convertor 解码,音符应还原。"""
    T = 64
    pred = torch.full((1, 32, T), -10.0)          # 全部"无音符"
    pred[:, H_TYPE:] = 0.0
    # lane1 @ frame 20: 强 onset,类型 Flick(class 3)
    pred[0, H_ONSET + 1, 20] = 10.0
    pred[0, H_TYPE + 4 * 1 + 3, 20] = 10.0        # lane1 的 Flick logit
    # lane3 @ frame 40: Tap
    pred[0, H_ONSET + 3, 40] = 10.0
    pred[0, H_TYPE + 4 * 3 + 0, 40] = 10.0

    note = to_note_array(pred).squeeze(0).numpy()
    conv  = Phigros4kConvertor(frame_ms=FRAME_MS, max_frame=T, from_logits=True)
    notes = conv.array_to_notes(note, bpm=120.0)
    assert len(notes) == 2, f"应解码出 2 个音符,得到 {len(notes)}"
    by_x = sorted(notes, key=lambda n: n["positionX"])
    assert by_x[0]["type"] == 4, "lane1 应为 Flick(4)"
    assert by_x[1]["type"] == 1, "lane3 应为 Tap(1)"
    print("[T3] note_array 闭环 ✓")


def test_onset_f1():
    T = 100
    tgt  = torch.zeros(1, 20, T)
    flag = torch.ones(1, T)
    for k, fr in [(0, 10), (1, 30), (2, 60)]:
        tgt[0, CH_IS_START + k, fr] = 1.0

    perfect = torch.full((1, 32, T), -10.0)
    for k, fr in [(0, 10), (1, 30), (2, 60)]:
        perfect[0, H_ONSET + k, fr] = 10.0
    f1, p, r = onset_f1(perfect, tgt, flag)
    assert f1 == pytest.approx(1.0), f"完美预测 F1 应为 1,得 {f1}"

    empty = torch.full((1, 32, T), -10.0)
    f1e, _, _ = onset_f1(empty, tgt, flag)
    assert f1e == 0.0

    # 容差:偏移 2 帧仍算命中,偏移 3 帧不算
    off2 = torch.full((1, 32, T), -10.0)
    off2[0, H_ONSET + 0, 12] = 10.0                # 10 → 12,±2 内
    f1o, _, _ = onset_f1(off2, tgt, flag, tol_frames=2)
    assert f1o > 0.0
    off3 = torch.full((1, 32, T), -10.0)
    off3[0, H_ONSET + 0, 13] = 10.0
    f1x, _, _ = onset_f1(off3, tgt, flag, tol_frames=2)
    assert f1x == 0.0
    print("[T4] onset_f1 ✓")


def test_type_ce_no_mean_collapse():
    """CE 头对极端类别(Flick)可自信输出——序数回归做不到这一点。"""
    T = 32
    torch.manual_seed(1)
    model = TranscriberNet(**CFG)
    loss_fn = TranscriptionLoss()
    mel  = torch.randn(1, 128, T * 2)
    note = torch.zeros(1, 20, T)
    for fr in range(0, T, 4):                      # lane0 全是 Flick
        note[0, CH_IS_START + 0, fr] = 1.0
        note[0, CH_NOTE_TYPE + 0, fr] = 1.0
    flag = torch.ones(1, T)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(60):
        loss, _ = loss_fn(model(mel), note, flag)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = model(mel)
        cls = pred[0, H_TYPE:H_TYPE + 4].argmax(dim=0)   # lane0 各帧类别
    onset_frames = list(range(0, T, 4))
    flick_rate = (cls[onset_frames] == 3).float().mean().item()
    assert flick_rate > 0.9, f"过拟合 60 步后 Flick 命中率应≈1,得 {flick_rate}"
    print(f"[T5] Flick 命中率 {flick_rate:.2f} ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

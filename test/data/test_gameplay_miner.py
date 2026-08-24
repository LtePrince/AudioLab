"""test_gameplay_miner.py — 玩法层挖掘的几何正确性测试

  [T1] 静态水平线:hit = (line_x + posX·0.05625, line_y)
  [T2] 旋转 90° 线:偏移全部转到 y 轴且带 16:9 纵横比修正(0.1/单位)
  [T3] 事件中点插值:移动线在 t 中点的位置
  [T4] formatVersion 1 打包坐标解包
  [T5] Hold 尾部随线漂移:tail_x 反映 hold 结束时刻的线位置
  [T6] JSONL 往返
"""

from __future__ import annotations

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest

from src.data.gameplay_miner import (
    GameplayNote, X_UNIT_H, X_UNIT_W, load_jsonl, mine_chart, save_jsonl,
)


def _chart(lines, fmt=3):
    return {"formatVersion": fmt, "offset": 0.0, "judgeLineList": lines}


def _line(notes_above, move=None, rotate=None, bpm=120.0):
    return {
        "bpm": bpm,
        "notesAbove": notes_above, "notesBelow": [],
        "speedEvents": [{"startTime": 0.0, "endTime": 1e9, "value": 2.0}],
        "judgeLineMoveEvents": move or [
            {"startTime": -1e6, "endTime": 1e9,
             "start": 0.5, "end": 0.5, "start2": 0.2, "end2": 0.2}],
        "judgeLineRotateEvents": rotate or [
            {"startTime": -1e6, "endTime": 1e9, "start": 0.0, "end": 0.0}],
        "judgeLineDisappearEvents": [
            {"startTime": -1e6, "endTime": 1e9, "start": 1.0, "end": 1.0}],
    }


def _note(time, type_=1, pos_x=0.0, hold=0.0):
    return {"type": type_, "time": time, "positionX": pos_x,
            "holdTime": hold, "speed": 1.0, "floorPosition": 0.0}


def _mine(chart) -> list[GameplayNote]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(chart, f)
        path = f.name
    try:
        return mine_chart(path)
    finally:
        os.unlink(path)


def test_static_horizontal():
    g = _mine(_chart([_line([_note(100, pos_x=4.0)])]))[0]
    assert g.hit_x == pytest.approx(0.5 + 4.0 * X_UNIT_W)
    assert g.hit_y == pytest.approx(0.2)
    assert g.theta == 0.0 and not g.moving and not g.rotating
    print("[T1] 静态水平线 ✓")


def test_rotated_vertical():
    rot = [{"startTime": -1e6, "endTime": 1e9, "start": 90.0, "end": 90.0}]
    g = _mine(_chart([_line([_note(100, pos_x=3.0)], rotate=rot)]))[0]
    assert g.hit_x == pytest.approx(0.5, abs=1e-9)
    assert g.hit_y == pytest.approx(0.2 + 3.0 * X_UNIT_H)   # 16:9 修正:0.1/单位
    assert g.theta == pytest.approx(90.0)
    print("[T2] 旋转 90° + 纵横比 ✓")


def test_midpoint_interpolation():
    move = [{"startTime": 0.0, "endTime": 200.0,
             "start": 0.0, "end": 1.0, "start2": 0.2, "end2": 0.6}]
    g = _mine(_chart([_line([_note(100)], move=move)]))[0]
    assert g.hit_x == pytest.approx(0.5) and g.hit_y == pytest.approx(0.4)
    assert g.moving
    print("[T3] 事件中点插值 ✓")


def test_format_v1_unpack():
    # v1 打包: v = trunc(x·880)·1000 + y·520 → (440, 104) = (0.5, 0.2)
    move = [{"startTime": -1e6, "endTime": 1e9,
             "start": 440 * 1000 + 104, "end": 440 * 1000 + 104}]
    line = _line([_note(50, pos_x=2.0)], move=move)
    g = _mine(_chart([line], fmt=1))[0]
    assert g.hit_x == pytest.approx(0.5 + 2.0 * X_UNIT_W)
    assert g.hit_y == pytest.approx(0.2)
    print("[T4] v1 打包坐标 ✓")


def test_hold_tail_drift():
    move = [{"startTime": 0.0, "endTime": 100.0,
             "start": 0.3, "end": 0.3, "start2": 0.2, "end2": 0.2},
            {"startTime": 100.0, "endTime": 200.0,
             "start": 0.3, "end": 0.7, "start2": 0.2, "end2": 0.2}]
    g = _mine(_chart([_line([_note(100, type_=3, hold=100.0)], move=move)]))[0]
    assert g.hit_x  == pytest.approx(0.3)      # 击打时线在 0.3
    assert g.tail_x == pytest.approx(0.7)      # 结束时线漂到 0.7
    assert g.dur == 100
    print("[T5] Hold 尾部漂移 ✓")


def test_jsonl_roundtrip(tmp_path):
    notes = _mine(_chart([_line([_note(10), _note(20, type_=3, pos_x=-2.5, hold=30.0)])]))
    p = tmp_path / "g.jsonl"
    save_jsonl(notes, p)
    assert load_jsonl(p) == notes
    print("[T6] JSONL 往返 ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

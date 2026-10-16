"""Regression tests across mining, audio labels, AR clocks and chart export."""
import json

import numpy as np
import pytest
import torch

from src.data.chart2array import Phigros4kConvertor, parse_phigros_file
from src.data.chart_tokenizer import ChartTokenizer
from src.data.dataset import PhigrosDataset
from src.data.gameplay_miner import load_jsonl, mine_chart, save_jsonl
from src.data.timing import TIMING_SCHEMA, check_checkpoint_timing
from src.models.chart_decoder import ChartDecoder, generate_tokens
from src.test_decoder import skeleton_from_chart, skeleton_from_transcriber
from src.train_decoder import TokenChartDataset
from script.eval_onset_f1 import onsets
from test.data.test_gameplay_miner import _chart, _line, _note


@pytest.fixture
def source(tmp_path):
    # Both notes start at audio 0.75 s, end at 1.25 s, on different lanes.
    chart = _chart([_line([_note(32, 3, -3.75, 32)], bpm=120),
                    _line([_note(64, 3, 3.75, 64)], bpm=240)])
    chart["offset"] = 0.25
    p = tmp_path / "song" / "chart.json"
    p.parent.mkdir()
    p.write_text(json.dumps(chart))
    return p


def test_mining_tokenizer_and_skeleton_share_bpm(source):
    notes = mine_chart(source)
    assert [(n.tick, n.dur) for n in notes] == [(32, 32), (32, 32)]
    tk = ChartTokenizer()
    assert [(n.tick, n.dur) for n in tk.decode_tokens(tk.encode_chart(str(source)))] == [(32, 32)] * 2
    assert skeleton_from_chart(str(source)) == [32]
    # Retarget into a zero-offset 240 BPM chart via absolute audio time.
    assert skeleton_from_chart(str(source), bpm=240, offset=0) == [96]
    assert [n[0] for n in onsets(str(source))] == [750, 750]


def test_hold_geometry_uses_unrounded_native_tail(tmp_path):
    move = [{"startTime": 0, "endTime": 200, "start": 0, "end": 1,
             "start2": 0.2, "end2": 0.2}]
    chart = _chart([_line([], bpm=120),
                    _line([_note(100.1, 3, 0, 0.4)], bpm=240, move=move)])
    p = tmp_path / "chart.json"
    p.write_text(json.dumps(chart))
    note = mine_chart(p)[0]
    assert note.tick == 50 and note.dur == 0
    assert note.hit_x == pytest.approx(100.1 / 200)
    assert note.tail_x == pytest.approx(100.5 / 200)


@pytest.mark.parametrize("rate", [1.0, 2.0])
def test_array_audio_offset_rate_and_export_roundtrip(source, tmp_path, rate):
    conv = Phigros4kConvertor(frame_ms=10, max_frame=200, rate=rate, offset_ms=7)
    flat = conv.flatten(parse_phigros_file(str(source)))
    lane, frame = np.where(flat.note_array[:4] > 0.5)
    times_ms = (frame + flat.note_array[lane + 4, frame]) * 10
    assert times_ms == pytest.approx([750 / rate + 7] * 2, abs=1e-4)
    out = tmp_path / "out.json"
    conv.save_phigros_file(flat.note_array, 120, str(out), offset=0.25)
    notes = json.loads(out.read_text())["judgeLineList"][0]["notesAbove"]
    assert [(n["time"], n["holdTime"]) for n in notes] == [(32, 32)] * 2
    assert [n[0] for n in onsets(str(out))] == [750, 750]


def test_negative_audio_onset_does_not_wrap_array(source):
    d = json.loads(source.read_text())
    d["offset"] = -0.501
    source.write_text(json.dumps(d))
    flat = Phigros4kConvertor(frame_ms=10, max_frame=100).flatten(parse_phigros_file(str(source)))
    assert flat.note_array[:4].sum() == 0


def test_mined_metadata_rejects_stale_data(source, tmp_path):
    p = tmp_path / "mined.jsonl"
    notes = mine_chart(source)
    save_jsonl(notes, p)
    with pytest.raises(ValueError, match="Missing mining metadata"):
        load_jsonl(p, expected_timing=(120, 0.25))
    save_jsonl(notes, p, source_path=source)
    assert load_jsonl(p, expected_timing=(120, 0.25)) == notes
    with pytest.raises(ValueError, match="mismatched"):
        load_jsonl(p, expected_timing=(240, 0.25))
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="mismatched"):
        load_jsonl(p, expected_timing=(120, 0.25))
    save_jsonl(notes, p, source_path=source)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Source chart changed"):
        load_jsonl(p, expected_timing=(120, 0.25))


@pytest.mark.parametrize("freeform", [False, True])
def test_ar_training_frames_match_audio_labels(source, tmp_path, freeform):
    listing = tmp_path / "data.txt"
    listing.write_text(f"{source},missing.ogg,IN\n")
    gameplay = tmp_path / "gameplay"
    if freeform:
        save_jsonl(mine_chart(source), gameplay / "song.jsonl", source_path=source)
    ds = TokenChartDataset(str(listing), None, frame_ms=10, max_frame=200,
                          hop_length=512, sr=22050, n_mels=128,
                          gameplay_dir=str(gameplay) if freeform else None)
    ds.base._load_mel = lambda _: torch.zeros(128, 400)
    item = ds[0]
    assert item["frames"][0] == 25  # BOS at chart start, audio 0.25 s
    # Input DT advances to the note, which predicts NOTE at audio 0.75 s.
    for tok, frame in zip(item["tokens"].tolist(), item["frames"].tolist()):
        if ds.tk.kind(tok) in ("DTB", "NOTE", "DURB"):
            assert frame == pytest.approx(75)
    assert torch.nonzero(item["note"][:4] > 0.5)[:, 1].tolist() == [75, 75]


def test_chart_cache_invalidates_when_source_changes(source, tmp_path):
    listing = tmp_path / "data.txt"
    listing.write_text(f"{source},missing.ogg,IN\n")
    ds = PhigrosDataset(str(listing), convertor_params={"frame_ms": 10, "max_frame": 200},
                       chart_cache_dir=str(tmp_path / "cache"), augment=False)
    ds._load_mel = lambda _: torch.zeros(128, 400)
    before = ds[0]
    old_key = ds._chart_cache_path(str(source))
    d = json.loads(source.read_text()); d["offset"] = 0.35
    source.write_text(json.dumps(d))
    after = ds[0]
    assert old_key != ds._chart_cache_path(str(source))
    assert before["meta"]["offset"] == 0.25 and after["meta"]["offset"] == 0.35
    assert torch.nonzero(after["note"][:4] > 0.5)[:, 1].tolist() == [85, 85]


def test_generation_clock_matches_teacher_forcing_and_clips_skeleton(monkeypatch):
    tk = ChartTokenizer()
    torch.manual_seed(5)
    dec = ChartDecoder(vocab_size=tk.vocab_size, hidden_dim=32, depth=1,
                       num_heads=4, mem_dim=16, time_dim=16).eval()
    memory = dec.prepare_memory(torch.randn(1, 1000, 16))
    valid = torch.ones(1, 1000, dtype=torch.bool)
    seen_tokens, seen_frames = [], []
    original_step = dec.step

    def record(token, frame, *args, **kwargs):
        seen_tokens.append(int(token.item())); seen_frames.append(float(frame.item()))
        return original_step(token, frame, *args, **kwargs)

    monkeypatch.setattr(dec, "step", record)
    tokens = generate_tokens(dec, memory, valid, 120, 10, 1000, offset=0.25,
                             skeleton_ticks=[-1, 32, 600, 9999], max_tokens=300,
                             generator=torch.Generator().manual_seed(9))
    assert sorted({n.tick for n in tk.decode_tokens(tokens)}) == [32, 600]
    assert seen_tokens == tokens[:-1]
    expected = [t * 1.5625 + 25 for t in tk.ticks_per_position(seen_tokens)]
    assert seen_frames == pytest.approx(expected)


def test_transcriber_skeleton_subtracts_offset_once():
    from src.models.transcriber import H_ONSET, H_SOFF
    pred = torch.full((1, 32, 150), -100.0)
    pred[0, H_ONSET, 75] = 100
    pred[0, H_ONSET, 10] = 100  # before chart start; skip
    pred[0, H_SOFF:H_SOFF + 4] = -100  # zero sub-frame offset
    skeleton = skeleton_from_transcriber(torch.zeros(1), lambda _: pred, 120, 10, offset=0.25)
    assert skeleton == [32]


def test_checkpoint_timing_version():
    check_checkpoint_timing({"timing_schema": TIMING_SCHEMA}, resume=True)
    with pytest.raises(ValueError, match="Start a new training run"):
        check_checkpoint_timing({}, resume=True)
    with pytest.warns(UserWarning, match="legacy"):
        check_checkpoint_timing({})

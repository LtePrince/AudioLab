"""Exercise the real audio -> decoder -> constrained JSON export on CPU."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.chart_tokenizer import ChartTokenizer, EOS
from src.models.chart_decoder import ChartDecoder
from src.models.transcriber import TranscriberNet
from src.test_decoder import generate_chart


def test_contact_batch_refuses_unverified_existing_output(tmp_path, monkeypatch):
    import sys
    from src import test_decoder
    source = tmp_path/'song'/'chart.json'; source.parent.mkdir()
    source.write_text(json.dumps({'offset': 0, 'judgeLineList': [{'bpm': 120}]}))
    listing = tmp_path/'list.txt'; listing.write_text(f'{source},unused.wav,IN\n')
    output = tmp_path/'output'; output.mkdir(); (output/'song.json').write_text('{}')
    monkeypatch.setattr(test_decoder, 'load_models', lambda *args: (None, None))
    monkeypatch.setattr(sys, 'argv', ['test_decoder', '--ckpt', 'unused.pt', '--device', 'cpu',
        '--list', str(listing), '--out-dir', str(output),
        '--playability-profile', 'contacts-v1', '--max-contacts', '4'])
    with pytest.raises(ValueError, match='fresh output paths'):
        test_decoder.main()


@pytest.mark.parametrize('lanes', [4, 64])
@pytest.mark.parametrize('policy', ['flat', 'type_preserving'])
def test_export_repairs_hold_overlap_and_uses_waveform_end(tmp_path, lanes, policy):
    audio = tmp_path / 'half_second.wav'
    sf.write(audio, np.zeros(11025), 22050)
    tk = ChartTokenizer(n_lanes=lanes)
    enc = TranscriberNet(n_mels=128, hidden_dim=32, conv_blocks=1, depth=1, num_heads=2).eval()
    dec = ChartDecoder(vocab_size=tk.vocab_size, n_lanes=lanes, hidden_dim=32,
                       mem_dim=32, time_dim=32, depth=1, num_heads=2,
                       offset_head=lanes == 64).eval()
    with torch.no_grad():
        dec.lm_head.weight.zero_()
        dec.lm_head.bias.zero_()
        dec.lm_head.bias[EOS] = 100
        dec.lm_head.bias[tk.note(0, 3)] = 200
        dec.lm_head.bias[tk.dtb(1)] = 250
        dec.lm_head.bias[tk.durb(2)] = 300
    outputs = []
    for raw in [True, False]:
        target = tmp_path / f'{raw}.json'
        generate_chart(str(audio), str(target), enc=enc, dec=dec, device=torch.device('cpu'),
                       bpm=120, seed=42, skeleton_ticks=[0, 16, 10000], raw_output=raw, quiet=True,
                       sampling_policy=policy)
        outputs.append(json.loads(target.read_text())['judgeLineList'][0]['notesAbove'])
    raw, repaired = outputs
    assert [n['time'] for n in raw] == [0, 16]
    assert [n['holdTime'] for n in raw] == [64, 64]
    assert [n['holdTime'] for n in repaired] == [16, 16]
    assert [n['type'] for n in repaired] == [3, 3]
    assert [n['positionX'] for n in repaired] == [n['positionX'] for n in raw]
    target = tmp_path / 'contacts.json'
    count = generate_chart(str(audio), str(target), enc=enc, dec=dec, device=torch.device('cpu'),
        bpm=120, seed=42, skeleton_ticks=[0, 16], quiet=True,
        playability_profile='contacts-v1', max_contacts=4 if lanes == 64 else None)
    exported = json.loads(target.read_text())['judgeLineList'][0]['notesAbove']
    audit = json.loads(target.with_suffix('.contacts.json').read_text())
    assert count == len(exported) == len(audit['contacts'])
    assert audit['max_contacts'] == 4 and not any(audit['audit'].values())

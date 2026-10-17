from pathlib import Path
import copy

import pytest

from script.evaluate_contacts import model_inputs


def test_model_comparison_ignores_extra_models_but_rejects_missing_inputs():
    report = dict(config=dict(checkpoint_root='weights', data_list='val.txt'), fingerprints={
        str(Path(p).resolve()):digest for p,digest in [
            ('weights/decoder_freex/decoder_best.pt', 'free-weights'),
            ('weights/transcriber/transcriber_best.pt', 'skeleton-weights'),
            ('val.txt', 'song-list')]})
    combined = copy.deepcopy(report)
    combined['fingerprints']['unrelated/4k.pt'] = 'four-key-weights'
    assert model_inputs(report, 'decoder_freex') == model_inputs(combined, 'decoder_freex')
    key = str(Path('weights/transcriber/transcriber_best.pt').resolve())
    combined['fingerprints'][key] = 'different-skeleton'
    assert model_inputs(report, 'decoder_freex') != model_inputs(combined, 'decoder_freex')
    del combined['fingerprints'][key]
    with pytest.raises(ValueError, match='missing model input fingerprint'):
        model_inputs(combined, 'decoder_freex')

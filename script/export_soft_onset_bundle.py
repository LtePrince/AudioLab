"""Package trained experimental decoders with their exact frozen audio encoders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import torch

from src.product.experimental_models import SCHEMA, load_models
from src.product.model_bundle import _check_space, export_bundle, sha256
from src.soft_onset import decoder_from_checkpoint
from src.test_decoder import load_models as load_legacy_models


def portable_checkpoint(checkpoint, base_checkpoint, onset_checkpoint, mode):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    decoder = decoder_from_checkpoint(ck, 'cpu')
    if ck['mode'] != mode or decoder.n_lanes != (4 if mode == '4k' else 64):
        raise ValueError('decoder mode mismatch')
    sources = ck['frozen_sources']
    if (sources[mode]['sha256'] != sha256(base_checkpoint)
            or sources['onset']['sha256'] != sha256(onset_checkpoint)):
        raise ValueError('frozen frontend differs from the training checkpoint')
    inference = ck['inference_contract']
    if (inference['audio_features'] != 'each route frozen legacy encoder, full attention'
            or inference['onset_encoder'] != '1024 mel core + 128 context, same train and inference'):
        raise ValueError('unsupported training frontend contract')
    encoder, _ = load_legacy_models(base_checkpoint, torch.device('cpu'))
    base = torch.load(base_checkpoint, map_location='cpu', weights_only=True)
    keys = ('soft_onset_schema', 'timing_schema', 'condition', 'history_schema', 'relation_schema',
            'relation_weight', 'model', 'dec_config', 'mode', 'steps', 'seed',
            'training_scope', 'evaluation_scope')
    packed = {key: ck[key] for key in keys}
    packed.update(enc_config=base['enc_config'], encoder_model=encoder.state_dict(),
        source_checkpoint_sha256=sha256(checkpoint), source_encoder_checkpoint_sha256=sha256(base_checkpoint),
        runtime_contract=dict(schema=SCHEMA, mode=mode, layout_encoder='full',
            feature_storage_dtype=inference.get('cache_feature_dtype', 'float32'),
            onset_chunk_frames=1024, onset_context_frames=128, forced_skeleton=False,
            onset_checkpoint_sha256=sha256(onset_checkpoint)))
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model-root', required=True, type=Path)
    parser.add_argument('--decoder-4k', required=True, type=Path)
    parser.add_argument('--decoder-freex', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bundle-id', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists() or args.output.is_symlink():
        parser.error('output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Bound staging first. The archive exporter separately reserves its entire
    # output after staging exists; never count obsolete decoder states twice.
    estimated = sum(p.stat().st_size for p in (args.decoder_4k, args.decoder_freex))
    estimated += sum((args.base_model_root/p).stat().st_size for p in (
        'transcriber/transcriber_best.pt', 'decoder/decoder_best.pt', 'decoder_freex/decoder_best.pt'))
    _check_space(args.output.parent, estimated)
    with tempfile.TemporaryDirectory(prefix='.soft-onset-export-', dir=args.output.parent) as temporary:
        root = Path(temporary)
        onset = args.base_model_root/'transcriber/transcriber_best.pt'
        (root/'transcriber').mkdir()
        shutil.copyfile(onset, root/'transcriber/transcriber_best.pt')
        for mode, folder, source in (('4k', 'decoder', args.decoder_4k),
                                      ('freex', 'decoder_freex', args.decoder_freex)):
            packed = portable_checkpoint(source, args.base_model_root/folder/'decoder_best.pt', onset, mode)
            path = root/folder/'decoder_best.pt'
            path.parent.mkdir()
            torch.save(packed, path)
            enc, dec = load_models(path, torch.device('cpu'))
            del packed, enc, dec
        result = export_bundle(root, args.output, bundle_id=args.bundle_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

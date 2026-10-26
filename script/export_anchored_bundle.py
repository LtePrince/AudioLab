"""Export E14/E18 checkpoints into the existing three-file runtime ZIP format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import torch

from src.anchored_ar import AnchoredAR, SCHEMA as MODEL_SCHEMA, FREEX_SCHEMA
from src.product.anchored_models import SCHEMA
from src.product.experimental_models import load_models
from src.product.model_bundle import _check_space, export_bundle, sha256
from src.test_decoder import load_models as load_legacy_models


def portable_checkpoint(checkpoint, base_checkpoint, onset_checkpoint, mode, cache_manifest,
                        *, decoding='greedy'):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cache = json.loads(Path(cache_manifest).read_text())
    if (ck.get('schema') != (MODEL_SCHEMA if mode == '4k' else FREEX_SCHEMA)
            or ck['config']['full_cache_sha256'] != sha256(cache_manifest)
            or cache['models'][mode]['sha256'] != sha256(base_checkpoint)
            or cache['models']['onset']['sha256'] != sha256(onset_checkpoint)):
        raise ValueError('model mode or frozen frontend differs from the training checkpoint')
    model = AnchoredAR(ck['config']['decoder_config'])
    model.load_state_dict(ck['model'], strict=True)
    if model.decoder.n_lanes != (4 if mode == '4k' else 64):
        raise ValueError('decoder mode mismatch')
    encoder, _ = load_legacy_models(base_checkpoint, torch.device('cpu'))
    base = torch.load(base_checkpoint, map_location='cpu', weights_only=True)
    variant = ck['config']['variant']
    identity = ('AL-AR4K v0.5.1', 'E14') if mode == '4k' else ('AL-FreeX v0.4.0', 'E18')
    condition = {'audio': 'A', 'soft': 'S'}[variant]
    label = f'{identity[0]} / {condition} / {identity[1]}-{condition} / e{ck["epoch"]:04d}'
    return dict(schema=ck['schema'], config=ck['config'], model=ck['model'],
        epoch=ck['epoch'], steps=ck['steps'], enc_config=base['enc_config'],
        encoder_model=encoder.state_dict(), source_checkpoint_sha256=sha256(checkpoint),
        source_encoder_checkpoint_sha256=sha256(base_checkpoint),
        runtime_contract=dict(schema=SCHEMA, mode=mode, variant=variant, model_label=label,
            layout_encoder='full', feature_storage_dtype='float16',
            onset_chunk_frames=1024, onset_context_frames=128, forced_skeleton=False,
            onset_checkpoint_sha256=sha256(onset_checkpoint), decoding=decoding,
            termination_policy='learned_stop_required', repair_profile='none-model-output-v1'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model-root', type=Path, required=True)
    p.add_argument('--cache-manifest', type=Path, required=True)
    p.add_argument('--decoder-4k', type=Path, required=True)
    p.add_argument('--decoder-freex', type=Path, required=True)
    p.add_argument('--decoding-4k', choices=('greedy', 'nucleus'), default='greedy')
    p.add_argument('--decoding-freex', choices=('greedy', 'nucleus'), default='greedy')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bundle-id', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.output.exists() or args.output.is_symlink():
        p.error('output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [(args.base_model_root/n) for n in (
        'transcriber/transcriber_best.pt', 'decoder/decoder_best.pt', 'decoder_freex/decoder_best.pt')]
    _check_space(args.output.parent, sum(x.stat().st_size for x in sources) +
                 args.decoder_4k.stat().st_size + args.decoder_freex.stat().st_size)
    with tempfile.TemporaryDirectory(prefix='.anchored-export-', dir=args.output.parent) as temporary:
        root = Path(temporary)
        onset = args.base_model_root/'transcriber/transcriber_best.pt'
        (root/'transcriber').mkdir()
        shutil.copyfile(onset, root/'transcriber/transcriber_best.pt')
        for mode, folder, source, decoding in (
                ('4k', 'decoder', args.decoder_4k, args.decoding_4k),
                ('freex', 'decoder_freex', args.decoder_freex, args.decoding_freex)):
            packed = portable_checkpoint(source, args.base_model_root/folder/'decoder_best.pt',
                                         onset, mode, args.cache_manifest, decoding=decoding)
            path = root/folder/'decoder_best.pt'
            path.parent.mkdir()
            torch.save(packed, path)
            encoder, decoder = load_models(path, torch.device('cpu'))
            del packed, encoder, decoder
        result = export_bundle(root, args.output, bundle_id=args.bundle_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

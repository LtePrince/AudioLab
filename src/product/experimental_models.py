"""Explicit runtime contract for portable dense-onset/history experiments."""
from __future__ import annotations

import re

import torch

from src.models.transcriber import TranscriberNet
from src.soft_onset import decoder_from_checkpoint
from src.test_decoder import load_models as load_legacy_models

SCHEMA = 'audiolab-soft-onset-runtime-v1'


def _contract(ck):
    contract = ck.get('runtime_contract')
    if contract is None:
        if ck.get('soft_onset_schema') or ck.get('dec_config', {}).get('soft_onset_dim', 0):
            raise ValueError('export the experiment with script.export_soft_onset_bundle first')
        return None
    if (contract.get('schema') != SCHEMA or contract.get('mode') not in ('4k', 'freex')
            or ck.get('condition') not in ('soft', 'soft_history', 'soft_history_aux')
            or contract.get('mode') != ck.get('mode')
            or contract.get('layout_encoder') != 'full'
            or contract.get('feature_storage_dtype') not in ('float16', 'float32')
            or contract.get('onset_chunk_frames') != 1024
            or contract.get('onset_context_frames') != 128
            or contract.get('forced_skeleton') is not False
            or not re.fullmatch('[0-9a-f]{64}', contract.get('onset_checkpoint_sha256', ''))):
        raise ValueError('unsupported soft-onset runtime contract')
    return contract


def runtime_contract(checkpoint, *, audio_sha256, mode):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    contract = _contract(ck)
    if contract is not None and (contract['mode'] != mode
            or contract['onset_checkpoint_sha256'] != audio_sha256):
        raise ValueError('experimental route or frozen onset checkpoint does not match training')
    return contract


def load_models(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    contract = _contract(ck)
    if contract is None:
        del ck
        return load_legacy_models(checkpoint, device)
    decoder = decoder_from_checkpoint(ck, device)
    expected = 4 if contract['mode'] == '4k' else 64
    if decoder.n_lanes != expected:
        raise ValueError('runtime mode does not match decoder positions')
    encoder = TranscriberNet(**ck['enc_config'])
    encoder.load_state_dict(ck['encoder_model'], strict=True)
    if encoder.mel_per_frame != 2 or encoder.conv_in.in_channels != 128:
        raise ValueError('experimental encoder clock or mel shape differs from training')
    decoder.runtime_contract = contract
    return encoder.to(device).eval(), decoder

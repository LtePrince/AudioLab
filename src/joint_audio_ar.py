"""Pure-audio, end-to-end 4K window AR; no cached neural features in training."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.attention import rope1d
from src.models.transcriber import TranscriberNet
from src.soft_onset import MEMORY_SECONDS
from src.windowed_ar import WindowedAR

SCHEMA = 'audiolab-joint-audio-window-ar-v1'


class AcousticBackbone(TranscriberNet):
    """The complete pretrained feature path, without unused transcription heads.

    Recomputing individual blocks saves activations without cropping attention
    or detaching gradients. It preserves the pretrained feature clock/weights.
    """
    def __init__(self, config):
        super().__init__(**config)
        del self.head
        del self.rhythm_head
        self.gradient_checkpointing = True

    def _run(self, module, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def features(self, mel):
        h = self._run(self.conv_in, mel)
        for block in self.conv_blocks:
            h = self._run(block, h)
        h = self._run(self.pool, h)
        length = mel.shape[-1] // self.mel_per_frame
        x = h[..., :length].transpose(1, 2)
        freqs = rope1d(torch.arange(length, device=x.device).float(), self.head_dim, self.rope_theta)
        for block in self.blocks:
            x = self._run(block, x, freqs)
        return self.norm_out(x)


class JointAudioWindowAR(nn.Module):
    def __init__(self, encoder_config, decoder_config, *, width=10.):
        super().__init__()
        if decoder_config.get('soft_onset_dim', 0) or decoder_config.get('n_lanes', 4) != 4:
            raise ValueError('joint A requires pure audio and four positions')
        self.acoustic = AcousticBackbone(encoder_config)
        self.chart = WindowedAR(decoder_config, width=width)
        self.chart.decoder.gradient_checkpointing = True

    @classmethod
    def from_parents(cls, parent, frontend):
        config = parent['config']
        if config.get('variant') != 'audio' or config.get('architecture') != 'window':
            raise ValueError('joint A requires a windowed A parent')
        decoder_config = dict(config['decoder_config'], soft_onset_dim=0)
        model = cls(frontend['enc_config'], decoder_config)
        acoustic = {k.removeprefix('encoder.'): v for k, v in frontend['model'].items()
                    if k.startswith('encoder.') and not k.startswith(('encoder.head.', 'encoder.rhythm_head.'))}
        model.acoustic.load_state_dict(acoustic, strict=True)
        chart = {k: v for k, v in parent['model'].items() if k != 'decoder.soft_onset_proj.weight'}
        model.chart.load_state_dict(chart, strict=True)
        return model, decoder_config

    def condition(self, mel, duration):
        if mel.ndim != 2 or mel.shape[0] != 128 or mel.shape[-1] % 2:
            raise ValueError('expected complete padded 128-bin Mel')
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('positive physical duration required')
        length = math.ceil(duration / MEMORY_SECONDS)
        if mel.shape[-1] // 2 != length:
            raise ValueError('Mel and physical audio duration have different clocks')
        features = self.acoustic.features(mel[None].to(next(self.parameters()).device))
        # Preserve the parent's feature quantization. Both casts are differentiable;
        # no detach, CPU transfer, no_grad or stale neural-feature cache is involved.
        features = features.half().float()
        return dict(features=features, duration=float(duration),
                    valid=torch.ones(features.shape[:2], dtype=torch.bool, device=features.device),
                    soft_onsets=features.new_zeros(*features.shape[:2], 2))

    def components(self, target, mel, duration):
        return self.chart.components(target, self.condition(mel, duration), use_soft=False)

    def optimizer_groups(self, *, decoder_lr, encoder_lr):
        if decoder_lr <= 0 or encoder_lr <= 0 or not all(p.requires_grad for p in self.parameters()):
            raise ValueError('every A parameter must be trainable with positive LR')
        return [dict(name='decoder', params=list(self.chart.parameters()), lr=decoder_lr),
                dict(name='encoder', params=list(self.acoustic.parameters()), lr=encoder_lr)]


def gradient_audit(model):
    rows = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        rows.append(dict(name=name, parameters=parameter.numel(), requires_grad=parameter.requires_grad,
                         has_gradient=grad is not None,
                         finite=grad is not None and bool(torch.isfinite(grad).all()),
                         norm=float(grad.norm()) if grad is not None else None))
    if any(not r['requires_grad'] or not r['has_gradient'] or not r['finite'] for r in rows):
        raise RuntimeError('joint training has frozen, disconnected or nonfinite parameters')
    prefixes = ['acoustic.conv_in.', 'acoustic.pool.', 'acoustic.norm_out.', 'chart.']
    prefixes += [f'acoustic.conv_blocks.{i}.' for i in range(len(model.acoustic.conv_blocks))]
    prefixes += [f'acoustic.blocks.{i}.' for i in range(len(model.acoustic.blocks))]
    groups = {prefix:sum(r['norm'] ** 2 for r in rows if r['name'].startswith(prefix)) ** .5
              for prefix in prefixes}
    if not all(value > 0 for value in groups.values()):
        raise RuntimeError('an entire acoustic/decoder block receives no learning signal')
    return dict(all_trainable=True, all_connected=True, parameters=sum(r['parameters'] for r in rows),
                group_gradient_norms=groups, rows=rows)

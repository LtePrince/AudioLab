"""Isolated E16 K/E/C ablations; production checkpoint loading is unchanged."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from src.models.attention import rope1d
from src.models.transcriber import EncoderBlock
from src.windowed_ar import WindowedAR


class ConditionalNoteHead(nn.Module):
    """p(action|h) p(type|h,NOTE) p(position|h,type,NOTE).

    Returns normalized log probabilities in the *original* vocabulary order.
    Tokenization, causal history, grammar and sampling remain identical. Unlike
    ChartDecoder.factorized, this changes output probabilities, not embeddings.
    Position parameters are shared across types, with a learned type condition.
    """
    def __init__(self, hidden, vocab_size, n_lanes):
        super().__init__()
        self.note0, self.n_note, self.n_lanes = 50, 4*n_lanes, n_lanes
        self.out_features = vocab_size
        if vocab_size <= self.note0+self.n_note:
            raise ValueError('vocabulary does not contain the complete NOTE block')
        self.action = nn.Linear(hidden, vocab_size-self.n_note+1)
        self.kind = nn.Linear(hidden, 4)
        self.type_condition = nn.Embedding(4, hidden)
        self.position = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, n_lanes))

    def forward(self, hidden):
        action = F.log_softmax(self.action(hidden), -1)
        kind = F.log_softmax(self.kind(hidden), -1)
        positions = F.log_softmax(self.position(hidden.unsqueeze(-2)+self.type_condition.weight), -1)
        notes = action[..., -1, None, None]+kind.unsqueeze(-1)+positions
        # NOTE ids are lane-major: note0 + 4*position + type.
        notes = notes.transpose(-1, -2).flatten(-2)
        return torch.cat((action[..., :self.note0], notes, action[..., self.note0:-1]), -1)


class AcousticTail(nn.Module):
    """Train the real last bidirectional audio block and final normalization.

    Inputs are the frozen lower encoder's activations BEFORE this block. They
    cover the entire waveform, at its original resolution and absolute clock.
    """
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        dim, heads = config['hidden_dim'], config['num_heads']
        self.block = EncoderBlock(dim, heads, config['mlp_ratio'], config['dropout'])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    @classmethod
    def from_encoder(cls, encoder):
        block = encoder.blocks[-1]
        config = dict(hidden_dim=encoder.norm_out.normalized_shape[0], num_heads=encoder.num_heads,
            mlp_ratio=block.mlp[0].out_features/block.mlp[0].in_features,
            dropout=block.mlp[2].p, rope_theta=encoder.rope_theta)
        tail = cls(config)
        tail.block.load_state_dict(deepcopy(block.state_dict()))
        tail.norm.load_state_dict(deepcopy(encoder.norm_out.state_dict()))
        return tail

    def forward(self, prefix):
        frames = torch.arange(prefix.shape[1], device=prefix.device).float()
        freqs = rope1d(frames, self.config['hidden_dim']//self.config['num_heads'], self.config['rope_theta'])
        return self.norm(self.block(prefix, freqs))


class VariantAR(WindowedAR):
    def __init__(self, decoder_config, *, variant='base', width=10., tail_config=None):
        if variant not in ('base', 'kind_position', 'encoder_tail', 'capacity'):
            raise ValueError('unknown isolated ablation')
        config = dict(decoder_config)
        if variant == 'capacity':
            config['hidden_dim'] = 512
        super().__init__(config, width=width)
        self.variant, self.decoder_config = variant, config
        if variant == 'kind_position':
            self.decoder.lm_head = ConditionalNoteHead(config['hidden_dim'], config['vocab_size'], config.get('n_lanes', 4))
        if variant == 'encoder_tail':
            if tail_config is None:
                raise ValueError('encoder adaptation requires an explicit pretrained tail')
            self.audio_tail = AcousticTail(tail_config)

    def audio_condition(self, condition):
        if self.variant != 'encoder_tail':
            return condition
        device = next(self.parameters()).device
        features = self.audio_tail(condition['prefix_features'].to(device))
        # Match the cached baseline's feature precision. Casts retain gradients.
        return dict(condition, features=features.half().float())

    def components(self, target, condition, *, use_soft):
        return super().components(target, self.audio_condition(condition), use_soft=use_soft)

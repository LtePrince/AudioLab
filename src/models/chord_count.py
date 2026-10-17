"""Audio-conditioned chord cardinality, without a universal finger-count cap."""
from __future__ import annotations

import torch
from torch import nn

FEATURE_SCHEMA = 'audio-three-frame-windows-and-onset-gaps-v1'
FRAME_SECONDS = 512/22050*2


def event_features(encoded, times, frame_seconds=FRAME_SECONDS):
    """Offline acoustic windows plus neighboring skeleton gaps; no chart labels.

    encoded: [audio frames, D], times: strictly increasing absolute audio seconds.
    Frame floor matches the transcriber label clock; windows are +/- 2 frames.
    """
    times = torch.as_tensor(times, dtype=torch.float32, device=encoded.device)
    if encoded.ndim != 2 or not len(encoded) or frame_seconds <= 0:
        raise ValueError('nonempty frame features and positive frame duration required')
    if times.ndim != 1 or not torch.isfinite(times).all() or (times < 0).any():
        raise ValueError('finite nonnegative audio times required')
    if (times[1:] <= times[:-1]).any():
        raise ValueError('event times must be strictly increasing')
    frames = torch.floor(times/frame_seconds).long()
    indices = (frames[:, None]+torch.tensor([-2, 0, 2], device=encoded.device)).clamp(0, len(encoded)-1)
    acoustic = encoded[indices].flatten(1)
    gaps = times[1:]-times[:-1]
    # Boundary gap=0 indicates missing context, not an invented onset.
    previous = torch.cat([times.new_zeros(1), gaps]) if len(times) else times
    following = torch.cat([gaps, times.new_zeros(1)]) if len(times) else times
    context = torch.log1p(torch.stack([previous, following], -1)*10)
    return torch.cat([acoustic, context], -1)


class ChordCountHead(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.config = dict(input_dim=input_dim, n_classes=n_classes, hidden_dim=hidden_dim)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(hidden_dim, n_classes))

    def forward(self, features):
        return self.net(features)

    @torch.no_grad()
    def initialize_prior(self, counts):
        """Start near empirical TRAIN frequencies; do not discard rare multi-press labels."""
        if len(counts) != self.config['n_classes'] or (counts < 0).any():
            raise ValueError('class counts do not match output vocabulary')
        prior = (counts.float()+.05)/(counts.sum()+.05*len(counts))
        self.net[-1].bias.copy_(prior.log())
        self.net[-1].weight.mul_(.01)

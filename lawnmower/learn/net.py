"""Actor-critic network for the lawnmower environment.

Fully convolutional, so one network works on lawns of any size:

* a trunk of residual 3x3 convolutions at full resolution, with growing
  dilations (1, 2, 4, 8, 16, ...) so that the receptive field covers the whole
  lawn while every cell keeps its own feature vector (no pooling, no stride:
  the exact position of the mower and of each unmowed cell matters);
* the **policy** reads the features of the 3x3 neighbourhood of the mower,
  i.e. of the 8 cells it can move to, and outputs 8 logits; illegal moves are
  masked out;
* the **value** reads the mower's features, the mean and max of the features
  over the lawn, and the number of cells still to mow.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grid import DIRECTIONS

# position of each action's target cell inside the 3x3 neighbourhood
_NEIGHBOUR_INDEX = [(1 + dr) * 3 + (1 + dc) for dr, dc in DIRECTIONS]


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class CoverageNet(nn.Module):
    def __init__(self, in_channels: int = 4, channels: int = 32,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), hidden: int = 128):
        super().__init__()
        self.config = dict(in_channels=in_channels, channels=channels,
                           dilations=tuple(dilations), hidden=hidden)
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.trunk = nn.Sequential(*(ResBlock(channels, d) for d in dilations))
        # policy: features of the 9 cells around the mower -> 8 logits
        self.policy = nn.Sequential(nn.Linear(9 * channels, hidden), nn.ReLU(),
                                    nn.Linear(hidden, len(DIRECTIONS)))
        # value: mower features, mean & max over the lawn, cells left
        self.value = nn.Sequential(nn.Linear(3 * channels + 1, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy[-1].bias)
        self.register_buffer("nb_index", torch.tensor(_NEIGHBOUR_INDEX), persistent=False)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        """``obs`` (B, 4 or 5, H, W) as produced by the env; ``mask`` (B, 8) legal actions.

        Returns ``(logits, value)`` with shapes (B, 8) and (B,).
        """
        b, _, h, w = obs.shape
        feat = F.relu(self.trunk(self.stem(obs)))                 # (B, C, H, W)
        c = feat.shape[1]
        # mower position from channel 2
        flat_pos = obs[:, 2].reshape(b, -1).argmax(dim=1)
        r, col = flat_pos // w, flat_pos % w
        # 3x3 patch around the mower (zero padding outside the grid)
        padded = F.pad(feat, (1, 1, 1, 1))
        dr = torch.arange(3, device=obs.device)
        rows = (r[:, None] + dr[None, :])[:, :, None].expand(b, 3, 3)
        cols = (col[:, None] + dr[None, :])[:, None, :].expand(b, 3, 3)
        bidx = torch.arange(b, device=obs.device)[:, None, None].expand(b, 3, 3)
        patch = padded.permute(0, 2, 3, 1)[bidx, rows, cols]       # (B, 3, 3, C)
        patch = patch.reshape(b, 9, c)
        logits = self.policy(patch.reshape(b, 9 * c))
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        lawn = (1 - obs[:, 0]).unsqueeze(1)                        # (B, 1, H, W)
        n_lawn = lawn.sum(dim=(2, 3)).clamp(min=1)
        mean = (feat * lawn).sum(dim=(2, 3)) / n_lawn
        mx = (feat * lawn - (1 - lawn) * 1e4).amax(dim=(2, 3))
        left = (lawn[:, 0] * (1 - obs[:, 1])).sum(dim=(1, 2), keepdim=False)[:, None] / 100.0
        value = self.value(torch.cat([patch[:, 4], mean, mx, left], dim=1)).squeeze(-1)
        return logits, value

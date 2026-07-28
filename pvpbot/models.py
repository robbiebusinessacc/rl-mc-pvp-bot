"""Shared network architectures. DO NOT MODIFY on module branches.

Both the RL trainer and the BC prior train PolicyNet; perception trains
PerceptionCNN. Keeping the classes here means checkpoints interchange.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from pvpbot.spec import ACTION_HEAD_SIZES, FRAME_SHAPE, OBS_DIM, PERCEPTION_DIM


class PolicyNet(nn.Module):
    """MLP encoder -> GRU core -> per-head categorical logits + value."""

    HIDDEN = 256
    CORE = 128

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, self.HIDDEN), nn.ReLU(),
        )
        self.core = nn.GRUCell(self.HIDDEN, self.CORE)
        self.heads = nn.ModuleList(
            [nn.Linear(self.CORE, n) for n in ACTION_HEAD_SIZES]
        )
        self.value = nn.Linear(self.CORE, 1)

    def initial_state(self, batch: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.zeros(batch, self.CORE, device=device)

    def forward(
        self, obs: torch.Tensor, state: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """obs (B, OBS_DIM), state (B, CORE) -> (logits per head, value (B,), new state)."""
        z = self.encoder(obs)
        h = self.core(z, state)
        logits = [head(h) for head in self.heads]
        return logits, self.value(h).squeeze(-1), h

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (actions (B, NUM_ACTION_HEADS) int64, new state)."""
        logits, _, h = self.forward(obs, state)
        if deterministic:
            acts = [l.argmax(dim=-1) for l in logits]
        else:
            acts = [torch.distributions.Categorical(logits=l).sample() for l in logits]
        return torch.stack(acts, dim=-1), h


class PerceptionCNN(nn.Module):
    """Color frame FRAME_SHAPE (3, 96, 170) -> PERCEPTION_DIM state estimate.

    Color at ~2x the original grayscale resolution: cyan diamond armor on
    green/blue terrain is the strongest localization signal in the frame,
    and aim precision is bounded by this sensor.
    """

    def __init__(self, stack: int = 1):
        """stack > 1: input is `stack` consecutive frames concatenated on the
        channel axis (stack*3 channels) -- motion made directly visible;
        single-frame nets top out ~19 deg aim error on fast strafers because
        velocity is unobservable in one frame. Weights stay layout-compatible
        except conv1; loaders tile conv1 across the stack (divided by stack)
        to warm-start from single-frame checkpoints."""
        super().__init__()
        c, h, w = FRAME_SHAPE
        self.stack = int(stack)
        c = c * self.stack
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=1), nn.ReLU(),
            nn.Conv2d(96, 96, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat, 384), nn.ReLU(), nn.Linear(384, PERCEPTION_DIM)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if self.stack > 1 and frames.shape[1] * self.stack == self.net[0].in_channels:
            # single-frame input to a stacked net: replicate as a static
            # window. With conv1 warm-started as tiled/stack this is exactly
            # the single-frame net's output -- synth batches and legacy eval
            # fixtures need no changes.
            frames = frames.repeat(1, self.stack, 1, 1)
        return self.head(self.net(frames))

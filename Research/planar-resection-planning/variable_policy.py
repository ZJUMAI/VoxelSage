"""Size-preserving spatial policy components for padded planar-grid PPO."""

from __future__ import annotations

from typing import Callable

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PaddedSpatialExtractor(BaseFeaturesExtractor):
    """Encode a padded grid without pooling away per-cell structure."""

    def __init__(self, observation_space) -> None:
        channels, rows, cols = observation_space.shape
        self.channels = int(channels)
        self.rows = int(rows)
        self.cols = int(cols)
        super().__init__(observation_space, features_dim=32 * self.rows * self.cols)
        self.spatial = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.spatial(observations).flatten(start_dim=1)


class SpatialActionHead(torch.nn.Module):
    """Shared-convolution logits head: one logit per padded grid position."""

    def __init__(self, *, channels: int, rows: int, cols: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.rows = int(rows)
        self.cols = int(cols)
        self.scorer = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        feature_map = latent.reshape(-1, self.channels, self.rows, self.cols)
        return self.scorer(feature_map).flatten(start_dim=1)


class VariableSpatialPolicy(MaskableActorCriticPolicy):
    """Maskable PPO policy whose action logits are produced convolutionally.

    The feature extractor returns a flattened spatial map and an empty SB3 MLP
    keeps it unchanged.  The custom action head restores that map and scores
    each location with shared convolutions, avoiding a fixed 49-way action
    classifier.  The critic remains a scalar readout over the same full map.
    """

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        extractor = self.features_extractor
        if not isinstance(extractor, PaddedSpatialExtractor):
            raise TypeError("VariableSpatialPolicy requires PaddedSpatialExtractor")
        self.action_net = SpatialActionHead(
            channels=32, rows=extractor.rows, cols=extractor.cols,
        )
        if self.action_space.n != extractor.rows * extractor.cols:
            raise ValueError("Variable action space must equal padded grid area")
        self.action_net.apply(lambda module: self.init_weights(module, gain=0.01))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
        )

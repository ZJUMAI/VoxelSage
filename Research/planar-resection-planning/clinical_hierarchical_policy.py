"""Factorized clamp-control and spatial-target policy for v10."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from variable_policy import PaddedSpatialExtractor


class ClinicalHierarchicalActionHead(torch.nn.Module):
    """Two clamp logits plus one shared-convolution logit per target cell."""

    def __init__(self, *, channels: int, rows: int, cols: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.rows = int(rows)
        self.cols = int(cols)
        self.target_scorer = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=1),
        )
        # Average pooling summarizes overall progress; max pooling preserves
        # sparse high-risk vessel features that v9's mean-only END head diluted.
        self.clamp_scorer = torch.nn.Sequential(
            torch.nn.Linear(self.channels * 2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        feature_map = latent.reshape(-1, self.channels, self.rows, self.cols)
        target_logits = self.target_scorer(feature_map).flatten(start_dim=1)
        pooled = torch.cat(
            (feature_map.mean(dim=(2, 3)), feature_map.amax(dim=(2, 3))), dim=1
        )
        clamp_logits = self.clamp_scorer(pooled)
        return torch.cat((clamp_logits, target_logits), dim=1)

    def initialize_release(self, bias: float) -> None:
        final = self.clamp_scorer[-1]
        with torch.no_grad():
            final.weight[1].zero_()
            final.bias[1] = float(bias)
            final.weight[0].zero_()
            final.bias[0] = 0.0


class ClinicalHierarchicalPolicy(MaskableActorCriticPolicy):
    """Maskable MultiDiscrete policy with separate clamp and target heads."""

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        extractor = self.features_extractor
        if not isinstance(extractor, PaddedSpatialExtractor):
            raise TypeError("ClinicalHierarchicalPolicy requires PaddedSpatialExtractor")
        nvec = np.asarray(getattr(self.action_space, "nvec", ()), dtype=np.int64)
        expected = np.asarray([2, extractor.rows * extractor.cols], dtype=np.int64)
        if not np.array_equal(nvec, expected):
            raise ValueError(f"Hierarchical action space must be {expected.tolist()}, got {nvec.tolist()}")
        self.action_net = ClinicalHierarchicalActionHead(
            channels=32, rows=extractor.rows, cols=extractor.cols
        )
        self.action_net.apply(lambda module: self.init_weights(module, gain=0.01))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

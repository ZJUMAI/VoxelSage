"""Spatial target policy for the clinical macro-action environment."""

from __future__ import annotations

from typing import Callable

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from variable_policy import PaddedSpatialExtractor


class ClinicalMacroActionHead(torch.nn.Module):
    """One shared-convolution score per cell plus one pooled END score."""

    def __init__(self, *, channels: int, rows: int, cols: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.rows = int(rows)
        self.cols = int(cols)
        self.spatial_scorer = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=1),
        )
        self.end_scorer = torch.nn.Linear(self.channels, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        feature_map = latent.reshape(-1, self.channels, self.rows, self.cols)
        spatial = self.spatial_scorer(feature_map).flatten(start_dim=1)
        pooled = feature_map.mean(dim=(2, 3))
        end = self.end_scorer(pooled)
        return torch.cat((spatial, end), dim=1)

    def initialize_end(self, bias: float) -> None:
        with torch.no_grad():
            self.end_scorer.weight.zero_()
            self.end_scorer.bias.fill_(float(bias))


class ClinicalMacroSpatialPolicy(MaskableActorCriticPolicy):
    """Preserve per-cell structure while supporting one extra END action."""

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        extractor = self.features_extractor
        if not isinstance(extractor, PaddedSpatialExtractor):
            raise TypeError("ClinicalMacroSpatialPolicy requires PaddedSpatialExtractor")
        if self.action_space.n != extractor.rows * extractor.cols + 1:
            raise ValueError("Macro action space must equal padded grid area plus END")
        self.action_net = ClinicalMacroActionHead(
            channels=32,
            rows=extractor.rows,
            cols=extractor.cols,
        )
        self.action_net.apply(lambda module: self.init_weights(module, gain=0.01))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
        )

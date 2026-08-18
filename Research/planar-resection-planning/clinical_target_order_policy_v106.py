"""Budget-aware spatial candidate scorer and tail-risk heads for v10.6."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from clinical_target_order_features_v106 import CANDIDATE_FEATURE_DIM, GLOBAL_FEATURE_DIM

GRID_CHANNELS = 11  # ten bit-packed semantic masks + quantized transfer distance
DOMAIN_CHANNEL = 0
FRONTIER_CHANNEL = 5


def unpack_spatial(grid_bits: np.ndarray, transfer_q: np.ndarray) -> np.ndarray:
    """Decode a batch of compact teacher states to float32 (B,11,30,40)."""
    bits = np.unpackbits(grid_bits, axis=-1)[..., :1200]
    semantic = bits.reshape(*bits.shape[:-1], 30, 40).astype(np.float32)
    transfer = transfer_q.astype(np.float32)[:, None, :, :] / 255.0
    return np.concatenate([semantic, transfer], axis=1)


def masked_mean_max(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """True-mask pooling; padded cells cannot dilute context."""
    mask = mask.to(dtype=features.dtype)
    count = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    mean = (features * mask).sum(dim=(-2, -1)) / count
    neg_inf = torch.finfo(features.dtype).min
    maximum = features.masked_fill(mask == 0, neg_inf).amax(dim=(-2, -1))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, maximum


class TargetOrderScorerV106(nn.Module):
    def __init__(
        self,
        *,
        cand_dim: int = CANDIDATE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        hidden: int = 96,
        spatial: int = 32,
    ) -> None:
        super().__init__()
        self.cand_dim = cand_dim; self.global_dim = global_dim
        self.hidden = hidden; self.spatial_dim = spatial
        self.encoder = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, spatial, 3, padding=1), nn.ReLU(),
            nn.Conv2d(spatial, spatial, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(spatial, spatial, 3, padding=4, dilation=4), nn.ReLU(),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(cand_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden), nn.ReLU())
        combined = hidden * 2 + spatial * 5  # cand/global + local + four pooled contexts
        self.trunk = nn.Sequential(nn.Linear(combined, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.score_head = nn.Linear(hidden, 1)
        self.t_total_head = nn.Linear(hidden, 1)
        self.b_tail_head = nn.Linear(hidden, 1)
        self.b_total_head = nn.Linear(hidden, 1)
        self.completion_head = nn.Linear(hidden, 1)
        self.safe_head = nn.Linear(hidden, 1)

    def forward(
        self,
        grid: torch.Tensor,          # B,11,30,40
        candidate_features: torch.Tensor,  # B,K,Dc
        global_context: torch.Tensor,       # B,Dg
        targets: torch.Tensor,       # B,K,2 row/col
    ) -> dict[str, torch.Tensor]:
        spatial = self.encoder(grid)
        batch, candidates = targets.shape[:2]
        rows = targets[..., 0].clamp(0, spatial.shape[-2] - 1)
        cols = targets[..., 1].clamp(0, spatial.shape[-1] - 1)
        bi = torch.arange(batch, device=targets.device)[:, None].expand(batch, candidates)
        local = spatial[bi, :, rows, cols]
        domain_mean, domain_max = masked_mean_max(spatial, grid[:, DOMAIN_CHANNEL:DOMAIN_CHANNEL + 1])
        front_mean, front_max = masked_mean_max(spatial, grid[:, FRONTIER_CHANNEL:FRONTIER_CHANNEL + 1])
        pooled = torch.cat([domain_mean, domain_max, front_mean, front_max], dim=-1)
        pooled = pooled[:, None, :].expand(-1, candidates, -1)
        cand = self.candidate_mlp(candidate_features)
        glob = self.global_mlp(global_context)[:, None, :].expand(-1, candidates, -1)
        hidden = self.trunk(torch.cat([cand, glob, local, pooled], dim=-1))
        return {
            "score": self.score_head(hidden).squeeze(-1),
            "T_total": self.t_total_head(hidden).squeeze(-1),
            "B_tail": self.b_tail_head(hidden).squeeze(-1),
            "B_total": self.b_total_head(hidden).squeeze(-1),
            "completion_logit": self.completion_head(hidden).squeeze(-1),
            "safe_logit": self.safe_head(hidden).squeeze(-1),
        }

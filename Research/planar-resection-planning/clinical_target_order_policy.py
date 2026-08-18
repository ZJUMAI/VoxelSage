"""Global candidate-scoring policy for v10.4 target-order (guide Section 7.2).

Architecture (shared scorer across all candidates):

    candidate explicit features  -> candidate MLP  --------+
    global context scalars       -> global MLP    --------+
                                                          v
                                     concat -> scorer -> candidate logit

The scorer shares weights over every legal candidate; the action mask is still
enforced by the environment.  Continuous features are normalised with the
Train-only scales (guide 7.1).  ``make_selector`` returns an env->action
callable for deterministic rollout.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_target_order_features import (
    CANDIDATE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    candidate_features,
    global_context,
    normalize_features,
)
from plan_target_order_v104 import candidate_targets


class TargetOrderScorer(nn.Module):
    """Shared candidate scorer with global-context conditioning."""

    def __init__(
        self,
        *,
        cand_dim: int = CANDIDATE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.cand_dim = int(cand_dim)
        self.global_dim = int(global_dim)
        self.hidden = int(hidden)
        self.cand_mlp = nn.Sequential(
            nn.Linear(self.cand_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_dim, self.hidden), nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        cand_feat: torch.Tensor,  # (B, K, Dc)
        global_ctx: torch.Tensor,  # (B, Dg)
    ) -> torch.Tensor:
        ce = self.cand_mlp(cand_feat)                       # (B, K, H)
        ge = self.global_mlp(global_ctx).unsqueeze(1)       # (B, 1, H)
        ge = ge.expand(-1, cand_feat.shape[1], -1)
        x = torch.cat([ce, ge], dim=-1)
        return self.scorer(x).squeeze(-1)                   # (B, K)


def make_selector(
    model: nn.Module,
    scales: Mapping[str, Any],
    *,
    candidate_count: int = 6,
) -> Callable[[ClinicalMacroResectionEnv], int]:
    """Deterministic env->action selector: argmax scorer over legal candidates."""
    model.eval()

    def select(env: ClinicalMacroResectionEnv) -> int:
        targets = candidate_targets(env, count=candidate_count)
        if not targets:
            raise RuntimeError("scorer selector has no legal candidates")
        feats = np.stack([
            normalize_features(candidate_features(env, t)[0], scales)
            for t in targets
        ])
        gc = np.asarray([global_context(env)], dtype=np.float32)
        with torch.no_grad():
            logits = model(
                torch.from_numpy(feats).unsqueeze(0),
                torch.from_numpy(gc),
            ).squeeze(0)
        best = int(logits.argmax().item())
        cell = targets[best]
        return int(cell[0] * env.max_cols + cell[1])

    return select

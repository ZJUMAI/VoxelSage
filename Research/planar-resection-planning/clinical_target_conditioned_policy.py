"""v10.2 target-conditioned clamp policy.

The clamp scorer consumes four feature groups:

    1. global pooled feature map (mean + max over the spatial grid);
    2. local feature at the planned target cell;
    3. route-pooled feature over the planned transfer route;
    4. the scalar observation channels.

The spatial extractor splits into a frozen ``base_spatial`` (a copy of the
v10.1 BC target feature extractor) and a trainable ``plan_spatial`` branch
that sees only the 10 target-conditioned channels.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from clinical_target_conditioned_environment import (
    CLAMP_ACTION_COUNT,
    PLANNED_ROUTE_CHANNEL,
    PLANNED_TARGET_CHANNEL,
)

# Scalar channels (fill-constant over the grid), read at (0, 0):
#   17 clamp_elapsed_fraction, 18 unclamp_remaining_fraction,
#   19 elapsed_time_fraction, 20 no_progress_streak_fraction,
#   24 clinical_cost_fraction, 28..35 the v10.2 plan/ischemia scalars.
SCALAR_FEATURE_INDICES = (17, 18, 19, 20, 24, 28, 29, 30, 31, 32, 33, 34, 35)

# Clamp-only contract (reviewer fix #5): the frozen BC macro-target selector
# must be insensitive to clamp / bleeding / elapsed signals so the automatic
# transfer equals the baseline for ANY legal release sequence.  The channels
# below are zeroed before the BC model predicts:
#   13 expected_bleeding_rate   (nonzero only while unclamped + exposed)
#   17 clamped_phase
#   18 clamp_elapsed_fraction
#   19 unclamp_remaining_fraction
#   20 elapsed_time_fraction
#   24 clinical_cost_fraction
# This definition is frozen once and enforced by Test 20.
CLAMP_BLIND_CHANNELS = (13, 17, 18, 19, 20, 24)


class PaddedPlanSpatialExtractor(BaseFeaturesExtractor):
    """Split extractor: frozen base branch + trainable plan branch."""

    def __init__(
        self,
        observation_space,
        *,
        base_in_channels: int = 26,
        plan_in_channels: int = 10,
        base_conv: int = 32,
        plan_conv: int = 8,
    ) -> None:
        channels, rows, cols = observation_space.shape
        self.channels = int(channels)
        self.rows = int(rows)
        self.cols = int(cols)
        self.base_in_channels = int(base_in_channels)
        self.plan_in_channels = int(plan_in_channels)
        super().__init__(
            observation_space,
            features_dim=(base_conv + plan_conv) * self.rows * self.cols,
        )
        self.base_spatial = nn.Sequential(
            nn.Conv2d(base_in_channels, base_conv, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_conv, base_conv, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # base_spatial is a copy of the frozen BC target feature extractor.
        # clamp-only training must never update it (guide §10: freeze target
        # head + target feature extractor).
        for parameter in self.base_spatial.parameters():
            parameter.requires_grad_(False)
        self.plan_spatial = nn.Sequential(
            nn.Conv2d(plan_in_channels, plan_conv, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(plan_conv, plan_conv, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base = self.base_spatial(observations[:, : self.base_in_channels])
        plan = self.plan_spatial(observations[:, self.base_in_channels :])
        return torch.cat((base, plan), dim=1).flatten(start_dim=1)


class TargetConditionedClampActionHead(nn.Module):
    """Global + local + route + scalar clamp scorer."""

    def __init__(
        self,
        *,
        base_conv: int = 32,
        plan_conv: int = 8,
        rows: int = 30,
        cols: int = 40,
        n_scalars: int = len(SCALAR_FEATURE_INDICES),
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.base_conv = int(base_conv)
        self.plan_conv = int(plan_conv)
        self.rows = int(rows)
        self.cols = int(cols)
        feat = self.base_conv + self.plan_conv
        in_dim = (
            2 * feat  # global mean+max over base+plan
            + feat  # local feature at planned target
            + feat  # route-pooled feature
            + int(n_scalars)
        )
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, CLAMP_ACTION_COUNT),
        )

    def fused_features(
        self,
        latent: torch.Tensor,
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the (B, in_dim) fused feature vector used by the scorer."""
        B = latent.shape[0]
        F = latent.reshape(B, -1, self.rows, self.cols)
        Fb, Fp = F[:, : self.base_conv], F[:, self.base_conv :]
        global_pooled = torch.cat(
            (
                Fb.mean(dim=(2, 3)),
                Fb.amax(dim=(2, 3)),
                Fp.mean(dim=(2, 3)),
                Fp.amax(dim=(2, 3)),
            ),
            dim=1,
        )
        if observations is None:
            zero_local = torch.zeros(B, self.base_conv + self.plan_conv, device=latent.device)
            zero_route = torch.zeros(B, self.base_conv + self.plan_conv, device=latent.device)
            return torch.cat((global_pooled, zero_local, zero_route), dim=1)

        # Local feature at the planned target (recover from one-hot channel).
        pos = observations[:, PLANNED_TARGET_CHANNEL].reshape(B, -1).argmax(dim=1)
        tr = pos // self.cols
        tc = pos % self.cols
        batch = torch.arange(B, device=latent.device)
        local = torch.cat((Fb[batch, :, tr, tc], Fp[batch, :, tr, tc]), dim=1)

        # Route-pooled feature (masked mean over the route; empty route -> zeros).
        route_mask = observations[:, PLANNED_ROUTE_CHANNEL] > 0.0  # (B, R, C)
        denom = route_mask.float().sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(1)
        route = torch.cat(
            (
                (Fb * route_mask.unsqueeze(1)).sum(dim=(2, 3)) / denom,
                (Fp * route_mask.unsqueeze(1)).sum(dim=(2, 3)) / denom,
            ),
            dim=1,
        )

        scalars = observations[:, SCALAR_FEATURE_INDICES, 0, 0]  # (B, n)
        return torch.cat((global_pooled, local, route, scalars), dim=1)

    def forward(
        self,
        latent: torch.Tensor,
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.scorer(self.fused_features(latent, observations))

    def initialize_release(self, bias: float) -> None:
        final = self.scorer[-1]
        with torch.no_grad():
            # Keep the non-zero weight rows (from init_weights gain=0.01) so
            # BOTH actions receive gradients.  Only the bias offsets make
            # release initially unlikely; zeroing the weights entirely would
            # pin logits to constants and stop learning (Stage-1 test caught
            # this).
            final.bias[1] = float(bias)
            final.bias[0] = 0.0


class TargetConditionedClampPolicy(MaskableActorCriticPolicy):
    """Maskable Discrete(2) policy with the fused condition-aware clamp head."""

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)
        extractor = self.features_extractor
        if not isinstance(extractor, PaddedPlanSpatialExtractor):
            raise TypeError(
                "TargetConditionedClampPolicy requires PaddedPlanSpatialExtractor"
            )
        if self.action_space.n != CLAMP_ACTION_COUNT:
            raise ValueError(
                f"action space must be Discrete({CLAMP_ACTION_COUNT}), got {self.action_space.n}"
            )
        self.action_net = TargetConditionedClampActionHead(
            base_conv=32,
            plan_conv=8,
            rows=extractor.rows,
            cols=extractor.cols,
        )
        self.action_net.apply(lambda module: self.init_weights(module, gain=0.01))
        self.action_net.initialize_release(-4.0)
        # Stage-1 safety oracle: regression head predicting the normalized
        # (delta_blood / blood_scale, delta_ischemia / ischemia_scale,
        # counterfactual advantage) of a release, fused into the policy so it
        # is saved and loaded with the model (reviewer fix #6).
        self.regression_head = nn.Linear(173, 3)
        self.regression_head.apply(lambda module: self.init_weights(module, gain=0.01))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self._clamp_obs: torch.Tensor | None = None

    def get_distribution(self, observations, action_masks=None):
        self._clamp_obs = observations
        return super().get_distribution(observations, action_masks)

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        self._clamp_obs = obs
        return super().forward(obs, deterministic, action_masks)

    def evaluate_actions(self, obs, actions, action_masks=None):
        self._clamp_obs = obs
        return super().evaluate_actions(obs, actions, action_masks)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(
            latent_pi, observations=getattr(self, "_clamp_obs", None)
        )
        return self.action_dist.proba_distribution(action_logits=logits)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenBCMacroTargetPolicy:
    """Wraps the v10.1 Stage 1A BC checkpoint; only its target head is used.

    All parameters are frozen; this object is created once and injected into
    the environment so the clamp PPO can never train or overwrite it.
    """

    def __init__(self, bc_checkpoint, device: str = "auto") -> None:
        from sb3_contrib import MaskablePPO
        import clinical_hierarchical_policy  # noqa: F401

        self.checkpoint_path = str(bc_checkpoint)
        self.checkpoint_sha256 = _file_sha256(self.checkpoint_path)
        self.model = MaskablePPO.load(str(bc_checkpoint), device=device)
        for parameter in self.model.policy.parameters():
            parameter.requires_grad_(False)

    def select_target(self, env) -> int:
        from clinical_hierarchical_environment import (
            CLAMP_ACTION_COUNT,
            CLINICAL_HIERARCHICAL_MASK_SIZE,
        )

        base = env._base_observation()
        # Clamp-only contract: zero the clamp/bleeding/elapsed channels so the
        # target is identical for every legal clamp schedule (see
        # CLAMP_BLIND_CHANNELS above).
        for channel in CLAMP_BLIND_CHANNELS:
            base[channel] = 0.0
        mask = np.zeros(CLINICAL_HIERARCHICAL_MASK_SIZE, dtype=bool)
        mask[0] = True  # CLAMP_CONTINUE
        mask[1] = env._release_is_legal()
        for row, col in env._frontier():
            mask[CLAMP_ACTION_COUNT + row * env.max_cols + col] = True
        action, _ = self.model.predict(base, deterministic=True, action_masks=mask)
        return int(action[1])

    def parameter_sha256(self) -> str:
        """Robust content hash over the frozen tensor values.

        Unlike ``state_dict().__str__()`` (repr-dependent), this hashes the
        actual bytes of every tensor, so it is stable across printing formats
        and devices.
        """
        digest = hashlib.sha256()
        for key, value in self.model.policy.state_dict().items():
            digest.update(key.encode("utf-8"))
            data = value.detach().cpu().contiguous().view(-1)
            digest.update(data.numpy().tobytes())
        return digest.hexdigest()

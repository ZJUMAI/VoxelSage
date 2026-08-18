"""Train MaskablePPO with direction, macro, or hierarchical clinical control."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from clinical_window_environment import (
    ACTION_END_CLAMP_EARLY,
    CLINICAL_ENVIRONMENT_VERSION,
    CLINICAL_OBSERVATION_CHANNELS,
    ClinicalWindowScenarioPoolEnv,
)
from clinical_window_policy import ClinicalGridExtractor, LocalGlobalClinicalExtractor


@dataclass(frozen=True)
class ClinicalTrainingConfig:
    seed: int = 2026080401
    timesteps: int = 500_000
    n_envs: int = 8
    n_steps: int = 1024
    batch_size: int = 512
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.98
    ent_coef: float = 0.01
    clip_range: float = 0.2
    target_kl: float = 0.03
    device: str = "auto"
    torch_threads: int = 1
    mechanics_update_interval: int = 0
    checkpoint_global_interval: int | None = None
    init_model: str | None = None
    bc_scenarios: int = 0
    bc_epochs: int = 3
    bc_batch_size: int = 128
    bc_learning_rate: float = 1e-4
    bc_margin: float = 1.0
    bc_v_weight: float = 0.5
    share_features_extractor: bool = False
    rl_margin_coef: float = 0.0
    rl_margin_updates: int = 1
    rl_margin_batch_size: int = 256
    rl_margin_buffer_size: int = 2048
    end_action_initial_bias: float = -4.0
    features_extractor: str = "cnn"
    control_mode: str = "direction"
    freeze_target_head: bool = False
    freeze_features_extractor: bool = False


def _discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Return raw Monte-Carlo targets on the same scale used by PPO."""
    result = np.empty(len(rewards), dtype=np.float32)
    step_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        step_return = float(rewards[index]) + gamma * step_return
        result[index] = step_return
    return result


def _masked_margin_loss(
    logits,
    actions,
    masks,
    margin: float,
    n_teacher_actions: int = 4,
):
    """Hinge loss requiring the teacher logit to beat legal teacher actions.

    Only actions ``0..n_teacher_actions-1`` are negative samples. The final
    ``end_clamp_early`` action is never compared, so PPO may freely raise END
    to learn its own unclamp timing after that action is unmasked.
    """
    import torch

    teacher_logit = logits.gather(1, actions.unsqueeze(1)).squeeze(1)
    direction_logits = logits[:, :n_teacher_actions]
    direction_masks = masks[:, :n_teacher_actions]
    masked_dirs = direction_logits.masked_fill(~direction_masks.bool(), float("-inf"))
    other_logits = masked_dirs.clone()
    other_logits.scatter_(1, actions.unsqueeze(1), float("-inf"))
    max_other = other_logits.max(dim=1).values
    # A state with only one legal direction has no alternative to regularize.
    has_other = torch.isfinite(max_other)
    losses = torch.where(
        has_other,
        torch.clamp(margin + max_other - teacher_logit, min=0.0),
        torch.zeros_like(teacher_logit),
    )
    return losses.mean()


class _TeacherDemoReservoir:
    """Bounded, deterministic reservoir for alternating PPO/teacher updates."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.seen = 0
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.masks: list[np.ndarray] = []

    def add(self, observation: np.ndarray, action: int, mask: np.ndarray) -> None:
        self.seen += 1
        if len(self.observations) < self.capacity:
            index = len(self.observations)
            self.observations.append(np.asarray(observation, dtype=np.float16).copy())
            self.actions.append(int(action))
            self.masks.append(np.asarray(mask, dtype=np.bool_).copy())
            return
        index = self.rng.randrange(self.seen)
        if index >= self.capacity:
            return
        self.observations[index] = np.asarray(observation, dtype=np.float16).copy()
        self.actions[index] = int(action)
        self.masks[index] = np.asarray(mask, dtype=np.bool_).copy()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.stack(self.observations),
            np.asarray(self.actions, dtype=np.int64),
            np.stack(self.masks),
        )


def _margin_ppo_class():
    """Create the optional SB3 subclass lazily so data utilities stay lightweight."""
    import torch
    from sb3_contrib import MaskablePPO

    class MarginRegularizedMaskablePPO(MaskablePPO):
        """MaskablePPO with bounded teacher-margin updates after each PPO update.

        Alternating updates avoid copying the version-specific PPO training loop while
        still restoring the teacher decision boundary after every rollout update.
        The demonstration arrays are deliberately excluded from model checkpoints.
        """

        def __init__(
            self,
            *args,
            rl_margin_coef: float = 0.0,
            rl_margin: float = 1.0,
            rl_margin_updates: int = 1,
            rl_margin_batch_size: int = 256,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.rl_margin_coef = float(rl_margin_coef)
            self.rl_margin = float(rl_margin)
            self.rl_margin_updates = int(rl_margin_updates)
            self.rl_margin_batch_size = int(rl_margin_batch_size)
            self._margin_observations: np.ndarray | None = None
            self._margin_actions: np.ndarray | None = None
            self._margin_masks: np.ndarray | None = None
            seed_value = int(self.seed) if self.seed is not None else 0
            self._margin_rng = np.random.default_rng(seed_value ^ 0xA11CE)

        def _excluded_save_params(self) -> list[str]:
            return super()._excluded_save_params() + [
                "_margin_observations",
                "_margin_actions",
                "_margin_masks",
            ]

        def set_margin_demonstrations(
            self,
            observations: np.ndarray,
            actions: np.ndarray,
            masks: np.ndarray,
        ) -> None:
            if not (len(observations) == len(actions) == len(masks)):
                raise ValueError("Teacher demonstration arrays must have equal lengths")
            if len(observations) == 0:
                raise ValueError("Teacher demonstration buffer must not be empty")
            self._margin_observations = observations
            self._margin_actions = actions
            self._margin_masks = masks

        def train(self) -> None:
            super().train()
            if self.rl_margin_coef <= 0.0 or self._margin_observations is None:
                return
            self.policy.set_training_mode(True)
            losses: list[float] = []
            matches: list[float] = []
            buffer_size = len(self._margin_observations)
            for _ in range(self.rl_margin_updates):
                indices = self._margin_rng.integers(
                    0,
                    buffer_size,
                    size=min(self.rl_margin_batch_size, buffer_size),
                )
                observations = torch.as_tensor(
                    self._margin_observations[indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.as_tensor(
                    self._margin_actions[indices], dtype=torch.long, device=self.device
                )
                masks = torch.as_tensor(self._margin_masks[indices], device=self.device)
                logits = self.policy.get_distribution(
                    observations, action_masks=masks
                ).distribution.logits
                margin_loss = _masked_margin_loss(
                    logits,
                    actions,
                    masks,
                    self.rl_margin,
                    n_teacher_actions=logits.shape[1] - 1,
                )
                loss = self.rl_margin_coef * margin_loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                losses.append(float(margin_loss.detach().cpu()))
                matches.append(float((logits.argmax(dim=1) == actions).float().mean().cpu()))
            self.logger.record("train/teacher_margin_loss", float(np.mean(losses)))
            self.logger.record("train/teacher_argmax_match", float(np.mean(matches)))

    return MarginRegularizedMaskablePPO


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_direction_behavior_cloning(
    *,
    model,
    scenarios: Sequence[Mapping[str, Any]],
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float] | None,
    config: ClinicalTrainingConfig,
) -> dict[str, Any]:
    """Stream S-priority demonstrations without caching full observations."""
    import torch

    from clinical_window_environment import ClinicalWindowResectionEnv
    from clinical_window_evaluation import (
        serpentine_direction_policy,
        serpentine_hierarchical_policy,
        serpentine_macro_target_policy,
    )

    if config.control_mode == "hierarchical":
        from clinical_hierarchical_environment import ClinicalHierarchicalResectionEnv

        environment_class = ClinicalHierarchicalResectionEnv
        teacher_policy = serpentine_hierarchical_policy
    elif config.control_mode == "macro":
        from clinical_macro_environment import ClinicalMacroResectionEnv

        environment_class = ClinicalMacroResectionEnv
        teacher_policy = serpentine_macro_target_policy
    elif config.control_mode == "direction":
        environment_class = ClinicalWindowResectionEnv
        teacher_policy = serpentine_direction_policy
    else:
        raise ValueError(f"Unknown control_mode: {config.control_mode!r}")

    selected = list(scenarios[:config.bc_scenarios])
    if not selected:
        return {"enabled": False, "demonstration_count": 0}
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=config.bc_learning_rate)
    rng = random.Random(config.seed ^ 0xBC2026)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    epoch_summaries: list[dict[str, Any]] = []
    demonstration_count = 0
    completed_episodes = 0
    reservoir = (
        _TeacherDemoReservoir(config.rl_margin_buffer_size, config.seed ^ 0xD3A0)
        if config.rl_margin_coef > 0.0
        else None
    )
    model.policy.set_training_mode(True)

    optimize_network = config.bc_epochs > 0
    for epoch_index in range(max(config.bc_epochs, 1)):
        epoch_demo_start = demonstration_count
        epoch_loss_start = len(losses)
        epoch_policy_loss_start = len(policy_losses)
        epoch_value_loss_start = len(value_losses)
        epoch_scenarios = list(selected)
        rng.shuffle(epoch_scenarios)
        observations: list[np.ndarray] = []
        actions: list[Any] = []
        masks: list[np.ndarray] = []
        returns: list[float] = []

        def train_batch() -> None:
            if not observations:
                return
            observation_tensor = torch.as_tensor(np.stack(observations), device=model.device)
            action_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=model.device)
            mask_tensor = torch.as_tensor(np.stack(masks), device=model.device)
            if config.bc_margin > 0.0:
                # Margin-loss behavior cloning: directly push the teacher action's
                # logit above every other legal action, so the deterministic argmax
                # matches the teacher (MLE-BC plateaus around loss ~1.1 and leaves
                # argmax collapsed to transfers).
                action_distribution = model.policy.get_distribution(
                    observation_tensor,
                    action_masks=mask_tensor,
                )
                if config.control_mode == "hierarchical":
                    logits = action_distribution.distributions[1].logits
                    margin_actions = action_tensor[:, 1]
                    margin_masks = mask_tensor[:, 2:]
                else:
                    logits = action_distribution.distribution.logits
                    margin_actions = action_tensor
                    margin_masks = mask_tensor
                margin_term = _masked_margin_loss(
                    logits,
                    margin_actions,
                    margin_masks,
                    config.bc_margin,
                    n_teacher_actions=(
                        logits.shape[1]
                        if config.control_mode == "hierarchical"
                        else logits.shape[1] - 1
                    ),
                )
                _, log_prob, entropy = model.policy.evaluate_actions(
                    observation_tensor,
                    action_tensor,
                    action_masks=mask_tensor,
                )
                policy_loss = margin_term - 0.01 * entropy.mean()
            else:
                _, log_prob, entropy = model.policy.evaluate_actions(
                    observation_tensor,
                    action_tensor,
                    action_masks=mask_tensor,
                )
                policy_loss = -log_prob.mean() - 0.001 * entropy.mean()
            loss = policy_loss
            if config.bc_v_weight > 0.0:
                # PPO uses raw, unnormalized returns in this trainer, so BC must
                # fit targets on that exact scale as well. Per-batch z-scoring
                # creates a different value function target for every batch.
                returns_tensor = torch.as_tensor(
                    np.asarray(returns, dtype=np.float32), device=model.device
                )
                values = model.policy.predict_values(observation_tensor).squeeze(-1)
                v_loss = torch.nn.functional.mse_loss(values, returns_tensor.detach())
                loss = loss + config.bc_v_weight * v_loss
                value_losses.append(float(v_loss.detach().cpu()))
            optimizer.zero_grad()
            loss.backward()
            if model.policy.share_features_extractor:
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            else:
                # Keep a large raw-return critic gradient from shrinking the
                # independent actor gradient through a single global norm.
                actor_parameters = list(model.policy.pi_features_extractor.parameters())
                actor_parameters += list(model.policy.mlp_extractor.policy_net.parameters())
                actor_parameters += list(model.policy.action_net.parameters())
                critic_parameters = list(model.policy.vf_features_extractor.parameters())
                critic_parameters += list(model.policy.mlp_extractor.value_net.parameters())
                critic_parameters += list(model.policy.value_net.parameters())
                torch.nn.utils.clip_grad_norm_(actor_parameters, 0.5)
                torch.nn.utils.clip_grad_norm_(critic_parameters, 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            observations.clear()
            actions.clear()
            masks.clear()
            returns.clear()

        for scenario in epoch_scenarios:
            env = environment_class(
                scenario=scenario,
                clinical_config=clinical_config,
                reward_config=reward_config,
                mechanics_update_interval=0,
            )
            env.reset()
            ep_obs: list[np.ndarray] = []
            ep_masks: list[np.ndarray] = []
            ep_actions: list[Any] = []
            ep_rewards: list[float] = []
            while not env.terminated and not env.truncated:
                ep_obs.append(env._observation())
                ep_masks.append(env.action_masks())
                action = teacher_policy(env)
                ep_actions.append(action)
                if reservoir is not None:
                    reservoir.add(ep_obs[-1], action, ep_masks[-1])
                demonstration_count += 1
                _, reward, _, _, _ = env.step(action)
                ep_rewards.append(float(reward))
            completed_episodes += int(env.terminated and env.failure_reason is None)
            if not optimize_network:
                # bc_epochs=0: only build the teacher reservoir, never update weights.
                continue
            # Raw Monte-Carlo returns with the same gamma and reward scale as RL.
            ep_returns = _discounted_returns(ep_rewards, config.gamma).tolist()
            observations.extend(ep_obs)
            actions.extend(ep_actions)
            masks.extend(ep_masks)
            returns.extend(ep_returns)
            if len(observations) >= config.bc_batch_size:
                train_batch()
        if optimize_network:
            train_batch()
        epoch_losses = losses[epoch_loss_start:]
        epoch_policy_losses = policy_losses[epoch_policy_loss_start:]
        epoch_value_losses = value_losses[epoch_value_loss_start:]
        epoch_summary = {
            "epoch": epoch_index + 1,
            "demonstration_count": demonstration_count - epoch_demo_start,
            "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "mean_policy_loss": (
                float(np.mean(epoch_policy_losses)) if epoch_policy_losses else None
            ),
            "mean_value_loss": (
                float(np.mean(epoch_value_losses)) if epoch_value_losses else None
            ),
        }
        epoch_summaries.append(epoch_summary)
        print(json.dumps({"behavior_cloning": epoch_summary}, ensure_ascii=False), flush=True)

    model.policy.set_training_mode(False)
    if reservoir is not None and reservoir.observations:
        model.set_margin_demonstrations(*reservoir.arrays())
    return {
        "enabled": True,
        "teacher": (
            "mechanical_serpentine_hierarchical_continue_target"
            if config.control_mode == "hierarchical"
            else (
                "mechanical_serpentine_macro_target"
                if config.control_mode == "macro"
                else "mechanical_serpentine_direction_only"
            )
        ),
        "control_mode": config.control_mode,
        "scenario_count": len(selected),
        "epochs": config.bc_epochs,
        "bc_optimization_epochs": int(config.bc_epochs),
        "teacher_buffer_count": len(reservoir.observations) if reservoir else 0,
        "teacher_buffer_scenarios": len(selected),
        "demonstration_count": demonstration_count,
        "completed_teacher_episodes": completed_episodes,
        "batch_size": config.bc_batch_size,
        "learning_rate": config.bc_learning_rate,
        "value_target_scale": "raw_ppo_reward_scale",
        "teacher_margin_buffer_count": len(reservoir.observations) if reservoir else 0,
        "teacher_margin_buffer_seen": reservoir.seen if reservoir else 0,
        "epoch_summaries": epoch_summaries,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "mean_policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
        "mean_value_loss": float(np.mean(value_losses)) if value_losses else None,
    }


def run_training(
    *,
    train_scenarios: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: ClinicalTrainingConfig,
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    import stable_baselines3
    import sb3_contrib
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    if not train_scenarios:
        raise ValueError("train_scenarios must not be empty")
    if config.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    if config.bc_scenarios < 0 or config.bc_epochs < 0 or config.bc_batch_size <= 0:
        raise ValueError(
            "BC settings must use non-negative scenarios/epochs and positive batch size "
            "(bc_epochs=0 only builds the teacher reservoir without weight updates)"
        )
    if config.bc_v_weight < 0.0 or config.rl_margin_coef < 0.0:
        raise ValueError("BC value and RL margin weights must be non-negative")
    if config.rl_margin_updates <= 0 or config.rl_margin_batch_size <= 0:
        raise ValueError("RL margin updates and batch size must be positive")
    if config.rl_margin_buffer_size <= 0:
        raise ValueError("RL margin buffer size must be positive")
    if config.rl_margin_coef > 0.0 and config.bc_scenarios == 0:
        raise ValueError("RL margin regularization requires --bc-scenarios > 0")
    if config.control_mode not in ("direction", "macro", "hierarchical"):
        raise ValueError("control_mode must be direction, macro, or hierarchical")
    if config.control_mode == "hierarchical" and config.rl_margin_coef > 0.0:
        raise ValueError("hierarchical control does not support --rl-margin-coef; use 0")
    output_dir.mkdir(parents=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Optuna runs multiple trials in one interpreter; PyTorch only permits
        # setting inter-op threads before parallel work starts.
        pass

    def make_env(rank: int):
        def init():
            if config.control_mode == "hierarchical":
                from clinical_hierarchical_environment import (
                    ClinicalHierarchicalScenarioPoolEnv,
                )

                environment_class = ClinicalHierarchicalScenarioPoolEnv
            elif config.control_mode == "macro":
                from clinical_macro_environment import ClinicalMacroScenarioPoolEnv

                environment_class = ClinicalMacroScenarioPoolEnv
            else:
                environment_class = ClinicalWindowScenarioPoolEnv
            return environment_class(
                train_scenarios,
                seed=config.seed + rank,
                clinical_config=clinical_config,
                reward_config=reward_config,
                mechanics_update_interval=config.mechanics_update_interval,
            )
        return init

    constructors = [make_env(rank) for rank in range(config.n_envs)]
    raw_env = DummyVecEnv(constructors) if config.n_envs == 1 else SubprocVecEnv(
        constructors, start_method="fork",
    )
    # Time and blood terms already use frozen Train-only scales.  A second
    # adaptive reward normalization would make the documented weights drift.
    vec_env = VecNormalize(raw_env, norm_obs=False, norm_reward=False, clip_reward=10.0)
    if config.control_mode == "hierarchical":
        from clinical_hierarchical_policy import ClinicalHierarchicalPolicy
        from variable_policy import PaddedSpatialExtractor

        policy_class = ClinicalHierarchicalPolicy
        policy_kwargs = {
            "features_extractor_class": PaddedSpatialExtractor,
            "net_arch": [],
            "share_features_extractor": True,
        }
    elif config.control_mode == "macro":
        from clinical_macro_policy import ClinicalMacroSpatialPolicy
        from variable_policy import PaddedSpatialExtractor

        policy_class = ClinicalMacroSpatialPolicy
        policy_kwargs = {
            "features_extractor_class": PaddedSpatialExtractor,
            "net_arch": [],
            "share_features_extractor": True,
        }
    else:
        policy_class = "CnnPolicy"
        if config.features_extractor == "local_global":
            extractor_class = LocalGlobalClinicalExtractor
        elif config.features_extractor == "cnn":
            extractor_class = ClinicalGridExtractor
        else:
            raise ValueError(f"Unknown features_extractor: {config.features_extractor!r}")
        policy_kwargs = {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            "share_features_extractor": config.share_features_extractor,
        }
    MarginRegularizedMaskablePPO = _margin_ppo_class()
    if config.init_model is None:
        model = MarginRegularizedMaskablePPO(
            policy_class,
            vec_env,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            device=config.device,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            clip_range=config.clip_range,
            target_kl=config.target_kl,
            rl_margin_coef=config.rl_margin_coef,
            rl_margin=config.bc_margin,
            rl_margin_updates=config.rl_margin_updates,
            rl_margin_batch_size=config.rl_margin_batch_size,
            verbose=1,
        )
        # Conservative prior for the never-trained END logit: weight row zeroed,
        # bias negative. PPO may later move END freely; this is only the
        # unlock-time safe initial value.
        if config.end_action_initial_bias is not None:
            if hasattr(model.policy.action_net, "initialize_release"):
                model.policy.action_net.initialize_release(config.end_action_initial_bias)
            elif hasattr(model.policy.action_net, "initialize_end"):
                model.policy.action_net.initialize_end(config.end_action_initial_bias)
            else:
                with torch.no_grad():
                    model.policy.action_net.weight.data[ACTION_END_CLAMP_EARLY].zero_()
                    model.policy.action_net.bias.data[ACTION_END_CLAMP_EARLY] = float(
                        config.end_action_initial_bias
                    )
    else:
        # 4.5: build a fresh PPO instance from CLI hyper-parameters and copy only
        # the policy/value weights from the checkpoint, rebuilding the optimizer
        # and rollout buffer. This guarantees the requested lr/ent_coef/target_kl/
        # n_epochs truly take effect instead of being silently overridden by the
        # checkpoint's stored hyper-parameters.
        initial = Path(config.init_model)
        if not initial.is_file():
            raise FileNotFoundError(f"Initial clinical model does not exist: {initial}")
        checkpoint = MarginRegularizedMaskablePPO.load(str(initial), device=config.device)
        checkpoint_shape = tuple(checkpoint.observation_space.shape)
        current_shape = tuple(vec_env.observation_space.shape)
        checkpoint_actions = tuple(
            int(value) for value in getattr(checkpoint.action_space, "nvec", ())
        ) or (int(checkpoint.action_space.n),)
        current_actions = tuple(
            int(value) for value in getattr(vec_env.action_space, "nvec", ())
        ) or (int(vec_env.action_space.n),)
        if checkpoint_shape != current_shape or checkpoint_actions != current_actions:
            vec_env.close()
            raise ValueError(
                f"Initial checkpoint shape/actions {checkpoint_shape}/{checkpoint_actions} "
                f"are incompatible with {config.control_mode} control "
                f"{current_shape}/{current_actions}; use a checkpoint from the same control mode"
            )
        checkpoint_policy_state = checkpoint.policy.state_dict()
        requested_hyper = {
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "ent_coef": config.ent_coef,
            "clip_range": config.clip_range,
            "target_kl": config.target_kl,
            "n_epochs": config.n_epochs,
        }
        checkpoint_hyper = {
            "learning_rate": float(checkpoint.learning_rate),
            "gamma": float(checkpoint.gamma),
            "gae_lambda": float(checkpoint.gae_lambda),
            "ent_coef": float(checkpoint.ent_coef),
            "clip_range": float(checkpoint.clip_range(1.0)),
            "target_kl": float(getattr(checkpoint, "target_kl", float("nan"))),
            "n_epochs": int(checkpoint.n_epochs),
        }
        del checkpoint
        model = MarginRegularizedMaskablePPO(
            policy_class,
            vec_env,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            device=config.device,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            clip_range=config.clip_range,
            target_kl=config.target_kl,
            rl_margin_coef=config.rl_margin_coef,
            rl_margin=config.bc_margin,
            rl_margin_updates=config.rl_margin_updates,
            rl_margin_batch_size=config.rl_margin_batch_size,
            verbose=1,
        )
        model.policy.load_state_dict(checkpoint_policy_state)
        model.verbose = 1
        model.rl_margin_coef = config.rl_margin_coef
        model.rl_margin = config.bc_margin
        model.rl_margin_updates = config.rl_margin_updates
        model.rl_margin_batch_size = config.rl_margin_batch_size
        model._margin_rng = np.random.default_rng(config.seed ^ 0xA11CE)
        actual_hyper = {
            "learning_rate": float(model.learning_rate),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "ent_coef": float(model.ent_coef),
            "clip_range": float(model.clip_range(1.0)),
            "target_kl": float(getattr(model, "target_kl", float("nan"))),
            "n_epochs": int(model.n_epochs),
        }
        for name, requested in requested_hyper.items():
            if abs(float(requested) - float(actual_hyper[name])) > 1e-9:
                raise RuntimeError(
                    f"fresh-instance hyper-parameter mismatch: {name} requested "
                    f"{requested!r} but new model has {actual_hyper[name]!r}"
                )
        _write_json(
            output_dir / "init_model_hyper_check.json",
            {
                "requested": requested_hyper,
                "fresh_model_actual": actual_hyper,
                "checkpoint_stored": checkpoint_hyper,
                "consistent": True,
            },
        )

    if config.freeze_target_head or config.freeze_features_extractor:
        if config.control_mode != "hierarchical":
            vec_env.close()
            raise ValueError("freeze options are only valid for hierarchical control")
        if config.freeze_target_head:
            for parameter in model.policy.action_net.target_scorer.parameters():
                parameter.requires_grad_(False)
        if config.freeze_features_extractor:
            for parameter in model.policy.features_extractor.parameters():
                parameter.requires_grad_(False)

    if config.control_mode == "hierarchical":
        from clinical_hierarchical_environment import (
            CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION,
            CLINICAL_HIERARCHICAL_OBSERVATION_CHANNELS,
        )

        environment_version = CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION
        observation_description = (
            f"{len(CLINICAL_HIERARCHICAL_OBSERVATION_CHANNELS)}x30x40"
        )
        actions_description = "MultiDiscrete([continue/release, 1200 padded targets])"
        extractor_description = "padded_spatial_dual_head"
    elif config.control_mode == "macro":
        from clinical_macro_environment import (
            CLINICAL_MACRO_ENVIRONMENT_VERSION,
            CLINICAL_MACRO_OBSERVATION_CHANNELS,
        )

        environment_version = CLINICAL_MACRO_ENVIRONMENT_VERSION
        observation_description = f"{len(CLINICAL_MACRO_OBSERVATION_CHANNELS)}x30x40"
        actions_description: Any = "1200 padded frontier targets + end_clamp_early"
        extractor_description = "padded_spatial_macro"
    else:
        environment_version = CLINICAL_ENVIRONMENT_VERSION
        observation_description = f"{len(CLINICAL_OBSERVATION_CHANNELS)}x30x40"
        actions_description = ["up", "down", "left", "right", "end_clamp_early"]
        extractor_description = config.features_extractor
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "clinical_config": dict(clinical_config),
        "reward_config": dict(reward_config or {}),
        "train_scenario_count": len(train_scenarios),
        "train_scenario_ids": [item.get("scenario_id") for item in train_scenarios],
        "environment_version": environment_version,
        "control_mode": config.control_mode,
        "reward_normalization": False,
        "observation": observation_description,
        "features_extractor": extractor_description,
        "actual_share_features_extractor": bool(model.policy.share_features_extractor),
        "early_end_mode": str(clinical_config.get("early_end_mode", "full")),
        "early_end_minutes": float(clinical_config.get("early_end_minutes", 0.0)),
        "end_action_initial_bias": config.end_action_initial_bias,
        "actions": actions_description,
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "provenance": dict(provenance or {}),
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    bc_result = _run_direction_behavior_cloning(
        model=model,
        scenarios=train_scenarios,
        clinical_config=clinical_config,
        reward_config=reward_config,
        config=config,
    )
    _write_json(output_dir / "behavior_cloning.json", bc_result)
    if bc_result["enabled"]:
        model.save(str(output_dir / "pretrained_model"))
    if config.checkpoint_global_interval:
        save_freq = max(1, config.checkpoint_global_interval // max(config.n_envs, 1))
    else:
        save_freq = max(1, config.timesteps // max(4 * config.n_envs, 1))
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="clinical_window",
        save_vecnormalize=True,
    )

    class ClinicalHealthCallback(BaseCallback):
        """Persist rolling terminal metrics so failed navigation is visible early."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.episodes: deque[dict[str, float]] = deque(maxlen=100)

        def _on_step(self) -> bool:
            for done, info in zip(self.locals.get("dones", ()), self.locals.get("infos", ())):
                if not done:
                    continue
                directions = max(1, int(info.get("direction_action_count", 0)))
                self.episodes.append({
                    "completion": float(info.get("coverage", 0.0) >= 1.0),
                    "coverage": float(info.get("coverage", 0.0)),
                    "elapsed_minutes": float(info.get("elapsed_minutes", 0.0)),
                    "expected_blood_loss_ml": float(info.get("expected_blood_loss_ml", 0.0)),
                    "transfer_overhead": float(info.get("transfer_count", 0)) / directions,
                    "early_end_count": float(info.get("early_end_count", 0)),
                    "max_no_progress_streak": float(info.get("max_no_progress_streak", 0)),
                    "stagnation_failure": float(
                        str(info.get("failure_reason") or "").startswith("stagnation:")
                    ),
                    "max_same_edge_streak": float(info.get("max_same_edge_streak", 0)),
                    "two_cell_loop_failure": float(
                        str(info.get("failure_reason") or "").startswith("two-cell oscillation:")
                    ),
                })
            return True

        def _on_rollout_end(self) -> None:
            if not self.episodes:
                return
            keys = tuple(self.episodes[0])
            payload = {
                "timesteps": int(self.num_timesteps),
                "rolling_episode_count": len(self.episodes),
                **{
                    f"mean_{key}": float(np.mean([item[key] for item in self.episodes]))
                    for key in keys
                },
            }
            for key, value in payload.items():
                if key.startswith("mean_"):
                    self.logger.record(f"clinical/{key}", value)
            _write_json(output_dir / "training_health" / f"step_{self.num_timesteps}.json", payload)
            _write_json(output_dir / "training_health_latest.json", payload)

    callback = CallbackList([checkpoint_callback, ClinicalHealthCallback()])
    try:
        model.learn(total_timesteps=config.timesteps, callback=callback, progress_bar=False)
        model.save(str(output_dir / "final_model"))
        vec_env.save(str(output_dir / "vecnormalize.pkl"))
        result = {
            "actual_timesteps": int(model.num_timesteps),
            "output_dir": str(output_dir),
            "final_model": str(output_dir / "final_model.zip"),
        }
        _write_json(output_dir / "training_complete.json", result)
        return result
    finally:
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--time-cost", type=float, default=1.0)
    parser.add_argument("--blood-cost", type=float, default=1.0)
    parser.add_argument("--progress-bonus", type=float, default=5.0)
    parser.add_argument("--seal-progress-bonus", type=float, default=2.0)
    parser.add_argument("--stagnation-penalty-cap", type=float, default=0.05)
    parser.add_argument("--two-cell-loop-penalty", type=float, default=0.25)
    parser.add_argument("--clinical-cost-cap", type=float, default=10.0)
    parser.add_argument("--front-tension-cost", type=float, default=0.10)
    parser.add_argument("--organ-energy-cost", type=float, default=0.10)
    parser.add_argument("--vessel-strain-cost", type=float, default=1.0)
    parser.add_argument("--completion-bonus", type=float, default=20.0)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026080401)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--mechanics-update-interval", type=int, default=0)
    parser.add_argument("--checkpoint-global-interval", type=int, default=None)
    parser.add_argument("--early-end-mode", choices=("disabled", "threshold", "full"), default="full")
    parser.add_argument("--early-end-minutes", type=float, default=0.0)
    parser.add_argument("--stagnation-soft-start-steps", type=int, default=40)
    parser.add_argument("--stagnation-penalty-ramp-steps", type=int, default=24)
    parser.add_argument("--stagnation-limit-steps", type=int, default=96)
    parser.add_argument("--two-cell-loop-soft-start-traversals", type=int, default=6)
    parser.add_argument("--two-cell-loop-limit-traversals", type=int, default=12)
    parser.add_argument("--end-action-initial-bias", type=float, default=-4.0)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--bc-scenarios", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--bc-margin", type=float, default=1.0)
    parser.add_argument("--bc-v-weight", type=float, default=0.5)
    parser.add_argument(
        "--share-features-extractor",
        action="store_true",
        help="Share actor/critic features (legacy behavior; independent extractors are safer).",
    )
    parser.add_argument(
        "--rl-margin-coef",
        type=float,
        default=0.0,
        help="Teacher-margin regularization weight applied after each PPO update (0 disables).",
    )
    parser.add_argument("--rl-margin-updates", type=int, default=1)
    parser.add_argument("--rl-margin-batch-size", type=int, default=256)
    parser.add_argument("--rl-margin-buffer-size", type=int, default=2048)
    parser.add_argument("--features-extractor", choices=("cnn", "local_global"), default="cnn")
    parser.add_argument(
        "--control-mode",
        choices=("direction", "macro", "hierarchical"),
        default="direction",
    )
    parser.add_argument("--freeze-target-head", action="store_true")
    parser.add_argument("--freeze-features-extractor", action="store_true")
    args = parser.parse_args()

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    train_scenarios = list(split_payload["splits"]["train"])
    if args.train_limit is not None:
        train_scenarios = train_scenarios[:args.train_limit]
    clinical_config = {
        "bleeding_probability": 1.0,
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "early_end_mode": args.early_end_mode,
        "early_end_minutes": args.early_end_minutes,
        "stagnation_soft_start_steps": args.stagnation_soft_start_steps,
        "stagnation_penalty_ramp_steps": args.stagnation_penalty_ramp_steps,
        "stagnation_limit_steps": args.stagnation_limit_steps,
        "two_cell_loop_soft_start_traversals": args.two_cell_loop_soft_start_traversals,
        "two_cell_loop_limit_traversals": args.two_cell_loop_limit_traversals,
    }
    config = ClinicalTrainingConfig(
        seed=args.seed,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        device=args.device,
        torch_threads=args.torch_threads,
        mechanics_update_interval=args.mechanics_update_interval,
        checkpoint_global_interval=args.checkpoint_global_interval,
        init_model=str(args.init_model) if args.init_model is not None else None,
        bc_scenarios=args.bc_scenarios,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        bc_margin=args.bc_margin,
        bc_v_weight=args.bc_v_weight,
        share_features_extractor=args.share_features_extractor,
        end_action_initial_bias=args.end_action_initial_bias,
        rl_margin_coef=args.rl_margin_coef,
        rl_margin_updates=args.rl_margin_updates,
        rl_margin_batch_size=args.rl_margin_batch_size,
        rl_margin_buffer_size=args.rl_margin_buffer_size,
        features_extractor=args.features_extractor,
        control_mode=args.control_mode,
        freeze_target_head=args.freeze_target_head,
        freeze_features_extractor=args.freeze_features_extractor,
    )
    result = run_training(
        train_scenarios=train_scenarios,
        output_dir=args.output_dir,
        config=config,
        clinical_config=clinical_config,
        reward_config={
            "time_cost": args.time_cost,
            "blood_cost": args.blood_cost,
            "progress_bonus": args.progress_bonus,
            "seal_progress_bonus": args.seal_progress_bonus,
            "stagnation_penalty_cap": args.stagnation_penalty_cap,
            "two_cell_loop_penalty": args.two_cell_loop_penalty,
            "clinical_cost_cap": args.clinical_cost_cap,
            "front_tension_cost": args.front_tension_cost,
            "organ_energy_cost": args.organ_energy_cost,
            "vessel_strain_cost": args.vessel_strain_cost,
            "completion_bonus": args.completion_bonus,
            "failure_penalty": args.failure_penalty,
            "invalid_action_penalty": args.invalid_action_penalty,
        },
        provenance={
            "split_file": str(args.splits.resolve()),
            "split_sha256": _sha256(args.splits),
            "scale_file": str(args.scales.resolve()),
            "scale_sha256": _sha256(args.scales),
        },
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

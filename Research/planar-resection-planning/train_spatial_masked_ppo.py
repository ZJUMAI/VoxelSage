"""Train the compact spatial Maskable PPO policy for the 7x7 vessel curriculum."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from environment import (
    LOCAL_OBSERVATION_CHANNELS,
    LocalGridScenarioPoolEnv,
    PlanarResectionEnv,
    local_grid_action_masks,
    local_grid_observation,
)
from evaluation import evaluate_policy, serpentine_priority_policy
from generalization_evaluation import generate_generalization_splits, generate_training_scenarios
from planner import plan_resection
from train_masked_ppo import _dependency_versions, toy_scenarios


@dataclass(frozen=True)
class SpatialTrainingConfig:
    seed: int = 2026072901
    grid_size: int = 7
    total_timesteps: int = 100_000
    n_envs: int = 8
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 3
    learning_rate: float = 3e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef_start: float = 0.001
    ent_coef_end: float = 0.0001
    vf_coef: float = 0.5
    target_kl: float = 0.02
    bc_epochs: int = 15
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    checkpoint_frequency: int = 25_000
    gate_transfer_overhead: float = 0.8
    transfer_cost: float = 2.0
    lookahead_transfer_cost: float = 0.0
    teacher_planner_fraction: float = 0.5
    device: str = "auto"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def collect_serpentine_demonstrations(
    scenarios: Sequence[Mapping[str, Any]], grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Collect legal S-priority/release state-action pairs in local coordinates."""
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    episode_overheads: list[float] = []
    for scenario in scenarios:
        env = PlanarResectionEnv(scenario=scenario)
        env.reset()
        while not env.terminated and not env.truncated:
            observations.append(local_grid_observation(env, grid_size))
            masks.append(local_grid_action_masks(env, grid_size))
            canvas_action = serpentine_priority_policy(env)
            row, col = divmod(canvas_action, 50)
            actions.append(row * grid_size + col)
            env.step(canvas_action)
        transfers = sum(event["action"] == "transfer" for event in env.events)
        cuts = sum(event["action"] == "cut" for event in env.events)
        episode_overheads.append(transfers / cuts)
    return (
        np.stack(observations).astype(np.float32),
        np.asarray(actions, dtype=np.int64),
        np.stack(masks).astype(bool),
        {
            "demonstration_count": float(len(observations)),
            "episode_count": float(len(scenarios)),
            "mean_transfer_overhead": float(mean(episode_overheads)),
        },
    )


def collect_mixed_teacher_demonstrations(
    scenarios: Sequence[Mapping[str, Any]], grid_size: int, planner_fraction: float,
    transfer_cost: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Collect mixed S-scan and dynamic-planner demonstrations."""
    if not 0.0 <= planner_fraction <= 1.0:
        raise ValueError("planner_fraction must be between 0 and 1")
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    overheads: list[float] = []
    planner_episodes = 0
    for index, scenario in enumerate(scenarios):
        use_planner = ((index * 37) % 100) < round(planner_fraction * 100)
        sequence: list[int] | None = None
        if use_planner:
            planned = plan_resection(**{
                key: scenario[key]
                for key in ("rows", "cols", "domain_cells", "obstacle_cells", "start_cell")
            })
            sequence = [
                int(event["cell"][0]) * 50 + int(event["cell"][1])
                for event in planned["events"] if event["action"] == "cut"
            ]
            planner_episodes += 1
        env = PlanarResectionEnv(
            scenario=scenario,
            reward_config={
                "transfer_cost": transfer_cost,
                "lookahead_transfer_cost": 0.0,
            },
        )
        env.reset()
        step_index = 1
        while not env.terminated and not env.truncated:
            observations.append(local_grid_observation(env, grid_size))
            masks.append(local_grid_action_masks(env, grid_size))
            if sequence is not None and step_index < len(sequence):
                canvas_action = sequence[step_index]
            else:
                canvas_action = serpentine_priority_policy(env)
            row, col = divmod(canvas_action, 50)
            actions.append(row * grid_size + col)
            env.step(canvas_action)
            step_index += 1
        transfers = sum(event["action"] == "transfer" for event in env.events)
        cuts = sum(event["action"] == "cut" for event in env.events)
        overheads.append(transfers / cuts if cuts else 0.0)
    return (
        np.stack(observations).astype(np.float32),
        np.asarray(actions, dtype=np.int64),
        np.stack(masks).astype(bool),
        {
            "demonstration_count": float(len(observations)),
            "episode_count": float(len(scenarios)),
            "planner_episode_count": float(planner_episodes),
            "mean_transfer_overhead": float(mean(overheads)),
        },
    )


def _external_validation(model, scenarios: Sequence[Mapping[str, Any]], grid_size: int) -> dict[str, Any]:
    metrics = []
    for scenario in scenarios:
        def policy(env: PlanarResectionEnv) -> int:
            local_action, _ = model.predict(
                local_grid_observation(env, grid_size),
                deterministic=True,
                action_masks=local_grid_action_masks(env, grid_size),
            )
            row, col = divmod(int(local_action), grid_size)
            return row * 50 + col

        metrics.append(evaluate_policy(scenario, policy))
    return {
        "validation_count": len(metrics),
        "completion_rate": float(mean(float(item["completion"]) for item in metrics)),
        "legal_action_rate": float(mean(item["legal_action_rate"] for item in metrics)),
        "mean_transfer_overhead": float(mean(item["transfer_overhead"] for item in metrics)),
        "mean_total_reward": float(mean(item["total_reward"] for item in metrics)),
        "metrics": metrics,
    }


def run_training(
    *,
    train_scenarios: Sequence[Mapping[str, Any]],
    validation_scenarios: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: SpatialTrainingConfig,
) -> dict[str, Any]:
    try:
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    except ImportError as exc:
        raise RuntimeError("The compatible PPO training environment is required") from exc

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    # Keep each learner process bounded; the host is shared with other jobs.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config.seed)

    class SpatialGridExtractor(BaseFeaturesExtractor):
        """Preserve the complete local feature map before actor/critic separation."""

        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            channels, rows, cols = observation_space.shape
            self.spatial = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * rows * cols, features_dim),
                torch.nn.ReLU(),
            )

        def forward(self, observations):
            return self.spatial(observations)

    class EntropyScheduleCallback(BaseCallback):
        def _on_step(self) -> bool:
            fraction = min(1.0, self.num_timesteps / max(1, config.total_timesteps))
            self.model.ent_coef = (
                config.ent_coef_start
                + fraction * (config.ent_coef_end - config.ent_coef_start)
            )
            return True

    class ExternalValidationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_evaluation = config.checkpoint_frequency
            self.gate_passed: bool | None = None

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_evaluation:
                return True
            summary = _external_validation(
                self.model, validation_scenarios, config.grid_size,
            )
            summary["timesteps"] = self.num_timesteps
            summary["explained_variance"] = float(
                self.model.logger.name_to_value.get("train/explained_variance", float("nan"))
            )
            _write_json(
                output_dir / "validation" / f"step_{self.num_timesteps:09d}.json",
                summary,
            )
            if self.gate_passed is None:
                self.gate_passed = (
                    summary["completion_rate"] == 1.0
                    and summary["legal_action_rate"] == 1.0
                    and summary["mean_transfer_overhead"] <= config.gate_transfer_overhead
                )
                _write_json(output_dir / "quality_gate.json", {
                    "passed": self.gate_passed,
                    "reason": "25k external metric gate passed" if self.gate_passed else "25k external metric gate failed",
                    "thresholds": {
                        "completion_rate": 1.0,
                        "legal_action_rate": 1.0,
                        "mean_transfer_overhead_max": config.gate_transfer_overhead,
                    },
                    "observed": {
                        key: value for key, value in summary.items() if key != "metrics"
                    },
                })
                if not self.gate_passed:
                    return False
            self.next_evaluation += config.checkpoint_frequency
            return True

    def make_env(worker_rank: int):
        def init():
            return LocalGridScenarioPoolEnv(
                train_scenarios,
                grid_size=config.grid_size,
                seed=config.seed + worker_rank,
                reward_config={
                    "transfer_cost": config.transfer_cost,
                    "lookahead_transfer_cost": config.lookahead_transfer_cost,
                },
            )
        return init

    if config.n_envs == 1:
        raw_vec_env = DummyVecEnv([make_env(0)])
    else:
        raw_vec_env = SubprocVecEnv(
            [make_env(rank) for rank in range(config.n_envs)],
            # Fork avoids re-importing the mechanics/SciPy stack in every worker
            # on this Linux host, which is materially more stable under load.
            start_method="fork",
        )
    vec_env = VecNormalize(
        raw_vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=config.gamma,
    )
    policy_kwargs = {
        "features_extractor_class": SpatialGridExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "share_features_extractor": False,
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
        "activation_fn": torch.nn.ReLU,
    }
    model = MaskablePPO(
        "MlpPolicy",
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
        clip_range=config.clip_range,
        ent_coef=config.ent_coef_start,
        vf_coef=config.vf_coef,
        target_kl=config.target_kl,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "dependencies": _dependency_versions(),
        "architecture": {
            "name": "local-spatial-v2",
            "observation_channels": list(LOCAL_OBSERVATION_CHANNELS),
            "observation_shape": [len(LOCAL_OBSERVATION_CHANNELS), config.grid_size, config.grid_size],
            "action_count": config.grid_size * config.grid_size,
            "global_average_pooling": False,
            "shared_actor_critic_extractor": False,
            "reward_normalization": True,
        },
        "train_scenarios": list(train_scenarios),
        "validation_scenarios": list(validation_scenarios),
    }
    _write_json(output_dir / "run_metadata.json", metadata)

    demo_observations, demo_actions, demo_masks, demo_summary = collect_mixed_teacher_demonstrations(
        train_scenarios, config.grid_size, config.teacher_planner_fraction, config.transfer_cost,
    )
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=config.bc_learning_rate)
    observation_tensor = torch.as_tensor(demo_observations, device=model.device)
    action_tensor = torch.as_tensor(demo_actions, device=model.device)
    mask_tensor = torch.as_tensor(demo_masks, device=model.device)
    losses: list[float] = []
    model.policy.set_training_mode(True)
    for _ in range(config.bc_epochs):
        permutation = torch.randperm(len(action_tensor), device=model.device)
        for start in range(0, len(action_tensor), config.bc_batch_size):
            indices = permutation[start:start + config.bc_batch_size]
            _, log_prob, entropy = model.policy.evaluate_actions(
                observation_tensor[indices],
                action_tensor[indices],
                action_masks=mask_tensor[indices],
            )
            loss = -log_prob.mean() - 0.001 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    demo_summary.update({
        "bc_epochs": float(config.bc_epochs),
        "final_bc_loss": losses[-1] if losses else float("nan"),
    })
    _write_json(output_dir / "behavior_cloning.json", demo_summary)
    model.save(str(output_dir / "pretrained_model"))
    pretrain_validation = _external_validation(model, validation_scenarios, config.grid_size)
    _write_json(output_dir / "validation_pretrain.json", pretrain_validation)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, config.checkpoint_frequency // config.n_envs),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_spatial",
    )
    validation_callback = ExternalValidationCallback()
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList([
                checkpoint_callback,
                validation_callback,
                EntropyScheduleCallback(),
            ]),
            progress_bar=False,
        )
        model.save(str(output_dir / "final_model"))
        vec_env.save(str(output_dir / "vecnormalize.pkl"))
        summary = _external_validation(model, validation_scenarios, config.grid_size)
        summary["actual_timesteps"] = int(model.num_timesteps)
        summary["quality_gate_passed"] = validation_callback.gate_passed
        _write_json(output_dir / "validation.json", summary)
        return summary
    finally:
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026072901)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gate-overhead", type=float, default=0.8)
    parser.add_argument("--train-scenarios", type=int, default=256)
    parser.add_argument("--teacher-planner-fraction", type=float, default=0.5)
    parser.add_argument("--transfer-cost", type=float, default=2.0)
    parser.add_argument("--lookahead-transfer-cost", type=float, default=0.0)
    args = parser.parse_args()
    config = SpatialTrainingConfig(
        seed=args.seed,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        bc_epochs=args.bc_epochs,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        device=args.device,
        gate_transfer_overhead=args.gate_overhead,
        teacher_planner_fraction=args.teacher_planner_fraction,
        transfer_cost=args.transfer_cost,
        lookahead_transfer_cost=args.lookahead_transfer_cost,
    )
    train = generate_training_scenarios(count=args.train_scenarios, seed=args.seed + 31_000)
    # Keep in-training validation intentionally small; the full frozen
    # generalization suite is run in parallel after training.
    frozen = generate_generalization_splits(test_count=1, stress_count=1)
    validation = frozen["splits"]["test"] + frozen["splits"]["stress"]
    summary = run_training(
        train_scenarios=train,
        validation_scenarios=validation,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

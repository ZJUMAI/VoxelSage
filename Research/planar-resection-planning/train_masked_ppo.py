"""Reproducible Maskable PPO entry point for PlanarResectionEnv.

This file does not start training on import.  It is intentionally a CLI so a
run always records its frozen scenarios, configuration, package versions, and
validation metrics next to the resulting checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from environment import CANVAS_SIZE, OBSERVATION_CHANNELS, PlanarResectionEnv, ScenarioPoolEnv
from evaluation import evaluate_policy


@dataclass(frozen=True)
class TrainingConfig:
    environment_version: int = 1
    seed: int = 2026072901
    total_timesteps: int = 100_000
    n_envs: int = 4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    device: str = "auto"
    checkpoint_frequency: int = 25_000


def toy_scenarios(*, size: int, count: int, seed: int, with_vessel: bool = False) -> list[dict[str, Any]]:
    """Deterministic small-grid curriculum used before random-domain training."""
    if size not in (5, 7):
        raise ValueError("Toy curriculum currently supports 5x5 and 7x7 grids")
    rng = random.Random(seed)
    domain = [[row, col] for row in range(size) for col in range(size)]
    boundary = [[0, col] for col in range(size)] + [[row, size - 1] for row in range(1, size)]
    boundary += [[size - 1, col] for col in range(size - 2, -1, -1)] + [[row, 0] for row in range(size - 2, 0, -1)]
    obstacle = [[size // 2, size // 2]] if with_vessel else []
    return [
        {
            "scenario_id": f"toy{size}-{'vessel' if with_vessel else 'plain'}-{index:04d}",
            "seed": seed + index,
            "rows": size, "cols": size, "domain_cells": domain,
            "obstacle_cells": obstacle, "start_cell": boundary[rng.randrange(len(boundary))],
        }
        for index in range(count)
    ]


def _dependency_versions() -> dict[str, str]:
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch

    return {
        "python": sys.version.split()[0], "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__, "sb3_contrib": sb3_contrib.__version__,
        "torch": torch.__version__, "cuda_available": str(torch.cuda.is_available()),
    }


def _require_training_dependencies():
    try:
        import torch
        from gymnasium import spaces
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "Masked PPO dependencies are missing. Install compatible gymnasium, "
            "stable-baselines3 and sb3-contrib packages before running this script."
        ) from exc
    return torch, spaces, MaskablePPO, BaseCallback, CallbackList, CheckpointCallback, BaseFeaturesExtractor, SubprocVecEnv


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_training(
    *,
    train_scenarios: Sequence[Mapping[str, Any]],
    validation_scenarios: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Train once and persist all artefacts needed for exact reruns."""
    (torch, spaces, MaskablePPO, BaseCallback, CallbackList, CheckpointCallback,
     BaseFeaturesExtractor, SubprocVecEnv) = _require_training_dependencies()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    class ResectionCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            channels = observation_space.shape[0]
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 32, kernel_size=5, stride=2), torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=2), torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=2), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(64, features_dim), torch.nn.ReLU(),
            )

        def forward(self, observations):
            return self.cnn(observations)

    class ExternalValidationCallback(BaseCallback):
        """Evaluate every saved-checkpoint interval using external metrics only."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_evaluation = config.checkpoint_frequency

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_evaluation:
                return True
            metrics = []
            for scenario in validation_scenarios:
                def policy(env: PlanarResectionEnv) -> int:
                    action, _ = self.model.predict(
                        env._observation(), deterministic=True, action_masks=env.action_masks(),
                    )
                    return int(action)
                metrics.append(evaluate_policy(scenario, policy))
            summary = {
                "timesteps": self.num_timesteps,
                "completion_rate": sum(item["completion"] for item in metrics) / len(metrics),
                "mean_transfer_overhead": float(np.mean([item["transfer_overhead"] for item in metrics])),
                "metrics": metrics,
            }
            _write_json(output_dir / "validation" / f"step_{self.num_timesteps:09d}.json", summary)
            self.next_evaluation += config.checkpoint_frequency
            return True

    def make_env(worker_rank: int):
        def init():
            return ScenarioPoolEnv(train_scenarios, seed=config.seed + worker_rank)
        return init

    vec_env = SubprocVecEnv([make_env(rank) for rank in range(config.n_envs)], start_method="spawn")
    policy_kwargs = {"features_extractor_class": ResectionCNN, "features_extractor_kwargs": {"features_dim": 256}}
    model = MaskablePPO(
        "MlpPolicy", vec_env, policy_kwargs=policy_kwargs, seed=config.seed, device=config.device,
        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
        learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef, verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
    )
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "dependencies": _dependency_versions(), "observation_channels": list(OBSERVATION_CHANNELS),
        "action_count": CANVAS_SIZE * CANVAS_SIZE, "train_scenarios": list(train_scenarios),
        "validation_scenarios": list(validation_scenarios),
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, config.checkpoint_frequency // config.n_envs),
        save_path=str(output_dir / "checkpoints"), name_prefix="ppo",
    )
    validation_callback = ExternalValidationCallback()
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList([checkpoint_callback, validation_callback]), progress_bar=True,
        )
        model.save(str(output_dir / "final_model"))
        metrics = []
        for scenario in validation_scenarios:
            def policy(env: PlanarResectionEnv) -> int:
                action, _ = model.predict(env._observation(), deterministic=True, action_masks=env.action_masks())
                return int(action)
            metrics.append(evaluate_policy(scenario, policy))
        summary = {
            "validation_count": len(metrics),
            "completion_rate": sum(item["completion"] for item in metrics) / len(metrics),
            "mean_transfer_overhead": float(np.mean([item["transfer_overhead"] for item in metrics])),
            "metrics": metrics,
        }
        _write_json(output_dir / "validation.json", summary)
        return summary
    finally:
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("toy5", "toy7"), default="toy5")
    parser.add_argument("--with-vessel", action="store_true", help="Include a releasable centre vessel in toy scenarios")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026072901)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    size = 5 if args.stage == "toy5" else 7
    config = TrainingConfig(seed=args.seed, total_timesteps=args.timesteps, n_envs=args.n_envs, device=args.device)
    train = toy_scenarios(size=size, count=32, seed=args.seed, with_vessel=args.with_vessel)
    validation = toy_scenarios(size=size, count=16, seed=args.seed + 1_000_000, with_vessel=args.with_vessel)
    summary = run_training(train_scenarios=train, validation_scenarios=validation, output_dir=args.output_dir, config=config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

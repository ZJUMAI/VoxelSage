"""Smoke and curriculum PPO training for 4 mm variable-size planar grids."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from environment import VARIABLE_GRID_COLS, VARIABLE_GRID_ROWS, VariableGridScenarioPoolEnv
from variable_policy import PaddedSpatialExtractor, VariableSpatialPolicy
from variable_scenarios import CURRICULUM_RANGES, generate_curriculum_train_pool, generate_stage_pool
from variable_teacher import load_teacher_cache


@dataclass(frozen=True)
class VariableTrainingConfig:
    stage: str = "a"
    seed: int = 2026073001
    timesteps: int = 25_000
    n_envs: int = 8
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 3
    learning_rate: float = 3e-5
    bc_epochs: int = 10
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    transfer_cost: float = 2.0
    device: str = "auto"
    init_model: str | None = None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run_training(
    *, train_scenarios: Sequence[Mapping[str, Any]], teacher_cache: Path, output_dir: Path, config: VariableTrainingConfig,
) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    # The mechanics solver is CPU-bound; unbounded PyTorch threads can exhaust
    # the shared host before PPO begins collecting its first rollout.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def make_env(rank: int):
        def init():
            return VariableGridScenarioPoolEnv(
                train_scenarios, seed=config.seed + rank,
                max_rows=VARIABLE_GRID_ROWS, max_cols=VARIABLE_GRID_COLS,
                reward_config={"transfer_cost": config.transfer_cost},
            )
        return init

    constructors = [make_env(rank) for rank in range(config.n_envs)]
    raw_env = DummyVecEnv(constructors) if config.n_envs == 1 else SubprocVecEnv(constructors, start_method="fork")
    vec_env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    policy_kwargs = {
        "features_extractor_class": PaddedSpatialExtractor,
        "net_arch": [],
        "share_features_extractor": True,
    }
    if config.init_model is None:
        model = MaskablePPO(
            VariableSpatialPolicy, vec_env, policy_kwargs=policy_kwargs, seed=config.seed,
            device=config.device, n_steps=config.n_steps, batch_size=config.batch_size,
            n_epochs=config.n_epochs, learning_rate=config.learning_rate, gamma=0.99,
            gae_lambda=0.95, target_kl=0.02, verbose=1,
        )
    else:
        init_path = Path(config.init_model)
        if not init_path.is_file():
            raise FileNotFoundError(f"Initial model does not exist: {init_path}")
        model = MaskablePPO.load(str(init_path), env=vec_env, device=config.device)
        model.verbose = 1
    _write_json(output_dir / "run_metadata.json", {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "cell_size_mm": 4.0,
        "maximum_grid": [VARIABLE_GRID_ROWS, VARIABLE_GRID_COLS],
        "architecture": "padded-convolutional-per-cell-logits",
        "train_scenarios": list(train_scenarios),
    })
    observations, actions, masks, teacher_summary = load_teacher_cache(teacher_cache)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=config.bc_learning_rate)
    observation_tensor = torch.as_tensor(observations, device=model.device)
    action_tensor = torch.as_tensor(actions, device=model.device)
    mask_tensor = torch.as_tensor(masks, device=model.device)
    losses: list[float] = []
    model.policy.set_training_mode(True)
    for _ in range(config.bc_epochs):
        permutation = torch.randperm(len(action_tensor), device=model.device)
        for start in range(0, len(action_tensor), config.bc_batch_size):
            indices = permutation[start:start + config.bc_batch_size]
            _, log_prob, entropy = model.policy.evaluate_actions(
                observation_tensor[indices], action_tensor[indices], action_masks=mask_tensor[indices],
            )
            loss = -log_prob.mean() - 0.001 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    teacher_summary.update({"bc_epochs": float(config.bc_epochs), "final_bc_loss": losses[-1]})
    _write_json(output_dir / "behavior_cloning.json", teacher_summary)
    model.save(str(output_dir / "pretrained_model"))
    callback = CheckpointCallback(
        save_freq=max(1, config.timesteps // (4 * config.n_envs)),
        save_path=str(output_dir / "checkpoints"), name_prefix="variable_spatial",
    )
    try:
        model.learn(total_timesteps=config.timesteps, callback=callback, progress_bar=False)
        model.save(str(output_dir / "final_model"))
        vec_env.save(str(output_dir / "vecnormalize.pkl"))
        return {"actual_timesteps": int(model.num_timesteps), "output_dir": str(output_dir)}
    finally:
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=tuple(CURRICULUM_RANGES), default="a")
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--train-scenarios", type=int, default=128)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--teacher-cache", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026073001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-model", type=Path, help="Optional prior-stage PPO model used as the initialization.")
    args = parser.parse_args()
    config = VariableTrainingConfig(
        stage=args.stage, timesteps=args.timesteps, n_envs=args.n_envs,
        n_steps=args.n_steps, batch_size=args.batch_size,
        seed=args.seed, device=args.device, bc_epochs=args.bc_epochs,
        init_model=str(args.init_model) if args.init_model is not None else None,
    )
    train = generate_curriculum_train_pool(stage=config.stage, count=args.train_scenarios, seed=config.seed)
    print(json.dumps(run_training(
        train_scenarios=train, teacher_cache=args.teacher_cache,
        output_dir=args.output_dir, config=config,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()

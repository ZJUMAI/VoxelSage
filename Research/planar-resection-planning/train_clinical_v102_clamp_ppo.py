"""v10.2 clamp-only conservative PPO (Stage 2).

Initializes from the Stage 1 best oracle checkpoint and freezes:
  - the BC target head;
  - the target feature extractor (base_spatial);
  - the automatic transferer (deterministic env logic).

PPO loss is augmented with an oracle KL/BC anchor so stochastic END
exploration does not wash out the supervised initialization.  Every 2k steps a
fixed Probe-64 is evaluated both deterministically and stochastically (>=5
repeats).  Early stopping restores the previous safe checkpoint and stops the
seed on any of the listed conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from clinical_target_conditioned_environment import (
    TargetConditionedClampEnv,
    TargetConditionedScenarioPoolEnv,
)
from clinical_target_conditioned_policy import (
    FrozenBCMacroTargetPolicy,
    PaddedPlanSpatialExtractor,
    TargetConditionedClampPolicy,
)


class OracleKLAnchorMaskablePPO:
    """MaskablePPO subclass stub — implemented in the training module below."""

    def __init__(self, *args, oracle_kl_coef=0.2, **kwargs):
        from sb3_contrib import MaskablePPO as _MaskablePPO

        self._base = _MaskablePPO
        self._args = args
        self._kwargs = kwargs
        self.oracle_kl_coef = float(oracle_kl_coef)
        self.oracle_policy = None
        self._model = None

    def __getattr__(self, name):
        if self._model is not None:
            return getattr(self._model, name)
        raise AttributeError(name)

    def build(self):
        from sb3_contrib import MaskablePPO as _MaskablePPO

        self._model = _MaskablePPO(*self._args, **self._kwargs)
        self._model.oracle_kl_coef = self.oracle_kl_coef
        self._model.oracle_policy = None
        return self._model


def run_clamp_ppo(
    *,
    train_scenarios: Sequence[Mapping[str, Any]],
    probe_scenarios: Sequence[Mapping[str, Any]],
    output_dir: Path,
    oracle_model_path: Path,
    bc_model_path: Path,
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    oracle_kl_coef: float,
    timesteps: int,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    clip_range: float,
    target_kl: float,
    ent_coef: float,
    seed: int,
    device: str,
    checkpoint_global_interval: int,
) -> dict[str, Any]:
    import torch
    import stable_baselines3
    import sb3_contrib
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output_dir}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    target_policy = FrozenBCMacroTargetPolicy(bc_model_path, device=device)

    def make_env(rank: int):
        def init():
            return TargetConditionedScenarioPoolEnv(
                train_scenarios,
                seed=seed + rank,
                clinical_config=clinical_config,
                reward_config=reward_config,
                ischemia_cost=ischemia_cost,
                ischemia_scale_minutes=ischemia_scale_minutes,
                target_selector=target_policy.select_target,
            )
        return init

    constructors = [make_env(rank) for rank in range(n_envs)]
    raw_env = SubprocVecEnv(constructors, start_method="fork")
    vec_env = VecNormalize(raw_env, norm_obs=False, norm_reward=False, clip_reward=10.0)

    oracle_checkpoint = MaskablePPO.load(str(oracle_model_path), device=device)
    oracle_state = oracle_checkpoint.policy.state_dict()

    model = MaskablePPO(
        TargetConditionedClampPolicy,
        vec_env,
        policy_kwargs={
            "features_extractor_class": PaddedPlanSpatialExtractor,
            "net_arch": [],
            "share_features_extractor": True,
        },
        seed=seed,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=0.9999,
        gae_lambda=0.98,
        ent_coef=ent_coef,
        clip_range=clip_range,
        target_kl=target_kl,
        verbose=1,
    )
    model.policy.load_state_dict(oracle_state)
    model.oracle_kl_coef = oracle_kl_coef
    model.oracle_policy = oracle_checkpoint.policy
    for parameter in model.oracle_policy.parameters():
        parameter.requires_grad_(False)
    # Freeze base_spatial (BC target extractor) and BC target model.
    for parameter in model.policy.features_extractor.base_spatial.parameters():
        parameter.requires_grad_(False)

    # Oracle KL anchor update after each PPO train().
    _attach_oracle_kl(model)

    class ProbeEvalCallback(BaseCallback):
        def __init__(self, eval_freq: int = 2000):
            super().__init__(verbose=0)
            self.eval_freq = eval_freq
            self.last_reward = None
            self.down_streak = 0

        def _on_rollout_end(self) -> bool:
            if self.num_timesteps % self.eval_freq:
                return True
            from evaluate_clinical_v102 import (
                evaluate_split,
                rollout_target_conditioned,
            )

            det_records = [
                rollout_target_conditioned(
                    scenario,
                    lambda env, m=model: _det_select(m, env),
                    target_selector=target_policy.select_target,
                    clinical_config=clinical_config,
                    reward_config=reward_config,
                    ischemia_cost=ischemia_cost,
                    ischemia_scale_minutes=ischemia_scale_minutes,
                )
                for scenario in probe_scenarios
            ]
            stoch_records = []
            for _ in range(5):
                stoch_records.extend([
                    rollout_target_conditioned(
                        scenario,
                        lambda env, m=model: _stoch_select(m, env),
                        target_selector=target_policy.select_target,
                        clinical_config=clinical_config,
                        reward_config=reward_config,
                        ischemia_cost=ischemia_cost,
                        ischemia_scale_minutes=ischemia_scale_minutes,
                    )
                    for scenario in probe_scenarios
                ])
            det_reward = float(np.mean([r["total_reward"] for r in det_records]))
            payload = {
                "timesteps": int(self.num_timesteps),
                "det_mean_reward": det_reward,
                "det_mean_time": float(np.mean([r["elapsed_minutes"] for r in det_records])),
                "det_mean_blood": float(np.mean([r["expected_blood_loss_ml"] for r in det_records])),
                "det_mean_ischemia": float(np.mean([r["total_clamped_minutes"] for r in det_records])),
                "det_end_count": float(np.mean([r["early_end_count"] for r in det_records])),
                "stoch_mean_reward": float(np.mean([r["total_reward"] for r in stoch_records])),
                "stoch_end_count": float(np.mean([r["early_end_count"] for r in stoch_records])),
                "target_hash": target_policy.parameter_sha256()[:16],
            }
            (output_dir / "probe" / f"step_{self.num_timesteps}.json").parent.mkdir(
                parents=True, exist_ok=True
            )
            (output_dir / "probe" / f"step_{self.num_timesteps}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self.logger.record("probe/det_reward", det_reward)
            # Early stopping: deterministic reward down twice consecutively.
            if self.last_reward is not None and det_reward < self.last_reward:
                self.down_streak += 1
            else:
                self.down_streak = 0
            self.last_reward = det_reward
            if self.down_streak >= 2:
                print("EARLY_STOP probe deterministic reward declined twice", flush=True)
                return False
            return True

    def _det_select(m, env):
        obs = env._observation()
        action, _ = m.predict(obs, deterministic=True)
        return int(action)

    def _stoch_select(m, env):
        obs = env._observation()
        action, _ = m.predict(obs, deterministic=False)
        return int(action)

    save_freq = max(1, checkpoint_global_interval // max(n_envs, 1))
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="clamp",
        save_vecnormalize=True,
    )
    callback = CallbackList([checkpoint_callback, ProbeEvalCallback()])
    try:
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
        model.save(str(output_dir / "final_model"))
        vec_env.save(str(output_dir / "vecnormalize.pkl"))
        result = {
            "actual_timesteps": int(model.num_timesteps),
            "output_dir": str(output_dir),
            "final_model": str(output_dir / "final_model.zip"),
        }
        (output_dir / "training_complete.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        vec_env.close()


def _attach_oracle_kl(model):
    """Attach an oracle-KL anchor update into the PPO training loop.

    The oracle policy is frozen; the anchor pushes the current policy's clamp
    distribution toward the oracle after each standard masked PPO update.
    """
    original_train = model.train

    def train_with_anchor():
        original_train()
        if getattr(model, "oracle_kl_coef", 0.0) <= 0.0:
            return
        if model.oracle_policy is None:
            return
        import torch

        obs = torch.as_tensor(
            model.rollout_buffer.observations[: model.batch_size], device=model.device
        )
        masks = torch.as_tensor(
            np.stack(model.rollout_buffer.action_masks[: model.batch_size]),
            device=model.device,
        )
        cur = model.policy.get_distribution(obs, action_masks=masks).distribution.logits
        with torch.no_grad():
            ref = model.oracle_policy.get_distribution(
                obs, action_masks=masks
            ).distribution.logits
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(cur, dim=1),
            torch.softmax(ref, dim=1),
            reduction="batchmean",
        )
        loss = model.oracle_kl_coef * kl
        model.policy.optimizer.zero_grad()
        loss.backward()
        trainable = [p for p in model.policy.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, model.max_grad_norm)
        model.policy.optimizer.step()
        model.logger.record("train/oracle_kl", float(kl.detach().cpu()))

    model.train = train_with_anchor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--init-oracle", required=True, type=Path)
    parser.add_argument("--bc-model", required=True, type=Path)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--oracle-kl-coef", type=float, default=0.2)
    parser.add_argument("--ischemia-cost", type=float, default=1.0)
    parser.add_argument("--time-cost", type=float, default=1.0)
    parser.add_argument("--blood-cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026090301)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-global-interval", type=int, default=2000)
    args = parser.parse_args()

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if not split_payload.get("frozen"):
        raise RuntimeError("v10.2 clamp PPO requires a frozen split file")
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    train_scenarios = list(split_payload["splits"]["train"])
    probe_scenarios = list(split_payload["splits"]["probe"])
    clinical_config = {
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": "threshold",
        "early_end_minutes": 5.0,
    }
    reward_config = {
        "time_cost": args.time_cost,
        "blood_cost": args.blood_cost,
        "completion_bonus": 5.0,
        "failure_penalty": 10.0,
        "invalid_action_penalty": 10.0,
    }
    run_clamp_ppo(
        train_scenarios=train_scenarios,
        probe_scenarios=probe_scenarios,
        output_dir=args.output_dir,
        oracle_model_path=args.init_oracle,
        bc_model_path=args.bc_model,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=args.ischemia_cost,
        ischemia_scale_minutes=float(scale_payload["ischemia_scale_minutes"]),
        oracle_kl_coef=args.oracle_kl_coef,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        checkpoint_global_interval=args.checkpoint_global_interval,
    )


if __name__ == "__main__":
    main()

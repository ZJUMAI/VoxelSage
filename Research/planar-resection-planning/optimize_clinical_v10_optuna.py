"""Optuna multi-objective search for v10 hierarchical Stage 2A.

Trials minimize frozen tuning-set time, blood loss, and transfer overhead while
recording hard feasibility constraints.  This script trains models when run;
it is intended for the CUDA training agent, not for import-time execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from clinical_window_evaluation import (
    aggregate_clinical_records,
    make_ppo_selector,
    rollout_clinical_policy,
)
from train_clinical_window_ppo import ClinicalTrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--init-model", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--storage", default="sqlite:///clinical_v10_optuna.db")
    parser.add_argument("--study-name", default="clinical-v10-stage2a")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tuning-limit", type=int, default=32)
    parser.add_argument("--blood-safety-ratio", type=float, default=1.05)
    parser.add_argument("--early-end-minutes", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026081101)
    args = parser.parse_args()
    if not args.init_model.is_file():
        raise FileNotFoundError(args.init_model)
    if args.trials <= 0 or args.timesteps <= 0 or args.tuning_limit <= 0:
        parser.error("trials, timesteps, and tuning-limit must be positive")

    import optuna
    import torch
    from sb3_contrib import MaskablePPO
    import clinical_hierarchical_policy  # noqa: F401

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if "tuning" not in split_payload["splits"]:
        raise ValueError("v10 split file must contain an independent 'tuning' split")
    train_scenarios = list(split_payload["splits"]["train"])
    tuning_scenarios = list(split_payload["splits"]["tuning"])[: args.tuning_limit]
    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    baseline_blood = float(baseline["summary"]["mean_expected_blood_loss_ml"])
    blood_limit = baseline_blood * args.blood_safety_ratio
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clinical_config = {
        "time_scale_minutes": float(scales["time_scale_minutes"]),
        "blood_scale_ml": float(scales["blood_scale_ml"]),
        "weight_kg": float(scales.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": "threshold",
        "early_end_minutes": args.early_end_minutes,
        "stagnation_soft_start_steps": 40,
        "stagnation_penalty_ramp_steps": 24,
        "stagnation_limit_steps": 96,
        "two_cell_loop_soft_start_traversals": 6,
        "two_cell_loop_limit_traversals": 12,
    }

    def constraints(trial: optuna.trial.FrozenTrial):
        return trial.user_attrs.get("constraints", (1.0, 1.0, 1.0, 1.0))

    sampler = optuna.samplers.NSGAIISampler(seed=args.seed, constraints_func=constraints)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        trial_seed = args.seed + trial.number * 1009
        reward_config = {
            "time_cost": 1.0,
            "blood_cost": trial.suggest_float("blood_cost", 0.5, 8.0, log=True),
            # Macro actions structurally guarantee progress; Stage 2 removes the
            # v9 shaping terms that dominated small clinical-cost differences.
            "progress_bonus": 0.0,
            "seal_progress_bonus": 0.0,
            "stagnation_penalty_cap": 0.0,
            "two_cell_loop_penalty": 0.0,
            "clinical_cost_cap": 10.0,
            "front_tension_cost": 0.0,
            "organ_energy_cost": 0.0,
            "vessel_strain_cost": 0.0,
            "completion_bonus": 5.0,
            "failure_penalty": 10.0,
            "invalid_action_penalty": 10.0,
        }
        config = ClinicalTrainingConfig(
            seed=trial_seed,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=trial.suggest_categorical("n_epochs", [3, 5, 8]),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
            gamma=trial.suggest_float("gamma", 0.995, 0.9999, log=True),
            gae_lambda=trial.suggest_float("gae_lambda", 0.90, 0.99),
            ent_coef=trial.suggest_float("ent_coef", 1e-5, 3e-3, log=True),
            clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
            target_kl=trial.suggest_float("target_kl", 0.01, 0.08, log=True),
            device=args.device,
            init_model=str(args.init_model),
            bc_scenarios=0,
            bc_epochs=0,
            bc_v_weight=0.0,
            rl_margin_coef=0.0,
            control_mode="hierarchical",
            freeze_target_head=True,
            freeze_features_extractor=True,
        )
        run_dir = args.output_dir / "trials" / f"trial_{trial.number:04d}"
        result = run_training(
            train_scenarios=train_scenarios,
            output_dir=run_dir,
            config=config,
            clinical_config=clinical_config,
            reward_config=reward_config,
            provenance={
                "optuna_study": args.study_name,
                "optuna_trial": trial.number,
                "split_file": str(args.splits.resolve()),
            },
        )
        torch.set_num_threads(1)
        model = MaskablePPO.load(result["final_model"], device=args.device)
        selector = make_ppo_selector(model)
        records = [
            rollout_clinical_policy(
                scenario,
                selector,
                clinical_config=clinical_config,
                reward_config=reward_config,
                control_mode="hierarchical",
            )
            for scenario in tuning_scenarios
        ]
        summary = aggregate_clinical_records(records)
        constraints_value = (
            1.0 - float(summary["completion_rate"]),
            float(summary["mean_expected_blood_loss_ml"]) - blood_limit,
            1.0 - float(summary["mean_legal_action_rate"]),
            float(summary["mean_stagnation_failure"])
            + float(summary["mean_two_cell_loop_failure"]),
        )
        trial.set_user_attr("constraints", constraints_value)
        trial.set_user_attr("summary", summary)
        evaluation = {
            "trial": trial.number,
            "params": trial.params,
            "constraints": constraints_value,
            "blood_limit_ml": blood_limit,
            "summary": summary,
            "records": records,
        }
        (run_dir / "tuning_evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            float(summary["mean_elapsed_minutes"]),
            float(summary["mean_expected_blood_loss_ml"]),
            float(summary["mean_transfer_overhead"]),
        )

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    trials: list[dict[str, Any]] = []
    for trial in study.trials:
        trials.append({
            "number": trial.number,
            "state": trial.state.name,
            "values": trial.values,
            "params": trial.params,
            "constraints": trial.user_attrs.get("constraints"),
            "summary": trial.user_attrs.get("summary"),
        })
    pareto = [trial.number for trial in study.best_trials]
    (args.output_dir / "optuna_summary.json").write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "storage": args.storage,
                "baseline_blood_ml": baseline_blood,
                "blood_limit_ml": blood_limit,
                "pareto_trial_numbers": pareto,
                "trials": trials,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

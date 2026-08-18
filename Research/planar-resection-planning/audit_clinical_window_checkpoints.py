"""Audit PPO checkpoints on a small frozen Validation subset before retraining."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from clinical_window_evaluation import (
    aggregate_clinical_records,
    make_ppo_selector,
    rollout_clinical_policy,
)
from clinical_window_environment import CLINICAL_ENVIRONMENT_VERSION
from clinical_macro_environment import CLINICAL_MACRO_ENVIRONMENT_VERSION
from clinical_hierarchical_environment import CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION


STEP_PATTERN = re.compile(r"_(\d+)_steps\.zip$")


def _checkpoint_key(path: Path) -> tuple[int, str]:
    match = STEP_PATTERN.search(path.name)
    return (int(match.group(1)) if match else 10**18, path.name)


def discover_models(run_dir: Path, *, include_final: bool) -> list[Path]:
    models = sorted((run_dir / "checkpoints").glob("*_steps.zip"), key=_checkpoint_key)
    final_model = run_dir / "final_model.zip"
    if include_final and final_model.is_file():
        models.append(final_model)
    if not models:
        raise FileNotFoundError(f"No checkpoint models found under {run_dir}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--progress-bonus", type=float, default=5.0)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--blood-safety-ratio", type=float, default=1.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-steps-multiplier", type=float, default=8.0)
    parser.add_argument("--include-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint audit: {args.output_dir}")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.blood_safety_ratio < 1.0 or args.bootstrap_samples <= 0:
        parser.error("blood-safety-ratio must be >=1 and bootstrap-samples positive")

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    metadata_path = args.run_dir / "run_metadata.json"
    run_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file() else {}
    )
    stored_version = run_metadata.get("environment_version")
    control_mode = str(run_metadata.get("control_mode", "direction"))
    expected_version = {
        "direction": CLINICAL_ENVIRONMENT_VERSION,
        "macro": CLINICAL_MACRO_ENVIRONMENT_VERSION,
        "hierarchical": CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION,
    }.get(control_mode)
    if expected_version is None:
        raise ValueError(f"Unsupported control_mode in run metadata: {control_mode!r}")
    if stored_version is not None and stored_version != expected_version:
        raise ValueError(
            f"Run uses {stored_version!r}, but this auditor requires "
            f"{expected_version!r}; use the archived evaluation JSON for old runs"
        )
    scenarios = list(split_payload["splits"]["validation"])[:args.limit]
    baseline_records = None
    baseline_blood = None
    if args.baseline_evaluation is not None:
        baseline_payload = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
        baseline_records = {
            record["scenario_id"]: record for record in baseline_payload["records"]
        }
        baseline_blood = float(
            baseline_payload["summary"]["mean_expected_blood_loss_ml"]
        )
    clinical_config = {
        **dict(run_metadata.get("clinical_config") or {}),
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "max_steps_multiplier": args.max_steps_multiplier,
    }
    reward_config = dict(run_metadata.get("reward_config") or {})
    if not reward_config:
        reward_config = {"progress_bonus": args.progress_bonus}

    import torch
    from sb3_contrib import MaskablePPO
    if control_mode == "hierarchical":
        import clinical_hierarchical_policy  # noqa: F401
    elif control_mode == "macro":
        import clinical_macro_policy  # noqa: F401

    torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True)
    audits: list[dict[str, Any]] = []
    for model_path in discover_models(args.run_dir, include_final=args.include_final):
        model = MaskablePPO.load(str(model_path), device="auto")
        selector = make_ppo_selector(model)
        records = [
            rollout_clinical_policy(
                scenario,
                selector,
                clinical_config=clinical_config,
                reward_config=reward_config,
                control_mode=control_mode,
            )
            for scenario in scenarios
        ]
        summary = aggregate_clinical_records(records)
        paired = None
        safety_feasible = True
        if baseline_records is not None and baseline_blood is not None:
            differences = np.asarray([
                float(record["expected_blood_loss_ml"])
                - float(baseline_records[record["scenario_id"]]["expected_blood_loss_ml"])
                for record in records
            ])
            rng = np.random.default_rng(2026081201)
            indices = rng.integers(
                0, len(differences), size=(args.bootstrap_samples, len(differences))
            )
            bootstrap = differences[indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            allowed_increase = baseline_blood * (args.blood_safety_ratio - 1.0)
            paired = {
                "mean_blood_difference_ml": float(differences.mean()),
                "median_blood_difference_ml": float(np.median(differences)),
                "improved_scenarios": int((differences < 0).sum()),
                "tied_scenarios": int((differences == 0).sum()),
                "bootstrap_95_ci_ml": [float(lower), float(upper)],
                "allowed_mean_increase_ml": allowed_increase,
            }
            safety_feasible = bool(upper <= allowed_increase)
        hard_feasible = bool(
            float(summary["completion_rate"]) == 1.0
            and float(summary["mean_legal_action_rate"]) == 1.0
            and float(summary["mean_stagnation_failure"]) == 0.0
            and float(summary["mean_two_cell_loop_failure"]) == 0.0
        )
        feasible = hard_feasible and safety_feasible
        payload = {
            "model": str(model_path.resolve()),
            "scenario_ids": [item["scenario_id"] for item in scenarios],
            "environment_version": expected_version,
            "control_mode": control_mode,
            "reward_config": reward_config,
            "summary": summary,
            "paired_baseline": paired,
            "hard_feasible": hard_feasible,
            "blood_safety_feasible": safety_feasible,
            "feasible": feasible,
            "records": records,
        }
        destination = args.output_dir / f"{model_path.stem}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audits.append({
            "model": str(model_path),
            "output": str(destination),
            "feasible": feasible,
            "paired_baseline": paired,
            **summary,
        })
        print(json.dumps(audits[-1], ensure_ascii=False), flush=True)

    ranked = sorted(
        audits,
        key=lambda item: (
            float(item["feasible"]),
            float(item["completion_rate"]),
            float(item["mean_coverage"]),
            -float(item["mean_elapsed_minutes"]),
            -float(item["mean_expected_blood_loss_ml"]),
            -float(item["mean_transfer_overhead"]),
        ),
        reverse=True,
    )
    manifest = {
        "run_dir": str(args.run_dir.resolve()),
        "split_file": str(args.splits.resolve()),
        "scale_file": str(args.scales.resolve()),
        "validation_limit": args.limit,
        "environment_version": expected_version,
        "control_mode": control_mode,
        "reward_config": reward_config,
        "ranking_rule": (
            "feasible safety gate, completion, coverage, lower time, lower blood, "
            "lower transfer overhead"
        ),
        "baseline_evaluation": (
            str(args.baseline_evaluation.resolve())
            if args.baseline_evaluation is not None else None
        ),
        "blood_safety_ratio": args.blood_safety_ratio,
        "best_model": (
            next((item["model"] for item in ranked if item["feasible"]), None)
            if args.baseline_evaluation is not None else ranked[0]["model"]
        ),
        "decision": (
            "GO" if any(item["feasible"] for item in ranked) else "NO-GO"
        ),
        "ranked": ranked,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

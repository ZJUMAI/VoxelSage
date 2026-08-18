"""v10.1 final model one-shot evaluation on Test-64 and Stress-64.

Uses the frozen final model (stage2c/trial19/seed_2026082204, threshold 5min)
and evaluates once on each held-out split.  Reports absolute metrics plus
paired blood difference vs the serpentine baseline of the same split with
bootstrap 95% CI.  No tuning is performed on Test/Stress.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinical_window_evaluation import (
    aggregate_clinical_records,
    make_ppo_selector,
    rollout_clinical_policy,
)
from clinical_hierarchical_environment import CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "stress"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--blood-safety-ratio", type=float, default=1.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    import torch
    from sb3_contrib import MaskablePPO
    import clinical_hierarchical_policy  # noqa: F401

    torch.set_num_threads(args.torch_threads)

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    scenarios = list(split_payload["splits"][args.split])[:args.limit]

    baseline = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    baseline_records = {r["scenario_id"]: r for r in baseline["records"]}
    baseline_blood = float(baseline["summary"]["mean_expected_blood_loss_ml"])
    allowed_increase = baseline_blood * (args.blood_safety_ratio - 1.0)

    metadata = json.loads((args.model.parent / "run_metadata.json").read_text(encoding="utf-8"))
    run_cc = dict(metadata.get("clinical_config") or {})
    run_cc.update({
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
    })
    reward_config = dict(metadata.get("reward_config") or {})

    model = MaskablePPO.load(str(args.model), device="auto")
    selector = make_ppo_selector(model)
    records = [
        rollout_clinical_policy(
            scenario,
            selector,
            clinical_config=run_cc,
            reward_config=reward_config,
            control_mode="hierarchical",
        )
        for scenario in scenarios
    ]
    summary = aggregate_clinical_records(records)
    differences = np.asarray([
        float(r["expected_blood_loss_ml"])
        - float(baseline_records[r["scenario_id"]]["expected_blood_loss_ml"])
        for r in records
    ])
    rng = np.random.default_rng(2026083001)
    indices = rng.integers(0, len(differences), size=(args.bootstrap_samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    feasible = bool(
        float(summary["completion_rate"]) == 1.0
        and float(summary["mean_legal_action_rate"]) == 1.0
        and float(summary["mean_stagnation_failure"]) == 0.0
        and float(summary["mean_two_cell_loop_failure"]) == 0.0
        and upper <= allowed_increase
    )
    result = {
        "split": args.split,
        "model": str(args.model),
        "summary": summary,
        "baseline_blood_ml": baseline_blood,
        "allowed_mean_increase_ml": allowed_increase,
        "paired": {
            "mean_blood_difference_ml": float(differences.mean()),
            "median_blood_difference_ml": float(np.median(differences)),
            "bootstrap_95_ci_ml": [float(lower), float(upper)],
        },
        "feasible": feasible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "split": args.split,
        "time_min": float(summary["mean_elapsed_minutes"]),
        "blood_ml": float(summary["mean_expected_blood_loss_ml"]),
        "baseline_blood_ml": baseline_blood,
        "blood_diff": float(differences.mean()),
        "ci": [float(lower), float(upper)],
        "feasible": feasible,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

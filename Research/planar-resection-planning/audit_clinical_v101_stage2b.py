"""v10.1 Stage 2B: validate Pareto candidates on frozen Validation-64.

For each candidate (trial5 / trial17 / trial18) with 3 seeds, evaluate the
trained model on Validation-64 using the run's own clinical/reward config
(early_end_mode=threshold).  Aggregate per-candidate statistics:
  - mean and worst-seed blood/time/transfer across the 3 seeds
  - paired blood difference vs baseline with bootstrap 95% CI
Safety gate: upper CI <= baseline blood * 1.05.
Select the feasible candidate with the lowest mean surgery time.
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
from clinical_hierarchical_environment import CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION


CANDIDATES = {
    "trial5": ["2026081701", "2026081704", "2026081707"],
    "trial17": ["2026081702", "2026081705", "2026081708"],
    "trial18": ["2026081703", "2026081706", "2026081709"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--stage2b-dir", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--blood-safety-ratio", type=float, default=1.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    import torch
    from sb3_contrib import MaskablePPO
    import clinical_hierarchical_policy  # noqa: F401

    torch.set_num_threads(args.torch_threads)

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    scenarios = list(split_payload["splits"]["validation"])[:args.limit]

    baseline = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
    baseline_records = {
        r["scenario_id"]: r for r in baseline["records"]
    }
    baseline_blood = float(baseline["summary"]["mean_expected_blood_loss_ml"])
    allowed_increase = baseline_blood * (args.blood_safety_ratio - 1.0)

    candidate_results: dict[str, Any] = {}
    for cand, seeds in CANDIDATES.items():
        per_seed: list[dict[str, Any]] = []
        for seed in seeds:
            run_dir = args.stage2b_dir / cand / f"seed_{seed}"
            model_path = run_dir / "final_model.zip"
            if not model_path.is_file():
                print(f"SKIP missing model: {model_path}", flush=True)
                continue
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            run_cc = dict(metadata.get("clinical_config") or {})
            run_cc.update({
                "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
                "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
                "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
                "bleeding_probability": 1.0,
            })
            reward_config = dict(metadata.get("reward_config") or {})

            model = MaskablePPO.load(str(model_path), device="auto")
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
            rng = np.random.default_rng(2026082001)
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
            entry = {
                "seed": seed,
                "model": str(model_path),
                "summary": summary,
                "paired": {
                    "mean_blood_difference_ml": float(differences.mean()),
                    "median_blood_difference_ml": float(np.median(differences)),
                    "bootstrap_95_ci_ml": [float(lower), float(upper)],
                    "allowed_mean_increase_ml": allowed_increase,
                },
                "feasible": feasible,
                "time_min": float(summary["mean_elapsed_minutes"]),
                "blood_ml": float(summary["mean_expected_blood_loss_ml"]),
                "transfer_overhead": float(summary["mean_transfer_overhead"]),
            }
            per_seed.append(entry)
            print(json.dumps({"candidate": cand, **entry}, ensure_ascii=False), flush=True)

        if per_seed:
            times = np.asarray([e["time_min"] for e in per_seed])
            bloods = np.asarray([e["blood_ml"] for e in per_seed])
            all_feasible = all(e["feasible"] for e in per_seed)
            candidate_results[cand] = {
                "n_seeds": len(per_seed),
                "seeds": per_seed,
                "mean_time_min": float(times.mean()),
                "worst_seed_time_min": float(times.max()),
                "mean_blood_ml": float(bloods.mean()),
                "worst_seed_blood_ml": float(bloods.max()),
                "all_feasible": all_feasible,
                "decision": "feasible" if all_feasible else "infeasible",
            }

    # Selection: among candidates where ALL seeds feasible, pick lowest mean time.
    feasible_cands = {c: r for c, r in candidate_results.items() if r["all_feasible"]}
    best = None
    if feasible_cands:
        best = min(feasible_cands, key=lambda c: feasible_cands[c]["mean_time_min"])

    summary = {
        "stage": "v10.1-stage2b",
        "validation_split": "validation",
        "validation_limit": args.limit,
        "baseline_evaluation": str(args.baseline_evaluation.resolve()),
        "baseline_blood_ml": baseline_blood,
        "blood_safety_ratio": args.blood_safety_ratio,
        "allowed_mean_increase_ml": allowed_increase,
        "candidates": candidate_results,
        "selection_rule": (
            "among candidates with all 3 seeds feasible, pick lowest mean surgery time"
        ),
        "best_candidate": best,
        "best_mean_time_min": feasible_cands[best]["mean_time_min"] if best else None,
        "decision": "GO" if best else "NO-GO",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": summary["decision"], "best_candidate": best,
                      "best_mean_time_min": summary["best_mean_time_min"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

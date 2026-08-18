"""Paired comparison of PPO and serpentine clinical-window evaluations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


METRICS = (
    "elapsed_minutes",
    "expected_blood_loss_ml",
    "peak_expected_bleeding_rate_ml_per_min",
    "unclamped_exposed_minutes",
    "transfer_count",
)


def _index(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        scenario_id = str(record["scenario_id"])
        if scenario_id in indexed:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        indexed[scenario_id] = record
    return indexed


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int, draws: int) -> list[float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    rng = random.Random(seed)
    estimates = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    )
    low = estimates[int(0.025 * (draws - 1))]
    high = estimates[int(0.975 * (draws - 1))]
    return [float(low), float(high)]


def compare(
    ppo: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    bootstrap_draws: int = 10_000,
    seed: int = 2026080405,
) -> dict[str, Any]:
    ppo_records = _index(ppo["records"])
    baseline_records = _index(baseline["records"])
    if set(ppo_records) != set(baseline_records):
        only_ppo = sorted(set(ppo_records) - set(baseline_records))
        only_baseline = sorted(set(baseline_records) - set(ppo_records))
        raise ValueError(
            f"Evaluation scenario sets differ; only PPO={only_ppo[:5]}, "
            f"only baseline={only_baseline[:5]}"
        )
    scenario_ids = sorted(ppo_records)
    paired: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        differences = [
            float(ppo_records[item][metric]) - float(baseline_records[item][metric])
            for item in scenario_ids
        ]
        baseline_values = [float(baseline_records[item][metric]) for item in scenario_ids]
        ppo_values = [float(ppo_records[item][metric]) for item in scenario_ids]
        paired[metric] = {
            "ppo_mean": mean(ppo_values),
            "baseline_mean": mean(baseline_values),
            "mean_paired_difference_ppo_minus_baseline": mean(differences),
            "median_paired_difference_ppo_minus_baseline": median(differences),
            "mean_difference_bootstrap_95_ci": _bootstrap_mean_ci(
                differences,
                seed=seed + metric_index,
                draws=bootstrap_draws,
            ),
            "ppo_better_count": sum(value < 0 for value in differences),
            "tie_count": sum(value == 0 for value in differences),
            "ppo_worse_count": sum(value > 0 for value in differences),
        }
    ppo_time = paired["elapsed_minutes"]["ppo_mean"]
    baseline_time = paired["elapsed_minutes"]["baseline_mean"]
    return {
        "split": ppo.get("split"),
        "scenario_count": len(scenario_ids),
        "ppo_completion_rate": mean(float(ppo_records[item]["completion"]) for item in scenario_ids),
        "baseline_completion_rate": mean(
            float(baseline_records[item]["completion"]) for item in scenario_ids
        ),
        "ppo_legal_action_rate": mean(
            float(ppo_records[item]["legal_action_rate"]) for item in scenario_ids
        ),
        "time_ratio_ppo_over_baseline": ppo_time / baseline_time if baseline_time else None,
        "passes_105_percent_time_gate": bool(baseline_time and ppo_time <= 1.05 * baseline_time),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "paired_metrics": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026080405)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison output: {args.output}")
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    ppo = json.loads(args.ppo.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    result = compare(
        ppo,
        baseline,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Frozen evaluation for the 4 mm variable-size PPO curriculum.

Run PPO and rule baselines as separate processes.  This deliberately keeps
PyTorch inference out of the mechanics workers and avoids native-library
contention observed in one-process replay.

Every record and the aggregate summary include the external risk metrics
from the handoff contract (``累计血管应变 / CVaR / 阈值以上步数比例 /
最大风险峰值`` and their front-tension / organ-energy counterparts):

- ``cumulative_vessel_strain``: sum over steps of per-step peak vessel strain
- ``worst_10pct_vessel_strain``: CVaR (mean of the worst 10% of steps)
- ``fraction_steps_above_safe`` / ``fraction_steps_above_tear``: share of
  steps whose peak strain exceeds the mechanics ``safe``/``tear`` thresholds
- ``max_vessel_strain`` / ``max_risk_peak``: per-step worst and the max
  across front tension, organ energy and vessel strain

These are external evaluation metrics only; they are not part of the reward.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from environment import VariableGridScenarioPoolEnv
from evaluation import evaluate_policy, serpentine_priority_policy
from mechanics import DEFAULT_MECHANICS
from variable_scenarios import CURRICULUM_RANGES, generate_stage_pool
from variable_teacher import _planner_selector

SAFE_VESSEL_STRAIN = float(DEFAULT_MECHANICS["safe_vessel_strain"])
TEAR_VESSEL_STRAIN = float(DEFAULT_MECHANICS["tear_vessel_strain"])

# Canonical per-episode metric fields shared by PPO and rule baselines.
_METRIC_FIELDS = (
    "mean_vessel_strain", "cumulative_vessel_strain", "worst_10pct_vessel_strain",
    "fraction_steps_above_safe", "fraction_steps_above_tear", "max_vessel_strain",
    "mean_front_tension", "worst_10pct_front_tension",
    "mean_organ_energy", "worst_10pct_organ_energy", "max_risk_peak",
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _worst_tenth(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    count = max(1, ceil(len(values) * 0.10))
    return mean(sorted(values, reverse=True)[:count])


def _risk_metrics(
    vessel_strains: Sequence[float],
    front_tensions: Sequence[float],
    organ_energies: Sequence[float],
) -> dict[str, float]:
    """Compute the external risk metrics from per-step peak-value series."""
    return {
        "mean_vessel_strain": mean(vessel_strains) if vessel_strains else 0.0,
        "cumulative_vessel_strain": sum(vessel_strains),
        "worst_10pct_vessel_strain": _worst_tenth(vessel_strains),
        "fraction_steps_above_safe": (
            mean(float(v > SAFE_VESSEL_STRAIN) for v in vessel_strains) if vessel_strains else 0.0
        ),
        "fraction_steps_above_tear": (
            mean(float(v > TEAR_VESSEL_STRAIN) for v in vessel_strains) if vessel_strains else 0.0
        ),
        "max_vessel_strain": max(vessel_strains, default=0.0),
        "mean_front_tension": mean(front_tensions) if front_tensions else 0.0,
        "worst_10pct_front_tension": _worst_tenth(front_tensions),
        "mean_organ_energy": mean(organ_energies) if organ_energies else 0.0,
        "worst_10pct_organ_energy": _worst_tenth(organ_energies),
        "max_risk_peak": max(front_tensions + organ_energies + vessel_strains, default=0.0),
    }


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "count": float(len(records)),
        "completion_rate": mean(float(item["completion"]) for item in records),
        "legal_action_rate": mean(float(item["legal_action_rate"]) for item in records),
        "transfer_overhead": mean(float(item["transfer_overhead"]) for item in records),
        **{field: mean(float(item[field]) for item in records) for field in _METRIC_FIELDS},
    }


def _record_from_evaluation(scenario_id: str, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Build a canonical record from :func:`evaluation.evaluate_policy` output."""
    return {
        "scenario_id": scenario_id,
        "completion": evaluation["completion"],
        "legal_action_rate": evaluation["legal_action_rate"],
        "transfer_overhead": evaluation["transfer_overhead"],
        "failure_reason": evaluation["failure_reason"],
        **{field: evaluation[field] for field in _METRIC_FIELDS},
    }


def _evaluate_baseline_one(scenario: Mapping[str, Any], method: str) -> dict[str, Any]:
    if method == "serpentine":
        selector: Callable[[Mapping[str, Any]], Callable] = lambda _: serpentine_priority_policy
    elif method == "planner":
        selector = _planner_selector
    else:
        raise ValueError(f"Unknown baseline: {method}")
    return _record_from_evaluation(
        scenario["scenario_id"], evaluate_policy(scenario, selector(scenario))
    )


def evaluate_baseline(scenarios: list[dict[str, Any]], method: str, workers: int = 1) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_evaluate_baseline_one(scenario, method) for scenario in scenarios]
    # executor.map preserves the input order, so records stay paired with
    # their scenario_id exactly as in the sequential path.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as executor:
        return list(executor.map(_evaluate_baseline_one, scenarios, [method] * len(scenarios)))


def evaluate_ppo(scenarios: list[dict[str, Any]], model_path: Path, workers: int) -> list[dict[str, Any]]:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import variable_policy  # Register the custom policy class before loading.

    def factory(scenario: dict[str, Any], rank: int):
        def init():
            return VariableGridScenarioPoolEnv([scenario], seed=rank)
        return init

    model = MaskablePPO.load(model_path, device="cpu")
    records: list[dict[str, Any]] = []
    for offset in range(0, len(scenarios), workers):
        batch = scenarios[offset:offset + workers]
        environment = SubprocVecEnv([factory(item, index) for index, item in enumerate(batch)], start_method="fork")
        try:
            observation = environment.reset()
            done = np.zeros(len(batch), dtype=bool)
            terminal: list[dict[str, Any] | None] = [None] * len(batch)
            vessel_strains: list[list[float]] = [[] for _ in batch]
            front_tensions: list[list[float]] = [[] for _ in batch]
            organ_energies: list[list[float]] = [[] for _ in batch]
            while not done.all():
                masks = np.stack(environment.env_method("action_masks"))
                actions, _ = model.predict(observation, deterministic=True, action_masks=masks)
                observation, _, ended, infos = environment.step(actions)
                for index, info in enumerate(infos):
                    if not done[index]:
                        vessel_strains[index].append(float(info["peak_vessel_strain"]))
                        front_tensions[index].append(float(info["peak_front_tension"]))
                        organ_energies[index].append(float(info["peak_organ_energy"]))
                for index in np.flatnonzero(ended & ~done):
                    terminal[index] = dict(infos[index])
                    done[index] = True
            for index, (scenario, info) in enumerate(zip(batch, terminal)):
                assert info is not None
                records.append({
                    "scenario_id": scenario["scenario_id"],
                    "completion": info["coverage"] == 1.0 and info["failure_reason"] is None,
                    "legal_action_rate": 1.0,
                    "transfer_overhead": info["transfer_count"] / info["cut_count"],
                    "failure_reason": info["failure_reason"],
                    **_risk_metrics(
                        vessel_strains[index], front_tensions[index], organ_energies[index],
                    ),
                })
        finally:
            environment.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ppo", "serpentine", "planner"), required=True)
    parser.add_argument("--stage", choices=tuple(CURRICULUM_RANGES), default="a")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.method == "ppo" and args.model_path is None:
        parser.error("--model-path is required for PPO evaluation")
    scenarios = generate_stage_pool(stage=args.stage, count=args.count, seed=args.seed, split="frozen")
    records = (
        evaluate_ppo(scenarios, args.model_path, args.workers)
        if args.method == "ppo" else evaluate_baseline(scenarios, args.method, args.workers)
    )
    _write(args.output, {"method": args.method, "stage": args.stage, "seed": args.seed, "summary": _summary(records), "records": records})
    print(json.dumps(_summary(records), ensure_ascii=False))


if __name__ == "__main__":
    main()

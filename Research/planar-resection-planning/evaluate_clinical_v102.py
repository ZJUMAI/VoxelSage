"""v10.2 frozen evaluation: fixed Probe / Validation / Test / Stress.

The deterministic and stochastic evaluations are always written to separate
files so the exploration-deployment gap is explicit.  Final evaluations save
per-scene records plus paired time/blood/ischemia bootstrap 95% CI versus the
mechanical 15/5 baseline under the same frozen BC target policy.

Selection paths reject ``test`` and ``stress``; those splits may only be read
once after the architecture, weights, seeds, and checkpoint rules are fully
frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from clinical_target_conditioned_environment import (
    CLAMP_CONTINUE,
    CLAMP_RELEASE,
    TargetConditionedClampEnv,
)
from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy

# Selection paths may only read these splits for their pre-authorized use.
SELECTION_SPLITS = ("oracle_dev", "probe", "tuning", "validation")
# Test/Stress may only be read once by an explicit one-shot final evaluation.
FINAL_SPLITS = ("test", "stress")


def guard_split_access(split: str, *, final_confirmed: bool = False) -> None:
    """Selection/probe/API paths must never read Test/Stress (guide 8).

    ``final_confirmed`` is the explicit one-shot confirmation that the
    architecture, weights, seeds and checkpoint rules are fully frozen.
    """
    if split in FINAL_SPLITS and not final_confirmed:
        raise PermissionError(
            f"split {split!r} is one-shot final-only; pass final_confirmed=True "
            "only after the architecture is fully frozen"
        )
    if split in FINAL_SPLITS and final_confirmed:
        return
    if split not in SELECTION_SPLITS:
        raise ValueError(f"unknown split {split!r}")


def make_target_conditioned_ppo_selector(model):
    """Deterministic clamp selector for a trained MaskablePPO.

    The release mask is passed to ``predict`` so the model can never select an
    illegal release; the rollout still records any such attempt as unsafe
    (guide 8 / reviewer fix #4).
    """

    def select(env: TargetConditionedClampEnv) -> int:
        observation = env._observation()
        action, _ = model.predict(
            observation, deterministic=True, action_masks=env.action_masks()
        )
        return int(action)

    return select


def make_target_conditioned_stochastic_selector(model):
    """Stochastic clamp selector (fixed-seed sampling) with action masking."""

    def select(env: TargetConditionedClampEnv) -> int:
        observation = env._observation()
        action, _ = model.predict(
            observation, deterministic=False, action_masks=env.action_masks()
        )
        return int(action)

    return select


def baseline_selector(env: TargetConditionedClampEnv) -> int:
    """Mechanical 15/5: always continue; env enforces the clamp schedule."""
    return CLAMP_CONTINUE


def rollout_target_conditioned(
    scenario: Mapping[str, Any],
    selector: Callable[[TargetConditionedClampEnv], int],
    *,
    target_selector: Callable[[TargetConditionedClampEnv], int],
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    n_stochastic: int = 1,
) -> dict[str, Any]:
    """Run a target-conditioned clamp episode, returning a rich record."""
    env = TargetConditionedClampEnv(
        scenario=scenario,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=ischemia_cost,
        ischemia_scale_minutes=ischemia_scale_minutes,
        target_selector=target_selector,
    )
    env.reset()
    rewards: list[float] = []
    reward_terms: dict[str, float] = {}
    release_probs: list[float] = []
    steps = 0
    proposed = 0
    illegal = 0
    unsafe_release_count = 0
    while not env.terminated and not env.truncated:
        mask = env.action_masks()
        action = selector(env)
        proposed += 1
        if not mask[action]:
            illegal += 1
            if action == CLAMP_RELEASE:
                unsafe_release_count += 1
        _, reward, _, _, info = env.step(action)
        rewards.append(float(reward))
        for name, value in info.get("reward_terms", {}).items():
            reward_terms[name] = reward_terms.get(name, 0.0) + float(value)
        release_probs.append(float(action == 1))
        steps += 1
    record = {
        "scenario_id": scenario.get("scenario_id"),
        "completion": env.terminated and env.failure_reason is None,
        "coverage": len(env.cut) / len(env.domain),
        "legal_action_rate": (proposed - illegal) / proposed if proposed else 1.0,
        "elapsed_minutes": env.elapsed_minutes,
        "expected_blood_loss_ml": env.expected_blood_loss_ml,
        "total_clamped_minutes": env.total_clamped_minutes,
        "total_unclamped_minutes": env.total_unclamped_minutes,
        "unclamped_exposed_minutes": env.unclamped_exposed_minutes,
        "clamp_cycle_count": env.clamp_cycle_count,
        "early_end_count": env.early_end_count,
        "transfer_overhead": env.transfer_count / max(1, env.direction_action_count),
        "stagnation_failure": str(env.failure_reason or "").startswith("stagnation:"),
        "two_cell_loop_failure": str(env.failure_reason or "").startswith("two-cell oscillation:"),
        "mean_release_probability": float(np.mean(release_probs)) if release_probs else 0.0,
        "total_reward": float(sum(rewards)),
        "reward_components": dict(sorted(reward_terms.items())),
        "unsafe_release_count": unsafe_release_count,
    }
    return record


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty record list")
    from statistics import mean, median

    fields = (
        "completion",
        "coverage",
        "elapsed_minutes",
        "expected_blood_loss_ml",
        "total_clamped_minutes",
        "total_unclamped_minutes",
        "unclamped_exposed_minutes",
        "clamp_cycle_count",
        "early_end_count",
        "transfer_overhead",
        "stagnation_failure",
        "two_cell_loop_failure",
        "mean_release_probability",
        "total_reward",
    )
    summary = {"episode_count": len(records), "completion_rate": mean(float(r["completion"]) for r in records)}
    for field in fields:
        values = [float(record[field]) for record in records]
        summary[f"mean_{field}"] = mean(values)
        summary[f"median_{field}"] = median(values)
    return summary


def _paired_bootstrap(
    records: Sequence[Mapping[str, Any]],
    baseline_records: Mapping[str, Any],
    field: str,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 2026090101,
) -> dict[str, float]:
    differences = np.asarray([
        float(record[field]) - float(baseline_records[record["scenario_id"]][field])
        for record in records
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": [float(lower), float(upper)],
    }


def evaluate_probe_separated(*, det_path: Path, stoch_path: Path, **kwargs: Any) -> None:
    """Write deterministic and stochastic probe evaluations to separate files."""
    det = evaluate_split(**kwargs, n_stochastic=1)
    stoch = evaluate_split(**kwargs, n_stochastic=5)
    det_path.write_text(json.dumps(det, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stoch_path.write_text(json.dumps(stoch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_target_selector(target_selector):
    """Accept either a callable(env)->int or an object with .select_target."""
    if target_selector is None:
        from clinical_target_conditioned_environment import serpentine_target_cell

        return serpentine_target_cell
    if hasattr(target_selector, "select_target"):
        return target_selector.select_target
    return target_selector


def evaluate_split(
    scenarios: Sequence[Mapping[str, Any]],
    model,
    *,
    target_selector=None,
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    n_stochastic: int = 1,
    baseline_records: Mapping[str, Any] | None = None,
    bootstrap_samples: int = 10_000,
    split: str | None = None,
    final_confirmed: bool = False,
) -> dict[str, Any]:
    """Evaluate a clamp model on a split; return per-scene records + paired.

    ``split`` (when given) is guarded at the API layer so selection/Optuna/
    early-stop callers can never read Test/Stress without the one-shot
    ``final_confirmed`` flag.  ``target_selector`` may be a callable(env)->int
    or a FrozenBCMacroTargetPolicy (``.select_target`` is used).
    """
    if split is not None:
        guard_split_access(split, final_confirmed=final_confirmed)
    selector_fn = _resolve_target_selector(target_selector)
    det_selector = make_target_conditioned_ppo_selector(model)
    stoch_selector = make_target_conditioned_stochastic_selector(model)

    det_records = [
        rollout_target_conditioned(
            scenario,
            det_selector,
            target_selector=selector_fn,
            clinical_config=clinical_config,
            reward_config=reward_config,
            ischemia_cost=ischemia_cost,
            ischemia_scale_minutes=ischemia_scale_minutes,
        )
        for scenario in scenarios
    ]
    stoch_records = []
    for _ in range(n_stochastic):
        stoch_records.extend([
            rollout_target_conditioned(
                scenario,
                stoch_selector,
                target_selector=selector_fn,
                clinical_config=clinical_config,
                reward_config=reward_config,
                ischemia_cost=ischemia_cost,
                ischemia_scale_minutes=ischemia_scale_minutes,
            )
            for scenario in scenarios
        ])

    result = {
        "det_summary": _aggregate(det_records),
        "stoch_summary": _aggregate(stoch_records),
        "det_records": det_records,
        "stoch_records": stoch_records,
    }
    if baseline_records is not None:
        result["paired"] = {
            field: _paired_bootstrap(
                det_records, baseline_records, field,
                bootstrap_samples=bootstrap_samples,
            )
            for field in ("elapsed_minutes", "expected_blood_loss_ml", "total_clamped_minutes")
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bc-model", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--split", choices=("probe", "validation", "test", "stress"), required=True)
    parser.add_argument(
        "--final-evaluation-confirmed",
        action="store_true",
        help="one-shot confirmation required to read test/stress (architecture fully frozen)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-stochastic", type=int, default=5)
    parser.add_argument("--ischemia-cost", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO

    from clinical_target_conditioned_policy import TargetConditionedClampPolicy  # noqa: F401

    # Selection paths must never read Test/Stress (guide 8): only the one-shot
    # final CLI with an explicit --final-evaluation-confirmed flag may.
    guard_split_access(args.split, final_confirmed=args.final_evaluation_confirmed)

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if not split_payload.get("frozen"):
        raise RuntimeError("v10.2 evaluation requires a frozen split file")
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    scenarios = list(split_payload["splits"][args.split])
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    clinical_config = {
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": "threshold",
        "early_end_minutes": 5.0,
    }
    reward_config = {
        "time_cost": 1.0,
        "blood_cost": 1.0,
        "completion_bonus": 5.0,
        "failure_penalty": 10.0,
        "invalid_action_penalty": 10.0,
    }

    model = MaskablePPO.load(str(args.model), device=args.device)
    target_policy = FrozenBCMacroTargetPolicy(args.bc_model, device=args.device)

    baseline_records = None
    if args.baseline_evaluation is not None:
        baseline = json.loads(args.baseline_evaluation.read_text(encoding="utf-8"))
        baseline_records = {
            record["scenario_id"]: record for record in baseline["det_records"]
        }

    result = evaluate_split(
        scenarios,
        model,
        target_selector=target_policy,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=args.ischemia_cost,
        ischemia_scale_minutes=float(scale_payload["ischemia_scale_minutes"]),
        n_stochastic=args.n_stochastic,
        baseline_records=baseline_records,
        bootstrap_samples=args.bootstrap_samples,
        split=args.split,
        final_confirmed=args.final_evaluation_confirmed,
    )
    result["split"] = args.split
    result["model"] = str(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "det": {
            "time_min": result["det_summary"]["mean_elapsed_minutes"],
            "blood_ml": result["det_summary"]["mean_expected_blood_loss_ml"],
            "clamp_min": result["det_summary"]["mean_total_clamped_minutes"],
            "end_count": result["det_summary"]["mean_early_end_count"],
            "release_prob": result["det_summary"]["mean_release_probability"],
        },
        "stoch": {
            "time_min": result["stoch_summary"]["mean_elapsed_minutes"],
            "blood_ml": result["stoch_summary"]["mean_expected_blood_loss_ml"],
        },
        "paired": result.get("paired"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

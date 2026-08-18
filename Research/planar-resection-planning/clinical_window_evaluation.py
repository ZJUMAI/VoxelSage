"""Baselines, calibration, and frozen evaluation for the clinical-window environment."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from clinical_window_environment import (
    ACTION_DELTAS,
    ACTION_END_CLAMP_EARLY,
    CLINICAL_ENVIRONMENT_VERSION,
    ClinicalWindowResectionEnv,
)
from planner import neighbors4


ClinicalSelector = Callable[[ClinicalWindowResectionEnv], Any]


def _scan_rank(env: ClinicalWindowResectionEnv, cell: tuple[int, int]) -> tuple[int, int, int]:
    row, col = cell
    scan_col = col if row % 2 == 0 else env.cols - 1 - col
    return row * env.cols + scan_col, row, col


def _action_to_neighbor(current: tuple[int, int], target: tuple[int, int]) -> int:
    delta = target[0] - current[0], target[1] - current[1]
    for action, action_delta in ACTION_DELTAS.items():
        if delta == action_delta:
            return action
    raise ValueError(f"Cells are not four-neighbours: {current} -> {target}")


def _cut_path_to_frontier(
    env: ClinicalWindowResectionEnv,
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    goals = {neighbor for neighbor in env.cut if target in set(neighbors4(neighbor))}
    if env.current in goals:
        return [env.current]
    queue: deque[tuple[int, int]] = deque([env.current])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {env.current: None}
    goal = None
    while queue:
        cell = queue.popleft()
        if cell in goals:
            goal = cell
            break
        for neighbor in neighbors4(cell):
            if neighbor in env.cut and neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    if goal is None:
        return []
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    return list(reversed(path))


def serpentine_direction_policy(env: ClinicalWindowResectionEnv) -> int:
    """Mechanical S-priority controller that never ends clamping early."""
    mask = env.action_masks()
    legal_directions = [action for action in ACTION_DELTAS if mask[action]]
    if not legal_directions:
        raise RuntimeError("Serpentine controller has no legal direction action")

    uncut_neighbors = []
    for action in legal_directions:
        delta_row, delta_col = ACTION_DELTAS[action]
        target = env.current[0] + delta_row, env.current[1] + delta_col
        if target not in env.cut:
            uncut_neighbors.append((target, action))
    if uncut_neighbors:
        return min(uncut_neighbors, key=lambda item: _scan_rank(env, item[0]))[1]

    for target in sorted(env._frontier(), key=lambda cell: _scan_rank(env, cell)):
        path = _cut_path_to_frontier(env, target)
        if len(path) >= 2:
            return _action_to_neighbor(env.current, path[1])
        if len(path) == 1:
            return _action_to_neighbor(env.current, target)
    return legal_directions[0]


def serpentine_macro_target_policy(env: ClinicalWindowResectionEnv) -> int:
    """Select the legal frontier target with mechanical S-order priority."""
    from clinical_macro_environment import CLINICAL_MACRO_GRID_ACTIONS

    legal_targets = np.flatnonzero(env.action_masks()[:CLINICAL_MACRO_GRID_ACTIONS])
    if not len(legal_targets):
        raise RuntimeError("Serpentine macro controller has no legal target action")

    def rank(action: int) -> tuple[int, int, int]:
        row, col = divmod(int(action), env.max_cols)
        return _scan_rank(env, (row, col))

    return int(min(legal_targets, key=rank))


def serpentine_hierarchical_policy(env: ClinicalWindowResectionEnv) -> np.ndarray:
    """Mechanical S target with clamp continuation; environment enforces 15/5."""
    from clinical_hierarchical_environment import CLAMP_CONTINUE, CLAMP_ACTION_COUNT

    mask = env.action_masks()
    legal_targets = np.flatnonzero(mask[CLAMP_ACTION_COUNT:])
    if not len(legal_targets):
        raise RuntimeError("Hierarchical serpentine controller has no legal target")

    def rank(action: int) -> tuple[int, int, int]:
        row, col = divmod(int(action), env.max_cols)
        return _scan_rank(env, (row, col))

    return np.asarray([CLAMP_CONTINUE, int(min(legal_targets, key=rank))], dtype=np.int64)


def rollout_clinical_policy(
    scenario: Mapping[str, Any],
    selector: ClinicalSelector,
    *,
    clinical_config: Mapping[str, float] | None = None,
    reward_config: Mapping[str, float] | None = None,
    include_replay: bool = False,
    include_step_trace: bool = False,
    mechanics_update_interval: int = 0,
    control_mode: str = "direction",
) -> dict[str, Any]:
    if control_mode == "hierarchical":
        from clinical_hierarchical_environment import ClinicalHierarchicalResectionEnv

        environment_class = ClinicalHierarchicalResectionEnv
    elif control_mode == "macro":
        from clinical_macro_environment import ClinicalMacroResectionEnv

        environment_class = ClinicalMacroResectionEnv
    elif control_mode == "direction":
        environment_class = ClinicalWindowResectionEnv
    else:
        raise ValueError(f"Unknown control_mode: {control_mode!r}")
    env = environment_class(
        scenario=scenario,
        clinical_config=clinical_config,
        reward_config=reward_config,
        mechanics_update_interval=mechanics_update_interval,
    )
    env.reset()
    rewards: list[float] = []
    reward_components: dict[str, float] = {}
    step_trace: list[dict[str, Any]] = []
    legal = 0
    proposed = 0
    expected_rates: list[float] = []
    while not env.terminated and not env.truncated:
        selected = selector(env)
        action = (
            np.asarray(selected, dtype=np.int64).reshape(-1)
            if control_mode == "hierarchical"
            else int(selected)
        )
        proposed += 1
        mask = env.action_masks()
        if control_mode == "hierarchical":
            legal += int(
                action.shape == (2,)
                and 0 <= int(action[0]) < 2
                and 0 <= int(action[1]) < 1200
                and mask[int(action[0])]
                and mask[2 + int(action[1])]
            )
        else:
            legal += int(0 <= action < len(mask) and mask[action])
        _, reward, _, _, info = env.step(action)
        rewards.append(float(reward))
        if include_step_trace:
            trace_item: dict[str, Any] = {
                "step": int(env.step_count),
                "action": action.tolist() if isinstance(action, np.ndarray) else int(action),
                "reward": float(reward),
                "cumulative_reward": float(sum(rewards)),
                "elapsed_minutes": float(info["elapsed_minutes"]),
                "expected_blood_loss_ml": float(info["expected_blood_loss_ml"]),
                "expected_bleeding_rate_ml_per_min": float(
                    info["expected_bleeding_rate_ml_per_min"]
                ),
                "phase": str(info["phase"]),
                "coverage": float(info["coverage"]),
                "reward_terms": {
                    str(name): float(value)
                    for name, value in info["reward_terms"].items()
                },
            }
            if control_mode == "macro" and isinstance(action, int):
                trace_item["target_cell"] = [
                    int(action // env.max_cols),
                    int(action % env.max_cols),
                ]
            step_trace.append(trace_item)
        expected_rates.append(float(info["expected_bleeding_rate_ml_per_min"]))
        for name, value in info["reward_terms"].items():
            reward_components[name] = reward_components.get(name, 0.0) + float(value)
    result = {
        "scenario_id": scenario.get("scenario_id"),
        "status": "ok" if env.terminated and env.failure_reason is None else "failed",
        "failure_reason": env.failure_reason,
        "completion": env.cut == env.domain,
        "coverage": len(env.cut) / len(env.domain),
        "cut_cells": [list(cell) for cell in sorted(env.cut)],
        "legal_action_rate": legal / proposed if proposed else 1.0,
        "episode_steps": env.step_count,
        "direction_action_count": env.direction_action_count,
        "macro_action_count": int(
            env.step_count if control_mode in ("macro", "hierarchical") else 0
        ),
        "max_macro_duration_minutes": float(
            getattr(env, "max_macro_duration_minutes", 0.0)
        ),
        "transfer_count": env.transfer_count,
        "transfer_overhead": env.transfer_count / max(1, env.direction_action_count),
        "no_progress_streak": env.no_progress_streak,
        "max_no_progress_streak": env.max_no_progress_streak,
        "stagnation_failure": str(env.failure_reason or "").startswith("stagnation:"),
        "same_edge_streak": env.same_edge_streak,
        "max_same_edge_streak": env.max_same_edge_streak,
        "two_cell_loop_failure": str(env.failure_reason or "").startswith(
            "two-cell oscillation:"
        ),
        "elapsed_minutes": env.elapsed_minutes,
        "expected_blood_loss_ml": env.expected_blood_loss_ml,
        "peak_expected_bleeding_rate_ml_per_min": env.peak_expected_bleeding_rate_ml_per_min,
        "mean_expected_bleeding_rate_ml_per_min": mean(expected_rates) if expected_rates else 0.0,
        "unclamped_exposed_minutes": env.unclamped_exposed_minutes,
        "total_clamped_minutes": env.total_clamped_minutes,
        "total_unclamped_minutes": env.total_unclamped_minutes,
        "clamp_cycle_count": env.clamp_cycle_count,
        "early_end_count": env.early_end_count,
        "sealed_vessel_count": len(env.sealed_ids),
        "hidden_component_ids": sorted(env.hidden_ids),
        "exposed_component_ids": sorted(env.exposed_ids),
        "sealed_component_ids": sorted(env.sealed_ids),
        "components": [
            {
                "id": int(component["id"]),
                "cells": [list(cell) for cell in sorted(component["cells"])],
                "ring": [list(cell) for cell in sorted(component["ring"])],
                "cross_section_cells": int(component["cross_section_cells"]),
                "area_mm2": float(component["area_mm2"]),
                "is_large": bool(component["is_large"]),
            }
            for component in env.components
        ],
        "total_reward": sum(rewards),
        "reward_components": dict(sorted(reward_components.items())),
        "max_front_tension": float(env.mechanics["peak_front_tension"]),
        "max_organ_energy": float(env.mechanics["peak_organ_energy"]),
        "max_vessel_strain": float(env.mechanics["peak_vessel_strain"]),
        "clamp_rule_violations": 0,
        "unclamp_rule_violations": 0,
    }
    if include_replay:
        result["replay"] = env.episode_replay()
    if include_step_trace:
        result["reward_trace"] = step_trace
    return result


def aggregate_clinical_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty record list")
    numeric_fields = (
        "coverage",
        "legal_action_rate",
        "episode_steps",
        "direction_action_count",
        "macro_action_count",
        "max_macro_duration_minutes",
        "transfer_count",
        "transfer_overhead",
        "no_progress_streak",
        "max_no_progress_streak",
        "stagnation_failure",
        "same_edge_streak",
        "max_same_edge_streak",
        "two_cell_loop_failure",
        "elapsed_minutes",
        "expected_blood_loss_ml",
        "peak_expected_bleeding_rate_ml_per_min",
        "mean_expected_bleeding_rate_ml_per_min",
        "unclamped_exposed_minutes",
        "total_clamped_minutes",
        "total_unclamped_minutes",
        "clamp_cycle_count",
        "early_end_count",
    )
    summary: dict[str, Any] = {
        "episode_count": len(records),
        "completion_rate": mean(float(record["completion"]) for record in records),
    }
    for field in numeric_fields:
        values = [float(record[field]) for record in records]
        summary[f"mean_{field}"] = mean(values)
        summary[f"median_{field}"] = median(values)
    macro_durations = [float(record["max_macro_duration_minutes"]) for record in records]
    summary["p95_max_macro_duration_minutes"] = float(
        np.percentile(macro_durations, 95)
    )
    summary["max_max_macro_duration_minutes"] = max(macro_durations)
    return summary


def calibrate_global_scales(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    clinical_config: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate one Train-only global scale pair from completed Pilot traces."""
    records = [
        rollout_clinical_policy(item, serpentine_direction_policy, clinical_config=clinical_config)
        for item in scenarios
    ]
    completed = [record for record in records if record["completion"]]
    if not completed:
        raise RuntimeError("Scale calibration produced no completed episodes")
    times = [float(record["elapsed_minutes"]) for record in completed]
    nonzero_blood = [float(record["expected_blood_loss_ml"]) for record in completed if record["expected_blood_loss_ml"] > 0]
    if not nonzero_blood:
        raise RuntimeError(
            "Scale calibration produced no non-zero blood loss; use larger Train scenarios or inspect exposure semantics"
        )
    return {
        "calibration_version": 1,
        "source_split": "train",
        "policy": "mechanical_serpentine_for_scale_only",
        "episode_count": len(records),
        "completed_episode_count": len(completed),
        "time_scale_minutes": float(median(times)),
        "blood_scale_ml": float(max(median(nonzero_blood), 100.0)),
        "records": records,
    }


def _load_scenarios(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "splits" not in payload or split not in payload["splits"]:
        raise ValueError(f"Split file does not contain split {split!r}")
    return list(payload["splits"][split])


def make_ppo_selector(model) -> ClinicalSelector:
    def select(env: ClinicalWindowResectionEnv):
        observation = env._observation()
        model_shape = tuple(model.observation_space.shape)
        environment_shape = tuple(observation.shape)
        if model_shape != environment_shape:
            raise ValueError(
                f"Model observation shape {model_shape} is incompatible with "
                f"evaluation environment shape {environment_shape}"
            )
        action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
        values = np.asarray(action)
        return values.astype(np.int64) if values.size > 1 else int(values.item())
    return select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "tuning", "validation", "test", "stress"), default="validation")
    parser.add_argument("--algorithm", choices=("serpentine", "ppo"), default="serpentine")
    parser.add_argument(
        "--control-mode",
        choices=("direction", "macro", "hierarchical"),
        default="direction",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scales", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps-multiplier", type=float, default=8.0)
    parser.add_argument("--weight-kg", type=float)
    parser.add_argument("--bleeding-probability", type=float, default=1.0)
    parser.add_argument("--mechanics-update-interval", type=int, default=0)
    parser.add_argument("--progress-bonus", type=float, default=5.0)
    parser.add_argument("--seal-progress-bonus", type=float, default=2.0)
    parser.add_argument("--time-cost", type=float, default=1.0)
    parser.add_argument("--blood-cost", type=float, default=1.0)
    parser.add_argument("--stagnation-penalty-cap", type=float, default=0.05)
    parser.add_argument("--two-cell-loop-penalty", type=float, default=0.25)
    parser.add_argument("--clinical-cost-cap", type=float, default=10.0)
    parser.add_argument("--early-end-mode", choices=("disabled", "threshold", "full"), default="full")
    parser.add_argument("--early-end-minutes", type=float, default=0.0)
    parser.add_argument("--stagnation-soft-start-steps", type=int, default=40)
    parser.add_argument("--stagnation-penalty-ramp-steps", type=int, default=24)
    parser.add_argument("--stagnation-limit-steps", type=int, default=96)
    parser.add_argument("--two-cell-loop-soft-start-traversals", type=int, default=6)
    parser.add_argument("--two-cell-loop-limit-traversals", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    scenarios = _load_scenarios(args.splits, args.split)
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    clinical_config: dict[str, float] = {
        "max_steps_multiplier": args.max_steps_multiplier,
        "bleeding_probability": args.bleeding_probability,
        "early_end_mode": args.early_end_mode,
        "early_end_minutes": args.early_end_minutes,
        "stagnation_soft_start_steps": args.stagnation_soft_start_steps,
        "stagnation_penalty_ramp_steps": args.stagnation_penalty_ramp_steps,
        "stagnation_limit_steps": args.stagnation_limit_steps,
        "two_cell_loop_soft_start_traversals": args.two_cell_loop_soft_start_traversals,
        "two_cell_loop_limit_traversals": args.two_cell_loop_limit_traversals,
    }
    if args.scales is not None:
        scales = json.loads(args.scales.read_text(encoding="utf-8"))
        clinical_config.update({
            "time_scale_minutes": float(scales["time_scale_minutes"]),
            "blood_scale_ml": float(scales["blood_scale_ml"]),
            "weight_kg": float(scales.get("weight_kg", 70.0)),
        })
    if args.weight_kg is not None:
        clinical_config["weight_kg"] = args.weight_kg
    if args.algorithm == "serpentine":
        selector = {
            "direction": serpentine_direction_policy,
            "macro": serpentine_macro_target_policy,
            "hierarchical": serpentine_hierarchical_policy,
        }[args.control_mode]
    else:
        if args.model is None:
            parser.error("--model is required for --algorithm ppo")
        import torch
        from sb3_contrib import MaskablePPO
        if args.control_mode == "hierarchical":
            import clinical_hierarchical_policy  # noqa: F401
        elif args.control_mode == "macro":
            import clinical_macro_policy  # noqa: F401
        if args.torch_threads <= 0:
            parser.error("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
        model = MaskablePPO.load(str(args.model), device="auto")
        selector = make_ppo_selector(model)
    records = [
        rollout_clinical_policy(
            item,
            selector,
            clinical_config=clinical_config,
            reward_config={
                "time_cost": args.time_cost,
                "blood_cost": args.blood_cost,
                "progress_bonus": args.progress_bonus,
                "seal_progress_bonus": args.seal_progress_bonus,
                "stagnation_penalty_cap": args.stagnation_penalty_cap,
                "two_cell_loop_penalty": args.two_cell_loop_penalty,
                "clinical_cost_cap": args.clinical_cost_cap,
            },
            mechanics_update_interval=args.mechanics_update_interval,
            control_mode=args.control_mode,
        )
        for item in scenarios
    ]
    payload = {
        "split": args.split,
        "algorithm": args.algorithm,
        "control_mode": args.control_mode,
        "environment_version": {
            "direction": CLINICAL_ENVIRONMENT_VERSION,
            "macro": "clinical-macro-window-v1",
            "hierarchical": "clinical-hierarchical-window-v1",
        }[args.control_mode],
        "clinical_config_overrides": clinical_config,
        "summary": aggregate_clinical_records(records),
        "records": records,
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

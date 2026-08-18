"""Teacher-trajectory selection for variable-size planar PPO."""

from __future__ import annotations

from statistics import mean
from typing import Any, Callable, Mapping, Sequence
import json
from pathlib import Path

import numpy as np

from environment import PlanarResectionEnv, variable_grid_action_masks, variable_grid_observation
from evaluation import serpentine_priority_policy
from planner import plan_resection


def _planner_selector(scenario: Mapping[str, Any]) -> Callable[[PlanarResectionEnv], int]:
    planned = plan_resection(**{
        key: scenario[key]
        for key in ("rows", "cols", "domain_cells", "obstacle_cells", "start_cell")
    })
    cuts = [
        int(event["cell"][0]) * 50 + int(event["cell"][1])
        for event in planned["events"]
        if event["action"] == "cut" and event.get("reason") != "start"
    ]
    cursor = 0

    def select(env: PlanarResectionEnv) -> int:
        nonlocal cursor
        if cursor < len(cuts):
            action = cuts[cursor]
            cursor += 1
            return action
        return serpentine_priority_policy(env)

    return select


def _rollout(scenario: Mapping[str, Any], select: Callable[[PlanarResectionEnv], int]) -> dict[str, Any]:
    env = PlanarResectionEnv(scenario=scenario, reward_config={"transfer_cost": 2.0})
    env.reset()
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    vessel_strains: list[float] = []
    while not env.terminated and not env.truncated:
        observations.append(variable_grid_observation(env))
        masks.append(variable_grid_action_masks(env))
        canvas_action = select(env)
        row, col = divmod(canvas_action, 50)
        actions.append(row * 40 + col)
        env.step(canvas_action)
        vessel_strains.append(float(env.mechanics["peak_vessel_strain"]))
    cuts = sum(event["action"] == "cut" for event in env.events)
    transfers = sum(event["action"] == "transfer" for event in env.events)
    return {
        "observations": observations,
        "actions": actions,
        "masks": masks,
        "completion": env.cut == env.domain,
        "transfer_overhead": transfers / cuts if cuts else float("inf"),
        "mean_vessel_strain": mean(vessel_strains) if vessel_strains else 0.0,
    }


def collect_filtered_teacher_demonstrations(
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Keep planner traces only when they dominate S-priority on both proxies."""
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    planner_selected = 0
    overheads: list[float] = []
    strains: list[float] = []
    for scenario in scenarios:
        s_trace = _rollout(scenario, serpentine_priority_policy)
        planner_trace = _rollout(scenario, _planner_selector(scenario))
        use_planner = (
            planner_trace["completion"]
            and planner_trace["transfer_overhead"] <= s_trace["transfer_overhead"]
            and planner_trace["mean_vessel_strain"] <= s_trace["mean_vessel_strain"]
        )
        trace = planner_trace if use_planner else s_trace
        planner_selected += int(use_planner)
        observations.extend(trace["observations"])
        actions.extend(trace["actions"])
        masks.extend(trace["masks"])
        overheads.append(float(trace["transfer_overhead"]))
        strains.append(float(trace["mean_vessel_strain"]))
    if not observations:
        raise RuntimeError("Teacher collection produced no state-action pairs")
    return (
        np.stack(observations).astype(np.float32),
        np.asarray(actions, dtype=np.int64),
        np.stack(masks).astype(bool),
        {
            "episode_count": float(len(scenarios)),
            "demonstration_count": float(len(observations)),
            "planner_selected_episode_count": float(planner_selected),
            "mean_transfer_overhead": float(mean(overheads)),
            "mean_vessel_strain": float(mean(strains)),
        },
    )


def write_teacher_cache(path: str | Path, scenarios: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Generate a portable cache before importing or initializing PyTorch."""
    observations, actions, masks, summary = collect_filtered_teacher_demonstrations(scenarios)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        observations=observations,
        actions=actions,
        masks=masks,
        summary=np.asarray(json.dumps(summary, ensure_ascii=False)),
    )
    return summary


def load_teacher_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Load a teacher cache and verify its action-mask contract."""
    with np.load(Path(path), allow_pickle=False) as payload:
        observations = np.asarray(payload["observations"], dtype=np.float32)
        actions = np.asarray(payload["actions"], dtype=np.int64)
        masks = np.asarray(payload["masks"], dtype=bool)
        summary = json.loads(str(payload["summary"].item()))
    if observations.ndim != 4 or observations.shape[1:] != (18, 30, 40):
        raise ValueError(f"Unexpected cached observation shape: {observations.shape}")
    if masks.shape != (len(actions), 1200) or len(observations) != len(actions):
        raise ValueError("Teacher cache has inconsistent sample dimensions")
    if np.any(actions < 0) or np.any(actions >= 1200) or not np.all(masks[np.arange(len(actions)), actions]):
        raise ValueError("Teacher cache contains an action outside its legal mask")
    return observations, actions, masks, {str(key): float(value) for key, value in summary.items()}

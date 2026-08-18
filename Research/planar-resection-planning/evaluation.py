"""Frozen-policy baselines and external metrics for planar resection."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from environment import CANVAS_SIZE, PlanarResectionEnv
from mechanics import DEFAULT_MECHANICS

ActionSelector = Callable[[PlanarResectionEnv], int]

SAFE_VESSEL_STRAIN = float(DEFAULT_MECHANICS["safe_vessel_strain"])
TEAR_VESSEL_STRAIN = float(DEFAULT_MECHANICS["tear_vessel_strain"])


def row_major_frontier_policy(env: PlanarResectionEnv) -> int:
    """Outward-to-inward row baseline constrained by the current frontier.

    The rule picks the row-major first currently legal cell.  It never bypasses
    vessels, release, or transfer semantics; those remain environment rules.
    """
    legal = np.flatnonzero(env.action_masks())
    if not len(legal):
        raise RuntimeError("row-major policy was called with no legal action")
    return int(legal[0])


def serpentine_scan_policy(env: PlanarResectionEnv) -> int:
    """True continuous S-scan for a complete obstacle-free rectangle.

    The first cut must be the top-left corner.  Every subsequent selected cell
    is adjacent to the preceding one, so the transfer overhead is exactly zero.
    This deliberately narrow baseline is the correct control for the 5x5
    no-vessel curriculum; it must not be applied to arbitrary starts or vessel
    layouts where an uninterrupted Hamiltonian scan may be impossible.
    """
    expected_domain = {(row, col) for row in range(env.rows) for col in range(env.cols)}
    if env.domain != expected_domain or env.obstacles or env.start != (0, 0):
        raise ValueError("Serpentine baseline requires a complete obstacle-free rectangle with start_cell [0, 0]")
    order = [
        (row, col)
        for row in range(env.rows)
        for col in (range(env.cols) if row % 2 == 0 else range(env.cols - 1, -1, -1))
    ]
    if len(env.cut) >= len(order):
        raise RuntimeError("serpentine policy was called after all cells were cut")
    row, col = order[len(env.cut)]
    return row * CANVAS_SIZE + col


def serpentine_priority_policy(env: PlanarResectionEnv) -> int:
    """Deterministic S-order priority baseline for obstacle/release scenarios.

    Unlike :func:`serpentine_scan_policy`, this policy accepts arbitrary
    boundary starts and vessel obstacles.  It preserves the top-to-bottom S
    ordering whenever that cell is legally reachable; dynamic-frontier and
    release rules decide the required deterministic detour.
    """
    legal = np.flatnonzero(env.action_masks())
    if not len(legal):
        raise RuntimeError("serpentine-priority policy was called with no legal action")

    def rank(action: int) -> tuple[int, int, int]:
        row, col = divmod(int(action), CANVAS_SIZE)
        scan_col = col if row % 2 == 0 else env.cols - 1 - col
        return row * env.cols + scan_col, row, col

    return int(min(legal, key=rank))


def evaluate_policy(
    scenario: Mapping[str, Any],
    policy: ActionSelector,
    *,
    max_steps: int | None = None,
    mechanics_parameters: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Run one frozen policy and report external, non-reward metrics."""
    env = PlanarResectionEnv(
        scenario=scenario, max_steps=max_steps, mechanics_parameters=mechanics_parameters,
    )
    _, _ = env.reset()
    legal_actions = 0
    proposed_actions = 0
    front_tensions: list[float] = []
    organ_energies: list[float] = []
    vessel_strains: list[float] = []
    rewards: list[float] = []
    reward_components: Dict[str, list[float]] = {}
    while not env.terminated and not env.truncated:
        action = policy(env)
        proposed_actions += 1
        if 0 <= action < CANVAS_SIZE ** 2 and env.action_masks()[action]:
            legal_actions += 1
        _, reward, _, _, info = env.step(action)
        rewards.append(float(reward))
        for name, value in info["reward_terms"].items():
            reward_components.setdefault(name, []).append(float(value))
        front_tensions.append(float(env.mechanics["peak_front_tension"]))
        organ_energies.append(float(env.mechanics["peak_organ_energy"]))
        vessel_strains.append(float(env.mechanics["peak_vessel_strain"]))
    event_actions = [event["action"] for event in env.events]
    transfers = event_actions.count("transfer")
    cuts = event_actions.count("cut")

    def worst_tenth(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        count = max(1, math.ceil(len(values) * 0.10))
        return mean(sorted(values, reverse=True)[:count])

    def fraction_above(values: Sequence[float], threshold: float) -> float:
        return mean(float(value > threshold) for value in values) if values else 0.0

    return {
        "status": "ok" if env.terminated and env.failure_reason is None else "failed",
        "failure_reason": env.failure_reason,
        "coverage": len(env.cut) / len(env.domain),
        "completion": env.cut == env.domain,
        "legal_action_rate": legal_actions / proposed_actions if proposed_actions else 1.0,
        "total_transfer_count": transfers,
        "transfer_overhead": transfers / cuts if cuts else float("inf"),
        "total_movement_steps": transfers,
        "episode_length": env.step_count,
        "total_reward": sum(rewards),
        "mean_step_reward": mean(rewards) if rewards else 0.0,
        "reward_components": {name: sum(values) for name, values in sorted(reward_components.items())},
        "mean_front_tension": mean(front_tensions) if front_tensions else 0.0,
        "worst_10pct_front_tension": worst_tenth(front_tensions),
        "mean_organ_energy": mean(organ_energies) if organ_energies else 0.0,
        "worst_10pct_organ_energy": worst_tenth(organ_energies),
        "mean_vessel_strain": mean(vessel_strains) if vessel_strains else 0.0,
        "cumulative_vessel_strain": sum(vessel_strains),
        "worst_10pct_vessel_strain": worst_tenth(vessel_strains),
        "fraction_steps_above_safe": fraction_above(vessel_strains, SAFE_VESSEL_STRAIN),
        "fraction_steps_above_tear": fraction_above(vessel_strains, TEAR_VESSEL_STRAIN),
        "max_vessel_strain": max(vessel_strains, default=0.0),
        "max_risk_peak": max(front_tensions + organ_energies + vessel_strains, default=0.0),
        "release_rule_correct": all(event["action"] != "release" or event["ring"] for event in env.events),
        "replay": env.episode_replay(),
    }


def evaluate_row_baseline(scenario: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Evaluate the documented outward-to-inward, row-major baseline."""
    return evaluate_policy(scenario, row_major_frontier_policy, **kwargs)


def evaluate_serpentine_baseline(scenario: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Evaluate the formal no-transfer S-scan control on its valid scope."""
    return evaluate_policy(scenario, serpentine_scan_policy, **kwargs)


def evaluate_serpentine_priority_baseline(scenario: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Evaluate the deterministic S-order priority policy with obstacles."""
    return evaluate_policy(scenario, serpentine_priority_policy, **kwargs)

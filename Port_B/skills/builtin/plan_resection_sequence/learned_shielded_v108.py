"""v10.8 Port B bridge: lazy exact shield (C4L).

Adds a new algorithm ID ``clinical_v108_c4_learned_lazy_exact`` to the
plan_resection_sequence Skill.  Behaviour:

  * Algorithm semantics, model inputs, candidate set, tie-break and
    safety predicate are identical to the v10.7 ``learned_shielded``
    algorithm (call it "eager" C4).
  * Verification is performed in lazy order: only the highest-ranked
    candidate is exact-verified first, then the second if the first is
    unsafe, etc.  Falls back to the deterministic serpentine target
    if all candidates are unsafe (same as v10.7).
  * The old ``learned_shielded`` algorithm remains untouched and
    remains the production default.  This new entry is gated behind
    ``algorithm == 'learned_shielded_v108'``; flip the request
    parameter to opt in.

This is the v10.8 E10 Port B bridge.  Do not enable in production until
plan Gate A-E all pass.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from skills.builtin.plan_resection_sequence.learned_shielded import (
    EXPECTED_CHECKPOINT_SHA256,
    POLICY_ID,
    _repo_root,
    _research_modules,
    build_scenario_from_control_points,
    configured_checkpoint_path,
    validate_checkpoint,
)

POLICY_ID_V108 = "clinical_v108_c4_learned_lazy_exact"
LAZY_VERIFY_MODE = "lazy"  # "lazy" or "audit"


def _v108_modules():
    """Resolve the v10.8 lazy rollout helper from the planar-resection-planning
    folder (local VoxelSage Research/ tree).  Mirrors _research_modules but
    for the v10.8 module path.
    """
    root = _repo_root()
    candidates = [
        root / "Research" / "planar-resection-planning",
        root / "贪吃蛇" / "planar_simulator",
    ]
    for base in candidates:
        if (base / "lazy_confirmation_controllers_v108.py").is_file():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            from lazy_confirmation_controllers_v108 import rollout_controller
            return rollout_controller
    raise FileNotFoundError(
        "Cannot find v10.8 lazy module.  Looked under:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def plan_learned_shielded_v108(
    surface_control_points: np.ndarray,
    grid_resolution: tuple[int, int],
    *,
    margin_ml: float = 16.07054347826075,
    verify_mode: str = LAZY_VERIFY_MODE,
) -> dict[str, Any]:
    """Run the v10.8 C4L (lazy) controller on the saved 3D surface.

    Reuses ``build_scenario`` and the v10.7 baseline helper to obtain a
    baseline blood value (the same procedure as v10.7
    ``plan_learned_shielded``); only the controller call is replaced.
    """
    _, v107_rollout, _ = _research_modules()
    rollout_v108 = _v108_modules()

    checkpoint, sha = validate_checkpoint()
    rows, cols = int(grid_resolution[0]), int(grid_resolution[1])
    scenario = build_scenario_from_control_points(
        surface_control_points, grid_resolution=(rows, cols),
    )

    t0 = time.time()
    baseline_res = v107_rollout(
        "C0", scenario, baseline_blood=0.0, margin_ml=margin_ml,
    )
    baseline_blood = float(baseline_res["realized_episode_B_ml"])
    baseline_t = time.time() - t0

    t1 = time.time()
    lazy_res = rollout_v108(
        "C4L", scenario,
        baseline_blood=baseline_blood, margin_ml=margin_ml,
        checkpoint_path=str(checkpoint),
        verify_mode=verify_mode,
    )
    lazy_t = time.time() - t1

    return {
        "policy_id": POLICY_ID_V108,
        "checkpoint_sha256": sha,
        "completion": bool(lazy_res.get("completion", False)),
        "failure_reason": lazy_res.get("failure_reason"),
        "elapsed_minutes": float(lazy_res.get("elapsed_minutes", 0.0)),
        "realized_episode_B_ml": float(lazy_res.get("realized_episode_B_ml", 0.0)),
        "budget_ml": float(baseline_blood + margin_ml),
        "margin_ml": float(margin_ml),
        "selected_max_B_total_ml": float(lazy_res.get("selected_max_B_total_ml", 0.0)),
        "shield_intervention_count": int(lazy_res.get("shield_intervention_count", 0)),
        "safety_invariant_violations": int(lazy_res.get("safety_invariant_violations", 0)),
        "verified_count_mean": float(lazy_res.get("verified_count_mean", 0.0)),
        "verified_count_max": int(lazy_res.get("verified_count_max", 0)),
        "wall_seconds": float(lazy_res.get("wall_seconds", 0.0)),
        "wall_seconds_baseline": float(baseline_t),
        "wall_seconds_lazy": float(lazy_t),
        "actions": lazy_res.get("actions", []),
        "action_sequence_hash": lazy_res.get("action_sequence_hash", ""),
    }

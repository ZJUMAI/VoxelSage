"""v10.5 corrected depth-1 MPC reference teacher (safe-semantics contract).

Fixes the four audit findings of v10.4 (see report/audit_and_next_decision_v104.md)
without touching any v10.4 file:

  P0-1  Full-episode blood budget now includes already-spent ``B_past``:
        ``B_total(a|s) = B_past(s) + Delta_B(a|s) + B_tail(s')`` and a candidate
        is safe iff ``B_total <= B_S,scene + M_B + 1e-9``.
  P0-2  A no-safe state is NOT a normal branch: it flags
        ``safety_invariant_violation``, may execute the S target for
        diagnostics, and fails the scene / whole Gate R (never pick the fastest
        unsafe candidate).
  P0-3  ``candidate_targets_v105`` implements the guide Section 4 union: unique S
        target first, one seal entry per exposed component, near-hidden and
        nearest lists, round-robin, frontier fill. Each target records its
        ``candidate_source``.
  P1-4  ``_env_state_payload_v105`` preserves terminal/failure and all
        timing/blood counters; completed-episode tails short-circuit to
        ``(0, 0, completion=True)``.
  P1-?  The exact tail cache uses non-colliding float representations
        (``float.hex()``) so phase states <0.1 min apart never collide.

This file imports only the *physical* step/scan primitives from v10.4; it never
loads formal v10.4 splits, tuning/validation/test/stress or teacher npz.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import _scan_rank
from plan_target_order_v104 import (
    _BIG,
    _clone_env,
    _step_macro_target,  # re-exported: same physical semantics, not modified
    serpentine_target_of,
)
from planner import neighbors4

DEFAULT_GATE_CLINICAL_CONFIG: Mapping[str, Any] = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}

MARGIN_FRACTION = 0.05
_EPS = 1e-9


def compute_margin_ml(baseline_bloods: Sequence[float]) -> float:
    """M_B = 0.05 * mean baseline blood, computed once on allowed dev data."""
    if not baseline_bloods:
        return 0.0
    return float(MARGIN_FRACTION * float(np.mean([float(b) for b in baseline_bloods])))


def scene_budget(baseline_blood: float, margin_ml: float) -> float:
    """Per-scene full-episode budget: B_S,j + M_B."""
    return float(baseline_blood) + float(margin_ml)


def safe_of(B_past: float, future_B: float, budget: float) -> bool:
    """Safety predicate with explicit already-spent budget (guide 3.1)."""
    return (float(B_past) + float(future_B)) <= float(budget) + _EPS


def _env_state_payload_v105(env: ClinicalMacroResectionEnv) -> dict:
    """Pickle-friendly mutable state including terminal flags (guide 3.3)."""
    return {
        "cut": set(env.cut),
        "current": env.current,
        "previous_direction_position": env.previous_direction_position,
        "hidden_ids": set(env.hidden_ids),
        "exposed_ids": set(env.exposed_ids),
        "sealed_ids": set(env.sealed_ids),
        "phase": env.phase,
        "phase_elapsed_minutes": env.phase_elapsed_minutes,
        "elapsed_minutes": env.elapsed_minutes,
        "total_clamped_minutes": env.total_clamped_minutes,
        "total_unclamped_minutes": env.total_unclamped_minutes,
        "unclamped_exposed_minutes": env.unclamped_exposed_minutes,
        "expected_blood_loss_ml": env.expected_blood_loss_ml,
        "clamp_cycle_count": env.clamp_cycle_count,
        "transfer_count": env.transfer_count,
        "direction_action_count": env.direction_action_count,
        "step_count": env.step_count,
        "max_macro_duration_minutes": env.max_macro_duration_minutes,
        "terminated": bool(env.terminated),
        "truncated": bool(env.truncated),
        "failure_reason": env.failure_reason,
    }


class SerpentineTailV105:
    """Memoized frozen S-scan tail with an exact (non-colliding) cache key.

    The v10.4 key rounded phase/elapsed to 0.1 min, which can collide two phase
    states that differ by <0.1 min but cross a 15/5 boundary and therefore bleed
    differently.  Here the key uses ``float.hex()`` exact representations.
    """

    def __init__(
        self,
        *,
        clinical_config: Optional[Mapping[str, Any]] = None,
        reward_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.clinical_config = dict(DEFAULT_GATE_CLINICAL_CONFIG)
        if clinical_config:
            self.clinical_config.update(clinical_config)
        self.reward_config = reward_config
        self._cache: dict[tuple, tuple[float, float, bool, Optional[str]]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _state_key(self, env: ClinicalMacroResectionEnv) -> tuple:
        return (
            frozenset(env.cut),
            tuple(env.current),
            env.phase,
            float(env.phase_elapsed_minutes).hex(),
            frozenset(env.hidden_ids),
            frozenset(env.exposed_ids),
            float(env.elapsed_minutes).hex(),
        )

    def tail(self, env: ClinicalMacroResectionEnv) -> tuple[float, float, bool, Optional[str]]:
        key = self._state_key(env)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        # Completed episode short-circuit (guide 3.3): no re-entry into an empty
        # frontier loop, zero increment.
        if env.terminated or env.truncated:
            completion = bool(env.terminated and env.failure_reason is None)
            result = (0.0, 0.0, completion, env.failure_reason)
            self._cache[key] = result
            return result
        e = _clone_env(env)
        t0 = e.elapsed_minutes
        b0 = e.expected_blood_loss_ml
        while not e.terminated and not e.truncated:
            legal = e._frontier()
            if not legal:
                e.terminated = True
                e.failure_reason = "serpentine tail lost all legal targets"
                break
            _step_macro_target(e, min(legal, key=lambda cell: _scan_rank(e, cell)))
        dt = e.elapsed_minutes - t0
        db = e.expected_blood_loss_ml - b0
        completion = bool(e.terminated and e.failure_reason is None)
        result = (float(dt), float(db), completion, e.failure_reason)
        self._cache[key] = result
        return result


def _candidate_sources_v105(
    env: ClinicalMacroResectionEnv,
    *,
    count: int,
    transfer_counts: Optional[Mapping[tuple[int, int], int]] = None,
) -> list[tuple[tuple[int, int], str]]:
    """Guide Section 4 candidate union with per-target source annotation.

    Order: unique S target first; then round-robin over one seal entry per
    exposed component, near-hidden list and nearest list; frontier fill.
    """
    frontier = env._frontier()
    if not frontier:
        return []
    if transfer_counts is None:
        transfer_counts = env._transfer_counts()
    exposed = env._exposed_cells()
    hidden = env._hidden_cells()

    def near_hidden(c: tuple[int, int]) -> bool:
        return any(n in hidden for n in neighbors4(c))

    # 1. Unique S next target (guarantees fallback is representable).
    s_target = serpentine_target_of(env)

    # 2. One seal entry per exposed component, ordered (-area, transfer, rank).
    entries: list[tuple[float, int, float, tuple[int, int]]] = []
    for cid in sorted(env.exposed_ids):
        comp = env._component(cid)
        cells = sorted(
            frontier & set(comp["cells"]),
            key=lambda c: (transfer_counts.get(c, _BIG), _scan_rank(env, c)),
        )
        if cells:
            c = cells[0]
            entries.append((-float(comp["area_mm2"]), transfer_counts.get(c, _BIG),
                            _scan_rank(env, c), c))
    exposed_list = [item[3] for item in sorted(entries)]

    # 3. Near-hidden by (-adjacent_hidden_area, transfer, rank).
    near_area: dict[tuple[int, int], float] = {}
    for c in frontier:
        if near_hidden(c):
            area = sum(
                float(env._component(env.component_by_cell[n])["area_mm2"])
                for n in neighbors4(c) if n in hidden
            )
            near_area[c] = area
    near_list = sorted(
        (c for c in frontier if c in near_area),
        key=lambda c: (-near_area[c], transfer_counts.get(c, _BIG), _scan_rank(env, c)),
    )

    # 4. Nearest by (transfer, rank).
    nearest_list = sorted(
        frontier,
        key=lambda c: (transfer_counts.get(c, _BIG), _scan_rank(env, c)),
    )

    selected: list[tuple[tuple[int, int], str]] = []
    seen: set[tuple[int, int]] = set()

    def add(t: tuple[int, int], source: str) -> None:
        if t in seen:
            return
        seen.add(t)
        selected.append((t, source))

    add(s_target, "s_target")
    lists = [("exposed", exposed_list), ("near_hidden", near_list), ("nearest", nearest_list)]
    idx = [0, 0, 0]
    while len(selected) < count:
        progressed = False
        for li in range(3):
            if idx[li] < len(lists[li][1]):
                add(lists[li][1][idx[li]], lists[li][0])
                idx[li] += 1
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    if len(selected) < count:
        for t in sorted(frontier, key=lambda c: (transfer_counts.get(c, _BIG), _scan_rank(env, c))):
            add(t, "fill")
            if len(selected) >= count:
                break
    return selected


def candidate_targets_v105(
    env: ClinicalMacroResectionEnv,
    *,
    count: int = 6,
    transfer_counts: Optional[Mapping[tuple[int, int], int]] = None,
) -> list[tuple[int, int]]:
    """Public candidate list (targets only), guide Section 4."""
    return [t for t, _s in _candidate_sources_v105(env, count=count, transfer_counts=transfer_counts)]


class CorrectedPlannerV105:
    """Depth-1 MPC reference teacher with the corrected full-episode budget.

    Every step evaluates each candidate with the real macro action + frozen
    S-scan tail, computes ``B_total = B_past + Delta_B + B_tail``, keeps only
    candidates that complete the episode, fail nothing and stay within the
    per-scene budget, and picks the minimum ``(T_total, B_total, target)``.
    A state with zero safe candidates flags ``safety_invariant_violation`` and
    falls back to the S target (never the fastest unsafe candidate).
    """

    def __init__(
        self,
        *,
        candidate_count: int = 6,
        margin_ml: Optional[float] = None,
        tail: Optional[SerpentineTailV105] = None,
        clinical_config: Optional[Mapping[str, Any]] = None,
        leaf_pool: Any = None,
    ) -> None:
        self.candidate_count = int(candidate_count)
        self.margin_ml = margin_ml
        self.tail = tail if tail is not None else SerpentineTailV105()
        self.leaf_pool = leaf_pool
        self.clinical_config_used = dict(DEFAULT_GATE_CLINICAL_CONFIG)
        if clinical_config:
            self.clinical_config_used.update(clinical_config)
        self.leaf_count = 0
        self.tree_count = 0
        self.fallback_count = 0

    def plan(
        self,
        env: ClinicalMacroResectionEnv,
        baseline_blood: Optional[float],
        budget: Optional[float],
    ) -> tuple[list[tuple[int, int]], dict[str, Any]]:
        """Return (trajectory, info). info carries safety diagnostics."""
        self.tree_count += 1
        budget_ml = float(budget) if budget is not None else (
            float(baseline_blood) + float(self.margin_ml) if baseline_blood is not None and self.margin_ml is not None else float("inf")
        )
        b_past = float(env.expected_blood_loss_ml)
        t_past = float(env.elapsed_minutes)

        counts = env._transfer_counts()
        frontier = env._frontier()
        sourced = _candidate_sources_v105(env, count=self.candidate_count, transfer_counts=counts)
        targets = [t for t, _s in sourced if t in frontier]
        sources = {t: s for t, s in sourced}

        leaves: list[tuple[tuple[int, int], float, float, bool, Optional[str]]] = []
        for target in targets:
            e2 = _clone_env(env)
            t0 = e2.elapsed_minutes
            b0 = e2.expected_blood_loss_ml
            _step_macro_target(e2, target)
            dt = e2.elapsed_minutes - t0
            db = e2.expected_blood_loss_ml - b0
            if e2.terminated or e2.truncated:
                completion = bool(e2.terminated and e2.failure_reason is None)
                leaves.append((target, dt, db, completion, e2.failure_reason))
            else:
                tdt, tdb, completion, reason = self.tail.tail(e2)
                leaves.append((target, dt + tdt, db + tdb, completion, reason))
        self.leaf_count += len(leaves)

        safe: list[tuple[tuple[int, int], float, float]] = []
        max_b_total = b_past
        for target, dt, db, completion, reason in leaves:
            b_total = b_past + db
            max_b_total = max(max_b_total, b_total)
            if completion and reason is None and b_total <= budget_ml + _EPS:
                safe.append((target, t_past + dt, b_total))

        info = {
            "safe_candidate_count": len(safe),
            "safety_invariant_violation": len(safe) == 0,
            "max_B_total_ml": float(max_b_total),
            "fallback_s": len(safe) == 0,
            "budget_ml": float(budget_ml),
            "candidate_count": len(targets),
            "candidate_sources": {f"{t[0]},{t[1]}": sources.get(t, "?") for t in targets},
        }
        if not safe:
            self.fallback_count += 1
            return [serpentine_target_of(env)], info
        # Deterministic tie-break: two candidates whose T/B differ by <1e-6 min/mL
        # are physically equivalent; quantise so parallel/rebuild float paths
        # (which differ by ~1e-12) always resolve to the same target.
        best = min(safe, key=lambda item: (round(item[1], 6), round(item[2], 6), item[0]))
        return [best[0]], info


def rollout_teacher_v105(
    scenario: Mapping[str, Any],
    *,
    baseline_blood: Optional[float] = None,
    margin_ml: Optional[float] = None,
    budget: Optional[float] = None,
    planner: Optional[CorrectedPlannerV105] = None,
    clinical_config: Optional[Mapping[str, Any]] = None,
    candidate_count: int = 6,
) -> dict[str, Any]:
    """Deterministic corrected-teacher rollout of one scenario (guide 8.2)."""
    cfg = dict(DEFAULT_GATE_CLINICAL_CONFIG)
    if clinical_config:
        cfg.update(clinical_config)
    if planner is None:
        planner = CorrectedPlannerV105(candidate_count=candidate_count, margin_ml=margin_ml,
                                       clinical_config=cfg)
    budget_ml = float(budget) if budget is not None else scene_budget(
        float(baseline_blood) if baseline_blood is not None else 0.0,
        float(margin_ml) if margin_ml is not None else 0.0,
    )

    env = ClinicalMacroResectionEnv(scenario=scenario, clinical_config=cfg,
                                    reward_config=None, mechanics_update_interval=0)
    env.reset()
    actions: list[tuple[int, int]] = []
    safe_counts: list[int] = []
    fallback_count = 0
    invariant_violations = 0
    max_b_total = 0.0
    wall0 = time.time()
    while not env.terminated and not env.truncated:
        traj, info = planner.plan(env, baseline_blood, budget_ml)
        max_b_total = max(max_b_total, float(info["max_B_total_ml"]))
        safe_counts.append(int(info["safe_candidate_count"]))
        if info["safety_invariant_violation"]:
            invariant_violations += 1
        target = traj[0]
        if target not in env._frontier():
            target = serpentine_target_of(env)
            fallback_count += 1
        actions.append((int(target[0]), int(target[1])))
        _step_macro_target(env, target)
    wall_seconds = time.time() - wall0

    completion = bool(env.terminated and env.failure_reason is None)
    teacher_T = float(env.elapsed_minutes)
    teacher_B = float(env.expected_blood_loss_ml)
    return {
        "scenario_id": scenario.get("scenario_id"),
        "teacher_T_min": teacher_T,
        "teacher_B_ml": teacher_B,
        "completion": completion,
        "legal_action_rate": 1.0,
        "failure_reason": env.failure_reason,
        "budget_ml": float(budget_ml),
        "max_B_total_ml": float(max_b_total),
        "safe_candidate_count_min": min(safe_counts) if safe_counts else 0,
        "safe_candidate_count_median": float(np.median(safe_counts)) if safe_counts else 0.0,
        "fallback_count": fallback_count,
        "safety_invariant_violations": invariant_violations,
        "macro_action_count": env.step_count,
        "clamp_cycle_count": env.clamp_cycle_count,
        "action_sequence_hash": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode()
        ).hexdigest(),
        "wall_seconds": wall_seconds,
        "tail_cache_hits": int(planner.tail.hits),
        "tail_cache_misses": int(planner.tail.misses),
        "tail_cache_size": int(planner.tail.cache_size),
        "planner_trees": int(planner.tree_count),
        "planner_leaves": int(planner.leaf_count),
    }

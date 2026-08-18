"""v10.4 target-order planning for Gate A (Train-only strong-planner audit).

Implements the three fair-comparison branches of the v10.4 guide Section 6.1,
all sharing the same macro environment (`ClinicalMacroResectionEnv`), action
mask, underlying shortest-path transfer, per-cell transfer billing, cutting
duration, blood-loss integration and the mechanical 15/5 state machine:

    1. ``serpentine_fixed15_5``       mechanical S-scan (guide baseline)
    2. ``nearest_frontier_fixed15_5`` greedy nearest frontier
    3. ``window_aware_planner_fixed15_5`` finite look-ahead beam/MPC planner

The window-aware planner follows guide Section 6.2: for each candidate branch
it clones the environment, executes the real macro action, then completes the
episode with a frozen S-scan tail under the *same* transfer/billing/blood code,
producing an end-of-episode (T, B).  The branch is then chosen by the frozen
rule order (feasibility -> blood non-inferiority -> shortest time -> less blood).
State/tail memoization is used so repeated S-scan tails are not re-simulated.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import (
    _scan_rank,
    rollout_clinical_policy,
    serpentine_macro_target_policy,
)
from clinical_window_environment import _EPSILON
from planner import neighbors4

DEFAULT_GATE_CLINICAL_CONFIG: Mapping[str, Any] = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}


def masked_global_pool(
    region: np.ndarray,
    mask: set[tuple[int, int]],
) -> float:
    """Mean over a masked set of grid cells, ignoring everything outside.

    Used by the (future) global-context candidate scorer; the mask is the real
    domain or frontier set so padded/empty grid cells can never dilute the
    pooled value (guide Section 10, item 11).  ``region`` is (H, W) float32.
    """
    if not mask:
        return 0.0
    values = [float(region[row, col]) for row, col in mask]
    return float(sum(values) / len(values))


def reference_global_scorer(
    local_patch: np.ndarray,
    global_frontier_frac: float,
    global_phase_frac: float,
) -> float:
    """Minimal global-context reference scorer for the Section 10.12 contract.

    Confirms that the *same* local patch can produce a different logit when the
    pooled global frontier / phase context differs.  Gate B's shared scorer must
    keep this property (local neighbourhood identical, global context different
    -> different candidate logit).
    """
    local = float(np.asarray(local_patch, dtype=np.float32).mean())
    return local + 0.5 * float(global_frontier_frac) + 0.3 * float(global_phase_frac)

# Tail cache phase_elapsed quantization (minutes). 0.1 min keeps phase
# boundaries accurate enough for blood integration while making repeats hit.
PHASE_ELAPSED_ROUND = 1
# Elapsed-time quantization for the truncation guard (max_episode_minutes).
ELAPSED_ROUND = 1
_BIG = 1e9


def make_gate_rollout(
    selector: Callable[[ClinicalMacroResectionEnv], int],
    *,
    clinical_config: Optional[Mapping[str, Any]] = None,
    reward_config: Optional[Mapping[str, Any]] = None,
    control_mode: str = "macro",
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Return a single-scenario rollout bound to the shared gate config."""
    cfg = dict(DEFAULT_GATE_CLINICAL_CONFIG)
    if clinical_config:
        cfg.update(clinical_config)

    def run(scenario: Mapping[str, Any]) -> dict[str, Any]:
        return rollout_clinical_policy(
            scenario,
            selector,
            clinical_config=cfg,
            reward_config=reward_config,
            mechanics_update_interval=0,
            control_mode=control_mode,
        )

    return run


def nearest_frontier_macro_policy(env: ClinicalMacroResectionEnv) -> int:
    """Greedy nearest frontier (transfer distance through already-cut region)."""
    legal = sorted(env._frontier())
    if not legal:
        raise RuntimeError("Nearest-frontier controller has no legal target")
    counts = env._transfer_counts()

    def key(cell):
        return (counts.get(cell, _BIG), _scan_rank(env, cell))

    best = min(legal, key=key)
    return int(best[0] * env.max_cols + best[1])


def serpentine_target_of(env: ClinicalMacroResectionEnv) -> tuple[int, int]:
    """The mechanical S-scan next target as a cell, for planner fallback."""
    legal = sorted(env._frontier())
    if not legal:
        raise RuntimeError("No legal frontier target for S-scan fallback")
    return min(legal, key=lambda cell: _scan_rank(env, cell))


def _mpc_tail_worker(payload: tuple) -> tuple[float, float, bool, Optional[str]]:
    """Worker for parallel depth-1 MPC: rebuild env from state, run one real
    macro action, then complete the episode with the frozen S-scan tail.

    Returns (delta_T_min, delta_B_ml, completion, failure_reason) for the whole
    episode measured from the given state.  Payload is
    ``(scenario, state_dict, target, clinical_config)``.
    """
    scenario, state, target, clinical_config = payload
    e = ClinicalMacroResectionEnv(scenario=scenario, clinical_config=clinical_config)
    e.reset()
    e.__dict__.update(state)
    e.events = []
    t_start = e.elapsed_minutes
    b_start = e.expected_blood_loss_ml
    _step_macro_target(e, target)
    if e.terminated or e.truncated:
        completion = bool(e.terminated and e.failure_reason is None)
        return (e.elapsed_minutes - t_start, e.expected_blood_loss_ml - b_start,
                completion, e.failure_reason)
    while not e.terminated and not e.truncated:
        legal = e._frontier()
        if not legal:
            e.terminated = True
            e.failure_reason = "mpc tail lost all legal targets"
            break
        _step_macro_target(e, min(legal, key=lambda cell: _scan_rank(e, cell)))
    completion = bool(e.terminated and e.failure_reason is None)
    return (e.elapsed_minutes - t_start, e.expected_blood_loss_ml - b_start,
            completion, e.failure_reason)


def _env_state_payload(env: ClinicalMacroResectionEnv) -> dict:
    """Pickle-friendly mutable state of an env for the parallel tail worker."""
    return {
        "cut": set(env.cut),
        "current": env.current,
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
    }


def _clone_env(env: ClinicalMacroResectionEnv) -> ClinicalMacroResectionEnv:
    """Cheap deterministic clone: share immutable geometry, copy mutable state."""
    e = copy.copy(env)
    e.cut = set(env.cut)
    e.hidden_ids = set(env.hidden_ids)
    e.exposed_ids = set(env.exposed_ids)
    e.sealed_ids = set(env.sealed_ids)
    e.current = env.current
    e.previous_direction_position = env.previous_direction_position
    e.events = []  # dropped for speed; tail only needs T/B/completion
    e.mechanics = dict(env.mechanics)
    # components / component_by_cell / clinical_config / reward_config are
    # treated as read-only after reset and are shared deliberately.
    return e


def _step_macro_target(
    env: ClinicalMacroResectionEnv,
    target: tuple[int, int],
) -> None:
    """Execute one macro target with the exact physical semantics of
    ``ClinicalMacroResectionEnv.step`` (transfer via already-cut region, per-cell
    transfer billing, cutting/sealing, exposure release, phase integration) but
    WITHOUT building the observation, action-mask, reward or mechanics outputs.

    This is the fast inner loop used by the S-scan tail and the beam search.
    ``tests/test_clinical_v104.py`` verifies it matches ``env.step`` exactly on
    elapsed time, blood loss, cut set, phase and transfer counters.
    """
    route = env._transfer_path(target)
    step_events: list[dict[str, Any]] = []
    if not route:
        env.terminated = True
        env.failure_reason = "no cut-region transfer path to selected frontier"
        return
    for cell in route[1:]:
        source = env.current
        env._advance_time(env.base_action_minutes, step_events)
        env.current = cell
        env.previous_direction_position = source
        env.transfer_count += 1
        env.direction_action_count += 1
    source = env.current
    duration = env._action_duration(target)
    env._advance_time(duration, step_events)
    if target in env._exposed_cells():
        component_id = env.component_by_cell[target]
        component = env._component(component_id)
        env.exposed_ids.remove(component_id)
        env.sealed_ids.add(component_id)
        env.cut.update(component["cells"])
    else:
        env.cut.add(target)
    env.current = target
    env.previous_direction_position = source
    env.direction_action_count += 1
    env._release_ready_components(step_events)
    env.step_count += 1
    if env.cut == env.domain:
        env.terminated = True
    elif env.elapsed_minutes >= env.clinical_config["max_episode_minutes"] - _EPSILON:
        env.truncated = True
        env.failure_reason = "maximum episode time reached"
    elif env.step_count >= env.max_steps:
        env.truncated = True
        env.failure_reason = f"maximum macro step count ({env.max_steps}) reached"


class SerpentineTail:
    """Memoized frozen S-scan tail completing an episode from a given state.

    ``tail(env)`` returns ``(delta_T_min, delta_B_ml, completion, failure_reason)``
    for the deterministic S-scan policy from the *current* env state onwards.
    The cache key captures everything that affects the tail's future evolution:
    the cut set, current cell, clamp phase, phase elapsed time, vessel
    hidden/exposed partition, and elapsed time (for the 240 min truncation).
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

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _state_key(self, env: ClinicalMacroResectionEnv) -> tuple:
        return (
            frozenset(env.cut),
            tuple(env.current),
            env.phase,
            round(env.phase_elapsed_minutes, PHASE_ELAPSED_ROUND),
            frozenset(env.hidden_ids),
            frozenset(env.exposed_ids),
            round(env.elapsed_minutes, ELAPSED_ROUND),
        )

    def tail(self, env: ClinicalMacroResectionEnv) -> tuple[float, float, bool, Optional[str]]:
        key = self._state_key(env)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        e = _clone_env(env)
        t0 = e.elapsed_minutes
        b0 = e.expected_blood_loss_ml
        while not e.terminated and not e.truncated:
            legal = e._frontier()
            if not legal:
                e.terminated = True
                e.failure_reason = "serpentine tail lost all legal targets"
                break
            target = min(legal, key=lambda cell: _scan_rank(e, cell))
            if target not in e._frontier():
                e.terminated = True
                e.failure_reason = "serpentine tail selected non-frontier target"
                break
            _step_macro_target(e, target)
        dt = e.elapsed_minutes - t0
        db = e.expected_blood_loss_ml - b0
        completion = bool(e.terminated and e.failure_reason is None)
        result = (float(dt), float(db), completion, e.failure_reason)
        self._cache[key] = result
        return result


def candidate_targets(
    env: ClinicalMacroResectionEnv,
    *,
    count: int,
    transfer_counts: Optional[Mapping[tuple[int, int], int]] = None,
) -> list[tuple[int, int]]:
    """Candidate frontier union per guide Section 6.3.

    Candidates are drawn from four overlapping ideas: nearest (low transfer),
    sealable exposed vessel cells (immediate sealing), near-hidden-vessel cells
    (high exposure risk / early sealing), and the S-scan ordering.  Duplicates
    are dropped preserving first-seen order, then truncated to ``count``.
    """
    frontier = env._frontier()
    if not frontier:
        return []
    if transfer_counts is None:
        transfer_counts = env._transfer_counts()
    exposed = env._exposed_cells()
    hidden = env._hidden_cells()

    def near_hidden(cell: tuple[int, int]) -> bool:
        return any(neighbor in hidden for neighbor in neighbors4(cell))

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(cells: Sequence[tuple[int, int]]) -> None:
        for cell in cells:
            if cell not in seen:
                seen.add(cell)
                ordered.append(cell)

    # 1. Sealable exposed vessel cells: sealing removes the bleeding source, so
    #    these are the highest-value targets whenever a vessel is exposed.
    add(sorted(frontier & exposed, key=lambda c: (transfer_counts.get(c, _BIG), _scan_rank(env, c))))
    # 2. Nearest (lowest transfer distance through already-cut region).
    add(sorted(frontier, key=lambda c: (transfer_counts.get(c, _BIG), _scan_rank(env, c))))
    # 3. Near-hidden-vessel cells (early exposure / near-term sealing).
    add(sorted((c for c in frontier if near_hidden(c)), key=lambda c: (transfer_counts.get(c, _BIG), _scan_rank(env, c))))
    # 4. S-scan ordering as a conservative fallback.
    add(sorted(frontier, key=lambda c: _scan_rank(env, c)))
    return ordered[:count]


class WindowAwarePlanner:
    """Finite look-ahead beam/MPC planner (guide Sections 6.2-6.3).

    ``select(env, baseline_blood)`` performs a receding-horizon search: it
    expands each beam node's candidate branches one real macro action, prunes
    intermediate layers with a cheap cumulative-cost heuristic, and on the leaf
    layer completes every branch with the frozen S-scan tail.  The best first
    target is chosen by the frozen rule order (feasibility -> blood margin ->
    shortest total time -> less blood).  If no feasible branch exists it falls
    back to the S-scan next action.
    """

    def __init__(
        self,
        *,
        candidate_count: int = 8,
        beam_width: int = 8,
        lookahead_depth: int = 4,
        margin_blood_ml: Optional[float] = None,
        tail: Optional[SerpentineTail] = None,
        leaf_pool: Any = None,
        clinical_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.candidate_count = int(candidate_count)
        self.beam_width = int(beam_width)
        self.lookahead_depth = int(lookahead_depth)
        self.margin_blood_ml = margin_blood_ml
        self.tail = tail if tail is not None else SerpentineTail()
        self.leaf_pool = leaf_pool
        self.clinical_config_used = dict(DEFAULT_GATE_CLINICAL_CONFIG)
        if clinical_config:
            self.clinical_config_used.update(clinical_config)
        self.nodes_expanded = 0
        self.leaf_count = 0
        self.tree_count = 0

    def select(self, env: ClinicalMacroResectionEnv, baseline_blood: Optional[float] = None) -> int:
        traj = self._plan(env, baseline_blood)
        first = traj[0] if traj else serpentine_target_of(env)
        return int(first[0] * env.max_cols + first[1])

    def plan_sequence(
        self,
        env: ClinicalMacroResectionEnv,
        n_actions: int,
        baseline_blood: Optional[float] = None,
    ) -> list[tuple[int, int]]:
        """Return the first ``n_actions`` targets of the best plan (receding
        horizon of width ``n_actions``; re-plan every ``n_actions`` steps)."""
        traj = self._plan(env, baseline_blood)
        if not traj:
            return [serpentine_target_of(env)] * n_actions
        if len(traj) >= n_actions:
            return traj[:n_actions]
        return traj + [traj[-1]] * (n_actions - len(traj))

    def _plan(
        self,
        env: ClinicalMacroResectionEnv,
        baseline_blood: Optional[float],
    ) -> list[tuple[int, int]]:
        self.tree_count += 1
        self._baseline_blood = baseline_blood
        if self.lookahead_depth <= 0:
            return [serpentine_target_of(env)]
        if self.lookahead_depth == 1:
            return self._plan_mpc(env, baseline_blood)

        # beam items: (env_clone, trajectory, cum_T, cum_B, done)
        beam: list[tuple[Any, list[tuple[int, int]], float, float, bool]] = [
            (_clone_env(env), [], 0.0, 0.0, False)
        ]
        for depth in range(self.lookahead_depth):
            expanded: list[tuple[Any, list[tuple[int, int]], float, float, bool]] = []
            for e, traj, cum_t, cum_b, done in beam:
                if done:
                    expanded.append((e, traj, cum_t, cum_b, done))
                    continue
                counts = e._transfer_counts()
                frontier = e._frontier()
                for target in candidate_targets(e, count=self.candidate_count, transfer_counts=counts):
                    if target not in frontier:
                        continue
                    e2 = _clone_env(e)
                    t0 = e2.elapsed_minutes
                    b0 = e2.expected_blood_loss_ml
                    _step_macro_target(e2, target)
                    dt = e2.elapsed_minutes - t0
                    db = e2.expected_blood_loss_ml - b0
                    done2 = bool(e2.terminated or e2.truncated)
                    expanded.append((e2, traj + [target], cum_t + dt, cum_b + db, done2))
            if not expanded:
                break
            if depth == self.lookahead_depth - 1:
                leaves: list[tuple[list[tuple[int, int]], float, float, bool, Optional[str]]] = []
                for e, traj, cum_t, cum_b, done in expanded:
                    if done:
                        completion = bool(e.terminated and e.failure_reason is None)
                        leaves.append((traj, cum_t, cum_b, completion, e.failure_reason))
                    else:
                        tail_t, tail_b, completion, reason = self.tail.tail(e)
                        leaves.append((traj, cum_t + tail_t, cum_b + tail_b, completion, reason))
                self.leaf_count += len(leaves)
                chosen = self._pick(leaves, baseline_blood)
                if chosen is not None:
                    return chosen
                # No feasible branch -> S-scan fallback (guide Section 6.2).
                return [serpentine_target_of(env)]
            beam = self._prune(expanded, self.beam_width)
        return [serpentine_target_of(env)]

    def _plan_mpc(
        self,
        env: ClinicalMacroResectionEnv,
        baseline_blood: Optional[float],
    ) -> list[tuple[int, int]]:
        """Depth-1 MPC (guide Section 6.2): clone env, execute the real macro
        action, complete the episode with the frozen S-scan tail, then pick by
        the frozen rule order (feasibility -> blood margin -> shortest time).

        This avoids the intermediate-beam pruning short-sightedness: sealing an
        exposed vessel costs one slower step now but removes the entire future
        bleeding source, which only a full-tail evaluation can see.
        """
        counts = env._transfer_counts()
        frontier = env._frontier()
        targets = [
            t for t in candidate_targets(env, count=self.candidate_count, transfer_counts=counts)
            if t in frontier
        ]
        if self.leaf_pool is not None and len(targets) > 1:
            state = _env_state_payload(env)
            payloads = [(env.scenario, state, tuple(target), dict(self.clinical_config_used))
                        for target in targets]
            results = self.leaf_pool.map(_mpc_tail_worker, payloads)
            leaves = [
                ([target], dt, db, completion, reason)
                for target, (dt, db, completion, reason) in zip(targets, results)
            ]
        else:
            leaves = []
            for target in targets:
                e2 = _clone_env(env)
                t0 = e2.elapsed_minutes
                b0 = e2.expected_blood_loss_ml
                _step_macro_target(e2, target)
                dt = e2.elapsed_minutes - t0
                db = e2.expected_blood_loss_ml - b0
                if e2.terminated or e2.truncated:
                    completion = bool(e2.terminated and e2.failure_reason is None)
                    leaves.append(([target], dt, db, completion, e2.failure_reason))
                else:
                    tail_t, tail_b, completion, reason = self.tail.tail(e2)
                    leaves.append(([target], dt + tail_t, db + tail_b, completion, reason))
        self.leaf_count += len(leaves)
        chosen = self._pick(leaves, baseline_blood)
        if chosen is not None:
            return chosen
        return [serpentine_target_of(env)]

    def _prune(
        self,
        expanded: Sequence[tuple[Any, list[tuple[int, int]], float, float, bool]],
        width: int,
    ) -> list[tuple[Any, list[tuple[int, int]], float, float, bool]]:
        """Prune inside the blood margin, then by cumulative time.

        The final pick only ever selects branches whose full-episode blood stays
        within the margin, so intermediate pruning must NOT throw away the fast
        branches -- only branches that already exceed the per-scene blood
        threshold (baseline blood + margin) are deprioritised.  Within the
        threshold the fastest branches are kept first (guide: shortest total
        time is the primary objective subject to blood non-inferiority)."""
        threshold = None
        base = getattr(self, "_baseline_blood", None)
        if base is not None and self.margin_blood_ml is not None:
            threshold = base + self.margin_blood_ml

        def key(item):
            e, _traj, cum_t, cum_b, done = item
            done_flag = 0 if done else 1
            fail_flag = 0 if (done and e.failure_reason is None) else (1 if done else 0)
            over = 0 if (threshold is None or cum_b <= threshold) else 1
            return (done_flag, fail_flag, over, cum_t, cum_b)

        ranked = sorted(expanded, key=key)
        self.nodes_expanded += len(ranked)
        return ranked[:width]

    def _pick(
        self,
        leaves: Sequence[tuple[list[tuple[int, int]], float, float, bool, Optional[str]]],
        baseline_blood: Optional[float],
    ) -> Optional[list[tuple[int, int]]]:
        """Frozen branch-selection rule order (guide Section 6.2). Returns the
        full best trajectory (whose first target is executed), or None when no
        feasible branch exists (caller falls back to the S-scan next action --
        never select a branch known to violate the blood margin)."""
        feasible = [leaf for leaf in leaves if leaf[3] and leaf[4] is None]
        if not feasible:
            return None
        if baseline_blood is not None and self.margin_blood_ml is not None:
            within = [leaf for leaf in feasible if leaf[2] <= baseline_blood + self.margin_blood_ml]
            if not within:
                return None  # all branches exceed the margin -> S-scan fallback
            feasible = within
        return min(feasible, key=lambda leaf: (leaf[1], leaf[2]))[0]


def rollout_planner(
    scenario: Mapping[str, Any],
    planner: WindowAwarePlanner,
    *,
    baseline_blood: Optional[float] = None,
    replan_interval: int = 8,
    clinical_config: Optional[Mapping[str, Any]] = None,
    reward_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Deterministic window-aware planner rollout for one scenario.

    Receding-horizon control: every ``replan_interval`` macro actions the
    planner re-searches and returns a short sequence; the intermediate actions
    are executed without re-planning.  ``replan_interval=1`` reduces to strict
    per-step re-planning (guide Section 6.2 semantics).
    """
    cfg = dict(DEFAULT_GATE_CLINICAL_CONFIG)
    if clinical_config:
        cfg.update(clinical_config)
    from clinical_macro_environment import ClinicalMacroResectionEnv as _Env

    env = _Env(scenario=scenario, clinical_config=cfg, reward_config=reward_config,
               mechanics_update_interval=0)
    env.reset()
    pending: list[tuple[int, int]] = []
    expected_rates: list[float] = []
    while not env.terminated and not env.truncated:
        # Re-plan immediately when a horizon target becomes invalid (e.g. a
        # vessel seal cut several cells at once, absorbing the next targets).
        # Do NOT pay an S-scan step here -- that would bake in the baseline's
        # slowness; the guide's S-scan fallback is reserved for "no feasible
        # branch" from _pick, which already happened inside plan_sequence.
        while not pending and not env.terminated and not env.truncated:
            pending = planner.plan_sequence(env, n_actions=replan_interval,
                                            baseline_blood=baseline_blood)
            if not pending:
                break
        target = pending.pop(0)
        if target not in env._frontier():
            # Still stale after a fresh plan; drop this horizon and re-plan.
            pending = []
            continue
        expected_rates.append(float(env._expected_bleeding_rate()))
        _step_macro_target(env, target)

    record = {
        "scenario_id": scenario.get("scenario_id"),
        "status": "ok" if env.terminated and env.failure_reason is None else "failed",
        "failure_reason": env.failure_reason,
        "completion": env.cut == env.domain,
        "coverage": len(env.cut) / len(env.domain),
        "legal_action_rate": 1.0,
        "episode_steps": env.step_count,
        "direction_action_count": env.direction_action_count,
        "macro_action_count": env.step_count,
        "max_macro_duration_minutes": float(env.max_macro_duration_minutes),
        "transfer_count": env.transfer_count,
        "transfer_overhead": env.transfer_count / max(1, env.direction_action_count),
        "no_progress_streak": env.no_progress_streak,
        "max_no_progress_streak": env.max_no_progress_streak,
        "stagnation_failure": str(env.failure_reason or "").startswith("stagnation:"),
        "same_edge_streak": env.same_edge_streak,
        "max_same_edge_streak": env.max_same_edge_streak,
        "two_cell_loop_failure": str(env.failure_reason or "").startswith("two-cell oscillation:"),
        "elapsed_minutes": env.elapsed_minutes,
        "expected_blood_loss_ml": env.expected_blood_loss_ml,
        "peak_expected_bleeding_rate_ml_per_min": env.peak_expected_bleeding_rate_ml_per_min,
        "mean_expected_bleeding_rate_ml_per_min": float(
            sum(expected_rates) / len(expected_rates) if expected_rates else 0.0
        ),
        "unclamped_exposed_minutes": env.unclamped_exposed_minutes,
        "total_clamped_minutes": env.total_clamped_minutes,
        "total_unclamped_minutes": env.total_unclamped_minutes,
        "clamp_cycle_count": env.clamp_cycle_count,
        "early_end_count": env.early_end_count,
        "sealed_vessel_count": len(env.sealed_ids),
        "total_reward": 0.0,
        "reward_components": {},
        "max_front_tension": float(env.mechanics["peak_front_tension"]),
        "max_organ_energy": float(env.mechanics["peak_organ_energy"]),
        "max_vessel_strain": float(env.mechanics["peak_vessel_strain"]),
        "clamp_rule_violations": 0,
        "unclamp_rule_violations": 0,
        "planner_trees": planner.tree_count,
        "planner_leaves": planner.leaf_count,
        "planner_nodes": planner.nodes_expanded,
        "tail_cache_size": planner.tail.cache_size,
    }
    return record

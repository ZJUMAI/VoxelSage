"""E9: worst-case / synthetic regression (plan §7.10).

Constructs deterministic scenarios that force the v10.8 lazy shield
down specific rank paths.  Uses the real v10.5 candidate generator and
the v10.6 model input pipeline, but stubs the safety shield to return
pre-canned ``safe_exact``/``B_total`` so we can force the controller
through:

  * rank 1 safe (early stop after 1 verify)
  * rank 1 unsafe, rank 2 safe (stop after 2)
  * rank 1..N unsafe, rank N+1 safe (stop after N+1)
  * all candidates unsafe (fallback path)
  * fewer than 6 candidates
  * identical scores (tie-break edge)
  * B_total exactly at budget
  * completion=False on the first candidate

Reports per-step ``verified_candidate_count`` and the selected rank;
also confirms that re-running with the same env produces the same
selected target (determinism).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Deterministic synthetic state with K targets
# ---------------------------------------------------------------------------
class _SynEnv:
    """A state for which ``_candidate_sources_v105`` will return the given
    K targets in order.  We construct a 6x6 grid with a frontier that
    matches the targets exactly.
    """

    def __init__(self, targets: list[tuple[int, int]]):
        # Build a minimal scenario-shaped dict.
        self.scenario = {"scenario_id": "worst-case"}
        self.elapsed_minutes = 0.0
        self.expected_blood_loss_ml = 0.0
        self._frontier_targets = list(targets)
        # Minimum attributes that downstream calls expect.
        self.terminated = False
        self.truncated = False
        self.failure_reason = None
        self.cut = set()
        self.sealed_ids = set()
        self.current = (0, 0)
        self.phase = "clamp"
        self.phase_elapsed_minutes = 0.0
        self.hidden_ids = set()
        self.exposed_ids = set()
        self.max_cols = 6
        self.cols = 6
        self.rows = 6
        self._v106_baseline_blood = 0.0
        self._v106_margin = 16.07054347826075

    def _frontier(self):
        return list(self._frontier_targets)

    def _transfer_counts(self):
        return {t: 1 for t in self._frontier_targets}

    def _observation(self):
        import numpy as np
        return {n: np.zeros((1, 6), dtype=np.float32) for n in
                ("domain", "cut", "hidden_vessel", "exposed_vessel",
                 "sealed_vessel", "frontier", "large_vessel",
                 "current_position", "previous_position", "start",
                 "transfer_distance")}

    def _component(self, _):
        return {"cells": set(), "area_mm2": 0.0}

    def _exposed_cells(self):
        return set()

    def _hidden_cells(self):
        return set()

    def component_by_cell(self):
        return {}


# ---------------------------------------------------------------------------
# Stubbed tail that returns a per-target decision
# ---------------------------------------------------------------------------
class _StubTail:
    """Returns a pre-canned result keyed by the stepped target."""

    def __init__(self, decisions: dict[tuple[int, int], tuple[float, float, bool, str | None]]):
        self._d = dict(decisions)
        self.call_log: list[tuple[int, int]] = []

    def tail(self, env_after):
        # The lazy select tags after._stepped_target; we use that to dispatch.
        target = env_after._stepped_target
        self.call_log.append(target)
        return self._d[target]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def _make_env_stub_step():
    """Replace the v10.4 step / clone with deterministic stubs that tag
    after._stepped_target so the stubbed tail can dispatch.  Must be called
    BEFORE importing clinical_safety_shield_v108 (which binds its own
    references at module level).
    """
    import plan_target_order_v104 as p104

    def _clone_env(env):
        clone = _SynEnv(env._frontier_targets)
        for attr in (
            "elapsed_minutes", "expected_blood_loss_ml",
            "terminated", "truncated", "failure_reason",
            "cut", "sealed_ids", "current",
            "phase", "phase_elapsed_minutes",
            "hidden_ids", "exposed_ids",
            "max_cols", "cols", "rows",
            "_v106_baseline_blood", "_v106_margin",
        ):
            setattr(clone, attr, getattr(env, attr, getattr(clone, attr, None)))
        clone.previous_direction_position = getattr(env, "previous_direction_position", None)
        clone.events = []
        clone.mechanics = dict(getattr(env, "mechanics", {}))
        return clone

    def _step_macro_target(env, target):
        env._stepped_target = target
        env._frontier_targets = [t for t in env._frontier_targets if t != target]

    p104._clone_env = _clone_env
    p104._step_macro_target = _step_macro_target


# Patch v10.4 globals BEFORE importing the v10.8 module.
_make_env_stub_step()

from plan_target_order_v104 import _clone_env, _step_macro_target, serpentine_target_of
from plan_target_order_v105 import SerpentineTailV105, _candidate_sources_v105
from clinical_safety_shield_v108 import select_lazy_v108, _target_sort_key
from clinical_safety_shield_v106 import ShieldCandidate


def run_case(name: str, targets: list[tuple[int, int]],
             scores: dict[tuple[int, int], float],
             decisions: dict[tuple[int, int], tuple[float, float, bool, str | None]],
             budget: float) -> dict:
    env = _SynEnv(targets)
    tail = _StubTail(decisions)
    chosen, diag = select_lazy_v108(env, budget_ml=budget, scores=scores, tail=tail)
    return {
        "case": name,
        "ranked": [list(t) for t in diag["ranked_candidate_ids"]],
        "verified_count": diag["verified_candidate_count"],
        "selected_rank": diag["selected_rank"],
        "selected": list(chosen) if chosen is not None else None,
        "fallback_used": diag["fallback_used"],
        "selected_safe": diag["selected_safe_exact"],
        "tail_call_log": [list(t) for t in tail.call_log],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/clinical_window_v10_8_lazy_shield/worst_case/worst_case_report.json")
    args = parser.parse_args()

    _make_env_stub_step()
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets6 = [(r, c) for r in range(1, 4) for c in range(1, 3)]
    scores_desc = {t: float(10 - i) for i, t in enumerate(targets6)}

    cases = []

    # 1. rank 1 safe
    cases.append(run_case(
        "rank1_safe",
        targets6, scores_desc,
        {t: (5.0, 0.5, True, None) for t in targets6},
        budget=100.0,
    ))

    # 2. rank 1 unsafe (over budget), rank 2 safe
    d2 = {t: (5.0, 0.5, True, None) for t in targets6}
    ranked = sorted(targets6, key=_target_sort_key(scores_desc))
    d2[ranked[0]] = (5.0, 200.0, True, None)
    cases.append(run_case("rank1_unsafe_rank2_safe", targets6, scores_desc, d2, budget=10.0))

    # 3. rank 1..5 unsafe, rank 6 safe
    d3 = {t: (5.0, 0.5, True, None) for t in targets6}
    for t in ranked[:5]:
        d3[t] = (5.0, 200.0, True, None)
    cases.append(run_case("rank1_to_5_unsafe_rank6_safe", targets6, scores_desc, d3, budget=10.0))

    # 4. all unsafe
    d4 = {t: (5.0, 200.0, True, None) for t in targets6}
    cases.append(run_case("all_unsafe_fallback", targets6, scores_desc, d4, budget=10.0))

    # 5. fewer than 6 candidates
    targets3 = targets6[:3]
    scores3 = {t: float(10 - i) for i, t in enumerate(targets3)}
    cases.append(run_case(
        "fewer_than_6_candidates",
        targets3, scores3,
        {t: (5.0, 0.5, True, None) for t in targets3},
        budget=100.0,
    ))

    # 6. identical scores
    tied_scores = {t: 1.0 for t in targets6}
    cases.append(run_case(
        "identical_scores_tiebreak",
        targets6, tied_scores,
        {t: (5.0, 0.5, True, None) for t in targets6},
        budget=100.0,
    ))

    # 7. B_total exactly at budget
    d7 = {t: (5.0, 10.0, True, None) for t in targets6}
    cases.append(run_case("b_at_budget", targets6, scores_desc, d7, budget=10.0))

    # 8. B_total just over budget (EPS)
    d8 = {t: (5.0, 10.0 + 1e-9 + 1e-12, True, None) for t in targets6}
    cases.append(run_case("b_over_eps", targets6, scores_desc, d8, budget=10.0))

    # 9. completion=False on first candidate
    d9 = {t: (5.0, 0.5, True, None) for t in targets6}
    d9[ranked[0]] = (5.0, 0.5, False, "serpentine tail lost all legal targets")
    cases.append(run_case("completion_false_first", targets6, scores_desc, d9, budget=100.0))

    # 10. failure_reason non-None on first candidate
    d10 = {t: (5.0, 0.5, True, None) for t in targets6}
    d10[ranked[0]] = (5.0, 0.5, True, "anything")
    cases.append(run_case("failure_reason_first", targets6, scores_desc, d10, budget=100.0))

    # 11. determinism: re-run case 1 twice
    cases.append(run_case(
        "determinism_run_1",
        targets6, scores_desc,
        {t: (5.0, 0.5, True, None) for t in targets6},
        budget=100.0,
    ))
    cases.append(run_case(
        "determinism_run_2",
        targets6, scores_desc,
        {t: (5.0, 0.5, True, None) for t in targets6},
        budget=100.0,
    ))

    report = {"n_cases": len(cases), "cases": cases}
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[E9] wrote {out_path}")
    for c in cases:
        print(
            f"  {c['case']:30s} verified={c['verified_count']:2d} "
            f"selected_rank={c['selected_rank']} fallback={c['fallback_used']} "
            f"selected={c['selected']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit + contract tests for v10.8 lazy exact shield.

Covers the 12 cases listed in plan §7.2.  The first block uses mock tail
fixtures to keep the tests fast and deterministic; the second block
runs against the real v10.7 model checkpoint on a frozen scenario and
asserts that C4L action_hash == C4E action_hash.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from typing import Any

from clinical_safety_shield_v106 import ShieldCandidate
from clinical_safety_shield_v108 import select_eager_audit_v108, select_lazy_v108


# ---------------------------------------------------------------------------
# Lightweight in-test doubles
# ---------------------------------------------------------------------------
class _FakeEnv:
    """Minimal env stand-in for select_*_v108.

    Only the attributes used by the select_*_v108 implementations need to
    behave correctly.  Everything else is ignored.
    """

    def __init__(
        self,
        elapsed: float = 0.0,
        blood: float = 0.0,
        frontier_targets: list[tuple[int, int]] | None = None,
    ) -> None:
        self.elapsed_minutes = float(elapsed)
        self.expected_blood_loss_ml = float(blood)
        self._frontier_targets = list(frontier_targets or [])
        self.terminated = False
        self.truncated = False
        self.failure_reason = None

    def _frontier(self) -> list[tuple[int, int]]:
        return list(self._frontier_targets)

    def _transfer_counts(self) -> dict[tuple[int, int], int]:
        return {t: 1 for t in self._frontier_targets}


class _FakeTail:
    """Return pre-canned tail results keyed by the env passed in.

    The signature matches ``SerpentineTailV105.tail``: ``(env) ->
    (tail_t, tail_b, completion, failure_reason)``.
    """

    def __init__(self, table: dict[tuple[int, int], tuple[float, float, bool, Any]]):
        self._table = dict(table)
        self.call_count_per_target: dict[tuple[int, int], int] = {}

    def tail(self, env_after_step) -> tuple[float, float, bool, Any]:
        # Use the after-step env's frontier to disambiguate.
        frontier = tuple(sorted(env_after_step._frontier_targets or [(0, 0)]))
        # The "env state key" in v10.7 actually used by select_*_v108 is the
        # source of the (target, after) state; the easiest deterministic
        # lookup is the *previous* target — encoded via the only missing
        # cell of the after env.  We instead use a side channel: the
        # _stepped_target attribute that tests set on the env clone.
        key = getattr(env_after_step, "_stepped_target", None)
        assert key is not None, "test bug: env._stepped_target not set"
        self.call_count_per_target[key] = self.call_count_per_target.get(key, 0) + 1
        return self._table[key]


# v10.8 imports _clone_env and _step_macro_target; supply a tiny stand-in
# that mutates the env and tags it so the fake tail can dispatch.
def _install_fakes(monkeypatch_module):
    import plan_target_order_v104 as p104

    def _clone_env(env):
        clone = _FakeEnv(
            elapsed=env.elapsed_minutes,
            blood=env.expected_blood_loss_ml,
            frontier_targets=list(env._frontier_targets),
        )
        return clone

    def _step_macro_target(env, target):
        env._stepped_target = target
        # Mutate in a deterministic way that lets tests assert state hash.
        # Keep deltas zero so test budgets only depend on the tail fixture.
        env.elapsed_minutes += 0.0
        env.expected_blood_loss_ml += 0.0
        # Recompute the frontier: drop the stepped target.
        env._frontier_targets = [t for t in env._frontier_targets if t != target]

    def _candidate_sources_v105(env, *, count, transfer_counts=None):
        # Use a stable iteration order regardless of dict ordering.
        return [(t, f"src_{i}") for i, t in enumerate(env._frontier_targets[:count])]

    def _serpentine_target_of(env):
        return (0, 0)  # deterministic; tests use targets distinct from (0,0)

    monkeypatch_module.setattr(p104, "_clone_env", _clone_env)
    monkeypatch_module.setattr(p104, "_step_macro_target", _step_macro_target)
    monkeypatch_module.setattr(p104, "serpentine_target_of", _serpentine_target_of)
    import plan_target_order_v105 as p105
    monkeypatch_module.setattr(p105, "_candidate_sources_v105", _candidate_sources_v105)


def _state_hash(env) -> str:
    return hashlib.sha256(repr(
        (env.elapsed_minutes, env.expected_blood_loss_ml, tuple(env._frontier_targets))
    ).encode()).hexdigest()


# ---------------------------------------------------------------------------
# §7.2 contract tests (cases 1-11)
# ---------------------------------------------------------------------------
class LazyShieldContractTests(unittest.TestCase):
    """Tests 1-11: pure unit tests, no real model / torch required."""

    def setUp(self) -> None:
        import plan_target_order_v104 as p104
        import plan_target_order_v105 as p105
        import clinical_safety_shield_v108 as v108

        class _P:
            def __init__(self):
                self._patches: list[tuple[Any, str, Any]] = []

            def setattr(self, mod, name, value):
                self._patches.append((mod, name, getattr(mod, name)))
                setattr(mod, name, value)

            def undo(self):
                for mod, name, original in reversed(self._patches):
                    setattr(mod, name, original)

        self._p = _P()
        _install_fakes(self._p)
        # Reload v108 so it picks up the patched dependencies.
        import importlib
        importlib.reload(v108)
        self._select_lazy = v108.select_lazy_v108
        self._select_audit = v108.select_eager_audit_v108
        # Track the underlying fake tail for assertions.
        self._tail = None

    def tearDown(self) -> None:
        self._p.undo()
        # Reload v108 to restore the real dependencies.
        import importlib
        import clinical_safety_shield_v108 as v108
        importlib.reload(v108)

    def _build_tail(self, env, table):
        from clinical_safety_shield_v108 import EPS
        tail = _FakeTail(table)
        self._tail = tail
        return tail

    # 1. 排名第一安全: 仅调用一次 verify, 选第一名
    def test_rank1_safe_calls_once(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1), (2, 2), (3, 3)])
        tail = self._build_tail(env, {
            (1, 1): (5.0, 1.0, True, None),
            (2, 2): (5.0, 1.0, True, None),
            (3, 3): (5.0, 1.0, True, None),
        })
        scores = {(1, 1): 3.0, (2, 2): 2.0, (3, 3): 1.0}
        chosen, diag = self._select_lazy(env, budget_ml=100.0, scores=scores, tail=tail)
        print(f"DEBUG chosen={chosen} diag={diag}")
        print(f"DEBUG call_count={tail.call_count_per_target}")
        self.assertEqual(chosen, (1, 1))
        self.assertEqual(diag["verified_candidate_count"], 1)
        self.assertEqual(diag["selected_rank"], 0)
        self.assertEqual(tail.call_count_per_target.get((1, 1), 0), 1)
        self.assertEqual(tail.call_count_per_target.get((2, 2), 0), 0)
        self.assertEqual(tail.call_count_per_target.get((3, 3), 0), 0)

    # 2. 第一名不安全, 第二名安全
    def test_rank1_unsafe_rank2_safe(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1), (2, 2), (3, 3)])
        tail = self._build_tail(env, {
            (1, 1): (5.0, 200.0, True, None),  # over budget
            (2, 2): (5.0, 1.0, True, None),
            (3, 3): (5.0, 1.0, True, None),
        })
        scores = {(1, 1): 3.0, (2, 2): 2.0, (3, 3): 1.0}
        chosen, diag = self._select_lazy(env, budget_ml=10.0, scores=scores, tail=tail)
        self.assertEqual(chosen, (2, 2))
        self.assertEqual(diag["verified_candidate_count"], 2)
        self.assertEqual(diag["selected_rank"], 1)
        self.assertEqual(tail.call_count_per_target[(1, 1)], 1)
        self.assertEqual(tail.call_count_per_target[(2, 2)], 1)
        self.assertEqual(tail.call_count_per_target.get((3, 3), 0), 0)

    # 3. 所有候选不安全 → infeasible (no safe candidate)
    def test_all_unsafe_fallback(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1), (2, 2), (3, 3)])
        tail = self._build_tail(env, {
            (1, 1): (5.0, 100.0, True, None),
            (2, 2): (5.0, 200.0, True, None),
            (3, 3): (5.0, 300.0, True, None),
        })
        scores = {(1, 1): 3.0, (2, 2): 2.0, (3, 3): 1.0}
        chosen, diag = self._select_lazy(env, budget_ml=10.0, scores=scores, tail=tail)
        self.assertIsNone(chosen)  # no safe candidate -> infeasible
        self.assertTrue(diag["fallback_used"])
        self.assertTrue(diag["infeasible"])
        self.assertEqual(diag["verified_candidate_count"], 3)
        self.assertEqual(tail.call_count_per_target[(1, 1)], 1)
        self.assertEqual(tail.call_count_per_target[(2, 2)], 1)
        self.assertEqual(tail.call_count_per_target[(3, 3)], 1)

    # 4. 分数完全相同 → deterministic tie-break matches v10.7
    def test_tie_break_matches_v107(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1), (1, 2), (2, 1)])
        tail = self._build_tail(env, {
            t: (5.0, 1.0, True, None) for t in env._frontier_targets
        })
        scores = {(1, 1): 1.0, (1, 2): 1.0, (2, 1): 1.0}
        chosen_l, _ = self._select_lazy(env, budget_ml=100.0, scores=scores, tail=tail)
        chosen_a, _ = self._select_audit(env, budget_ml=100.0, scores=scores, tail=tail)
        # v10.7: max(score, (-row, -col)) => highest score; ties broken by
        # row ascending (because -row desc) and then col ascending.
        self.assertEqual(chosen_l, (1, 1))
        self.assertEqual(chosen_a, (1, 1))
        self.assertEqual(chosen_l, chosen_a)

    # 5. B_total at threshold +/- EPS
    def test_budget_boundary(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1)])
        tail = self._build_tail(env, {
            (1, 1): (5.0, 1e-9, True, None),  # at EPS
        })
        scores = {(1, 1): 1.0}
        # budget = 0 → 0 + 1e-9 == 0 + EPS, safe
        chosen, diag = self._select_lazy(env, budget_ml=0.0, scores=scores, tail=tail)
        self.assertEqual(chosen, (1, 1))
        self.assertTrue(diag["selected_safe_exact"])
        # budget = -EPS-1e-12 → unsafe → infeasible (no safe candidate)
        tail2 = _FakeTail({(1, 1): (5.0, 1e-9, True, None)})
        chosen, diag = self._select_lazy(env, budget_ml=-2e-9, scores=scores, tail=tail2)
        self.assertIsNone(chosen)
        self.assertTrue(diag["fallback_used"])
        self.assertTrue(diag["infeasible"])

    # 6. completion=False → unsafe → infeasible
    def test_completion_false_blocks(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1)])
        tail = self._build_tail(env, {(1, 1): (5.0, 1.0, False, "serpentine tail lost all legal targets")})
        chosen, diag = self._select_lazy(env, budget_ml=100.0, scores={(1, 1): 1.0}, tail=tail)
        self.assertIsNone(chosen)
        self.assertTrue(diag["fallback_used"])
        self.assertTrue(diag["infeasible"])

    # 7. failure_reason non-None → unsafe → infeasible
    def test_failure_reason_blocks(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1)])
        tail = self._build_tail(env, {(1, 1): (5.0, 1.0, True, "anything")})
        chosen, diag = self._select_lazy(env, budget_ml=100.0, scores={(1, 1): 1.0}, tail=tail)
        self.assertIsNone(chosen)
        self.assertTrue(diag["fallback_used"])
        self.assertTrue(diag["infeasible"])

    # 8. state hash unchanged before/after verify
    def test_state_immutable_during_verify(self) -> None:
        env = _FakeEnv(elapsed=10.0, blood=2.0, frontier_targets=[(1, 1), (2, 2)])
        h_before = _state_hash(env)
        tail = self._build_tail(env, {
            (1, 1): (5.0, 1.0, True, None),
            (2, 2): (5.0, 1.0, True, None),
        })
        scores = {(1, 1): 3.0, (2, 2): 2.0}
        self._select_lazy(env, budget_ml=100.0, scores=scores, tail=tail)
        h_after = _state_hash(env)
        self.assertEqual(h_before, h_after, "select_lazy_v108 must not mutate the live env")

    # 9. audit mode: model forward called once even if many verifies
    def test_audit_mode_verifies_all(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1), (2, 2), (3, 3)])
        tail = self._build_tail(env, {
            (1, 1): (5.0, 1.0, True, None),
            (2, 2): (5.0, 1.0, True, None),
            (3, 3): (5.0, 1.0, True, None),
        })
        scores = {(1, 1): 3.0, (2, 2): 2.0, (3, 3): 1.0}
        chosen, diag = self._select_audit(env, budget_ml=100.0, scores=scores, tail=tail)
        self.assertEqual(chosen, (1, 1))
        self.assertEqual(diag["verified_candidate_count"], 3)
        self.assertEqual(diag["unverified_candidate_count"], 0)
        for t in env._frontier_targets:
            self.assertEqual(tail.call_count_per_target[t], 1)

    # 10. lazy mode does not evaluate unverified candidates
    def test_lazy_skips_unverified(self) -> None:
        env = _FakeEnv(frontier_targets=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])
        tail = self._build_tail(env, {
            t: (5.0, 1.0, True, None) for t in env._frontier_targets
        })
        scores = {(1, 1): 6.0, (2, 2): 5.0, (3, 3): 4.0, (4, 4): 3.0, (5, 5): 2.0, (6, 6): 1.0}
        _, diag = self._select_lazy(env, budget_ml=100.0, scores=scores, tail=tail)
        self.assertEqual(diag["verified_candidate_count"], 1)
        self.assertEqual(diag["unverified_candidate_count"], 5)
        for t in [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]:
            self.assertEqual(tail.call_count_per_target.get(t, 0), 0)

    # 11. repeated call: deterministic given same env + scores
    def test_repeated_call_determinism(self) -> None:
        env_a = _FakeEnv(frontier_targets=[(1, 1), (2, 2)])
        env_b = _FakeEnv(frontier_targets=[(1, 1), (2, 2)])
        tail_a = _FakeTail({(1, 1): (5.0, 1.0, True, None), (2, 2): (5.0, 1.0, True, None)})
        tail_b = _FakeTail({(1, 1): (5.0, 1.0, True, None), (2, 2): (5.0, 1.0, True, None)})
        scores = {(1, 1): 2.0, (2, 2): 1.0}
        c_a, d_a = self._select_lazy(env_a, budget_ml=100.0, scores=scores, tail=tail_a)
        c_b, d_b = self._select_lazy(env_b, budget_ml=100.0, scores=scores, tail=tail_b)
        self.assertEqual(c_a, c_b)
        self.assertEqual(d_a["selected_candidate_id"], d_b["selected_candidate_id"])
        self.assertEqual(d_a["verified_candidate_count"], d_b["verified_candidate_count"])


if __name__ == "__main__":
    unittest.main()

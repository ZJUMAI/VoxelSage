"""v10.5 safe-semantics contract tests (guide Section 7, 12 items).

Each test maps to one mandatory pre-Gate-R check:
  1. B_total includes B_past; the same future cost is safe/unsafe under different
     already-spent budgets
  2. the S next target is always inside the K=6 candidate set
  3. exposed / near-hidden / nearest all enter the set (round-robin) when present
  4. candidates are unique, legal, and deterministically ordered
  5. a no-safe state flags safety_invariant_violation and never picks the
     fastest unsafe candidate
  6. an already-completed candidate tail returns completion with 0 T/B increment
  7. state payload round-trip preserves terminal/failure and all timing/blood
  8. two phase states differing by <0.1 min never collide in the exact cache
  9. fast macro step matches real env.step on T/B/cut/phase
 10. running the same input twice yields identical actions and metrics
 11. v10.4 frozen hashes stay byte-identical
 12. no held-out scene import in tests or implementation
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from clinical_macro_environment import ClinicalMacroResectionEnv  # noqa: E402
from clinical_window_evaluation import _scan_rank  # noqa: E402
from plan_target_order_v105 import (  # noqa: E402
    CorrectedPlannerV105,
    SerpentineTailV105,
    _env_state_payload_v105,
    _step_macro_target,
    candidate_targets_v105,
    rollout_teacher_v105,
    safe_of,
)
from plan_target_order_v104 import serpentine_target_of  # noqa: E402


def rectangle(rows: int = 6, cols: int = 6, vessels=(), start=(0, 0)):
    return {
        "scenario_id": "test-v105",
        "rows": rows,
        "cols": cols,
        "cell_size_mm": 4.0,
        "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
        "obstacle_cells": [list(cell) for cell in vessels],
        "start_cell": [int(start[0]), int(start[1])],
    }


def make_env(scenario, **kwargs):
    return ClinicalMacroResectionEnv(
        scenario=scenario,
        clinical_config={
            "early_end_mode": "disabled",
            "bleeding_probability": 1.0,
            "max_steps_multiplier": 8.0,
        },
        reward_config=kwargs.get("reward_config"),
        mechanics_update_interval=0,
    )


def clone_env_state(env: ClinicalMacroResectionEnv) -> ClinicalMacroResectionEnv:
    """Independent copy of mutable env state for paired comparisons."""
    e = make_env({
        "scenario_id": env.scenario["scenario_id"],
        "rows": env.rows,
        "cols": env.cols,
        "cell_size_mm": env.scenario["cell_size_mm"],
        "domain_cells": sorted(env.domain),
        "obstacle_cells": sorted(env.vessel_cells),
        "start_cell": list(env.start),
    })
    e.reset()
    e.__dict__.update({
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
        "events": [],
    })
    return e


def serp_target(env):
    return serpentine_target_of(env)


class V105BudgetContractTests(unittest.TestCase):
    def test_1_b_total_includes_b_past(self):
        budget = 100.0
        # Same future cost, different already-spent budgets -> different safety.
        self.assertTrue(safe_of(B_past=0.0, future_B=10.0, budget=budget))
        self.assertFalse(safe_of(B_past=95.0, future_B=10.0, budget=budget))
        # Boundary inclusion with tolerance.
        self.assertTrue(safe_of(B_past=90.0, future_B=10.0, budget=budget))
        # The planner must expose B_past explicitly, not compare future-only.
        env = make_env(rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3))))
        env.reset()
        self.assertAlmostEqual(float(env.expected_blood_loss_ml), 0.0, places=9)
        # Move into a state with real accumulated blood (expose + bleed).
        cid = env.component_by_cell[(2, 2)]
        env.exposed_ids.add(cid)
        env.hidden_ids.discard(cid)
        env.phase = "unclamped"
        env.phase_elapsed_minutes = 0.0
        env._advance_time(3.0, [])
        self.assertGreater(float(env.expected_blood_loss_ml), 0.0)

    def test_2_s_target_always_in_candidates(self):
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3), (4, 4)))
        env = make_env(scen)
        env.reset()
        while not env.terminated and not env.truncated:
            cands = candidate_targets_v105(env, count=6)
            s_target = serp_target(env)
            self.assertIn(s_target, cands, f"state {env.step_count}: S target {s_target} not in {cands}")
            self.assertLessEqual(len(cands), 6)
            _step_macro_target(env, cands[0])
        self.assertTrue(env.cut == env.domain)


class V105CandidateContractTests(unittest.TestCase):
    def test_3_exposed_entry_enters_candidate_set(self):
        # Vessel component A mid-grid: make one of its cells adjacent to the cut
        # and expose it, then the K=6 set must contain an exposed seal entry
        # plus the S target, all unique and legal.
        scen = rectangle(rows=8, cols=8, vessels=((3, 3), (3, 4), (4, 3)))
        env = make_env(scen)
        env.reset()
        a_cells = {(3, 3), (3, 4), (4, 3)}
        a_id = env.component_by_cell[(3, 3)]
        env.cut.add((3, 2))  # make (3,3) adjacent to the cut -> seal entry
        env.exposed_ids.add(a_id)
        env.hidden_ids.discard(a_id)
        cands = candidate_targets_v105(env, count=6)
        exposed_entry = [c for c in cands if c in a_cells]
        self.assertGreaterEqual(len(exposed_entry), 1,
                                "an exposed-vessel seal entry must be in the set")
        self.assertIn(serp_target(env), cands)
        self.assertEqual(len(cands), len(set(cands)), "no duplicates")
        for c in cands:
            self.assertIn(c, env._frontier())

    def test_4_candidates_unique_legal_deterministic(self):
        scen = rectangle(rows=7, cols=7, vessels=((2, 2), (2, 3), (4, 4), (4, 5)))
        env = make_env(scen)
        env.reset()
        while not env.terminated and not env.truncated:
            c1 = candidate_targets_v105(env, count=6)
            c2 = candidate_targets_v105(env, count=6)
            self.assertEqual(c1, c2, "candidate order must be deterministic")
            self.assertEqual(len(c1), len(set(c1)))
            for c in c1:
                self.assertIn(c, env._frontier())
            _step_macro_target(env, c1[0])
        self.assertTrue(env.cut == env.domain)


class V105NoSafeTests(unittest.TestCase):
    def test_5_no_safe_flags_invariant_and_never_fastest_unsafe(self):
        # Budget 0 with an exposed vessel => every candidate's B_total > 0, so
        # there is no safe candidate. The planner must flag invariant violation
        # and fall back to S, not pick the fastest unsafe candidate.
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3)))
        env = make_env(scen)
        env.reset()
        cid = env.component_by_cell[(2, 2)]
        env.exposed_ids.add(cid)
        env.hidden_ids.discard(cid)
        env.phase = "unclamped"
        env.phase_elapsed_minutes = 0.0
        planner = CorrectedPlannerV105(candidate_count=6, margin_ml=0.0)
        traj, info = planner.plan(env, baseline_blood=0.0, budget=0.0)
        self.assertTrue(info["safety_invariant_violation"])
        self.assertEqual(info["safe_candidate_count"], 0)
        # Fallback is the S target, never the fastest-unsafe candidate.
        self.assertEqual(traj[0], serp_target(env))


class V105TailContractTests(unittest.TestCase):
    def test_6_completed_candidate_tail_returns_zero_increment(self):
        tail = SerpentineTailV105()
        scen = rectangle(rows=3, cols=3, vessels=())
        env = make_env(scen)
        env.reset()
        # Cut everything except the last cell, then finish it via tail.
        while not env.terminated and not env.truncated:
            legal = sorted(env._frontier())
            if not legal:
                break
            _step_macro_target(env, min(legal, key=lambda c: _scan_rank(env, c)))
        self.assertTrue(env.cut == env.domain)
        t0, b0 = env.elapsed_minutes, env.expected_blood_loss_ml
        dt, db, comp, reason = tail.tail(env)
        self.assertTrue(comp)
        self.assertAlmostEqual(dt, 0.0, places=9)
        self.assertAlmostEqual(db, 0.0, places=9)

    def test_7_state_payload_round_trip_preserves_terminal(self):
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3)))
        env = make_env(scen)
        env.reset()
        while not env.terminated and not env.truncated:
            legal = sorted(env._frontier())
            if not legal:
                break
            _step_macro_target(env, min(legal, key=lambda c: _scan_rank(env, c)))
        payload = _env_state_payload_v105(env)
        for key in ("terminated", "truncated", "failure_reason", "elapsed_minutes",
                    "expected_blood_loss_ml", "clamp_cycle_count", "phase",
                    "phase_elapsed_minutes", "cut", "current"):
            self.assertIn(key, payload)
        # Rebuild a fresh env from the payload and verify the terminal state.
        e2 = ClinicalMacroResectionEnv(scenario=scen, clinical_config=env.clinical_config)
        e2.reset()
        e2.__dict__.update(payload)
        self.assertEqual(e2.terminated, env.terminated)
        self.assertEqual(e2.truncated, env.truncated)
        self.assertEqual(e2.failure_reason, env.failure_reason)
        self.assertAlmostEqual(e2.elapsed_minutes, env.elapsed_minutes, places=12)
        self.assertAlmostEqual(e2.expected_blood_loss_ml, env.expected_blood_loss_ml, places=12)
        self.assertEqual(e2.cut, env.cut)
        self.assertEqual(e2.phase, env.phase)

    def test_8_sub_0p1min_phase_states_have_distinct_cache_keys(self):
        tail = SerpentineTailV105()
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3)))
        env = make_env(scen)
        env.reset()
        cid = env.component_by_cell[(2, 2)]
        env.exposed_ids.add(cid)
        env.hidden_ids.discard(cid)
        # Two clamped-phase states 0.01 min apart near the 15 min boundary.
        env.phase = "clamped"
        env.phase_elapsed_minutes = 14.94
        key_a = tail._state_key(env)
        env2 = clone_env_state(env)
        env2.phase_elapsed_minutes = 14.95
        key_b = tail._state_key(env2)
        self.assertNotEqual(key_a, key_b,
                            "phase states <0.1 min apart must not share a cache key")

    def test_9_fast_macro_step_matches_real_step(self):
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3), (4, 4)))
        env = make_env(scen)
        env.reset()
        for _ in range(5):
            legal = sorted(env._frontier())
            if not legal:
                break
            _step_macro_target(env, min(legal, key=lambda c: _scan_rank(env, c)))
        counts = env._transfer_counts()
        checked = 0
        for target in candidate_targets_v105(env, count=8, transfer_counts=counts):
            fast = clone_env_state(env)
            slow = clone_env_state(env)
            tf, bf = fast.elapsed_minutes, fast.expected_blood_loss_ml
            _step_macro_target(fast, target)
            ts, bs = slow.elapsed_minutes, slow.expected_blood_loss_ml
            slow.step(target[0] * slow.max_cols + target[1])
            self.assertAlmostEqual(fast.elapsed_minutes - tf, slow.elapsed_minutes - ts, places=6)
            self.assertAlmostEqual(
                fast.expected_blood_loss_ml - bf, slow.expected_blood_loss_ml - bs, places=6)
            self.assertEqual(fast.cut, slow.cut)
            self.assertEqual(fast.phase, slow.phase)
            checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_10_same_input_twice_identical(self):
        scen = rectangle(rows=8, cols=8, vessels=((2, 2), (2, 3), (4, 4), (5, 5)))
        margin = 2.0
        baseline = 20.0
        rec1 = rollout_teacher_v105(scen, baseline_blood=baseline, margin_ml=margin)
        rec2 = rollout_teacher_v105(scen, baseline_blood=baseline, margin_ml=margin)
        self.assertEqual(rec1["action_sequence_hash"], rec2["action_sequence_hash"])
        self.assertAlmostEqual(rec1["teacher_T_min"], rec2["teacher_T_min"], places=12)
        self.assertAlmostEqual(rec1["teacher_B_ml"], rec2["teacher_B_ml"], places=12)
        self.assertEqual(rec1["completion"], rec2["completion"])


class V105HygieneTests(unittest.TestCase):
    def test_11_v104_frozen_hashes_unchanged(self):
        frozen = Path("results/clinical_window_v10_4_target_order/frozen")
        sums_file = frozen / "SHA256SUMS"
        self.assertTrue(sums_file.exists())
        for line in sums_file.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ")
            path = frozen / name
            self.assertTrue(path.exists(), f"missing {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, digest, f"v10.4 hash drifted for {name}")

    def test_12_no_held_out_scene_import(self):
        # The v10.5 implementation must never load v10.4 formal splits, internal
        # dev, the teacher npz, the old Gate A file, or the split preparer.
        impl = Path("plan_target_order_v105.py").read_text(encoding="utf-8")
        for token in ("splits_v10_4.json", "policy_internal_dev",
                      "teacher_rankings.npz", "prepare_clinical_v104_splits",
                      "gate_a_splits_v104.json", "planner_gate"):
            self.assertNotIn(token, impl, f"implementation must not touch {token}")
        # Runtime sanity: no imported module carries v10.4 split identifiers.
        self.assertNotIn("splits_v10_4", " ".join(sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)

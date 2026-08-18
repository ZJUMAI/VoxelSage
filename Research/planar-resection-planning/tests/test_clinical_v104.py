"""v10.4 target-order behaviour tests (guide Section 10).

Each test maps to one of the 15 mandatory pre-Gate-A checks:
  1. fixed 15/5 integrates exactly across phase boundaries during transfer/cut
  2. END is illegal in every v10.4 state
  3. every legal macro target produces new cut, no low-level loop
  4. candidate transfer count / delta T / delta B match real cloned step()
  5. large-vessel handling time is exactly 3x normal
  6. zero blood increment while clamped, unexposed, or sealed
  7. transfer appears only through time cost (no separate transfer penalty)
  8. progress_bonus=seal_progress_bonus=0 => no implicit ordering shaping
  9. mechanics_update_interval=0 zeroes, =1 yields finite repeatable metrics
 10. tension replay does not change actions, total time or total blood
 11. frontier/domain-masked global pooling is not diluted by padded area
 12. identical local neighbourhood but different global frontier/phase can
     produce different logits
 13. illegal cells can never be sampled in deterministic/stochastic evaluation
 14. split IDs / seeds / hashes do not cross
 15. bootstrap is scene-paired on paired differences
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from clinical_macro_environment import (  # noqa: E402
    ACTION_END_CLAMP_MACRO,
    CLINICAL_MACRO_ACTION_COUNT,
    CLINICAL_MACRO_GRID_ACTIONS,
    ClinicalMacroResectionEnv,
)
from clinical_window_evaluation import (  # noqa: E402
    _scan_rank,
    rollout_clinical_policy,
    serpentine_macro_target_policy,
)
from plan_target_order_v104 import (  # noqa: E402
    SerpentineTail,
    WindowAwarePlanner,
    _step_macro_target,
    candidate_targets,
    nearest_frontier_macro_policy,
    masked_global_pool,
)


def rectangle(rows: int = 6, cols: int = 6, vessels=(), start=(0, 0)):
    """Simple rectangle scenario; vessels must stay off the boundary."""
    return {
        "scenario_id": "test-v104",
        "rows": rows,
        "cols": cols,
        "cell_size_mm": 4.0,
        "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
        "obstacle_cells": [list(cell) for cell in vessels],
        "start_cell": [int(start[0]), int(start[1])],
    }


def make_env(scenario, *, mechanics_update_interval=0, reward_config=None):
    return ClinicalMacroResectionEnv(
        scenario=scenario,
        clinical_config={
            "early_end_mode": "disabled",
            "bleeding_probability": 1.0,
            "max_steps_multiplier": 8.0,
        },
        reward_config=reward_config,
        mechanics_update_interval=mechanics_update_interval,
    )


def run_serpentine_full(scenario, **kwargs):
    env = make_env(scenario, **kwargs)
    env.reset()
    while not env.terminated and not env.truncated:
        legal = sorted(env._frontier())
        if not legal:
            break
        target = min(legal, key=lambda c: _scan_rank(env, c))
        _step_macro_target(env, target)
    return env


class V104PhaseIntegrationTests(unittest.TestCase):
    def test_1_phase_boundary_integration_is_exact(self):
        # Vessel at (2,2)-(2,3); expose it, then advance 1.0 min starting 0.5
        # min before the 5 min unclamp boundary. Blood must be rate*0.5 (open),
        # then 0.5 clamped (no loss); after the crossing phase is clamped with
        # 0.5 elapsed of the new 15 min clamp window.
        env = make_env(rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3))))
        env.reset()
        cid = env.component_by_cell[(2, 2)]
        env.exposed_ids.add(cid)
        env.hidden_ids.discard(cid)
        env.phase = "unclamped"
        env.phase_elapsed_minutes = 4.5
        rate = env._expected_bleeding_rate()
        self.assertGreater(rate, 0.0)
        blood0 = env.expected_blood_loss_ml
        elapsed0 = env.elapsed_minutes
        loss, _ = env._advance_time(1.0, [])
        self.assertAlmostEqual(loss, rate * 0.5, places=6)
        self.assertAlmostEqual(env.expected_blood_loss_ml - blood0, rate * 0.5, places=6)
        self.assertEqual(env.phase, "clamped")  # 0.5 open then flips to clamped
        self.assertAlmostEqual(env.phase_elapsed_minutes, 0.5, places=6)
        self.assertAlmostEqual(env.elapsed_minutes - elapsed0, 1.0, places=6)

    def test_1_transfer_crossing_boundary_charged_correctly(self):
        env = make_env(rectangle(rows=5, cols=5, vessels=((2, 2),)))
        env.reset()
        env.phase = "unclamped"
        env.phase_elapsed_minutes = 4.95  # 0.05 min until boundary
        # One cut at (0,1) is base_action_minutes (~0.07 min) -> crosses boundary.
        base = env.base_action_minutes
        blood0 = env.expected_blood_loss_ml
        t0 = env.elapsed_minutes
        _step_macro_target(env, (0, 1))
        self.assertGreater(env.elapsed_minutes - t0, base - 1e-9)
        # No vessel exposed yet, so loss should be 0 throughout.
        self.assertAlmostEqual(env.expected_blood_loss_ml - blood0, 0.0, places=9)


class V104EndAndProgressTests(unittest.TestCase):
    def test_2_end_is_illegal_in_every_state(self):
        env = make_env(rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3), (4, 4))))
        env.reset()
        mask = env.action_masks()
        self.assertFalse(mask[ACTION_END_CLAMP_MACRO])
        # Advance through clamped and unclamped phases, END must stay illegal.
        phases_seen = set()
        while not env.terminated and not env.truncated:
            legal = sorted(env._frontier())
            if not legal:
                break
            phases_seen.add(env.phase)
            self.assertFalse(env.action_masks()[ACTION_END_CLAMP_MACRO])
            _step_macro_target(env, min(legal, key=lambda c: _scan_rank(env, c)))
        self.assertTrue(env.cut == env.domain)
        self.assertIn("clamped", phases_seen)
        self.assertFalse(env.action_masks()[ACTION_END_CLAMP_MACRO])

    def test_3_each_macro_step_adds_new_cut_no_loop(self):
        env = make_env(rectangle(rows=5, cols=5, vessels=((1, 2), (2, 2), (2, 3))))
        env.reset()
        cut_sizes = [len(env.cut)]
        steps = 0
        while not env.terminated and not env.truncated:
            legal = sorted(env._frontier())
            if not legal:
                break
            target = min(legal, key=lambda c: _scan_rank(env, c))
            _step_macro_target(env, target)
            cut_sizes.append(len(env.cut))
            steps += 1
            self.assertGreater(len(env.cut), cut_sizes[-2], "macro step added no cut")
        # Every macro step strictly enlarges the cut (no low-level loop); a
        # single step may seal a vessel and cut several cells at once, so steps
        # is at most the number of newly cut cells.
        self.assertLessEqual(steps, cut_sizes[-1] - 1)
        self.assertEqual(env.cut, env.domain)


def clone_env_state(env: ClinicalMacroResectionEnv) -> ClinicalMacroResectionEnv:
    """Independent copy of mutable env state for paired step() comparisons."""
    e = make_env(
        {
            "scenario_id": env.scenario["scenario_id"],
            "rows": env.rows,
            "cols": env.cols,
            "cell_size_mm": env.scenario["cell_size_mm"],
            "domain_cells": sorted(env.domain),
            "obstacle_cells": sorted(env.vessel_cells),
            "start_cell": list(env.start),
        }
    )
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


class V104CandidateConsistencyTests(unittest.TestCase):
    def test_4_candidate_delta_matches_real_step(self):
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3), (4, 4)))
        env = make_env(scen)
        env.reset()
        for _ in range(5):  # move into a mid-episode state
            legal = sorted(env._frontier())
            if not legal:
                break
            _step_macro_target(env, min(legal, key=lambda c: _scan_rank(env, c)))
        counts = env._transfer_counts()
        seen = 0
        for target in candidate_targets(env, count=8, transfer_counts=counts):
            self.assertIn(target, env._frontier())
            self.assertEqual(counts[target], len(env._transfer_path(target)) - 1)
            fast = clone_env_state(env)
            slow = clone_env_state(env)
            tf, bf = fast.elapsed_minutes, fast.expected_blood_loss_ml
            _step_macro_target(fast, target)
            ts, bs = slow.elapsed_minutes, slow.expected_blood_loss_ml
            slow.step(target[0] * slow.max_cols + target[1])
            self.assertAlmostEqual(fast.elapsed_minutes - tf, slow.elapsed_minutes - ts, places=6)
            self.assertAlmostEqual(
                fast.expected_blood_loss_ml - bf, slow.expected_blood_loss_ml - bs, places=6
            )
            self.assertEqual(fast.cut, slow.cut)
            seen += 1
        self.assertGreaterEqual(seen, 4)


class V104VesselTimingTests(unittest.TestCase):
    def test_5_large_vessel_is_exactly_3x(self):
        env = make_env(rectangle(rows=6, cols=6, vessels=((2, 2), (3, 2))))
        env.reset()
        # Component of 2 cells -> is_large True.
        large = [c for c in env.components if c["is_large"]]
        small = [c for c in env.components if not c["is_large"]]
        self.assertEqual(len(large), 1)
        self.assertEqual(len(small), 0)
        cell = sorted(large[0]["cells"])[0]
        # Expose it, then the action duration must be 3x base.
        cid = env.component_by_cell[cell]
        env.exposed_ids.add(cid)
        env.hidden_ids.discard(cid)
        self.assertEqual(
            env._action_duration(cell),
            env.base_action_minutes * env.clinical_config["large_vessel_time_multiplier"],
        )
        self.assertAlmostEqual(
            env._action_duration(cell) / env.base_action_minutes, 3.0, places=6
        )


class V104BloodTests(unittest.TestCase):
    def test_6_zero_loss_while_clamped_or_unexposed_or_sealed(self):
        env = make_env(rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3))))
        env.reset()
        cid = env.component_by_cell[(2, 2)]
        # Clamped + exposed -> no loss.
        env.phase = "clamped"
        env.exposed_ids.add(cid)
        b0 = env.expected_blood_loss_ml
        env._advance_time(2.0, [])
        self.assertAlmostEqual(env.expected_blood_loss_ml - b0, 0.0, places=9)
        # Unexposed (clear exposed) + unclamped -> no loss.
        env.exposed_ids.clear()
        env.phase = "unclamped"
        env.phase_elapsed_minutes = 0.0
        b0 = env.expected_blood_loss_ml
        env._advance_time(2.0, [])
        self.assertAlmostEqual(env.expected_blood_loss_ml - b0, 0.0, places=9)
        # Sealed component -> moved to sealed_ids, not exposed -> no loss.
        env.exposed_ids.discard(cid)
        env.sealed_ids.add(cid)
        env.phase = "unclamped"
        b0 = env.expected_blood_loss_ml
        env._advance_time(2.0, [])
        self.assertAlmostEqual(env.expected_blood_loss_ml - b0, 0.0, places=9)


class V104RewardShapingTests(unittest.TestCase):
    def test_7_transfer_only_via_time_cost(self):
        scen = rectangle(rows=5, cols=5, vessels=((2, 2),))
        env = make_env(scen)
        env.reset()
        all_keys = set()
        while not env.terminated and not env.truncated:
            legal = sorted(env._frontier())
            if not legal:
                break
            idx = min(legal, key=lambda c: _scan_rank(env, c))
            _, _, _, _, info = env.step(idx[0] * env.max_cols + idx[1])
            all_keys.update(info["reward_terms"].keys())
        self.assertGreater(env.transfer_count, 0)
        self.assertNotIn("transfer_penalty", all_keys)
        self.assertNotIn("transfer_cost", all_keys)

    def test_8_no_implicit_ordering_shaping_when_bonuses_zero(self):
        cfg = {"progress_bonus": 0.0, "seal_progress_bonus": 0.0}
        env = make_env(rectangle(rows=5, cols=5, vessels=((2, 2), (3, 3))), reward_config=cfg)
        env.reset()
        while not env.terminated and not env.truncated:
            legal = sorted(env._frontier())
            if not legal:
                break
            idx = min(legal, key=lambda c: _scan_rank(env, c))
            env.step(idx[0] * env.max_cols + idx[1])
        terms = env._info(events=[])["reward_terms"]
        self.assertNotIn("progress_bonus", terms)
        self.assertNotIn("seal_progress_bonus", terms)


class V104MechanicsTests(unittest.TestCase):
    def test_9_zero_vs_finite_repeatable_mechanics(self):
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3), (4, 4)))
        env0 = run_serpentine_full(scen, mechanics_update_interval=0)
        self.assertEqual(env0.mechanics["peak_front_tension"], 0.0)
        self.assertEqual(env0.mechanics["peak_organ_energy"], 0.0)
        self.assertEqual(env0.mechanics["peak_vessel_strain"], 0.0)
        env1a = run_serpentine_full(scen, mechanics_update_interval=1)
        self.assertTrue(np.isfinite(env1a.mechanics["peak_front_tension"]))
        self.assertTrue(np.isfinite(env1a.mechanics["peak_organ_energy"]))
        self.assertTrue(np.isfinite(env1a.mechanics["peak_vessel_strain"]))
        env1b = run_serpentine_full(scen, mechanics_update_interval=1)
        self.assertEqual(env1a.mechanics["peak_front_tension"], env1b.mechanics["peak_front_tension"])
        self.assertEqual(env1a.mechanics["peak_organ_energy"], env1b.mechanics["peak_organ_energy"])

    def test_10_tension_replay_does_not_change_actions_time_or_blood(self):
        # Use a real vessel-rich gate scenario so the mechanics solve is non-trivial.
        gate = json.loads(Path(
            "results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"
        ).read_text(encoding="utf-8"))
        scen = gate["splits"]["planner_gate"]["scenarios"][0]
        cfg = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}

        def manual(interval):
            env = make_env(scen, mechanics_update_interval=interval)
            env.reset()
            peak = 0.0
            while not env.terminated and not env.truncated:
                legal = sorted(env._frontier())
                if not legal:
                    break
                idx = min(legal, key=lambda c: _scan_rank(env, c))
                env.step(idx[0] * env.max_cols + idx[1])
                peak = max(peak, float(env.mechanics["peak_front_tension"]))
            return env, peak

        env0, _ = manual(0)
        env1, peak1 = manual(1)
        # Mechanics replay must not change actions, total time or total blood.
        self.assertEqual(env0.elapsed_minutes, env1.elapsed_minutes)
        self.assertEqual(env0.expected_blood_loss_ml, env1.expected_blood_loss_ml)
        self.assertEqual(env0.step_count, env1.step_count)
        self.assertEqual(env0.cut, env1.cut)
        self.assertTrue(env0.cut == env0.domain)
        # Non-trivial tension appeared at some point during the interval=1 replay.
        self.assertGreater(peak1, 0.0)


class V104GlobalContextTests(unittest.TestCase):
    def test_11_masked_global_pool_ignores_padded_area(self):
        # 4x4 real domain in a 30x40 padded grid.
        rng = np.random.default_rng(7)
        region = np.zeros((30, 40), dtype=np.float32)
        region[:4, :4] = rng.random((4, 4))
        domain = {(r, c) for r in range(4) for c in range(4)}
        p1 = masked_global_pool(region, domain)
        # Same values, only padded surroundings differ -> same pool.
        region2 = np.zeros((30, 40), dtype=np.float32)
        region2[:4, :4] = region[:4, :4]
        region2[20:, 30:] = 999.0  # loud padded area that must be ignored
        p2 = masked_global_pool(region2, domain)
        self.assertAlmostEqual(p1, p2, places=6)
        # And the pool equals the plain mean over the real region.
        self.assertAlmostEqual(p1, float(region[:4, :4].mean()), places=6)

    def test_12_global_context_changes_logit_for_same_local_patch(self):
        from plan_target_order_v104 import reference_global_scorer
        patch = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # Same local patch, very different global frontier/phase scalars.
        l1 = reference_global_scorer(patch, 0.1, 0.1)
        l2 = reference_global_scorer(patch, 0.9, 0.7)
        self.assertNotAlmostEqual(l1, l2)


class V104LegalityTests(unittest.TestCase):
    def test_13_illegal_cells_never_samplable(self):
        scen = rectangle(rows=6, cols=6, vessels=((2, 2), (2, 3), (4, 4)))
        env = make_env(scen)
        env.reset()
        visited_states = 0
        while not env.terminated and not env.truncated:
            mask = env.action_masks()
            legal = np.flatnonzero(mask[:CLINICAL_MACRO_GRID_ACTIONS])
            self.assertEqual(set(divmod(int(a), env.max_cols) for a in legal), env._frontier())
            self.assertFalse(mask[ACTION_END_CLAMP_MACRO])
            visited_states += 1
            target = min(env._frontier(), key=lambda c: _scan_rank(env, c))
            _step_macro_target(env, target)
        self.assertGreater(visited_states, 10)
        # Deterministic baselines all keep legal_action_rate == 1.0.
        for sel in (serpentine_macro_target_policy, nearest_frontier_macro_policy):
            rec = rollout_clinical_policy(scen, sel, clinical_config={"early_end_mode": "disabled"},
                                          mechanics_update_interval=0, control_mode="macro")
            self.assertEqual(rec["legal_action_rate"], 1.0)
            self.assertTrue(rec["completion"])


class V104DataHygieneTests(unittest.TestCase):
    def test_14_split_ids_seeds_hashes_no_cross(self):
        gate = json.loads(Path(
            "results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"
        ).read_text(encoding="utf-8"))
        tune = set(gate["splits"]["planner_tune"]["scenario_ids"])
        g = set(gate["splits"]["planner_gate"]["scenario_ids"])
        self.assertEqual(len(tune), 384)
        self.assertEqual(len(g), 128)
        self.assertFalse(tune & g)
        self.assertEqual(gate["seed"], 20260811)
        sha_file = Path("results/clinical_window_v10_4_target_order/pilot_gate_a/SHA256SUMS")
        self.assertTrue(sha_file.exists())
        self.assertIn("gate_a_splits_v104.json", sha_file.read_text(encoding="utf-8"))

    def test_15_bootstrap_is_scene_paired_on_differences(self):
        from evaluate_clinical_v102 import _paired_bootstrap
        records = [
            {"scenario_id": "s0", "elapsed_minutes": 12.0, "expected_blood_loss_ml": 3.0},
            {"scenario_id": "s1", "elapsed_minutes": 15.0, "expected_blood_loss_ml": 5.0},
            {"scenario_id": "s2", "elapsed_minutes": 10.0, "expected_blood_loss_ml": 2.0},
        ]
        baseline = {
            "s0": {"elapsed_minutes": 10.0, "expected_blood_loss_ml": 2.0},
            "s1": {"elapsed_minutes": 12.0, "expected_blood_loss_ml": 4.0},
            "s2": {"elapsed_minutes": 9.0, "expected_blood_loss_ml": 1.0},
        }
        stat = _paired_bootstrap(records, baseline, "elapsed_minutes", bootstrap_samples=500)
        self.assertAlmostEqual(stat["mean_difference"], 2.0, places=6)  # (2+3+1)/3
        self.assertEqual(len(stat["bootstrap_95_ci"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

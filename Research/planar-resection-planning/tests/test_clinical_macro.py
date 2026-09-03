from __future__ import annotations

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
    CLINICAL_MACRO_OBSERVATION_CHANNELS,
    ClinicalMacroResectionEnv,
)
from clinical_window_evaluation import (  # noqa: E402
    rollout_clinical_policy,
    serpentine_macro_target_policy,
)
from clinical_window_scenarios import generate_clinical_stage_pool  # noqa: E402


def rectangle(rows: int = 5, cols: int = 5, vessels=()):
    return {
        "scenario_id": "test-clinical-macro",
        "rows": rows,
        "cols": cols,
        "cell_size_mm": 4.0,
        "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
        "obstacle_cells": [list(cell) for cell in vessels],
        "start_cell": [0, 0],
    }


def action(row: int, col: int) -> int:
    return row * 40 + col


class ClinicalMacroEnvironmentTests(unittest.TestCase):
    def test_mask_contains_frontier_targets_plus_end_slot(self):
        env = ClinicalMacroResectionEnv(scenario=rectangle())
        observation, _ = env.reset()
        self.assertEqual(
            observation.shape,
            (len(CLINICAL_MACRO_OBSERVATION_CHANNELS), 30, 40),
        )
        self.assertEqual(env.action_space.n, CLINICAL_MACRO_ACTION_COUNT)
        legal = np.flatnonzero(env.action_masks()).tolist()
        self.assertEqual(legal, [action(0, 1), action(1, 0)])
        self.assertFalse(env.action_masks()[ACTION_END_CLAMP_MACRO])

    def test_shortest_transfer_is_automatic_but_charged_per_cell(self):
        env = ClinicalMacroResectionEnv(scenario=rectangle())
        env.reset()
        env.step(action(0, 1))
        elapsed_before = env.elapsed_minutes
        _, _, _, _, info = env.step(action(1, 0))
        self.assertAlmostEqual(
            env.elapsed_minutes - elapsed_before,
            2 * env.base_action_minutes,
        )
        self.assertEqual(
            [event["action"] for event in info["events"] if event["action"] in ("transfer", "cut")],
            ["transfer", "cut"],
        )
        self.assertEqual(env.transfer_count, 1)
        self.assertEqual(env.current, (1, 0))
        self.assertAlmostEqual(
            info["max_macro_duration_minutes"], 2 * env.base_action_minutes,
        )

    def test_transfer_time_bleeds_when_exposed_and_unclamped(self):
        env = ClinicalMacroResectionEnv(
            scenario=rectangle(vessels=((2, 2),)),
            clinical_config={"bleeding_probability": 0.1},
        )
        env.reset()
        env.hidden_ids.clear()
        env.exposed_ids.add(0)
        env.phase = "unclamped"
        env.cut.update({(0, 1), (1, 0)})
        env.current = (0, 1)
        rate = env._expected_bleeding_rate()
        env.step(action(2, 0))
        self.assertAlmostEqual(
            env.expected_blood_loss_ml,
            rate * 3 * env.base_action_minutes,
        )
        self.assertEqual(env.transfer_count, 2)

    def test_end_is_same_zero_time_clinical_decision(self):
        env = ClinicalMacroResectionEnv(scenario=rectangle())
        env.reset()
        env.step(action(0, 1))
        elapsed = env.elapsed_minutes
        self.assertTrue(env.action_masks()[ACTION_END_CLAMP_MACRO])
        _, _, _, _, info = env.step(ACTION_END_CLAMP_MACRO)
        self.assertEqual(env.elapsed_minutes, elapsed)
        self.assertEqual(info["phase"], "unclamped")

    def test_serpentine_macro_completes_without_navigation_stagnation(self):
        scenario = generate_clinical_stage_pool(
            stage="d", count=1, seed=19, split="test",
        )[0]
        result = rollout_clinical_policy(
            scenario,
            serpentine_macro_target_policy,
            clinical_config={"early_end_mode": "disabled"},
            control_mode="macro",
        )
        self.assertTrue(result["completion"])
        self.assertEqual(result["legal_action_rate"], 1.0)
        self.assertEqual(result["max_no_progress_streak"], 0)
        self.assertEqual(result["max_same_edge_streak"], 0)

    def test_standard_vessel_rules_accept_and_complete_boundary_barrier(self):
        result = rollout_clinical_policy(
            rectangle(
                rows=5,
                cols=5,
                vessels=tuple((row, 2) for row in range(5)),
            ),
            serpentine_macro_target_policy,
            clinical_config={"early_end_mode": "disabled"},
            control_mode="macro",
            include_replay=True,
        )

        self.assertTrue(result["completion"])
        self.assertEqual(result["sealed_vessel_count"], 1)
        self.assertEqual(result["legal_action_rate"], 1.0)
        exposure = next(
            event for event in result["replay"]["events"]
            if event["action"] == "expose_vessel"
        )
        self.assertEqual(exposure["release_rule"], "boundary_frontier_deadlock")
        self.assertLess(
            exposure["required_ring_cell_count"],
            exposure["full_ring_cell_count"],
        )

    def test_spatial_policy_emits_grid_targets_plus_end(self):
        import torch
        from sb3_contrib import MaskablePPO

        from clinical_macro_policy import ClinicalMacroSpatialPolicy
        from variable_policy import PaddedSpatialExtractor

        torch.set_num_threads(1)
        env = ClinicalMacroResectionEnv(scenario=rectangle())
        model = MaskablePPO(
            ClinicalMacroSpatialPolicy,
            env,
            policy_kwargs={
                "features_extractor_class": PaddedSpatialExtractor,
                "net_arch": [],
                "share_features_extractor": True,
            },
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            device="cpu",
        )
        observation, _ = env.reset()
        distribution = model.policy.get_distribution(
            torch.as_tensor(observation[None]),
            action_masks=torch.as_tensor(env.action_masks()[None]),
        )
        self.assertEqual(
            distribution.distribution.logits.shape,
            (1, CLINICAL_MACRO_ACTION_COUNT),
        )


if __name__ == "__main__":
    unittest.main()

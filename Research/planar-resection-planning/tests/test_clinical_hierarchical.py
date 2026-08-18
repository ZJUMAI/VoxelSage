from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from clinical_hierarchical_environment import (  # noqa: E402
    CLAMP_ACTION_COUNT,
    CLAMP_CONTINUE,
    CLAMP_RELEASE,
    CLINICAL_HIERARCHICAL_MASK_SIZE,
    ClinicalHierarchicalResectionEnv,
)


def rectangle(rows: int = 5, cols: int = 5, vessels=()):
    return {
        "scenario_id": "test-clinical-hierarchical",
        "rows": rows,
        "cols": cols,
        "cell_size_mm": 4.0,
        "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
        "obstacle_cells": [list(cell) for cell in vessels],
        "start_cell": [0, 0],
    }


def target(row: int, col: int) -> int:
    return row * 40 + col


class ClinicalHierarchicalEnvironmentTests(unittest.TestCase):
    def test_factorized_mask_has_clamp_and_target_parts(self):
        env = ClinicalHierarchicalResectionEnv(
            scenario=rectangle(), clinical_config={"early_end_mode": "disabled"}
        )
        env.reset()
        mask = env.action_masks()
        self.assertEqual(mask.shape, (CLINICAL_HIERARCHICAL_MASK_SIZE,))
        self.assertTrue(mask[CLAMP_CONTINUE])
        self.assertFalse(mask[CLAMP_RELEASE])
        legal_targets = np.flatnonzero(mask[CLAMP_ACTION_COUNT:]).tolist()
        self.assertEqual(legal_targets, [target(0, 1), target(1, 0)])

    def test_release_and_target_share_one_macro_step(self):
        env = ClinicalHierarchicalResectionEnv(
            scenario=rectangle(), clinical_config={"early_end_mode": "full"}
        )
        env.reset()
        env.step(np.asarray([CLAMP_CONTINUE, target(0, 1)]))
        elapsed = env.elapsed_minutes
        steps = env.step_count
        _, _, _, _, info = env.step(np.asarray([CLAMP_RELEASE, target(0, 2)]))
        self.assertEqual(env.step_count, steps + 1)
        self.assertEqual(env.early_end_count, 1)
        self.assertEqual(env.phase, "unclamped")
        self.assertGreater(env.elapsed_minutes, elapsed)
        self.assertIn((0, 2), env.cut)
        self.assertTrue(any(e.get("clamp_decision") == "release_now" for e in info["events"]))

    def test_exposed_unsealed_vessel_masks_release(self):
        env = ClinicalHierarchicalResectionEnv(
            scenario=rectangle(vessels=((2, 2),)),
            clinical_config={"early_end_mode": "full"},
        )
        env.reset()
        env.step(np.asarray([CLAMP_CONTINUE, target(0, 1)]))
        self.assertTrue(env.action_masks()[CLAMP_RELEASE])
        env.hidden_ids.clear()
        env.exposed_ids.add(0)
        self.assertFalse(env.action_masks()[CLAMP_RELEASE])

    def test_timing_oracle_compares_deepcopied_branches(self):
        from train_clamp_timing_oracle import counterfactual_release_advantage

        env = ClinicalHierarchicalResectionEnv(
            scenario=rectangle(rows=3, cols=3),
            clinical_config={"early_end_mode": "full"},
        )
        env.reset()
        env.step(np.asarray([CLAMP_CONTINUE, target(0, 1)]))
        advantage, details = counterfactual_release_advantage(
            env, target(0, 2), time_cost=1.0, blood_cost=1.0
        )
        self.assertTrue(np.isfinite(advantage))
        self.assertIn("continue_blood", details)
        self.assertIn("release_blood", details)

    def test_policy_emits_two_plus_grid_logits(self):
        import torch
        from sb3_contrib import MaskablePPO

        from clinical_hierarchical_policy import ClinicalHierarchicalPolicy
        from variable_policy import PaddedSpatialExtractor

        torch.set_num_threads(1)
        env = ClinicalHierarchicalResectionEnv(scenario=rectangle())
        model = MaskablePPO(
            ClinicalHierarchicalPolicy,
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
        self.assertEqual(len(distribution.distributions), 2)
        self.assertEqual(distribution.distributions[0].logits.shape, (1, 2))
        self.assertEqual(distribution.distributions[1].logits.shape, (1, 1200))

    def test_hierarchical_behavior_cloning_updates_target_head(self):
        import torch
        from sb3_contrib import MaskablePPO

        from clinical_hierarchical_policy import ClinicalHierarchicalPolicy
        from train_clinical_window_ppo import (
            ClinicalTrainingConfig,
            _run_direction_behavior_cloning,
        )
        from variable_policy import PaddedSpatialExtractor

        torch.set_num_threads(1)
        torch.manual_seed(0)
        env = ClinicalHierarchicalResectionEnv(
            scenario=rectangle(), clinical_config={"early_end_mode": "disabled"}
        )
        model = MaskablePPO(
            ClinicalHierarchicalPolicy,
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
        before = model.policy.action_net.target_scorer[-1].weight.detach().clone()
        result = _run_direction_behavior_cloning(
            model=model,
            scenarios=[rectangle()],
            clinical_config={"early_end_mode": "disabled"},
            reward_config={"progress_bonus": 5.0, "seal_progress_bonus": 2.0},
            config=ClinicalTrainingConfig(
                control_mode="hierarchical",
                bc_scenarios=1,
                bc_epochs=1,
                bc_batch_size=16,
                bc_learning_rate=1e-3,
                bc_margin=1.0,
                bc_v_weight=0.0,
                device="cpu",
            ),
        )
        after = model.policy.action_net.target_scorer[-1].weight.detach()
        self.assertTrue(result["enabled"])
        self.assertGreater(result["demonstration_count"], 0)
        self.assertFalse(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()

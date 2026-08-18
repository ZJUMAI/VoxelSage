"""Stage 1 supervised-learning behavioural tests (decision-maker mandate).

Covers:
  1. plan_spatial receives non-zero gradients (no torch.no_grad in training);
  2. a positive-blood sample is NEVER labelled release (v2-safe rule);
  3. both baseline and oracle occupancy enter the dataset;
  4. the three regression targets are normalized by their scales;
  5. regression_head round-trips through model save/load;
  6. Dev metrics and rollout metrics are computed from real records
     (unsafe-FPR / AUROC / calibration / regret + tracked unsafe releases);
  + robust BC checkpoint + tensor hashes; improved/equal/worsened buckets are
    mutually exclusive and sum to the scene count.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import torch  # noqa: E402

from clinical_target_conditioned_environment import (  # noqa: E402
    CLAMP_RELEASE,
    TargetConditionedClampEnv,
    serpentine_target_cell,
)
from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy  # noqa: E402

import train_target_conditioned_clamp_oracle as oracle_mod  # noqa: E402

BC_MODEL_PATH = str(
    SIMULATOR_DIR
    / "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"
)


def rectangle(rows: int = 5, cols: int = 5, vessels=(), scenario_id: str = "t"):
    return {
        "scenario_id": scenario_id,
        "rows": rows,
        "cols": cols,
        "cell_size_mm": 4.0,
        "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
        "obstacle_cells": [list(cell) for cell in vessels],
        "start_cell": [0, 0],
    }


def v102_config():
    clinical = {
        "time_scale_minutes": 39.83,
        "blood_scale_ml": 1274.85,
        "weight_kg": 70.0,
        "bleeding_probability": 1.0,
        "early_end_mode": "threshold",
        "early_end_minutes": 10.0,
    }
    reward = {
        "time_cost": 1.0,
        "blood_cost": 1.0,
        "completion_bonus": 5.0,
        "failure_penalty": 10.0,
        "invalid_action_penalty": 10.0,
    }
    return clinical, reward


class Test1PlanSpatialGradient(unittest.TestCase):
    """plan_spatial must receive gradients; base_spatial stays frozen."""

    @classmethod
    def setUpClass(cls):
        cls.bc = FrozenBCMacroTargetPolicy(BC_MODEL_PATH, device="cpu")

    def test_plan_spatial_gradients_and_base_frozen(self):
        clinical, reward = v102_config()
        model = oracle_mod._build_frozen_base_clamp_model(
            seed=7, device="cpu", target_policy=self.bc,
            scenario=rectangle(rows=5, cols=5), clinical_config=clinical,
            reward_config=reward, ischemia_cost=1.0, ischemia_scale_minutes=20.0,
        )
        obs = torch.randn(2, 36, 30, 40)
        features = model.policy.extract_features(obs)
        latent_pi, _ = model.policy.mlp_extractor(features)
        fused = model.policy.action_net.fused_features(latent_pi, obs)
        logits = model.policy.action_net.scorer(fused)
        logits.sum().backward()
        plan_grads = [
            p.grad for p in model.policy.features_extractor.plan_spatial.parameters()
        ]
        self.assertTrue(all(g is not None for g in plan_grads))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in plan_grads))
        base_grads = [
            p.grad for p in model.policy.features_extractor.base_spatial.parameters()
        ]
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in base_grads))


class Test2PositiveBloodNeverRelease(unittest.TestCase):
    """A positive-blood counterfactual must never be labelled release."""

    def test_positive_blood_label_zero(self):
        details = {"delta_blood": 5.0, "delta_ischemia": -10.0}
        label, reg, db, di = oracle_mod._stage1_label_and_reg(
            1.0, details, epsilon_ischemia=1e-6, blood_scale=100.0,
            ischemia_scale=10.0,
        )
        self.assertEqual(label, 0)
        self.assertEqual(db, 5.0)
        self.assertEqual(di, -10.0)

    def test_safe_negative_blood_label_one(self):
        details = {"delta_blood": -5.0, "delta_ischemia": -10.0}
        label, _, _, _ = oracle_mod._stage1_label_and_reg(
            2.0, details, epsilon_ischemia=1e-6, blood_scale=100.0,
            ischemia_scale=10.0,
        )
        self.assertEqual(label, 1)


class Test3BaselineAndOracleOccupancy(unittest.TestCase):
    """Both baseline (always-continue) and safe-oracle passes enter the data."""

    def test_both_policies_in_dataset(self):
        clinical, reward = v102_config()
        scenarios = [rectangle(rows=6, cols=6, scenario_id="s1")]
        obs, labels, audit = oracle_mod.collect_stage1_examples(
            scenarios,
            target_selector=serpentine_target_cell,
            clinical_config=clinical,
            reward_config=reward,
            ischemia_cost=1.0,
            ischemia_scale_minutes=20.0,
            epsilon_ischemia=1e-6,
            advantage_margin=1e-6,
            seed=1,
        )
        policies = set(bool(a["oracle_policy"]) for a in audit)
        self.assertEqual(policies, {False, True})
        # obs are float16 36-channel observations.
        self.assertEqual(obs[0].dtype, np.float16)
        self.assertEqual(obs[0].shape, (36, 30, 40))


class Test4RegressionNormalized(unittest.TestCase):
    """Regression targets are normalized by blood / ischemia scales."""

    def test_normalization(self):
        details = {"delta_blood": -150.0, "delta_ischemia": -2.0}
        label, reg, _, _ = oracle_mod._stage1_label_and_reg(
            1.5, details, epsilon_ischemia=1e-6, blood_scale=298.0,
            ischemia_scale=30.0,
        )
        self.assertAlmostEqual(reg[0], -150.0 / 298.0, places=6)
        self.assertAlmostEqual(reg[1], -2.0 / 30.0, places=6)
        self.assertAlmostEqual(reg[2], 1.5, places=6)


class Test5RegressionHeadRoundtrip(unittest.TestCase):
    """regression_head persists through model save/load."""

    @classmethod
    def setUpClass(cls):
        cls.bc = FrozenBCMacroTargetPolicy(BC_MODEL_PATH, device="cpu")

    def test_save_load_consistent(self):
        from sb3_contrib import MaskablePPO

        clinical, reward = v102_config()
        model = oracle_mod._build_frozen_base_clamp_model(
            seed=7, device="cpu", target_policy=self.bc,
            scenario=rectangle(rows=5, cols=5), clinical_config=clinical,
            reward_config=reward, ischemia_cost=1.0, ischemia_scale_minutes=20.0,
        )
        with torch.no_grad():
            model.policy.regression_head.weight.fill_(0.0)
            model.policy.regression_head.bias.fill_(1.0)
        tmp = Path(tempfile.mkdtemp())
        model.save(str(tmp / "m"))
        loaded = MaskablePPO.load(str(tmp / "m.zip"), device="cpu")
        x = torch.randn(3, 173)
        self.assertTrue(torch.allclose(
            model.policy.regression_head(x),
            loaded.policy.regression_head(x),
        ))


class Test6DevAndRolloutMetricsReal(unittest.TestCase):
    """Dev metrics come from real records; rollout unsafe releases are tracked."""

    def test_dev_metrics_from_real_records(self):
        audit = [
            {"release_legal": True, "delta_blood": -1.0, "delta_ischemia": -1.0,
             "advantage": 1.0},
            {"release_legal": True, "delta_blood": 1.0, "delta_ischemia": -1.0,
             "advantage": -0.5},  # genuinely unsafe (delta_blood > 0)
            {"release_legal": False, "delta_blood": 0.0, "delta_ischemia": 0.0,
             "advantage": 0.0},
            {"release_legal": True, "delta_blood": -2.0, "delta_ischemia": -2.0,
             "advantage": 2.0},
        ]
        y_true = np.array([1, 0, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.5, 0.8])
        m = oracle_mod.evaluate_stage1_metrics(y_true, y_prob, audit, threshold=0.5)
        self.assertIn("auroc", m)
        self.assertIn("auprc", m)
        self.assertIn("balanced_accuracy", m)
        self.assertIn("unsafe_release_false_positive_rate", m)
        self.assertIn("ece", m)
        self.assertIn("regret_total", m)
        # legal subset = indices 0,1,3 -> yt=[1,0,1], pred=[1,0,1] -> perfect
        self.assertAlmostEqual(m["balanced_accuracy"], 1.0)
        # unsafe state = index 1 (db=1>0); model predicts continue there -> FPR 0
        self.assertAlmostEqual(m["unsafe_release_false_positive_rate"], 0.0)
        # unsafe prediction: flip prob at index 1 to 0.95 -> model releases an
        # unsafe state -> unsafe-FPR becomes 1/1 = 1.0
        m2 = oracle_mod.evaluate_stage1_metrics(
            y_true, np.array([0.9, 0.95, 0.5, 0.8]), audit, threshold=0.5
        )
        self.assertAlmostEqual(m2["unsafe_release_false_positive_rate"], 1.0)

    def test_rollout_unsafe_release_tracked(self):
        from evaluate_clinical_v102 import rollout_target_conditioned

        clinical, reward = v102_config()

        def always_release(env):
            return CLAMP_RELEASE

        rec = rollout_target_conditioned(
            rectangle(rows=5, cols=5, scenario_id="r1"),
            always_release,
            target_selector=serpentine_target_cell,
            clinical_config=clinical,
            reward_config=reward,
            ischemia_cost=1.0,
            ischemia_scale_minutes=20.0,
        )
        self.assertGreater(rec["unsafe_release_count"], 0)
        self.assertLess(rec["legal_action_rate"], 1.0)


class Test7RobustHashes(unittest.TestCase):
    """BC checkpoint file hash + robust tensor hash both recorded."""

    @classmethod
    def setUpClass(cls):
        cls.bc = FrozenBCMacroTargetPolicy(BC_MODEL_PATH, device="cpu")

    def test_checkpoint_and_tensor_hashes(self):
        self.assertEqual(len(self.bc.checkpoint_sha256), 64)
        param_sha = self.bc.parameter_sha256()
        self.assertEqual(len(param_sha), 64)
        # Robust: recomputing gives the same value, and it differs from the
        # raw file hash (different objects).
        self.assertEqual(param_sha, self.bc.parameter_sha256())
        self.assertNotEqual(param_sha, self.bc.checkpoint_sha256)


class Test8BucketsMutuallyExclusive(unittest.TestCase):
    """improved + equal + worsened == n under one shared tolerance."""

    def test_buckets_sum_to_n(self):
        base1 = {
            "scenario_id": "s1", "completion": True, "coverage": 1.0,
            "legal_action_rate": 1.0, "elapsed_minutes": 100.0,
            "expected_blood_loss_ml": 100.0, "total_clamped_minutes": 60.0,
            "early_end_count": 0, "unsafe_end_count": 0, "failure_reason": None,
        }
        orac1 = dict(base1)
        orac1.update({"expected_blood_loss_ml": 80.0, "early_end_count": 1})
        base2 = dict(base1)
        base2.update({"scenario_id": "s2", "elapsed_minutes": 90.0})
        orac2 = dict(base2)
        orac2.update({"expected_blood_loss_ml": 90.0})
        result = oracle_mod.evaluate_gate_a_policy(
            [base1, base2], [orac1, orac2], bootstrap_samples=50, seed=1
        )
        for key in ("blood", "ischemia", "time"):
            f = result["fields"][key]
            self.assertEqual(
                f["n_improved"] + f["n_equal"] + f["n_worsened"],
                result["n_scenarios"],
                key,
            )


if __name__ == "__main__":
    unittest.main()

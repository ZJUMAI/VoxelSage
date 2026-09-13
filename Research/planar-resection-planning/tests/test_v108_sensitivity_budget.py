"""Regression tests for same-condition C0 budgets in the v10.8 runner."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_safety_shield_v106 import ExactSafetyShieldV106
from confirmation_controllers_v107 import _clinical_config, _make_env, rollout_C0
from evaluate_v108_sensitivity import (
    SENSITIVITY_CONDITIONS, _existing_shard_matches, _freeze_condition_baseline, main,
)
from plan_target_order_v104 import serpentine_target_of


def synthetic_scene():
    # This geometry exposes its vessel across the 10/5 boundary, while C0
    # incurs no blood loss under 15/5. It uses no held-out experiment data.
    return {
        "scenario_id": "synthetic-condition-budget",
        "rows": 16, "cols": 16, "cell_size_mm": 4.0,
        "domain_cells": [[r, c] for r in range(16) for c in range(16)],
        "obstacle_cells": [[8, 8], [8, 9]], "start_cell": [0, 0],
    }


class SensitivityBudgetTests(unittest.TestCase):
    def test_same_condition_budget_keeps_initial_serpentine_feasible(self):
        scene = synthetic_scene()
        s0 = rollout_C0(scene, _clinical_config(SENSITIVITY_CONDITIONS["S0"]))
        cfg = _clinical_config(SENSITIVITY_CONDITIONS["S2"])
        s2 = rollout_C0(scene, cfg)
        env = _make_env(scene, cfg)
        margin = 16.07054347826075
        old_budget = s0["realized_episode_B_ml"] + margin
        records = ExactSafetyShieldV106(clinical_config=cfg).evaluate(env, budget_ml=old_budget)
        s = next(r for r in records if r.target == serpentine_target_of(env))
        self.assertTrue(s.completion)
        self.assertAlmostEqual(s.B_total, s2["realized_episode_B_ml"], places=9)
        self.assertFalse(s.safe_exact)
        self.assertLessEqual(s.B_total, s2["realized_episode_B_ml"] + margin)

    def test_runner_does_not_use_s0_reference_as_other_condition_budget(self):
        scene = synthetic_scene()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            reference = root / "reference_s0.json"
            checkpoint = root / "unused-checkpoint.pt"
            split.write_text(json.dumps({"scenarios": [scene]}), encoding="utf-8")
            # A deliberately incompatible reference must never affect budgets.
            reference.write_text(json.dumps({"records": {
                scene["scenario_id"]: {"expected_blood_loss_ml": 999.0},
            }}), encoding="utf-8")
            checkpoint.write_bytes(b"C0 does not load a model")
            arguments = [
                "--split-file", str(split), "--baseline-file", str(reference),
                "--checkpoint", str(checkpoint), "--output-root", str(root / "out"),
                "--conditions", "S0,S2", "--controllers", "C0",
                "--scene-workers", "1", "--limit", "1", "--margin", "16",
            ]
            with patch("evaluate_v108_sensitivity._repository_commit", return_value="test"):
                self.assertEqual(main(arguments), 0)
                self.assertEqual(main(arguments), 0)  # Same-provenance resume is safe.
            rows = []
            for condition in ("S0", "S2"):
                path = root / "out" / condition / "C0" / f"{scene['scenario_id']}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(row["completion"])
                self.assertEqual(row["budget_ml"], row["realized_episode_B_ml"] + 16.0)
                self.assertEqual(row["evaluation_metadata"]["condition"], condition)
                rows.append(row)
            self.assertNotEqual(rows[0]["budget_ml"], rows[1]["budget_ml"])
            self.assertNotEqual(
                rows[0]["evaluation_metadata"]["baseline_sha256"],
                rows[1]["evaluation_metadata"]["baseline_sha256"],
            )

    def test_frozen_baseline_rejects_changed_condition_or_subset(self):
        scene = synthetic_scene()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cfg = SENSITIVITY_CONDITIONS["S2"]
            records, _ = _freeze_condition_baseline("S2", [scene], cfg, out, "split-a")
            self.assertGreater(records[scene["scenario_id"]]["realized_episode_B_ml"], 0)
            with self.assertRaisesRegex(ValueError, "baseline mismatch"):
                _freeze_condition_baseline("S2", [scene], SENSITIVITY_CONDITIONS["S0"], out, "split-a")
            with self.assertRaisesRegex(ValueError, "baseline mismatch"):
                _freeze_condition_baseline("S2", [], cfg, out, "split-a")
            with self.assertRaisesRegex(ValueError, "baseline mismatch"):
                _freeze_condition_baseline("S2", [scene], cfg, out, "split-b")

    def test_legacy_or_different_budget_shards_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            metadata = {"baseline_sha256": "correct-condition"}
            self.assertFalse(_existing_shard_matches(path, metadata))
            for row in ({}, {"evaluation_metadata": {"baseline_sha256": "s0"}}):
                path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    _existing_shard_matches(path, metadata)


if __name__ == "__main__":
    unittest.main()

"""v10.7 confirmatory experiment contract tests (guide Gates A/B)."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from confirmation_controllers_v107 import (
    _score_myopic,
    rollout_controller,
)
from prepare_clinical_v107_confirmation import (
    BOOTSTRAP_SEED,
    MASTER_SEED,
    SPLIT_COUNTS,
    SENSITIVITY_CONDITIONS,
    content_hash,
    make_scenario,
)
from clinical_macro_environment import ClinicalMacroResectionEnv
from plan_target_order_v104 import _step_macro_target, serpentine_target_of

MARGIN = 16.07054347826075
CFG = {"early_end_mode": "disabled", "early_end_minutes": 0.0,
       "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}


class V107DataTests(unittest.TestCase):
    def test_1_master_seed_and_counts(self):
        self.assertEqual(MASTER_SEED, 2026081707)
        self.assertEqual(SPLIT_COUNTS, {"dev_smoke": 32, "replication": 256, "sensitivity_base": 128})

    def test_2_five_sensitivity_conditions_frozen(self):
        self.assertEqual(set(SENSITIVITY_CONDITIONS), {"S0", "S1", "S2", "S3", "S4"})
        self.assertEqual(SENSITIVITY_CONDITIONS["S0"]["max_clamp_minutes"], 15.0)
        self.assertEqual(SENSITIVITY_CONDITIONS["S2"]["max_clamp_minutes"], 10.0)
        self.assertEqual(SENSITIVITY_CONDITIONS["S4"]["bleeding_probability"], 0.25)

    def test_3_content_hash_ignores_scenario_id(self):
        a = make_scenario("replication", 0)
        b = dict(a); b["scenario_id"] = "renamed-0000"
        self.assertEqual(content_hash(a), content_hash(b))
        c = make_scenario("replication", 1)
        self.assertNotEqual(content_hash(a), content_hash(c))

    def test_4_scenario_ids_prefixes(self):
        for name in ("dev_smoke", "replication", "sensitivity_base"):
            s = make_scenario(name, 0)
            self.assertTrue(s["scenario_id"].startswith(f"clinical-d-v10.7-{name}-"))
            self.assertEqual(s["stage"], "d")


class V107ControllerTests(unittest.TestCase):
    def test_5_C2_myopic_does_not_read_teacher_tail(self):
        # _score_myopic must only use current state and next-macro costs.
        env = ClinicalMacroResectionEnv(scenario=make_scenario("dev_smoke", 0),
                                        clinical_config=CFG, mechanics_update_interval=0)
        env.reset()
        from clinical_safety_shield_v106 import ExactSafetyShieldV106
        records = ExactSafetyShieldV106().evaluate(env, budget_ml=1e9)
        scores = _score_myopic(env, records)
        self.assertEqual(set(scores.keys()), {r.target for r in records})

    def test_6_C0_and_C1_agree_on_dev_smoke(self):
        for i in range(3):
            s = make_scenario("dev_smoke", i)
            r0 = rollout_controller("C0", s, baseline_blood=0.0, margin_ml=MARGIN)
            base = r0["realized_episode_B_ml"]
            r1 = rollout_controller("C1", s, baseline_blood=base, margin_ml=MARGIN)
            self.assertEqual(r0["action_sequence_hash"], r1["action_sequence_hash"],
                             f"scene {i}: C0/C1 action hash differ")
            self.assertAlmostEqual(r0["elapsed_minutes"], r1["elapsed_minutes"], places=9)
            self.assertAlmostEqual(r0["realized_episode_B_ml"], r1["realized_episode_B_ml"], places=9)

    def test_7_C1_executed_trajectory_within_budget(self):
        s = make_scenario("dev_smoke", 2)
        r0 = rollout_controller("C0", s, baseline_blood=0.0, margin_ml=MARGIN)
        base = r0["realized_episode_B_ml"]
        r1 = rollout_controller("C1", s, baseline_blood=base, margin_ml=MARGIN)
        self.assertTrue(r1["selected_max_B_total_ml"] <= base + MARGIN + 1e-9)
        self.assertEqual(r1["safety_invariant_violations"], 0)


class V107DataAccessTests(unittest.TestCase):
    def test_8_train_modules_do_not_parse_heldout_splits(self):
        frozen = SIM / "results/clinical_window_v10_7_confirmation/frozen"
        self.assertTrue((frozen / "experiment_manifest.json").is_file())
        master = json.loads((frozen / "splits_v10_7.json").read_text(encoding="utf-8"))
        self.assertTrue(master["formal"])
        counts = master["counts"]
        self.assertEqual(counts, SPLIT_COUNTS)


if __name__ == "__main__":
    unittest.main()

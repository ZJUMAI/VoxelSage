"""Contract tests for the v10.7.1 sensitivity correction."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from aggregate_sensitivity_v1071 import upper_cvar10
from prepare_sensitivity_v1071 import (
    COUNT, CONDITIONS, MASTER_SEED, clinical_config, config_hash, content_hash, make_scene,
)


class V1071ContractTests(unittest.TestCase):
    def test_version_seed_and_count_are_new(self):
        self.assertEqual(MASTER_SEED, 20260817071)
        self.assertEqual(COUNT, 128)
        self.assertTrue(make_scene(0)["scenario_id"].startswith("clinical-d-v10.7.1-sensitivity-correction-"))

    def test_all_five_conditions_are_prespecified(self):
        self.assertEqual(set(CONDITIONS), {"S0", "S1", "S2", "S3", "S4"})
        self.assertEqual(CONDITIONS["S2"]["max_clamp_minutes"], 10.0)
        self.assertEqual(CONDITIONS["S4"]["bleeding_probability"], 0.25)

    def test_condition_hashes_are_distinct_when_config_differs(self):
        hashes = {condition: config_hash(condition) for condition in CONDITIONS}
        self.assertEqual(len(set(hashes.values())), 5)
        self.assertNotEqual(hashes["S0"], hashes["S1"])
        self.assertNotEqual(hashes["S0"], hashes["S3"])

    def test_early_end_remains_disabled(self):
        for condition in CONDITIONS:
            cfg = clinical_config(condition)
            self.assertEqual(cfg["early_end_mode"], "disabled")
            self.assertEqual(cfg["early_end_minutes"], 0.0)

    def test_content_hash_is_rename_safe(self):
        first = make_scene(0)
        renamed = dict(first); renamed["scenario_id"] = "renamed"
        self.assertEqual(content_hash(first), content_hash(renamed))
        self.assertNotEqual(content_hash(first), content_hash(make_scene(1)))

    def test_upper_cvar_uses_worst_not_best_tail(self):
        values = np.arange(1.0, 11.0)
        self.assertEqual(upper_cvar10(values), 10.0)
        values = np.arange(1.0, 21.0)
        self.assertEqual(upper_cvar10(values), 19.5)


if __name__ == "__main__":
    unittest.main()

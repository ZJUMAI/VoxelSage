"""v10.6 L0 budget-observation and exact-shield contract tests."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_safety_shield_v106 import ExactSafetyShieldV106
from clinical_target_order_features_v106 import (
    CANDIDATE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    budget_values,
    candidate_features_v106,
    global_context_v106,
)
from clinical_target_order_policy_v106 import masked_mean_max, unpack_spatial
from plan_target_order_v104 import _step_macro_target, serpentine_target_of
from plan_target_order_v105 import _candidate_sources_v105, _env_state_payload_v105
from prepare_clinical_v106_splits import HISTORICAL_GATE, SPLIT_COUNTS, SPLIT_SEEDS, make_scenario
from train_target_order_v106 import losses


def rectangle(rows=7, cols=7, vessels=((2, 2), (2, 3), (4, 4)), start=(0, 0)):
    return {
        "scenario_id": "test-v106",
        "rows": rows, "cols": cols, "cell_size_mm": 4.0,
        "domain_cells": [[r, c] for r in range(rows) for c in range(cols)],
        "obstacle_cells": [list(c) for c in vessels],
        "start_cell": list(start),
    }


def env_of(scene=None):
    env = ClinicalMacroResectionEnv(
        scenario=scene or rectangle(),
        clinical_config={"early_end_mode": "disabled", "bleeding_probability": 1.0,
                         "max_steps_multiplier": 8.0},
        mechanics_update_interval=0,
    )
    env.reset()
    return env


class V106ObservationTests(unittest.TestCase):
    def test_1_budget_observation_raw_and_normalized(self):
        env = env_of()
        env.expected_blood_loss_ml = 25.0
        vec, raw = global_context_v106(
            env, baseline_blood_ml=80.0, margin_ml=20.0, blood_scale_ml=50.0)
        self.assertEqual(vec.shape, (GLOBAL_FEATURE_DIM,))
        self.assertEqual(raw, {
            "B_past_ml": 25.0, "B_baseline_scene_ml": 80.0, "M_B_ml": 20.0,
            "B_budget_total_ml": 100.0, "B_remaining_ml": 75.0,
        })
        np.testing.assert_allclose(vec[-8:], [0.5, 1.6, 0.4, 2.0, 1.5, 0.25, 0.75, 0.0])

    def test_2_b_past_changes_complete_observation(self):
        a = env_of(); b = env_of()
        b.expected_blood_loss_ml = 7.0
        va, _ = global_context_v106(a, baseline_blood_ml=100, margin_ml=5, blood_scale_ml=100)
        vb, _ = global_context_v106(b, baseline_blood_ml=100, margin_ml=5, blood_scale_ml=100)
        self.assertFalse(np.array_equal(va, vb))

    def test_3_s_target_is_model_candidate_with_flag(self):
        env = env_of()
        sourced = _candidate_sources_v105(env, count=6)
        targets = [t for t, _ in sourced]
        s = serpentine_target_of(env)
        self.assertIn(s, targets)
        source = dict(sourced)[s]
        feat, _after, _dt, _db, meta = candidate_features_v106(env, s, source=source)
        self.assertEqual(feat.shape, (CANDIDATE_FEATURE_DIM,))
        self.assertTrue(meta["is_serpentine_fallback"])
        self.assertEqual(feat[-5], 1.0)

    def test_15_compact_spatial_round_trip(self):
        semantic = np.zeros((2, 10, 30, 40), dtype=np.uint8)
        semantic[0, 0, :7, :9] = 1
        semantic[1, 5, 4, 6] = 1
        bits = np.packbits(semantic.reshape(2, 10, -1), axis=-1)
        transfer = np.zeros((2, 30, 40), dtype=np.uint8)
        transfer[1, 4, 6] = 255
        decoded = unpack_spatial(bits, transfer)
        np.testing.assert_array_equal(decoded[:, :10], semantic.astype(np.float32))
        self.assertEqual(decoded[1, 10, 4, 6], 1.0)

    def test_16_masked_context_ignores_padding(self):
        feature = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8); mask[:, :, :4, :4] = 1
        mean_a, max_a = masked_mean_max(feature, mask)
        feature[:, :, 4:, :] = 1e6
        feature[:, :, :, 4:] = 1e6
        mean_b, max_b = masked_mean_max(feature, mask)
        torch.testing.assert_close(mean_a, mean_b)
        torch.testing.assert_close(max_a, max_b)

    def test_17_nan_padding_cannot_poison_training_loss(self):
        valid = torch.tensor([[True, True, False]])
        batch = {
            "valid": valid, "safe": torch.tensor([[True, True, False]]),
            "completion": torch.tensor([[1.0, 1.0, 0.0]]),
        }
        output = {"score": torch.zeros(1, 3),
                  "completion_logit": torch.zeros(1, 3),
                  "safe_logit": torch.zeros(1, 3)}
        for name in ("T_total", "B_tail", "B_total"):
            batch[name] = torch.tensor([[1.0, 2.0, float("nan")]])
            output[name] = torch.zeros(1, 3)
        weights = {"rank": 1.0, "T_total": 0.2, "B_tail": 0.3,
                   "B_total": 0.3, "completion": 0.1, "safe": 0.2}
        total, components = losses(output, batch, torch.tensor([0]), weights)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))

    def test_18_success_failure_reason_none_is_countable(self):
        rows = [
            {"completion": True, "failure_reason": None},
            {"completion": False, "failure_reason": "truncated"},
        ]
        failures = sum(
            bool((not row["completion"]) or (row["failure_reason"] is not None))
            for row in rows
        )
        self.assertEqual(failures, 1)


class V106ShieldTests(unittest.TestCase):
    def test_4_b_total_identity_for_every_candidate(self):
        env = env_of()
        env.expected_blood_loss_ml = 3.25
        shield = ExactSafetyShieldV106()
        records = shield.evaluate(env, budget_ml=1e9)
        self.assertGreater(len(records), 0)
        for rec in records:
            self.assertAlmostEqual(
                rec.B_total,
                3.25 + rec.delta_B_action + rec.B_tail,
                places=9,
            )

    def test_5_terminal_payload_and_tail_completion_preserved(self):
        env = env_of(rectangle(rows=3, cols=3, vessels=()))
        while not env.terminated and not env.truncated:
            _step_macro_target(env, serpentine_target_of(env))
        payload = _env_state_payload_v105(env)
        self.assertTrue(payload["terminated"])
        self.assertFalse(payload["truncated"])
        self.assertIsNone(payload["failure_reason"])

    def test_6_shield_rejects_over_budget(self):
        env = env_of()
        records = ExactSafetyShieldV106().evaluate(env, budget_ml=-1.0)
        self.assertTrue(records)
        self.assertTrue(all(not r.safe_exact for r in records))

    def test_7_model_risk_scores_cannot_bypass_shield(self):
        env = env_of(make_scenario("policy_train", 0))
        _step_macro_target(env, serpentine_target_of(env))
        shield = ExactSafetyShieldV106()
        # Freeze the budget at the least-blood exact tail. Costlier candidates
        # are unsafe and receive arbitrarily high model logits.
        probe = shield.evaluate(env, budget_ml=1e9)
        budget = min(r.B_total for r in probe)
        records = shield.evaluate(env, budget_ml=budget)
        unsafe = [r for r in records if not r.safe_exact]
        self.assertTrue(unsafe, "constructed state must contain an unsafe candidate")
        scores = {r.target: (1e9 if not r.safe_exact else -1e9) for r in records}
        selected, info = shield.select(env, budget_ml=budget, scores=scores)
        selected_record = next(r for r in info["records"] if r.target == selected)
        self.assertTrue(selected_record.safe_exact)
        self.assertTrue(info["shield_intervention"])

    def test_8_no_safe_is_invariant_and_s_fallback(self):
        env = env_of()
        shield = ExactSafetyShieldV106()
        selected, info = shield.select(env, budget_ml=-1.0, scores=None)
        self.assertEqual(selected, serpentine_target_of(env))
        self.assertTrue(info["safety_invariant_violation"])
        self.assertEqual(info["safe_candidate_count"], 0)

    def test_9_shield_deterministic(self):
        def run():
            env = env_of()
            shield = ExactSafetyShieldV106()
            actions = []
            while not env.terminated and not env.truncated:
                records = shield.evaluate(env, budget_ml=1e9)
                scores = {r.target: -r.T_total for r in records}
                target, _ = shield.select(env, budget_ml=1e9, scores=scores)
                actions.append(target)
                _step_macro_target(env, target)
            return hashlib.sha256(json.dumps(actions).encode()).hexdigest()
        self.assertEqual(run(), run())

    def test_19_exact_record_cache_is_semantically_transparent(self):
        env = env_of()
        cache = {}
        shield = ExactSafetyShieldV106(record_cache=cache)
        first = shield.evaluate(env, budget_ml=123.0)
        second = shield.evaluate(env, budget_ml=123.0)
        self.assertEqual(first, second)
        self.assertEqual(shield.record_cache_misses, 1)
        self.assertEqual(shield.record_cache_hits, 1)
        self.assertEqual(len(cache), 1)


class V106DataHygieneTests(unittest.TestCase):
    def test_10_new_split_ids_do_not_overlap_each_other_or_v105_gate(self):
        sample_ids = {
            name: {make_scenario(name, i)["scenario_id"] for i in range(min(3, count))}
            for name, count in SPLIT_COUNTS.items()
        }
        names = list(sample_ids)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                self.assertFalse(sample_ids[left] & sample_ids[right])
        gate = json.loads(HISTORICAL_GATE.read_text(encoding="utf-8"))
        historical = set(gate["splits"]["planner_gate"]["scenario_ids"])
        self.assertFalse(set().union(*sample_ids.values()) & historical)
        self.assertEqual(len(SPLIT_SEEDS), len(set(SPLIT_SEEDS.values())))

    def test_11_train_modules_do_not_name_held_out_splits(self):
        # Collector/trainer do not exist yet at L0; once added they are included
        # automatically and must not import or address held-out split names.
        for name in ("collect_target_order_teacher_v106.py", "train_target_order_v106.py"):
            path = SIM / name
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8").lower()
            for forbidden in ("validation", "test\"", "stress", "tuning"):
                self.assertNotIn(forbidden, source)

    def test_12_v104_v105_frozen_hashes_unchanged(self):
        v104 = SIM / "results/clinical_window_v10_4_target_order/frozen"
        for line in (v104 / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            path = v104 / filename.strip()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        # v10.5 INPUT_SHA256SUMS records hashes at their original source paths.
        source = SIM / "results/clinical_window_v10_4_target_order/pilot_gate_a"
        manifest = SIM / "results/clinical_window_v10_5_safe_planner/frozen_inputs/INPUT_SHA256SUMS"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            path = source / filename.strip()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_13_negative_remaining_budget_is_not_clipped(self):
        env = env_of(); env.expected_blood_loss_ml = 120.0
        raw = budget_values(env, baseline_blood_ml=90.0, margin_ml=10.0)
        self.assertEqual(raw["B_remaining_ml"], -20.0)
        vec, _ = global_context_v106(env, baseline_blood_ml=90, margin_ml=10, blood_scale_ml=50)
        self.assertLess(vec[-4], 0.0)
        self.assertEqual(vec[-1], 1.0)

    def test_14_formal_files_are_phase_isolated_and_hashed(self):
        frozen = SIM / "results/clinical_window_v10_6_shielded_learning/frozen"
        master = json.loads((frozen / "splits_v10_6.json").read_text(encoding="utf-8"))
        self.assertTrue(master["formal"])
        self.assertNotIn("splits", master, "master must not embed scenario payloads")
        self.assertEqual(master["counts"], SPLIT_COUNTS)
        all_ids = {name: set(ids) for name, ids in master["scenario_ids"].items()}
        for name, count in SPLIT_COUNTS.items():
            scene_payload = json.loads((frozen / f"split_{name}.json").read_text(encoding="utf-8"))
            base_payload = json.loads((frozen / f"baseline_{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(scene_payload["split"], name)
            self.assertEqual(base_payload["split"], name)
            self.assertEqual(len(scene_payload["scenarios"]), count)
            self.assertEqual(set(base_payload["records"]), all_ids[name])
            text = (frozen / f"split_{name}.json").read_text(encoding="utf-8")
            text += (frozen / f"baseline_{name}.json").read_text(encoding="utf-8")
            foreign = set().union(*(ids for other, ids in all_ids.items() if other != name))
            self.assertFalse(any(sid in text for sid in foreign))
        for line in (frozen / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            self.assertEqual(
                hashlib.sha256((frozen / filename.strip()).read_bytes()).hexdigest(), digest
            )


if __name__ == "__main__":
    unittest.main()

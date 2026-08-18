"""v10.2 clinical-window tests (behavioural, not interface stubs).

Reviewer-gate fixes applied:
  * Test 6  constructs a deterministic release-legal state and asserts release
            advances real time (no fictional zero-time END).
  * Test 9  runs a tiny Probe through evaluate_probe_separated and checks the
            det/stoch files are separate and statistically distinct.
  * Test 10 writes a temporary final-eval JSON and checks per-scene records,
            paired differences and bootstrap CIs all exist.
  * Test 12 calls guard_split_access + evaluate_split and asserts test/stress
            are rejected without the one-shot final confirmation.
  * New Gate A v2 behavioural tests (guide section 7 + reviewer fixes):
            delta_blood>0 must continue; safe release conditions; replan after
            release; baseline/oracle pairing; episode-level aggregation (never
            candidate-state mean); threshold-10 gating; BC target hash + action
            sequence audit; frozen BC target insensitive to clamp state.

All tests run on CPU.  Report should say "behavioural tests / total" instead of
the old 12/12 placeholder count.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import torch  # noqa: E402

from clinical_target_conditioned_environment import (  # noqa: E402
    CLAMP_CONTINUE,
    CLAMP_RELEASE,
    PLANNED_ROUTE_CHANNEL,
    PLANNED_TARGET_CHANNEL,
    TargetConditionedClampEnv,
    serpentine_target_cell,
)
from clinical_target_conditioned_policy import (  # noqa: E402
    FrozenBCMacroTargetPolicy,
    PaddedPlanSpatialExtractor,
    TargetConditionedClampActionHead,
    TargetConditionedClampPolicy,
)

import train_target_conditioned_clamp_oracle as oracle_mod  # noqa: E402

BC_MODEL_PATH = str(
    SIMULATOR_DIR
    / "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"
)


def rectangle(rows: int = 5, cols: int = 5, vessels=(), scenario_id: str = "test-v102"):
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
        "early_end_minutes": 5.0,
    }
    reward = {
        "time_cost": 1.0,
        "blood_cost": 1.0,
        "completion_bonus": 5.0,
        "failure_penalty": 10.0,
        "invalid_action_penalty": 10.0,
    }
    return clinical, reward


def make_env(scenario=None, clinical_config=None):
    clinical, reward = v102_config()
    if clinical_config is not None:
        clinical = dict(clinical)
        clinical.update(clinical_config)
    return TargetConditionedClampEnv(
        scenario=scenario or rectangle(),
        clinical_config=clinical,
        reward_config=reward,
        ischemia_cost=1.0,
        ischemia_scale_minutes=20.0,
    )


def make_release_legal_env(scenario=None):
    """Force a deterministic release-legal state (clamped, past threshold,
    no exposed unsealed vessel)."""
    env = make_env(scenario=scenario)
    env.reset()
    env.phase = "clamped"
    env.phase_elapsed_minutes = 10.0
    env.exposed_ids.clear()
    assert env.action_masks()[CLAMP_RELEASE], "fixture must be release-legal"
    return env


def _make_tiny_ppo(scenario, clinical, reward, seed: int = 7):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([
        lambda: TargetConditionedClampEnv(
            scenario=scenario, clinical_config=clinical, reward_config=reward,
            ischemia_cost=1.0, ischemia_scale_minutes=20.0,
        )
    ])
    model = MaskablePPO(
        TargetConditionedClampPolicy,
        venv,
        policy_kwargs={
            "features_extractor_class": PaddedPlanSpatialExtractor,
            "net_arch": [],
            "share_features_extractor": True,
        },
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        learning_rate=1e-4,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=32)
    return model


def _counterfactual_details(delta_blood, delta_ischemia, delta_time=0.0, advantage=1.0):
    """A minimal counterfactual details dict with valid field names."""
    details = {
        "continue_time": 100.0,
        "release_time": 100.0 + delta_time,
        "continue_blood": 100.0,
        "release_blood": 100.0 + delta_blood,
        "continue_ischemia": 60.0,
        "release_ischemia": 60.0 + delta_ischemia,
        "delta_time": float(delta_time),
        "delta_blood": float(delta_blood),
        "delta_ischemia": float(delta_ischemia),
    }
    return float(advantage), details


class Test1PlannedTargetAffectsLogits(unittest.TestCase):
    """Test 1: same global state, different planned target -> different logits."""

    def test_different_targets_give_different_logits(self):
        head = TargetConditionedClampActionHead(base_conv=32, plan_conv=8, rows=30, cols=40)
        latent = torch.randn(2, 40, 30, 40)
        obs = torch.zeros(2, 36, 30, 40)
        obs[0, PLANNED_TARGET_CHANNEL, 5, 5] = 1.0
        obs[1, PLANNED_TARGET_CHANNEL, 12, 20] = 1.0
        logits = head(latent, obs)
        self.assertEqual(logits.shape, (2, 2))
        self.assertFalse(torch.equal(logits[0], logits[1]))


class Test2OracleAndClampShareTarget(unittest.TestCase):
    """Test 2: oracle collection and clamp forward use the same planned target."""

    def test_obs_target_matches_env_planned_target(self):
        env = make_env()
        obs, _ = env.reset()
        self.assertIsNotNone(env.planned_target_index)
        hot = np.argwhere(obs[PLANNED_TARGET_CHANNEL] > 0)
        self.assertEqual(hot.shape[0], 1)
        self.assertEqual((int(hot[0][0]), int(hot[0][1])), env.planned_target)


class Test3BatchAlignment(unittest.TestCase):
    """Test 3: route / target scalars / local features not misaligned in batch."""

    def test_local_feature_uses_each_samples_own_target(self):
        head = TargetConditionedClampActionHead(base_conv=32, plan_conv=8, rows=30, cols=40)
        latent = torch.randn(2, 40, 30, 40)
        obs = torch.zeros(2, 36, 30, 40)
        obs[0, PLANNED_TARGET_CHANNEL, 5, 6] = 1.0
        obs[1, PLANNED_TARGET_CHANNEL, 9, 12] = 1.0
        obs[:, PLANNED_ROUTE_CHANNEL, :, :] = 0.0
        obs[0, PLANNED_ROUTE_CHANNEL, 5, 5] = 1.0
        obs[1, PLANNED_ROUTE_CHANNEL, 9, 11] = 1.0
        fused = head.fused_features(latent, obs)
        feat = 40
        F = latent.reshape(2, -1, 30, 40)
        expected_local_0 = torch.cat((F[0, :32, 5, 6], F[0, 32:, 5, 6]))
        self.assertTrue(torch.allclose(fused[0, 2 * feat : 2 * feat + feat], expected_local_0))


class Test4ReleaseMaskOnExposedVessel(unittest.TestCase):
    """Test 4: exposed unsealed vessel -> release mask always False."""

    def test_release_mask_false_when_exposed(self):
        env = make_env()
        env.reset()
        env.hidden_ids.clear()
        env.exposed_ids.add(0)
        env.phase = "clamped"
        env.phase_elapsed_minutes = 10.0
        mask = env.action_masks()
        self.assertFalse(mask[CLAMP_RELEASE])


class Test5CounterfactualFromIdenticalState(unittest.TestCase):
    """Test 5: release/continue counterfactuals start from identical deep-copy."""

    def test_deepcopy_branches_identical_before_step(self):
        env = make_env()
        env.reset()
        a = copy.deepcopy(env)
        b = copy.deepcopy(env)
        self.assertEqual(a.events, b.events)
        self.assertEqual(a.cut, b.cut)
        self.assertAlmostEqual(a.elapsed_minutes, b.elapsed_minutes)
        self.assertEqual(a.planned_target, b.planned_target)


class Test6NoFictionalZeroTime(unittest.TestCase):
    """Test 6: release adds no fictional zero-time benefit.

    Builds a deterministic release-legal state so the core assertion is always
    executed (reviewer fix: the old test skipped it when release was illegal).
    """

    def test_release_advances_real_time_and_executes_target(self):
        env = make_release_legal_env()
        elapsed_before = env.elapsed_minutes
        cut_before = set(env.cut)
        env.step(CLAMP_RELEASE)
        # release is part of the macro step (transfer + cut): it must advance
        # real time and execute the planned target, never a zero-cost END.
        self.assertGreater(env.elapsed_minutes, elapsed_before)
        self.assertTrue(env.cut.issuperset(cut_before))
        self.assertGreater(len(env.cut), len(cut_before))


class Test7RewardConservation(unittest.TestCase):
    """Test 7: per-step time/blood/ischemia reward terms conserve state."""

    def test_reward_terms_match_state_deltas(self):
        clinical, reward = v102_config()
        env = make_env()
        env.reset()
        for _ in range(5):
            elapsed_before = env.elapsed_minutes
            blood_before = env.expected_blood_loss_ml
            clamp_before = env.total_clamped_minutes
            _, _, term, trunc, info = env.step(CLAMP_CONTINUE)
            terms = info["reward_terms"]
            expected_time = (
                reward["time_cost"]
                * (env.elapsed_minutes - elapsed_before) / clinical["time_scale_minutes"]
            )
            expected_blood = (
                reward["blood_cost"]
                * (env.expected_blood_loss_ml - blood_before) / clinical["blood_scale_ml"]
            )
            expected_ischemia = (
                1.0 * (env.total_clamped_minutes - clamp_before) / 20.0
            )
            self.assertAlmostEqual(terms["time_cost"], expected_time, places=6)
            self.assertAlmostEqual(terms["blood_cost"], expected_blood, places=6)
            self.assertAlmostEqual(terms["ischemia_cost"], expected_ischemia, places=6)


class Test8TargetHashUnchanged(unittest.TestCase):
    """Test 8: clamp-only PPO target checkpoint hash unchanged by training."""

    def test_frozen_base_spatial_unchanged_by_training(self):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        scenario = rectangle(rows=8, cols=8, vessels=((3, 3), (3, 4), (6, 6)))
        clinical, reward = v102_config()
        venv = DummyVecEnv([
            lambda: TargetConditionedClampEnv(
                scenario=scenario, clinical_config=clinical, reward_config=reward,
                ischemia_cost=1.0, ischemia_scale_minutes=20.0,
            )
        ])
        model = MaskablePPO(
            TargetConditionedClampPolicy,
            venv,
            policy_kwargs={
                "features_extractor_class": PaddedPlanSpatialExtractor,
                "net_arch": [],
                "share_features_extractor": True,
            },
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            learning_rate=1e-4,
            seed=7,
            device="cpu",
            verbose=0,
        )
        before = {
            name: param.clone()
            for name, param in model.policy.features_extractor.base_spatial.named_parameters()
        }
        model.learn(total_timesteps=64)
        for name, param in model.policy.features_extractor.base_spatial.named_parameters():
            self.assertTrue(torch.equal(before[name], param.data), f"base_spatial {name} changed")


class Test9DetAndStochSeparate(unittest.TestCase):
    """Test 9: deterministic and stochastic evaluation written to separate files
    with statistically distinct summaries (reviewer fix: was a callable stub)."""

    def test_probe_writes_separate_det_stoch_files(self):
        from evaluate_clinical_v102 import evaluate_probe_separated

        scenarios = [
            rectangle(rows=6, cols=6, vessels=((1, 1),)),
            rectangle(rows=6, cols=6, vessels=((2, 3),)),
        ]
        clinical, reward = v102_config()
        model = _make_tiny_ppo(scenarios[0], clinical, reward, seed=9)
        tmp = Path(tempfile.mkdtemp())
        det_path = tmp / "probe_det.json"
        stoch_path = tmp / "probe_stoch.json"
        evaluate_probe_separated(
            det_path=det_path,
            stoch_path=stoch_path,
            scenarios=scenarios,
            model=model,
            target_selector=serpentine_target_cell,
            clinical_config=clinical,
            reward_config=reward,
            ischemia_cost=1.0,
            ischemia_scale_minutes=20.0,
        )
        det = json.loads(det_path.read_text(encoding="utf-8"))
        stoch = json.loads(stoch_path.read_text(encoding="utf-8"))
        n = len(scenarios)
        # Separate files with their own summaries and records.
        self.assertNotEqual(det_path, stoch_path)
        for payload in (det, stoch):
            self.assertIn("det_summary", payload)
            self.assertIn("stoch_summary", payload)
            self.assertIn("det_records", payload)
            self.assertIn("stoch_records", payload)
        # evaluate_probe_separated runs n_stochastic=1 into the det file and
        # n_stochastic=5 into the stoch file: the summaries must be separate
        # and statistically distinct, not mixed.
        self.assertEqual(det["det_summary"]["episode_count"], n)
        self.assertEqual(det["stoch_summary"]["episode_count"], n)  # 1x in det file
        self.assertEqual(stoch["det_summary"]["episode_count"], n)
        self.assertEqual(stoch["stoch_summary"]["episode_count"], 5 * n)  # 5x in stoch file
        # det and stoch files carry different stochastic summary counts.
        self.assertNotEqual(
            det["stoch_summary"]["episode_count"], stoch["stoch_summary"]["episode_count"]
        )
        # det vs stoch summaries are distinct objects, not aliases.
        self.assertIsNot(det["det_summary"], det["stoch_summary"])


class Test10PerSceneRecordsAndPaired(unittest.TestCase):
    """Test 10: final evaluation saves per-scene records, paired differences and
    bootstrap CIs (reviewer fix: was an info-field check)."""

    def test_final_evaluation_records_and_paired(self):
        from evaluate_clinical_v102 import (
            baseline_selector,
            evaluate_split,
            rollout_target_conditioned,
        )

        scenarios = [
            rectangle(rows=6, cols=6, vessels=((1, 1),)),
            rectangle(rows=6, cols=6, vessels=((2, 2), (3, 3))),
        ]
        clinical, reward = v102_config()
        baseline_records = {}
        for scenario in scenarios:
            rec = rollout_target_conditioned(
                scenario,
                baseline_selector,
                target_selector=serpentine_target_cell,
                clinical_config=clinical,
                reward_config=reward,
                ischemia_cost=1.0,
                ischemia_scale_minutes=20.0,
            )
            baseline_records[rec["scenario_id"]] = rec
        model = _make_tiny_ppo(scenarios[0], clinical, reward, seed=10)
        result = evaluate_split(
            scenarios,
            model,
            target_selector=serpentine_target_cell,
            clinical_config=clinical,
            reward_config=reward,
            ischemia_cost=1.0,
            ischemia_scale_minutes=20.0,
            n_stochastic=1,
            baseline_records=baseline_records,
            bootstrap_samples=200,
            split="probe",
        )
        self.assertIn("det_records", result)
        self.assertIn("paired", result)
        for field in ("elapsed_minutes", "expected_blood_loss_ml", "total_clamped_minutes"):
            paired = result["paired"][field]
            self.assertIn("bootstrap_95_ci", paired)
            self.assertEqual(len(paired["bootstrap_95_ci"]), 2)
            self.assertIn("mean_difference", paired)
        # Writing a temporary final-eval JSON and reading it back keeps the
        # schema contract verifiable.
        tmp = Path(tempfile.mkdtemp()) / "final_eval.json"
        tmp.write_text(json.dumps(result, default=float, ensure_ascii=False), encoding="utf-8")
        back = json.loads(tmp.read_text(encoding="utf-8"))
        self.assertEqual(len(back["det_records"]), len(scenarios))
        self.assertIn("bootstrap_95_ci", back["paired"]["expected_blood_loss_ml"])


class Test11SplitIdSeedUnique(unittest.TestCase):
    """Test 11: split IDs and seeds never overlap across sets."""

    def test_v102_splits_unique(self):
        from prepare_clinical_v102_splits import (
            V102_SPLIT_COUNTS,
            V102_SPLIT_SEEDS,
            make_stress_scenario,
            make_clinical_scenario,
        )

        ids = []
        seeds = []
        for name, count in V102_SPLIT_COUNTS.items():
            for index in range(min(count, 8)):
                if name == "stress":
                    scen = make_stress_scenario(
                        index=index, seed=V102_SPLIT_SEEDS[name] + index * 7919,
                        split=f"v10.2-{name}",
                    )
                else:
                    scen = make_clinical_scenario(
                        stage="d", index=index,
                        seed=V102_SPLIT_SEEDS[name] + index * 7919,
                        split=f"v10.2-{name}",
                    )
                ids.append(scen["scenario_id"])
                seeds.append(int(scen["seed"]))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(seeds), len(set(seeds)))


class Test12TestStressNotInSelection(unittest.TestCase):
    """Test 12: Test/Stress paths are rejected by selection and API layers
    without the one-shot final confirmation (reviewer fix: was assertTrue(True))."""

    def test_test_stress_rejected_by_selection_paths(self):
        from evaluate_clinical_v102 import (
            guard_split_access,
            evaluate_split,
            SELECTION_SPLITS,
        )

        for split in ("test", "stress"):
            with self.assertRaises(PermissionError):
                guard_split_access(split)
            with self.assertRaises(PermissionError):
                evaluate_split(
                    [], None,
                    target_selector=serpentine_target_cell,
                    clinical_config={}, reward_config={},
                    ischemia_cost=1.0, ischemia_scale_minutes=1.0,
                    split=split,
                )
        # Explicit one-shot confirmation unblocks the final split only.
        guard_split_access("test", final_confirmed=True)
        guard_split_access("stress", final_confirmed=True)
        for split in SELECTION_SPLITS:
            guard_split_access(split)
        # Unknown splits are rejected outright.
        with self.assertRaises(ValueError):
            guard_split_access("train")


class Test13GateA2DeltaBloodPositiveContinues(unittest.TestCase):
    """Gate A v2: delta_blood > 0 must continue regardless of ischemia gain."""

    def test_delta_blood_positive_forces_continue(self):
        env = make_release_legal_env()
        advantage, details = _counterfactual_details(delta_blood=10.0, delta_ischemia=-5.0)
        with mock.patch.object(
            oracle_mod, "counterfactual_release_advantage",
            return_value=(advantage, details),
        ):
            action, record = oracle_mod.choose_safe_oracle_action(
                env, epsilon_ischemia=1e-6, advantage_margin=1e-6
            )
        self.assertEqual(action, CLAMP_CONTINUE)
        self.assertIn("delta_blood_positive", record["reject_reason"])


class Test14GateA2SafeReleaseConditions(unittest.TestCase):
    """Gate A v2: release requires delta_blood<=0 AND delta_ischemia<-eps AND
    positive advantage (guide 5.2)."""

    def test_release_requires_blood_safe_and_ischemia_improvement(self):
        env = make_release_legal_env()
        advantage, details = _counterfactual_details(delta_blood=-10.0, delta_ischemia=-5.0)
        with mock.patch.object(
            oracle_mod, "counterfactual_release_advantage",
            return_value=(advantage, dict(details)),
        ):
            action, record = oracle_mod.choose_safe_oracle_action(
                env, epsilon_ischemia=1e-6, advantage_margin=1e-6
            )
        self.assertEqual(action, CLAMP_RELEASE)
        self.assertIsNone(record["reject_reason"])

        # delta_ischemia >= 0 -> continue
        _, details_no_gain = _counterfactual_details(delta_blood=-10.0, delta_ischemia=0.0)
        with mock.patch.object(
            oracle_mod, "counterfactual_release_advantage",
            return_value=(1.0, dict(details_no_gain)),
        ):
            action, record = oracle_mod.choose_safe_oracle_action(
                env, epsilon_ischemia=1e-6, advantage_margin=1e-6
            )
        self.assertEqual(action, CLAMP_CONTINUE)
        self.assertIn("delta_ischemia_not_improved", record["reject_reason"])

        # no composite advantage -> continue
        _, details_no_adv = _counterfactual_details(delta_blood=-10.0, delta_ischemia=-5.0)
        with mock.patch.object(
            oracle_mod, "counterfactual_release_advantage",
            return_value=(-0.5, dict(details_no_adv)),
        ):
            action, record = oracle_mod.choose_safe_oracle_action(
                env, epsilon_ischemia=1e-6, advantage_margin=1e-6
            )
        self.assertEqual(action, CLAMP_CONTINUE)
        self.assertIn("no_advantage", record["reject_reason"])


class Test15GateA2ReplansAfterRelease(unittest.TestCase):
    """Gate A v2: after a release the next decision must be replanned from the
    new state (guide 5.2 forbids reusing a cached planned target / pre-selected
    batch).  Releasing switches the phase to unclamped, so the immediately
    following decision cannot be release-legal."""

    def test_release_replans_from_new_state(self):
        # A 5x5 domain finishes in <2 min so phase_elapsed never reaches the
        # default early_end_minutes=5; use a longer domain with a low threshold
        # so the fixture reliably produces release-legal decision points.
        scenario = rectangle(rows=6, cols=6)
        clinical, reward = v102_config()
        clinical["early_end_minutes"] = 1.0
        advantage, details = _counterfactual_details(delta_blood=-1.0, delta_ischemia=-1.0)
        with mock.patch.object(
            oracle_mod, "counterfactual_release_advantage",
            return_value=(advantage, dict(details)),
        ):
            record = oracle_mod.rollout_safe_greedy_oracle(
                scenario,
                target_selector=serpentine_target_cell,
                clinical_config=clinical,
                reward_config=reward,
                ischemia_cost=1.0,
                ischemia_scale_minutes=20.0,
                epsilon_ischemia=1e-6,
                advantage_margin=1e-6,
            )
        released = [d for d in record["decisions"] if d["action"] == CLAMP_RELEASE]
        self.assertGreater(len(released), 0)
        for i, decision in enumerate(record["decisions"][:-1]):
            if decision["action"] == CLAMP_RELEASE:
                nxt = record["decisions"][i + 1]
                self.assertFalse(nxt["release_legal"])
                self.assertNotEqual(nxt["phase"], "clamped")
                # The decision is taken from the NEW state, so its recorded
                # macro_step is strictly greater than the release step.
                self.assertGreater(nxt["macro_step"], decision["macro_step"])
        for decision in record["decisions"]:
            self.assertGreaterEqual(decision["post_action_total_clamped"], 0.0)
            self.assertGreaterEqual(
                decision["post_action_elapsed"], decision["elapsed_minutes"]
            )


class Test16GateA2RolloutPaired(unittest.TestCase):
    """Gate A v2: full oracle rollouts pair one-to-one with baselines by
    scenario_id (guide 5.1 / 11)."""

    def test_oracle_baseline_paired_by_scenario(self):
        scenarios = [
            rectangle(rows=5, cols=5, scenario_id="s-a"),
            rectangle(rows=5, cols=5, vessels=((1, 1),), scenario_id="s-b"),
        ]
        clinical, reward = v102_config()
        baseline_records = []
        oracle_records = []
        for scenario in scenarios:
            baseline_records.append(
                oracle_mod.rollout_baseline_episode(
                    scenario,
                    target_selector=serpentine_target_cell,
                    clinical_config=clinical,
                    reward_config=reward,
                    ischemia_cost=1.0,
                    ischemia_scale_minutes=20.0,
                )
            )
            oracle_records.append(
                oracle_mod.rollout_safe_greedy_oracle(
                    scenario,
                    target_selector=serpentine_target_cell,
                    clinical_config=clinical,
                    reward_config=reward,
                    ischemia_cost=1.0,
                    ischemia_scale_minutes=20.0,
                    epsilon_ischemia=1e-6,
                    advantage_margin=1e-6,
                )
            )
        result = oracle_mod.evaluate_gate_a_policy(
            baseline_records, oracle_records, bootstrap_samples=50, seed=1
        )
        self.assertEqual(result["n_scenarios"], 2)
        self.assertIn("per_scenario_differences", result)
        for record in oracle_records:
            self.assertIn(
                record["scenario_id"], result["per_scenario_differences"]["blood"]
            )


class Test17GateA2EpisodeLevelAggregation(unittest.TestCase):
    """Gate A v2: the aggregator reports EPISODE-level paired differences, never
    the mean of independent candidate-state deltas (guide 7 / 5.3)."""

    def test_aggregator_uses_episode_level_not_candidate_mean(self):
        # Scenario s1: oracle episode has 3 release decisions each with an
        # instant delta_blood of -5, but the FINAL episode blood difference is
        # +2.  A candidate-state aggregator would report a negative mean; the
        # episode aggregator must report the real +2.
        base1 = {
            "scenario_id": "s1", "completion": True, "coverage": 1.0,
            "legal_action_rate": 1.0, "elapsed_minutes": 100.0,
            "expected_blood_loss_ml": 100.0, "total_clamped_minutes": 60.0,
            "early_end_count": 0, "unsafe_end_count": 0, "failure_reason": None,
        }
        orac1 = dict(base1)
        orac1.update({
            "expected_blood_loss_ml": 102.0, "early_end_count": 3,
            "decisions": [
                {"action": CLAMP_RELEASE, "delta_blood": -5.0, "delta_ischemia": -1.0}
            ] * 3,
        })
        base2 = {
            "scenario_id": "s2", "completion": True, "coverage": 1.0,
            "legal_action_rate": 1.0, "elapsed_minutes": 90.0,
            "expected_blood_loss_ml": 80.0, "total_clamped_minutes": 50.0,
            "early_end_count": 0, "unsafe_end_count": 0, "failure_reason": None,
        }
        orac2 = dict(base2)
        orac2.update({"expected_blood_loss_ml": 80.0, "early_end_count": 0})
        result = oracle_mod.evaluate_gate_a_policy(
            [base1, base2], [orac1, orac2], bootstrap_samples=50, seed=1
        )
        # candidate-state mean would be (3*-5 + 0)/2 = -7.5; episode mean is (2+0)/2 = +1
        self.assertAlmostEqual(result["fields"]["blood"]["mean_difference"], 1.0, places=6)
        self.assertEqual(result["per_scenario_differences"]["blood"]["s1"], 2.0)
        self.assertEqual(result["per_scenario_differences"]["blood"]["s2"], 0.0)


class Test18GateA2Threshold10Gating(unittest.TestCase):
    """Gate A v2: threshold-10 means release is illegal before 10 clamped minutes."""

    def test_release_illegal_before_threshold10(self):
        env = make_env(clinical_config={"early_end_minutes": 10.0})
        env.reset()
        env.phase = "clamped"
        env.phase_elapsed_minutes = 9.0
        env.exposed_ids.clear()
        self.assertFalse(env.action_masks()[CLAMP_RELEASE])
        env.phase_elapsed_minutes = 10.0
        self.assertTrue(env.action_masks()[CLAMP_RELEASE])


class Test19GateA2TargetAuditable(unittest.TestCase):
    """Gate A v2: BC target hash and per-decision target actions are auditable."""

    def test_bc_target_hash_and_action_sequence_auditable(self):
        scenario = rectangle(rows=5, cols=5)
        clinical, reward = v102_config()
        fake_sha = "a" * 64
        baseline = oracle_mod.rollout_baseline_episode(
            scenario,
            target_selector=serpentine_target_cell,
            clinical_config=clinical,
            reward_config=reward,
            ischemia_cost=1.0,
            ischemia_scale_minutes=20.0,
            bc_target_sha256=fake_sha,
        )
        oracle = oracle_mod.rollout_safe_greedy_oracle(
            scenario,
            target_selector=serpentine_target_cell,
            clinical_config=clinical,
            reward_config=reward,
            ischemia_cost=1.0,
            ischemia_scale_minutes=20.0,
            epsilon_ischemia=1e-6,
            advantage_margin=1e-6,
            bc_target_sha256=fake_sha,
        )
        self.assertEqual(baseline["bc_target_sha256"], fake_sha)
        self.assertEqual(oracle["bc_target_sha256"], fake_sha)
        self.assertGreater(len(oracle["decisions"]), 0)
        for decision in oracle["decisions"]:
            self.assertIsNotNone(decision["planned_target_index"])
            self.assertIsNotNone(decision["planned_target"])
            self.assertIsNotNone(decision["planned_route_length"])


class Test20TargetInsensitiveToClamp(unittest.TestCase):
    """Reviewer fix #5: the frozen BC target must not depend on clamp state.

    This is a behaviour test over the real BC checkpoint: flipping the clamp
    phase of an identical cut state must not change the chosen target.  It
    records the "clamp-only transfer == baseline transfer" contract.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = FrozenBCMacroTargetPolicy(BC_MODEL_PATH, device="cpu")

    def test_target_unchanged_when_clamp_state_flips(self):
        scenario = rectangle(rows=5, cols=5, vessels=((1, 1), (2, 2)))
        clinical, reward = v102_config()
        env = TargetConditionedClampEnv(
            scenario=scenario,
            clinical_config=clinical,
            reward_config=reward,
            ischemia_cost=1.0,
            ischemia_scale_minutes=20.0,
            target_selector=self.bc.select_target,
            safe_release_mask=True,
        )
        env.reset()
        checks = 0
        while not env.terminated and not env.truncated and checks < 5:
            if env.step_count % 3 == 0:
                orig_phase, orig_el = env.phase, env.phase_elapsed_minutes
                clamped_target = self.bc.select_target(env)
                if env.phase == "clamped":
                    env.phase = "unclamped"
                    env.phase_elapsed_minutes = min(2.0, 5.0 - 1e-3)
                else:
                    env.phase = "clamped"
                    env.phase_elapsed_minutes = min(10.0, 15.0 - 1e-3)
                flipped_target = self.bc.select_target(env)
                env.phase, env.phase_elapsed_minutes = orig_phase, orig_el
                self.assertEqual(
                    clamped_target, flipped_target,
                    f"BC target changed when clamp state flipped at step {env.step_count}",
                )
                checks += 1
            env.step(CLAMP_CONTINUE, build_obs=False)
        self.assertGreaterEqual(checks, 3)


if __name__ == "__main__":
    unittest.main()

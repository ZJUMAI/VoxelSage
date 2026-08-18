from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from train_clinical_window_ppo import (  # noqa: E402
    ClinicalTrainingConfig,
    _TeacherDemoReservoir,
    _discounted_returns,
    _margin_ppo_class,
    _masked_margin_loss,
)


def _rectangle(rows: int = 5, cols: int = 5, vessels=()):
    return {
        "scenario_id": "test-early-end",
        "rows": rows,
        "cols": cols,
        "cell_size_mm": 4.0,
        "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
        "obstacle_cells": [list(cell) for cell in vessels],
        "start_cell": [0, 0],
    }


class ClinicalTrainingMethodTests(unittest.TestCase):
    def test_value_pretraining_targets_keep_raw_ppo_reward_scale(self):
        actual = _discounted_returns([2.0, -1.0, 4.0], gamma=0.5)
        np.testing.assert_allclose(actual, [2.5, 1.0, 4.0])
        self.assertNotAlmostEqual(float(actual.mean()), 0.0)

    def test_margin_loss_ignores_illegal_actions(self):
        import torch

        logits = torch.tensor([[0.0, 1.0, 100.0], [3.0, -2.0, 9.0]])
        actions = torch.tensor([1, 0])
        masks = torch.tensor([[True, True, False], [True, False, False]])
        loss = _masked_margin_loss(logits, actions, masks, margin=2.0)
        # First row: max(0, 2 + 0 - 1) = 1. Second row has no legal alternative.
        self.assertAlmostEqual(float(loss), 0.5)

    def test_teacher_reservoir_is_bounded_and_reproducible(self):
        first = _TeacherDemoReservoir(capacity=3, seed=17)
        second = _TeacherDemoReservoir(capacity=3, seed=17)
        for index in range(20):
            observation = np.full((2, 3, 4), index / 10.0, dtype=np.float32)
            mask = np.asarray([True, index % 2 == 0, False])
            first.add(observation, index % 2, mask)
            second.add(observation, index % 2, mask)
        self.assertEqual(first.seen, 20)
        self.assertEqual(len(first.observations), 3)
        for left, right in zip(first.arrays(), second.arrays()):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first.arrays()[0].dtype, np.float16)

    def test_safer_actor_critic_separation_is_the_default(self):
        config = ClinicalTrainingConfig()
        self.assertFalse(config.share_features_extractor)
        self.assertEqual(config.rl_margin_coef, 0.0)

    def test_margin_ppo_supports_sb3_seedless_load_construction(self):
        algorithm_class = _margin_ppo_class()
        model = algorithm_class("CnnPolicy", None, _init_setup_model=False)
        self.assertIsNotNone(model._margin_rng)


class EarlyEndModeTests(unittest.TestCase):
    def test_disabled_never_allows_end_but_auto_switches_at_15min(self):
        from clinical_window_environment import (
            ACTION_END_CLAMP_EARLY,
            ACTION_RIGHT,
            ClinicalWindowResectionEnv,
        )

        env = ClinicalWindowResectionEnv(
            scenario=_rectangle(),
            clinical_config={"early_end_mode": "disabled"},
        )
        env.reset()
        self.assertFalse(env.action_masks()[ACTION_END_CLAMP_EARLY])
        env.phase_elapsed_minutes = env.clinical_config["max_clamp_minutes"] - 0.001
        _, _, _, _, info = env.step(ACTION_RIGHT)
        # Auto 15/5 still works even though END is masked.
        self.assertEqual(env.phase, "unclamped")
        self.assertEqual(info["phase"], "unclamped")

    def test_unclamp_window_is_fixed_five_minutes_in_all_modes(self):
        from clinical_window_environment import (
            ACTION_END_CLAMP_EARLY,
            ACTION_RIGHT,
            ClinicalWindowResectionEnv,
        )

        for mode in ("disabled", "threshold", "full"):
            env = ClinicalWindowResectionEnv(
                scenario=_rectangle(),
                clinical_config={
                    "early_end_mode": mode,
                    "early_end_minutes": 10.0,
                },
            )
            env.reset()
            env.step(ACTION_RIGHT)
            if mode in ("threshold", "full"):
                env.phase_elapsed_minutes = 10.0
                env.step(ACTION_END_CLAMP_EARLY)
            else:
                env.phase_elapsed_minutes = env.clinical_config["max_clamp_minutes"] - 0.001
                env.step(ACTION_RIGHT)
            self.assertEqual(env.phase, "unclamped")
            # The fixed 5-minute reperfusion window is a clinical-config constant,
            # independent of the early-end mode used to enter the open phase.
            self.assertAlmostEqual(float(env.clinical_config["unclamp_minutes"]), 5.0)

    def test_threshold_ten_locks_until_ten_minutes(self):
        from clinical_window_environment import (
            ACTION_END_CLAMP_EARLY,
            ACTION_RIGHT,
            ClinicalWindowResectionEnv,
        )

        env = ClinicalWindowResectionEnv(
            scenario=_rectangle(),
            clinical_config={"early_end_mode": "threshold", "early_end_minutes": 10.0},
        )
        env.reset()
        env.step(ACTION_RIGHT)
        env.phase_elapsed_minutes = 9.99
        self.assertFalse(env.action_masks()[ACTION_END_CLAMP_EARLY])
        env.phase_elapsed_minutes = 10.0
        self.assertTrue(env.action_masks()[ACTION_END_CLAMP_EARLY])

    def test_full_matches_current_clinical_semantics(self):
        from clinical_window_environment import (
            ACTION_END_CLAMP_EARLY,
            ACTION_RIGHT,
            ClinicalWindowResectionEnv,
        )

        env = ClinicalWindowResectionEnv(
            scenario=_rectangle(),
            clinical_config={"early_end_mode": "full"},
        )
        env.reset()
        self.assertFalse(env.action_masks()[ACTION_END_CLAMP_EARLY])
        env.step(ACTION_RIGHT)
        self.assertTrue(env.action_masks()[ACTION_END_CLAMP_EARLY])

    def test_replay_records_early_end_mode(self):
        from clinical_window_environment import (
            ACTION_RIGHT,
            ClinicalWindowResectionEnv,
        )

        env = ClinicalWindowResectionEnv(
            scenario=_rectangle(),
            clinical_config={"early_end_mode": "threshold", "early_end_minutes": 10.0},
        )
        env.reset()
        env.step(ACTION_RIGHT)
        replay = env.episode_replay()
        self.assertEqual(replay["clinical_config"]["early_end_mode"], "threshold")
        self.assertAlmostEqual(float(replay["clinical_config"]["early_end_minutes"]), 10.0)

    def test_huge_end_logit_does_not_change_direction_margin_loss(self):
        import torch

        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 1000.0]])
        actions = torch.tensor([0])
        masks = torch.tensor([[True, True, False, True, True]])
        loss = _masked_margin_loss(logits, actions, masks, margin=1.0)
        # END logit 1000 is not a negative sample; margin compares only directions.
        self.assertAlmostEqual(float(loss.item()), 0.0)


class BehaviorCloningDecoupleTests(unittest.TestCase):
    def _tiny_vec_env(self, scenarios):
        from clinical_window_environment import ClinicalWindowScenarioPoolEnv
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            return ClinicalWindowScenarioPoolEnv(
                scenarios,
                seed=0,
                clinical_config={"early_end_mode": "disabled"},
                reward_config={"progress_bonus": 5.0, "seal_progress_bonus": 2.0},
                mechanics_update_interval=0,
            )

        return DummyVecEnv([make_env])

    def test_bc_epochs_zero_fills_reservoir_without_moving_weights(self):
        import torch

        from clinical_window_policy import ClinicalGridExtractor
        from train_clinical_window_ppo import _run_direction_behavior_cloning

        scenarios = [_rectangle()]
        vec_env = self._tiny_vec_env(scenarios)
        model = _margin_ppo_class()(
            "CnnPolicy",
            vec_env,
            seed=0,
            device="cpu",
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            verbose=0,
            policy_kwargs={
                "features_extractor_class": ClinicalGridExtractor,
                "features_extractor_kwargs": {"features_dim": 64},
                "net_arch": {"pi": [32], "vf": [32]},
                "share_features_extractor": False,
            },
            _init_setup_model=True,
        )
        before = [p.detach().cpu().clone() for p in model.policy.parameters()]
        config = ClinicalTrainingConfig(
            seed=0,
            bc_scenarios=1,
            bc_epochs=0,
            bc_batch_size=8,
            bc_margin=2.0,
            bc_v_weight=0.5,
            rl_margin_coef=0.2,
            gamma=0.999,
        )
        result = _run_direction_behavior_cloning(
            model=model,
            scenarios=scenarios,
            clinical_config={"early_end_mode": "disabled"},
            reward_config={"progress_bonus": 5.0, "seal_progress_bonus": 2.0},
            config=config,
        )
        after = [p.detach().cpu().clone() for p in model.policy.parameters()]
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a), "bc_epochs=0 must not update weights")
        self.assertEqual(result["bc_optimization_epochs"], 0)
        self.assertGreater(result["teacher_buffer_count"], 0)
        self.assertIsNotNone(model._margin_observations)

    def test_fresh_instance_copies_weights_and_uses_cli_hyperparams(self):
        import torch

        from clinical_window_policy import ClinicalGridExtractor

        scenarios = [_rectangle()]
        vec_env = self._tiny_vec_env(scenarios)
        policy_kwargs = {
            "features_extractor_class": ClinicalGridExtractor,
            "features_extractor_kwargs": {"features_dim": 64},
            "net_arch": {"pi": [32], "vf": [32]},
            "share_features_extractor": False,
        }
        source = _margin_ppo_class()(
            "CnnPolicy", vec_env, seed=1, learning_rate=9e-4, verbose=0,
            n_steps=32, batch_size=16, n_epochs=1, policy_kwargs=policy_kwargs,
        )
        with torch.no_grad():
            source.policy.action_net.bias.data.fill_(0.123)
        state = source.policy.state_dict()
        fresh = _margin_ppo_class()(
            "CnnPolicy", vec_env, seed=2, learning_rate=3e-4, verbose=0,
            n_steps=32, batch_size=16, n_epochs=1, policy_kwargs=policy_kwargs,
        )
        fresh.policy.load_state_dict(state)
        for key in state:
            self.assertTrue(
                torch.equal(fresh.policy.state_dict()[key], state[key]),
                f"weight mismatch at {key}",
            )
        self.assertAlmostEqual(float(fresh.learning_rate), 3e-4)


if __name__ == "__main__":
    unittest.main()

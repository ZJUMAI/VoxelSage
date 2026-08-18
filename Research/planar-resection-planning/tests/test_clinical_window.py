from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from clinical_window_environment import (  # noqa: E402
    ACTION_END_CLAMP_EARLY,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    CLINICAL_OBSERVATION_CHANNELS,
    ClinicalWindowResectionEnv,
)
from clinical_window_evaluation import rollout_clinical_policy, serpentine_direction_policy  # noqa: E402
from clinical_window_policy import LocalGlobalClinicalExtractor  # noqa: E402
from clinical_window_scenarios import generate_clinical_stage_pool  # noqa: E402


def rectangle(rows: int = 5, cols: int = 5, vessels=()):
    return {
        "scenario_id": "test-clinical",
        "rows": rows,
        "cols": cols,
        "cell_size_mm": 4.0,
        "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
        "obstacle_cells": [list(cell) for cell in vessels],
        "start_cell": [0, 0],
    }


class ClinicalWindowEnvironmentTests(unittest.TestCase):
    def test_reset_exposes_five_actions_and_padded_observation(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        observation, info = env.reset(seed=3)
        self.assertEqual(observation.shape, (len(CLINICAL_OBSERVATION_CHANNELS), 30, 40))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(env.action_space.n, 5)
        np.testing.assert_array_equal(env.action_masks(), [False, True, False, True, False])
        self.assertEqual(info["phase"], "clamped")

    def test_early_end_is_zero_time_and_reperfusion_is_fixed(self):
        env = ClinicalWindowResectionEnv(
            scenario=rectangle(),
            clinical_config={"unclamp_minutes": 0.1},
        )
        env.reset()
        env.step(ACTION_RIGHT)
        elapsed = env.elapsed_minutes
        self.assertTrue(env.action_masks()[ACTION_END_CLAMP_EARLY])
        _, _, _, _, info = env.step(ACTION_END_CLAMP_EARLY)
        self.assertEqual(env.elapsed_minutes, elapsed)
        self.assertEqual(info["phase"], "unclamped")
        self.assertFalse(env.action_masks()[ACTION_END_CLAMP_EARLY])
        env.step(ACTION_LEFT)
        env.step(ACTION_RIGHT)
        self.assertEqual(env.phase, "clamped")
        self.assertAlmostEqual(env.total_unclamped_minutes, 0.1, places=8)

    def _exposed_large_env(self, **clinical_overrides):
        config = {"bleeding_probability": 0.1, **clinical_overrides}
        env = ClinicalWindowResectionEnv(
            scenario=rectangle(vessels=((2, 2), (2, 3))),
            clinical_config=config,
        )
        env.reset()
        env.hidden_ids.clear()
        env.exposed_ids.add(0)
        env.current = (2, 1)
        env.cut.add((2, 1))
        env._update_mechanics(force=True)
        return env

    def test_large_vessel_is_sealed_once_with_triple_time(self):
        env = self._exposed_large_env()
        env.phase = "unclamped"
        rate = env._expected_bleeding_rate()
        _, _, _, _, info = env.step(ACTION_RIGHT)
        expected_duration = 3 * env.base_action_minutes
        self.assertAlmostEqual(env.elapsed_minutes, expected_duration)
        self.assertAlmostEqual(env.expected_blood_loss_ml, rate * expected_duration)
        self.assertEqual(info["sealed_component_ids"], [0])
        self.assertTrue({(2, 2), (2, 3)} <= env.cut)
        vessel_events = [event for event in info["events"] if event["action"] == "seal_and_cut_vessel"]
        self.assertEqual(len(vessel_events), 1)
        self.assertTrue(vessel_events[0]["is_large"])

    def test_action_crossing_clamp_boundary_only_bleeds_after_release(self):
        env = self._exposed_large_env()
        env.phase = "clamped"
        env.phase_elapsed_minutes = env.clinical_config["max_clamp_minutes"] - env.base_action_minutes / 2
        rate = (
            env.clinical_config["bleeding_probability"]
            * env.bleeding_beta
            * 2
            * env.cell_area_mm2
        )
        env.step(ACTION_RIGHT)
        self.assertAlmostEqual(
            env.expected_blood_loss_ml,
            rate * env.base_action_minutes * 2.5,
            places=8,
        )

    def test_uniform_probability_scales_expected_rate_linearly(self):
        low = self._exposed_large_env(bleeding_probability=0.2)
        high = self._exposed_large_env(bleeding_probability=0.4)
        low.phase = high.phase = "unclamped"
        self.assertAlmostEqual(high._expected_bleeding_rate(), 2 * low._expected_bleeding_rate())

    def test_transfer_is_logged_but_not_a_reward_term(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        env.reset()
        env.step(ACTION_RIGHT)
        _, _, _, _, info = env.step(ACTION_LEFT)
        self.assertEqual(env.transfer_count, 1)
        self.assertNotIn("transfer_cost", info["reward_terms"])
        self.assertGreater(info["reward_terms"]["time_cost"], 0)
        self.assertEqual(info["reward_terms"]["progress_bonus"], 0.0)

    def test_new_cut_gets_progress_but_transfer_does_not(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle(rows=3, cols=3))
        env.reset()
        _, cut_reward, _, _, cut_info = env.step(ACTION_RIGHT)
        _, transfer_reward, _, _, transfer_info = env.step(ACTION_LEFT)
        self.assertAlmostEqual(cut_info["reward_terms"]["progress_bonus"], -5.0 / 8.0)
        self.assertGreater(cut_reward, 0.0)
        self.assertEqual(transfer_info["reward_terms"]["progress_bonus"], 0.0)
        self.assertLess(transfer_reward, 0.0)

    def test_sealing_vessel_cross_section_gets_normalized_progress_bonus(self):
        env = self._exposed_large_env()
        env.phase = "clamped"
        _, _, _, _, info = env.step(ACTION_RIGHT)
        self.assertIn("seal_and_cut_vessel", [event["action"] for event in info["events"]])
        self.assertAlmostEqual(
            info["reward_terms"]["seal_progress_bonus"],
            -env.reward_config["seal_progress_bonus"],
        )

    def test_no_seal_progress_bonus_without_sealing(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        env.reset()
        _, _, _, _, info = env.step(ACTION_RIGHT)
        self.assertNotIn("seal_progress_bonus", info["reward_terms"])

    def test_seal_shaping_is_bounded_independent_of_component_count(self):
        env = ClinicalWindowResectionEnv(
            scenario=rectangle(rows=6, cols=6, vessels=((2, 2), (3, 4))),
        )
        env.reset()
        first = env._reward_terms(0.0, 0.0, 0, seal_count=1)
        second = env._reward_terms(0.0, 0.0, 0, seal_count=1)
        self.assertAlmostEqual(
            first["seal_progress_bonus"] + second["seal_progress_bonus"],
            -env.reward_config["seal_progress_bonus"],
        )

    def test_no_progress_streak_resets_on_cut_and_end_does_not_change_it(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        env.reset()
        env.step(ACTION_RIGHT)
        self.assertEqual(env.no_progress_streak, 0)
        env.step(ACTION_LEFT)
        self.assertEqual(env.no_progress_streak, 1)
        env.step(ACTION_END_CLAMP_EARLY)
        self.assertEqual(env.no_progress_streak, 1)

    def test_stagnation_penalty_has_grace_period_and_linear_ramp(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        env.reset()
        env.cut.update({(0, 1), (1, 1), (1, 0)})
        actions = (ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT, ACTION_UP)
        info = None
        for index in range(41):
            _, _, _, _, info = env.step(actions[index % len(actions)])
        self.assertIsNotNone(info)
        self.assertEqual(env.no_progress_streak, 41)
        self.assertAlmostEqual(
            info["reward_terms"]["stagnation_cost"],
            env.reward_config["stagnation_penalty_cap"] / 24.0,
        )

    def test_stagnation_truncates_before_global_time_limit(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        env.reset()
        env.cut.update({(0, 1), (1, 1), (1, 0)})
        actions = (ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT, ACTION_UP)
        for index in range(96):
            _, _, terminated, truncated, info = env.step(actions[index % len(actions)])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["failure_reason"].startswith("stagnation:"))
        self.assertEqual(info["max_no_progress_streak"], 96)
        self.assertLess(env.elapsed_minutes, env.clinical_config["max_episode_minutes"])

    def test_two_cell_loop_is_penalized_then_terminated_early(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        env.reset()
        env.step(ACTION_RIGHT)
        for index in range(12):
            action = ACTION_LEFT if index % 2 == 0 else ACTION_RIGHT
            _, _, terminated, truncated, info = env.step(action)
            if index < 5:
                self.assertNotIn("two_cell_loop_cost", info["reward_terms"])
            else:
                self.assertEqual(
                    info["reward_terms"]["two_cell_loop_cost"],
                    env.reward_config["two_cell_loop_penalty"],
                )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["failure_reason"].startswith("two-cell oscillation:"))
        self.assertEqual(info["max_same_edge_streak"], 12)
        self.assertEqual(info["no_progress_streak"], 12)

    def test_training_clinical_cost_is_capped_but_raw_metrics_are_not(self):
        env = ClinicalWindowResectionEnv(scenario=rectangle())
        env.reset()
        scale = env.clinical_config["time_scale_minutes"]
        cap = env.reward_config["clinical_cost_cap"]
        env.elapsed_minutes = (cap + 0.5) * scale
        crossing = env._reward_terms(scale, 0.0, 0)
        self.assertAlmostEqual(crossing["time_cost"], 0.5)
        env.elapsed_minutes = (cap + 1.5) * scale
        saturated = env._reward_terms(scale, 0.0, 0)
        self.assertAlmostEqual(saturated["time_cost"], 0.0)
        self.assertGreater(env.elapsed_minutes, cap * scale)
        observation = env._observation()
        channel = CLINICAL_OBSERVATION_CHANNELS.index("clinical_cost_fraction")
        self.assertTrue(np.all(observation[channel] == 1.0))

    def test_completed_episode_has_fixed_total_progress_bonus(self):
        item = generate_clinical_stage_pool(stage="a", count=1, seed=17, split="test")[0]
        result = rollout_clinical_policy(item, serpentine_direction_policy)
        self.assertTrue(result["completion"])
        self.assertAlmostEqual(result["reward_components"]["progress_bonus"], -5.0)

    def test_generated_large_components_use_complete_cross_section_threshold(self):
        scenarios = generate_clinical_stage_pool(stage="d", count=2, seed=9, split="test")
        env = ClinicalWindowResectionEnv(scenario=scenarios[0])
        env.reset()
        for component in env.components:
            self.assertEqual(component["is_large"], component["cross_section_cells"] >= 2)

    def test_mechanical_serpentine_completes_a_small_vessel_case(self):
        item = generate_clinical_stage_pool(stage="a", count=1, seed=11, split="test")[0]
        result = rollout_clinical_policy(item, serpentine_direction_policy)
        self.assertTrue(result["completion"])
        self.assertEqual(result["legal_action_rate"], 1.0)
        self.assertEqual(result["early_end_count"], 0)


class LocalGlobalClinicalExtractorTests(unittest.TestCase):
    CHANNELS = CLINICAL_OBSERVATION_CHANNELS
    CURRENT = LocalGlobalClinicalExtractor.CURRENT_POSITION_CHANNEL

    def _extractor(self):
        import gymnasium as gym

        space = gym.spaces.Box(low=0.0, high=1.0, shape=(len(self.CHANNELS), 30, 40), dtype=np.float32)
        return LocalGlobalClinicalExtractor(space, features_dim=256)

    def _synthetic_observation(self, position=(10, 12)):
        import torch

        observation = torch.zeros(1, len(self.CHANNELS), 30, 40)
        row, col = position
        observation[0, self.CURRENT, row, col] = 1.0
        channel = {name: index for index, name in enumerate(self.CHANNELS)}
        # 四邻域标记 domain；right 邻域额外标记 cut。
        for (delta_row, delta_col), name in (((-1, 0), "up"), ((1, 0), "down"), ((0, -1), "left"), ((0, 1), "right")):
            neighbor_row, neighbor_col = row + delta_row, col + delta_col
            if 0 <= neighbor_row < 30 and 0 <= neighbor_col < 40:
                observation[0, channel["domain"], neighbor_row, neighbor_col] = 1.0
        right_row, right_col = row, col + 1
        observation[0, channel["cut"], right_row, right_col] = 1.0
        # 四个整层全局标量通道填常量。
        for scalar_index in LocalGlobalClinicalExtractor.GLOBAL_SCALAR_CHANNELS:
            observation[0, scalar_index].fill_(float(scalar_index) / 100.0)
        return observation

    def test_forward_emits_features_dim(self):
        import torch

        extractor = self._extractor()
        output = extractor(self._synthetic_observation())
        self.assertEqual(output.shape, (1, 256))
        self.assertTrue(torch.isfinite(output).all())

    def test_local_neighborhood_features_follow_action_order(self):
        extractor = self._extractor()
        observation = self._synthetic_observation(position=(10, 12))
        local = extractor._local_neighborhood_features(observation)
        self.assertEqual(local.shape, (1, 5 * len(self.CHANNELS)))
        channel = {name: index for index, name in enumerate(self.CHANNELS)}
        # 顺序: self, up, down, left, right；每段长 len(CHANNELS)。
        self.assertAlmostEqual(float(local[0, 0 * len(self.CHANNELS) + self.CURRENT]), 1.0)
        self.assertAlmostEqual(float(local[0, 4 * len(self.CHANNELS) + channel["cut"]]), 1.0)
        # down 邻域 (11,12) 存在但只标了 domain，cut 应为 0。
        self.assertAlmostEqual(float(local[0, 2 * len(self.CHANNELS) + channel["cut"]]), 0.0)
        # domain 通道在四邻域都为 1（self 邻域是当前格，domain 未标应为 0）。
        self.assertAlmostEqual(float(local[0, 0 * len(self.CHANNELS) + channel["domain"]]), 0.0)
        self.assertAlmostEqual(float(local[0, 1 * len(self.CHANNELS) + channel["domain"]]), 1.0)
        self.assertAlmostEqual(float(local[0, 3 * len(self.CHANNELS) + channel["domain"]]), 1.0)

    def test_out_of_bounds_neighbor_is_zeroed(self):
        import torch

        extractor = self._extractor()
        # 起点在 (0,0)：up (-1,0) 与 left (0,-1) 越界。
        observation = self._synthetic_observation(position=(0, 0))
        local = extractor._local_neighborhood_features(observation)
        width = len(self.CHANNELS)
        self.assertTrue(torch.all(local[0, 1 * width:2 * width] == 0.0))  # up
        self.assertTrue(torch.all(local[0, 3 * width:4 * width] == 0.0))  # left
        self.assertAlmostEqual(float(local[0, 2 * width + 0]), 1.0)  # down domain

    def test_global_scalars_read_constant_layers(self):
        extractor = self._extractor()
        observation = self._synthetic_observation()
        scalars = extractor._global_scalars(observation)
        self.assertEqual(scalars.shape, (1, len(LocalGlobalClinicalExtractor.GLOBAL_SCALAR_CHANNELS)))
        for index, scalar_index in enumerate(LocalGlobalClinicalExtractor.GLOBAL_SCALAR_CHANNELS):
            self.assertAlmostEqual(float(scalars[0, index]), scalar_index / 100.0)


if __name__ == "__main__":
    unittest.main()

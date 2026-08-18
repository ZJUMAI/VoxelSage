from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from environment import (  # noqa: E402
    CANVAS_SIZE,
    LOCAL_OBSERVATION_CHANNELS,
    OBSERVATION_CHANNELS,
    LocalGridScenarioPoolEnv,
    PlanarResectionEnv,
)
from evaluation import (  # noqa: E402
    evaluate_row_baseline,
    evaluate_serpentine_baseline,
    evaluate_serpentine_priority_baseline,
)
from planner import plan_resection  # noqa: E402


def scenario(size: int = 5, obstacles=()):
    return {
        "rows": size,
        "cols": size,
        "domain_cells": [[row, col] for row in range(size) for col in range(size)],
        "obstacle_cells": [list(cell) for cell in obstacles],
        "start_cell": [0, 0],
    }


class PlanarResectionEnvironmentTests(unittest.TestCase):
    def test_reset_has_fixed_observation_and_frontier_mask(self):
        env = PlanarResectionEnv(scenario=scenario())
        observation, info = env.reset(seed=17)
        self.assertEqual(observation.shape, (len(OBSERVATION_CHANNELS), CANVAS_SIZE, CANVAS_SIZE))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(info["cut_count"], 1)
        mask = env.action_masks()
        self.assertTrue(mask[1])
        self.assertTrue(mask[CANVAS_SIZE])
        self.assertFalse(mask[0])
        self.assertEqual(mask.dtype, np.bool_)

    def test_action_is_a_cut_and_transfer_is_automatic(self):
        env = PlanarResectionEnv(scenario=scenario())
        env.reset()
        _, _, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual([event["action"] for event in info["events"]], ["cut"])
        _, _, _, _, info = env.step(CANVAS_SIZE)
        self.assertEqual([event["action"] for event in info["events"]], ["transfer", "cut"])
        self.assertEqual(info["events"][0]["cell"], [0, 0])

    def test_lookahead_transfer_penalty_reports_next_frontier_distance(self):
        item = {
            "rows": 3,
            "cols": 3,
            "domain_cells": [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]],
            "obstacle_cells": [],
            "start_cell": [0, 0],
        }
        env = PlanarResectionEnv(
            scenario=item,
            reward_config={"lookahead_transfer_cost": 1.0},
        )
        env.reset()
        env.step(1)
        _, _, terminated, truncated, info = env.step(2)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reward_terms"]["lookahead_transfer_cost"], 2.0)

    def test_invalid_action_is_rejected_by_environment_not_only_mask(self):
        env = PlanarResectionEnv(scenario=scenario())
        env.reset()
        _, reward, terminated, truncated, info = env.step(CANVAS_SIZE * CANVAS_SIZE - 1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, 0)
        self.assertIn("invalid action", info["failure_reason"])

    def test_release_event_precedes_cutting_released_vessel(self):
        env = PlanarResectionEnv(scenario=scenario(5, obstacles=[(2, 2)]))
        env.reset()
        release_seen = False
        for _ in range(24):
            mask = env.action_masks()
            if not mask.any():
                break
            action = int(np.flatnonzero(mask)[0])
            _, _, terminated, truncated, info = env.step(action)
            release_seen |= any(event["action"] == "release" for event in info["events"])
            if release_seen and mask[2 * CANVAS_SIZE + 2]:
                break
            if terminated or truncated:
                break
        self.assertTrue(release_seen)
        self.assertIn(0, info["released_component_ids"])

    def test_selected_planner_cuts_replay_with_same_event_semantics(self):
        item = scenario(5, obstacles=[(2, 2)])
        planner_result = plan_resection(**item)
        env = PlanarResectionEnv(scenario=item)
        env.reset()
        for event in planner_result["events"]:
            if event["action"] != "cut" or event["index"] == 0:
                continue
            cell = event["cell"]
            action = cell[0] * CANVAS_SIZE + cell[1]
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        self.assertTrue(env.terminated)
        self.assertIsNone(env.failure_reason)
        self.assertEqual(env.cut, {tuple(cell) for cell in planner_result["domain_cells"]})

    def test_row_baseline_uses_external_metrics_not_training_reward(self):
        result = evaluate_row_baseline(scenario(5, obstacles=[(2, 2)]))
        self.assertTrue(result["completion"])
        self.assertEqual(result["legal_action_rate"], 1.0)
        self.assertIn("mean_front_tension", result)
        self.assertNotIn("reward", result)

    def test_serpentine_baseline_has_zero_transfer_on_corner_started_rectangle(self):
        result = evaluate_serpentine_baseline(scenario(5))
        self.assertTrue(result["completion"])
        self.assertEqual(result["total_transfer_count"], 0)
        self.assertEqual(result["transfer_overhead"], 0.0)

    def test_serpentine_priority_baseline_handles_releasable_vessel(self):
        result = evaluate_serpentine_priority_baseline(scenario(5, obstacles=[(2, 2)]))
        self.assertTrue(result["completion"])
        self.assertEqual(result["legal_action_rate"], 1.0)
        self.assertIn("transfer_cost", result["reward_components"])

    def test_local_grid_view_preserves_actions_and_coordinates(self):
        env = LocalGridScenarioPoolEnv([scenario(7, obstacles=[(3, 3)])], grid_size=7, seed=11)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (len(LOCAL_OBSERVATION_CHANNELS), 7, 7))
        self.assertEqual(env.action_space.n, 49)
        self.assertTrue(info["action_mask"][1])
        self.assertTrue(info["action_mask"][7])
        self.assertEqual(observation[-2, 0, 0], 0.0)
        self.assertEqual(observation[-2, 6, 0], 1.0)
        self.assertEqual(observation[-1, 0, 6], 1.0)
        _, _, terminated, truncated, step_info = env.step(1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual([event["action"] for event in step_info["events"]], ["cut"])


if __name__ == "__main__":
    unittest.main()

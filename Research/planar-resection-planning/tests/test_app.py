from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

import app  # noqa: E402


class AppSmokeTests(unittest.TestCase):
    def test_frontend_copy_is_english_only(self):
        html_path = SIMULATOR_DIR / "static" / "index.html"
        html = html_path.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", html))

    def test_index_exists_and_contains_required_controls(self):
        html_path = SIMULATOR_DIR / "static" / "index.html"
        html = html_path.read_text(encoding="utf-8")
        for required_id in (
            'id="grid"',
            'id="mode-tension"',
            'id="lock-obstacles"',
            'id="run-plan"',
            'id="timeline"',
            'id="reward-chart"',
            'id="clinical-metrics"',
            'name="algorithm" value="serpentine"',
            'name="algorithm" value="planner"',
            'name="algorithm" value="clinical-v104"',
            'id="weight-distance" type="number" min="0" step="0.05" value="1"',
            'id="weight-risk"',
            'id="weight-shape"',
            'id="weight-exposure"',
            'id="tension-tool-cut"',
            'id="metric-vessel-strain"',
            'id="tension-tool-vessel"',
            'id="tension-tool-cut"',
        ):
            self.assertIn(required_id, html)
        for removed_control in (
            'id="load-ml-case"',
            'id="load-vessel-ml-case"',
            'id="load-variable-c-case"',
            'id="run-policy"',
            'resolveMlPolicyId',
            'state.policyId',
            'policy_id:',
        ):
            self.assertNotIn(removed_control, html)

    def test_v104_checkpoint_is_callable_and_returns_rollout_reward_trace(self):
        status = app.api_clinical_v104_status()
        if not status["available"]:
            self.skipTest("historical v10.4 checkpoint is not distributed")
        self.assertTrue(status["available"])
        self.assertEqual(status["gate_b_decision"], "NO-GO")
        domain = [[row, col] for row in range(6) for col in range(6)]
        result = app.api_clinical_v104_plan({
            "rows": 6,
            "cols": 6,
            "domain_cells": domain,
            "obstacle_cells": [[2, 2], [2, 3]],
            "start_cell": [0, 0],
        })
        self.assertEqual(result["policy_id"], "clinical_v104_target_order_bc")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["coverage"], 1.0)
        self.assertGreater(len(result["reward_trace"]), 0)
        self.assertEqual(
            result["reward_trace_kind"], "single_scenario_rollout"
        )
        self.assertAlmostEqual(
            result["reward_trace"][-1]["cumulative_reward"],
            result["total_reward"],
        )

    def test_generate_and_plan_handlers_do_not_need_medical_case(self):
        generated = app.api_generate({"seed": 77, "rows": 12, "cols": 12})
        planned = app.api_plan({
            "rows": generated["rows"],
            "cols": generated["cols"],
            "domain_cells": generated["domain_cells"],
            "obstacle_cells": [],
            "start_cell": generated["boundary_cells"][0],
        })
        self.assertEqual(planned["status"], "ok")
        self.assertEqual(planned["coverage"], 1.0)

    def test_serpentine_baseline_completes_a_vessel_layout_without_mechanics_rollout(self):
        domain = [[row, col] for row in range(12) for col in range(12)]
        result = app.api_plan_serpentine({
            "rows": 12,
            "cols": 12,
            "domain_cells": domain,
            "obstacle_cells": [[5, 5], [5, 6]],
            "start_cell": [0, 0],
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["release_count"], 1)
        self.assertGreaterEqual(result["transfer_count"], 0)

    def test_tension_handler_is_standalone(self):
        domain = [[row, col] for row in range(5) for col in range(5)]
        result = app.api_tension({
            "rows": 5,
            "cols": 5,
            "domain_cells": domain,
            "vessel_cells": [[2, 2]],
            "cut_cells": [[2, 1]],
        })
        self.assertEqual(result["model"], "2.5d-front-tension-v2")
        self.assertGreater(result["peak_normal_tension"], 0)

    def test_policy_endpoint_reports_scope_before_model_is_available(self):
        status = app.api_policy_status()
        self.assertIn("available", status)
        self.assertEqual(status["policy_id"], "variable_c_stage_c")
        self.assertIn("30x40", status["scope"])
        with self.assertRaises(Exception):
            app.api_policy_plan({
                "rows": 31, "cols": 6,
                "domain_cells": [[row, col] for row in range(31) for col in range(6)],
                "obstacle_cells": [], "start_cell": [0, 0],
            })

    def test_default_ml_scope_accepts_custom_connected_layout(self):
        from trained_policy import TrainedPolicyService

        domain = [[row, col] for row in range(8) for col in range(9)]
        domain.remove([7, 8])
        TrainedPolicyService._validate_scope({
            "rows": 8,
            "cols": 9,
            "domain_cells": domain,
            "obstacle_cells": [[2, 2], [4, 4]],
            "start_cell": [0, 0],
        })

    def test_default_ml_scope_accepts_30_by_40_boundary(self):
        from trained_policy import TrainedPolicyService

        domain = [[row, col] for row in range(30) for col in range(40)]
        TrainedPolicyService._validate_scope({
            "rows": 30, "cols": 40, "domain_cells": domain,
            "obstacle_cells": [[15, 20]], "start_cell": [0, 0],
        })

    def test_default_ml_scope_rejects_oversize_and_disconnected_domains(self):
        from trained_policy import TrainedPolicyService

        for rows, cols in ((31, 10), (10, 41)):
            with self.assertRaises(ValueError):
                TrainedPolicyService._validate_scope({
                    "rows": rows, "cols": cols,
                    "domain_cells": [[row, col] for row in range(rows) for col in range(cols)],
                    "obstacle_cells": [], "start_cell": [0, 0],
                })
        with self.assertRaises(ValueError):
            TrainedPolicyService._validate_scope({
                "rows": 10, "cols": 10,
                "domain_cells": [[0, 0], [9, 9]],
                "obstacle_cells": [], "start_cell": [0, 0],
            })

    def test_policy_request_cannot_override_the_registered_default(self):
        from trained_policy import TrainedPolicyService

        class FakeModel:
            def predict(self, *args, **kwargs):
                return 0, None

        service = TrainedPolicyService()
        service._model = FakeModel()
        service.load = lambda *args, **kwargs: {}
        result = service.plan({
            "policy_id": "toy5_plain",
            "rows": 1, "cols": 1, "domain_cells": [[0, 0]],
            "obstacle_cells": [], "start_cell": [0, 0],
        })
        self.assertEqual(result["policy_id"], "variable_c_stage_c")


if __name__ == "__main__":
    unittest.main()

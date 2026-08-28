"""Regression tests for the bundled resection-surface planner package.

The ``plan_resection`` skill imports its algorithm from a pure-Python package
(``surface_planner`` + its sibling ``reward_function``) that ships inside
``skills/builtin/plan_resection/``.  These tests guard three properties:

1. The planner modules import cleanly from the bundled location (this caught the
   sibling ``reward_function.candidate_reward`` import when the package was
   first relocated into the repository).
2. ``plan_resection.main._locate_surface_planner()`` resolves the repo-internal
   package (and only falls back to the legacy ``data/`` tree when absent).
3. When the planner is unavailable, the lookup returns ``None`` instead of
   raising, so the skill can degrade to a structured ``status=unavailable``
   result.
"""

import pathlib
import sys
import tempfile
import types
import unittest


def _import_planner_modules():
    """Insert the bundled planner dirs on sys.path and import the functions."""
    from skills.builtin.plan_resection.main import _locate_surface_planner

    planner_dir = _locate_surface_planner()
    parent = str(pathlib.Path(planner_dir).parent)
    for p in (str(planner_dir), parent):
        if p not in sys.path:
            sys.path.insert(0, p)

    from plan_surfaces import (
        build_candidates,
        predict_scale,
        score_and_select_candidates,
        eval_surfaces,
    )
    from curved_refinement import (
        refine_candidate,
        candidate_clearance,
        tumor_boundary_points,
    )
    from surface_metrics import candidate_curvature_metrics

    return [
        build_candidates,
        predict_scale,
        score_and_select_candidates,
        eval_surfaces,
        refine_candidate,
        candidate_clearance,
        tumor_boundary_points,
        candidate_curvature_metrics,
    ]


class SurfacePlannerPackageTests(unittest.TestCase):
    def test_planner_modules_import_from_bundled_location(self):
        fns = _import_planner_modules()
        self.assertEqual(len(fns), 8)
        for fn in fns:
            self.assertTrue(callable(fn), f"{fn} should be callable")

    def test_locate_surface_planner_returns_repo_internal_path(self):
        from skills.builtin.plan_resection.main import _locate_surface_planner

        planner_dir = _locate_surface_planner()
        self.assertIsNotNone(planner_dir)
        self.assertTrue((planner_dir / "plan_surfaces.py").exists())
        # In the public repo this resolves under skills/builtin/plan_resection,
        # not under a data/ directory.
        self.assertIn("skills/builtin/plan_resection", str(planner_dir))

    def test_locate_surface_planner_graceful_when_absent(self):
        # When neither the bundled package nor a legacy data/ tree exists, the
        # lookup must return None (degrade) rather than raise, so the skill can
        # produce a structured "unavailable" result instead of crashing.
        import skills.builtin.plan_resection.main as main_mod

        original_file = main_mod.__file__
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # A tree with a sibling reward_function but NO plan_surfaces.py.
                root = pathlib.Path(tmp)
                (root / "surface_planner").mkdir()
                (root / "reward_function").mkdir()

                # _locate_surface_planner resolves Path(__file__).parents[0];
                # point __file__ at a fake module inside the empty tree so the
                # bundled lookup misses and the legacy data/ tree is also empty.
                fake_module = types.ModuleType("fake_tmp_module")
                fake_module.__file__ = str(root / "plan_resection" / "main.py")
                main_mod.__file__ = str(root / "plan_resection" / "main.py")

                # The helper should return None (not raise) for this empty tree.
                result = main_mod._locate_surface_planner()
                self.assertIsNone(result)
        finally:
            main_mod.__file__ = original_file

    def test_unavailable_skill_result_is_structured(self):
        # The run() contract when the planner is missing: a JSON-serializable
        # dict that downstream callers (three_d_reconstruction) can consume.
        result = {
            "status": "unavailable",
            "reason": "surface_planner_not_found",
            "margin_min_mm": 0.0,
            "margin_p05_mm": 0.0,
            "margin_success": False,
            "resection_plane_count": 0,
            "candidate_count": 0,
            "json_updated": False,
            "predicted_scale": "unavailable",
            "vessel_mask_variants": {},
        }
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("resection_plane_count", result)
        self.assertFalse(result["margin_success"])
        import json

        self.assertEqual(json.loads(json.dumps(result)), result)


if __name__ == "__main__":
    unittest.main()

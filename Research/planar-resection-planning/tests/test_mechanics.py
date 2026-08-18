from __future__ import annotations

import sys
import unittest
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from mechanics import solve_tension  # noqa: E402


def square_domain(size: int):
    return [(row, col) for row in range(size) for col in range(size)]


class TensionMechanicsTests(unittest.TestCase):
    def base_payload(self):
        return {
            "rows": 5,
            "cols": 5,
            "domain_cells": square_domain(5),
        }

    def test_no_anchor_or_pull_is_required(self):
        result = solve_tension(**self.base_payload())
        self.assertEqual(result["model"], "2.5d-front-tension-v2")
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["peak_normal_tension"], 0)
        self.assertGreater(result["peak_organ_energy"], 0)

    def test_thickness_is_thin_at_boundary_and_thick_in_center(self):
        result = solve_tension(**self.base_payload())
        cells = {tuple(item["cell"]): item for item in result["cells"]}
        self.assertLess(cells[(0, 0)]["thickness"], cells[(2, 2)]["thickness"])

    def test_cut_breaks_only_normal_connection_and_recomputes(self):
        intact = solve_tension(**self.base_payload())
        cut = solve_tension(**self.base_payload(), cut_cells=[(2, 2)])
        self.assertEqual(cut["active_cell_count"], intact["active_cell_count"])
        self.assertLess(cut["spring_count"], intact["spring_count"])
        center = next(item for item in cut["cells"] if item["cell"] == [2, 2])
        self.assertTrue(center["is_cut"])
        self.assertEqual(center["normal_tension"], 0.0)

    def test_cut_front_has_nonzero_interface_and_front_tension(self):
        result = solve_tension(
            **self.base_payload(),
            cut_cells=[[2, 1], [2, 2], [2, 3]],
        )
        front = [item for item in result["cells"] if item["is_front"]]
        self.assertTrue(front)
        self.assertGreater(result["peak_front_tension"], 0)
        self.assertGreater(max(item["interface_tension"] for item in front), 0)
        self.assertTrue(any(item["is_tip"] for item in result["cells"] if item["is_cut"]))

    def test_straight_cuts_have_a_visible_front_at_multiple_depths(self):
        peaks = []
        for length in (1, 3, 5):
            result = solve_tension(
                rows=9,
                cols=9,
                domain_cells=square_domain(9),
                cut_cells=[[4, column] for column in range(1, 1 + length)],
            )
            peaks.append(result["peak_front_tension"])
            self.assertGreater(result["peak_front_tension"], 0)
            self.assertTrue(result["front_cells"])
        self.assertTrue(all(value > 0 for value in peaks))

    def test_zero_prestrain_has_zero_tension(self):
        result = solve_tension(
            **self.base_payload(),
            parameters={"prestrain": 0, "lateral_prestrain": 0},
        )
        self.assertEqual(result["peak_normal_tension"], 0.0)
        self.assertEqual(result["peak_shear_tension"], 0.0)
        self.assertEqual(result["peak_front_tension"], 0.0)

    def test_symmetric_cut_has_symmetric_front_response(self):
        result = solve_tension(
            rows=7,
            cols=7,
            domain_cells=square_domain(7),
            cut_cells=[[3, 2], [3, 3], [3, 4]],
        )
        values = {tuple(item["cell"]): item["front_tension"] for item in result["cells"]}
        self.assertAlmostEqual(values[(2, 2)], values[(4, 2)], places=6)
        self.assertAlmostEqual(values[(2, 3)], values[(4, 3)], places=6)

    def test_vessel_is_included_in_heatmap_and_has_strain(self):
        result = solve_tension(**self.base_payload(), vessel_cells=[(2, 2)])
        vessel = next(item for item in result["cells"] if item["cell"] == [2, 2])
        self.assertTrue(vessel["is_vessel"])
        self.assertGreaterEqual(vessel["vessel_strain"], 0)

    def test_vessel_cannot_be_cut(self):
        with self.assertRaisesRegex(ValueError, "Vessel cells cannot be cut"):
            solve_tension(**self.base_payload(), vessel_cells=[(2, 2)], cut_cells=[(2, 2)])

    def test_old_external_force_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not accept anchor_cells or tractions"):
            solve_tension(**self.base_payload(), anchor_cells=[(0, 0)])

    def test_same_input_is_deterministic(self):
        payload = {
            **self.base_payload(),
            "vessel_cells": [(2, 3)],
            "cut_cells": [(2, 2)],
        }
        self.assertEqual(solve_tension(**payload), solve_tension(**payload))


if __name__ == "__main__":
    unittest.main()

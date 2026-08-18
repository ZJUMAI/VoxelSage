from __future__ import annotations

import math
import sys
import unittest
from collections import deque
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from planner import (  # noqa: E402
    boundary_cells,
    generate_domain,
    is_connected,
    neighbors4,
    plan_resection,
)


def square_domain(size: int):
    return {(r, c) for r in range(size) for c in range(size)}


def has_hole(domain, rows, cols):
    outside = set()
    queue = deque()
    for r in range(rows):
        for c in (0, cols - 1):
            if (r, c) not in domain:
                outside.add((r, c))
                queue.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if (r, c) not in domain:
                outside.add((r, c))
                queue.append((r, c))
    while queue:
        current = queue.popleft()
        for nxt in neighbors4(current):
            r, c = nxt
            if 0 <= r < rows and 0 <= c < cols and nxt not in domain and nxt not in outside:
                outside.add(nxt)
                queue.append(nxt)
    return any(
        (r, c) not in domain and (r, c) not in outside
        for r in range(rows)
        for c in range(cols)
    )


class DomainGenerationTests(unittest.TestCase):
    def test_seed_is_reproducible_and_domain_has_required_shape(self):
        first = generate_domain(seed=20260728)
        second = generate_domain(seed=20260728)

        self.assertEqual(first, second)
        domain = {tuple(cell) for cell in first["domain_cells"]}
        self.assertTrue(is_connected(domain))
        self.assertFalse(has_hole(domain, first["rows"], first["cols"]))

        rs = [cell[0] for cell in domain]
        cs = [cell[1] for cell in domain]
        rectangle_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        self.assertLess(len(domain), rectangle_area)

    def test_random_domains_have_compact_smooth_outlines(self):
        for seed in (1, 9, 77, 20260728):
            generated = generate_domain(seed=seed, rows=24, cols=30)
            domain = {tuple(cell) for cell in generated["domain_cells"]}
            perimeter = sum(
                neighbor not in domain
                for cell in domain
                for neighbor in neighbors4(cell)
            )
            compactness = 4 * math.pi * len(domain) / perimeter ** 2
            self.assertGreater(compactness, 0.55, msg=f"seed={seed}")

    def test_explicit_size_and_limits(self):
        generated = generate_domain(seed=9, rows=12, cols=17)
        self.assertEqual((generated["rows"], generated["cols"]), (12, 17))
        with self.assertRaisesRegex(ValueError, "between 10 and 50"):
            generate_domain(seed=9, rows=9, cols=17)


class PlannerValidationTests(unittest.TestCase):
    def test_rejects_boundary_obstacle_and_interior_start(self):
        domain = square_domain(7)
        with self.assertRaisesRegex(ValueError, "boundary"):
            plan_resection(
                rows=7, cols=7, domain_cells=sorted(domain),
                obstacle_cells=[(0, 3)], start_cell=(0, 0),
            )
        with self.assertRaisesRegex(ValueError, "start_cell"):
            plan_resection(
                rows=7, cols=7, domain_cells=sorted(domain),
                obstacle_cells=[], start_cell=(3, 3),
            )

    def test_weight_validation(self):
        domain = square_domain(5)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            plan_resection(
                rows=5, cols=5, domain_cells=sorted(domain),
                obstacle_cells=[], start_cell=(0, 0),
                weights={"distance": -1},
            )


class DynamicPlanningTests(unittest.TestCase):
    def test_events_obey_frontier_transfer_and_release_invariants(self):
        domain = square_domain(7)
        obstacle = {(3, 3)}
        result = plan_resection(
            rows=7, cols=7, domain_cells=sorted(domain),
            obstacle_cells=sorted(obstacle), start_cell=(0, 0),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["release_count"], 1)

        cut = set()
        released = set()
        component = result["components"][0]
        ring = {tuple(cell) for cell in component["ring"]}
        obstacle_cut_index = None
        release_index = None
        for event in result["events"]:
            action = event["action"]
            if action == "transfer":
                self.assertIn(tuple(event["cell"]), cut)
            elif action == "cut":
                cell = tuple(event["cell"])
                if cut:
                    self.assertTrue(any(nxt in cut for nxt in neighbors4(cell)))
                if cell in obstacle:
                    obstacle_cut_index = event["index"]
                    self.assertIn(component["id"], released)
                cut.add(cell)
            elif action == "release":
                self.assertTrue(ring <= cut)
                release_index = event["index"]
                released.add(event["component_id"])

        self.assertIsNotNone(release_index)
        self.assertIsNotNone(obstacle_cut_index)
        self.assertLess(release_index, obstacle_cut_index)

    def test_enclosed_tissue_reports_partial_deadlock(self):
        domain = square_domain(9)
        center = (4, 4)
        obstacle_ring = {
            (r, c)
            for r in range(3, 6)
            for c in range(3, 6)
            if (r, c) != center
        }
        result = plan_resection(
            rows=9, cols=9, domain_cells=sorted(domain),
            obstacle_cells=sorted(obstacle_ring), start_cell=(0, 0),
        )

        self.assertEqual(result["status"], "partial")
        self.assertLess(result["coverage"], 1.0)
        self.assertEqual(result["release_count"], 0)
        self.assertIn([4, 4], result["uncovered_cells"])
        self.assertTrue(result["failure_reason"])

    def test_same_input_is_deterministic(self):
        generated = generate_domain(seed=1234, rows=18, cols=18)
        domain = {tuple(cell) for cell in generated["domain_cells"]}
        interior = sorted(domain - boundary_cells(domain))
        obstacles = interior[len(interior) // 2:len(interior) // 2 + 3]
        kwargs = dict(
            rows=18,
            cols=18,
            domain_cells=generated["domain_cells"],
            obstacle_cells=obstacles,
            start_cell=generated["boundary_cells"][0],
        )
        self.assertEqual(plan_resection(**kwargs), plan_resection(**kwargs))

    def test_weights_can_change_the_cut_order(self):
        generated = generate_domain(seed=31, rows=18, cols=18)
        domain = {tuple(cell) for cell in generated["domain_cells"]}
        interior = sorted(domain - boundary_cells(domain))
        obstacles = interior[len(interior) // 2:len(interior) // 2 + 4]
        common = dict(
            rows=18,
            cols=18,
            domain_cells=generated["domain_cells"],
            obstacle_cells=obstacles,
            start_cell=generated["boundary_cells"][0],
        )
        movement_only = plan_resection(
            **common,
            weights={"distance": 4, "vessel_risk": 0, "shape": 0, "exposure": 0},
        )
        vessel_first = plan_resection(
            **common,
            weights={"distance": 0, "vessel_risk": 8, "shape": 1, "exposure": 1},
        )
        movement_cuts = [event["cell"] for event in movement_only["events"] if event["action"] == "cut"]
        vessel_cuts = [event["cell"] for event in vessel_first["events"] if event["action"] == "cut"]
        self.assertNotEqual(movement_cuts, vessel_cuts)


if __name__ == "__main__":
    unittest.main()

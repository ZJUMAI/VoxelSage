from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from metrics import normalized_objective, validate_and_measure  # noqa: E402
from generalization_evaluation import generate_generalization_splits  # noqa: E402
from environment import PlanarResectionEnv  # noqa: E402
from planner import plan_resection  # noqa: E402
from run_four_weight_search import _coarse_candidates, _weight_key  # noqa: E402
from run_ga_four_weight_search import (  # noqa: E402
    GAConfig,
    _initial_population,
    _local_candidates,
    _next_generation,
    _weight_key as _ga_weight_key,
)
from scenarios import generate_experiment_splits, generate_pilot_scenarios  # noqa: E402
from train_masked_ppo import toy_scenarios  # noqa: E402


class PilotScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = generate_pilot_scenarios()

    def test_pilot_is_reproducible_and_has_documented_distribution(self):
        self.assertEqual(self.pilot, generate_pilot_scenarios())
        self.assertEqual(self.pilot["scenario_count"], 100)
        counts = Counter(scenario["scenario_type"] for scenario in self.pilot["scenarios"])
        self.assertEqual(counts, {
            "isolated": 20,
            "compact": 25,
            "elongated": 20,
            "multiple": 20,
            "stress_ring": 15,
        })
        self.assertTrue(all(len(scenario["starts"]) == 3 for scenario in self.pilot["scenarios"]))

    def test_regular_and_stress_scenarios_have_expected_behavior(self):
        regular = self.pilot["scenarios"][0]
        stress = self.pilot["scenarios"][-1]
        for scenario, expected in ((regular, "ok"), (stress, "partial")):
            result = plan_resection(
                rows=scenario["rows"], cols=scenario["cols"],
                domain_cells=scenario["domain_cells"],
                obstacle_cells=scenario["obstacle_cells"],
                start_cell=scenario["starts"][0]["cell"],
                weights={"distance": 1, "vessel_risk": 2, "shape": 1, "exposure": 0.75},
            )
            metric = validate_and_measure(result)
            self.assertEqual(result["status"], expected)
            self.assertTrue(metric["event_valid"])

    def test_baseline_normalizes_to_one(self):
        scenario = self.pilot["scenarios"][0]
        result = plan_resection(
            rows=scenario["rows"], cols=scenario["cols"],
            domain_cells=scenario["domain_cells"], obstacle_cells=scenario["obstacle_cells"],
            start_cell=scenario["starts"][0]["cell"],
            weights={"distance": 1, "vessel_risk": 0, "shape": 0, "exposure": 0},
        )
        metric = validate_and_measure(result)
        self.assertEqual(normalized_objective(metric, metric), 1.0)

    def test_experiment_splits_are_seed_disjoint_and_versioned(self):
        payload = generate_experiment_splits(train_count=3, validation_count=2, test_count=2, stress_count=2)
        self.assertEqual(payload["split_version"], 1)
        self.assertEqual(payload["counts"], {"train": 3, "validation": 2, "test": 2, "stress": 2})
        seeds = [scenario["seed"] for items in payload["splits"].values() for scenario in items]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_toy_scenarios_are_deterministic_and_match_curriculum_size(self):
        first = toy_scenarios(size=5, count=4, seed=91, with_vessel=False)
        self.assertEqual(first, toy_scenarios(size=5, count=4, seed=91, with_vessel=False))
        self.assertTrue(all(item["rows"] == 5 and item["cols"] == 5 for item in first))
        self.assertTrue(all(not item["obstacle_cells"] for item in first))

    def test_generalization_splits_are_deterministic_feasible_and_stratified(self):
        first = generate_generalization_splits(test_count=8, stress_count=8)
        self.assertEqual(first, generate_generalization_splits(test_count=8, stress_count=8))
        self.assertEqual(first["counts"], {"test": 8, "stress": 8})
        for split, scenarios in first["splits"].items():
            self.assertEqual(len({item["category"] for item in scenarios}), 4)
            for item in scenarios:
                self.assertEqual(item["split"], split)
                env = PlanarResectionEnv(scenario=item)
                env.reset()
                self.assertTrue(env.domain - env.obstacles)


class FourWeightSearchTests(unittest.TestCase):
    def test_four_weight_coarse_grid_has_documented_shape(self):
        candidates = _coarse_candidates()
        self.assertEqual(len(candidates), 567)
        self.assertEqual(candidates[0], {"distance": 1.0, "vessel_risk": 0.0, "shape": 0.0, "exposure": 0.0})
        self.assertEqual(candidates[-1], {"distance": 1.0, "vessel_risk": 4.0, "shape": 4.0, "exposure": 3.0})
        self.assertEqual(len({_weight_key(item) for item in candidates}), 567)

    def test_four_weight_runner_resumes_existing_candidate_records(self):
        """A resumed search must retain the completed candidate's atomic record."""
        runner = SIMULATOR_DIR / "run_four_weight_search.py"
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "four_weight"
            command = [
                sys.executable, str(runner), "--output-dir", str(output_dir),
                "--workers", "8", "--max-candidates", "1",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            candidate_path = output_dir / "candidates" / "risk_0.00_shape_0.00_exposure_0.00.json"
            first_record = candidate_path.read_bytes()

            resumed = subprocess.run(command, check=True, capture_output=True, text=True)

            self.assertIn("Reusing risk_0.00_shape_0.00_exposure_0.00", resumed.stdout)
            self.assertEqual(candidate_path.read_bytes(), first_record)
            self.assertTrue((output_dir / "summary.json").is_file())


class GeneticFourWeightSearchTests(unittest.TestCase):
    def test_ga_candidate_schedule_is_deterministic_bounded_and_unique(self):
        """The seeded GA must schedule reproducible, non-repeated valid candidates."""
        config = GAConfig()
        first = _initial_population(config)
        second = _initial_population(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)

        # Deterministic synthetic ranks stand in for completed Pilot-100 metrics.
        ranked = sorted(first, key=lambda item: _ga_weight_key(item["weights"]))
        next_first = _next_generation(config, 1, ranked, { _ga_weight_key(item["weights"]) for item in first })
        next_second = _next_generation(config, 1, ranked, { _ga_weight_key(item["weights"]) for item in second })
        scheduled = first + next_first
        self.assertEqual(next_first, next_second)
        self.assertEqual(len(next_first), 20)
        self.assertEqual(len({_ga_weight_key(item["weights"]) for item in scheduled}), 40)
        for candidate in scheduled:
            weights = candidate["weights"]
            self.assertEqual(weights["distance"], 1.0)
            self.assertGreaterEqual(weights["vessel_risk"], 0.0)
            self.assertLessEqual(weights["vessel_risk"], 4.0)
            self.assertGreaterEqual(weights["shape"], 0.0)
            self.assertLessEqual(weights["shape"], 4.0)
            self.assertGreaterEqual(weights["exposure"], 0.0)
            self.assertLessEqual(weights["exposure"], 3.0)

        # The full 20 × 8 schedule must remain unique even when rankings are
        # deterministic stand-ins for the expensive Pilot-100 evaluations.
        def full_schedule() -> list[dict[str, object]]:
            population = _initial_population(config)
            scheduled_population = list(population)
            used = {_ga_weight_key(item["weights"]) for item in population}
            for generation in range(1, config.generations):
                synthetic_ranked = sorted(population, key=lambda item: _ga_weight_key(item["weights"]))
                population = _next_generation(config, generation, synthetic_ranked, used)
                scheduled_population.extend(population)
                used.update(_ga_weight_key(item["weights"]) for item in population)
            return scheduled_population

        full_first = full_schedule()
        full_second = full_schedule()
        self.assertEqual(full_first, full_second)
        self.assertEqual(len(full_first), 160)
        self.assertEqual(len({_ga_weight_key(item["weights"]) for item in full_first}), 160)

    def test_local_grid_has_11_cubed_candidates_around_interior_ga_winner(self):
        winner = {"distance": 1.0, "vessel_risk": 2.0, "shape": 2.0, "exposure": 1.5}
        candidates = _local_candidates(winner)
        self.assertEqual(len(candidates), 11 ** 3)
        self.assertEqual(len({_ga_weight_key(item) for item in candidates}), 11 ** 3)
        self.assertEqual(candidates[0], {"distance": 1.0, "vessel_risk": 1.5, "shape": 1.5, "exposure": 1.0})
        self.assertEqual(candidates[-1], {"distance": 1.0, "vessel_risk": 2.5, "shape": 2.5, "exposure": 2.0})


if __name__ == "__main__":
    unittest.main()

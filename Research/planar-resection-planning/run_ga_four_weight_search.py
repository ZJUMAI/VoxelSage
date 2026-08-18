#!/usr/bin/env python3
"""Resumable GA screening followed by local 0.1 exhaustive weight search.

This is deliberately separate from the historical exhaustive-grid runner.  It
uses the same Pilot-100 scenarios, planner records, and aggregate semantics,
but records genetic-search provenance so a stopped run can be resumed and
audited without repeating a complete candidate evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from run_pilot import (  # noqa: E402
    BASELINE_WEIGHTS,
    _aggregate,
    _ensure_scenarios,
    _record_index,
    _run_records,
)
from scenarios import PILOT_BASE_SEED  # noqa: E402

SEED = 2026073001
TENTHS_PER_UNIT = 10
RISK_AND_SHAPE_MAX = 40
EXPOSURE_MAX = 30


@dataclass(frozen=True)
class GAConfig:
    """Frozen genetic-search configuration used in every resumable run."""

    seed: int = SEED
    population_size: int = 20
    generations: int = 8
    elite_count: int = 2
    tournament_size: int = 3
    mutation_radius_tenths: int = 5

    @property
    def total_candidates(self) -> int:
        return self.population_size * self.generations


def _write_json(path: Path, payload: Any) -> None:
    """Atomically publish a JSON checkpoint that is safe to reuse on restart."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CURRENT_DIR.parents[1], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _weight_key(weights: Mapping[str, float]) -> str:
    """Canonical key for the 0.1 lattice searched by both stages."""
    return (
        f"risk_{float(weights['vessel_risk']):.2f}_shape_{float(weights['shape']):.2f}_"
        f"exposure_{float(weights['exposure']):.2f}"
    )


def _units(weights: Mapping[str, float]) -> tuple[int, int, int]:
    return (
        int(round(float(weights["vessel_risk"]) * TENTHS_PER_UNIT)),
        int(round(float(weights["shape"]) * TENTHS_PER_UNIT)),
        int(round(float(weights["exposure"]) * TENTHS_PER_UNIT)),
    )


def _weights(units: tuple[int, int, int]) -> dict[str, float]:
    risk, shape, exposure = units
    return {
        "distance": 1.0,
        "vessel_risk": risk / TENTHS_PER_UNIT,
        "shape": shape / TENTHS_PER_UNIT,
        "exposure": exposure / TENTHS_PER_UNIT,
    }


def _clamp_units(units: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        min(RISK_AND_SHAPE_MAX, max(0, units[0])),
        min(RISK_AND_SHAPE_MAX, max(0, units[1])),
        min(EXPOSURE_MAX, max(0, units[2])),
    )


def _all_unit_points() -> Iterable[tuple[int, int, int]]:
    for risk in range(RISK_AND_SHAPE_MAX + 1):
        for shape in range(RISK_AND_SHAPE_MAX + 1):
            for exposure in range(EXPOSURE_MAX + 1):
                yield risk, shape, exposure


def _candidate(weights: Mapping[str, float], provenance: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _weights(_clamp_units(_units(weights)))
    return {"weights": canonical, "candidate_key": _weight_key(canonical), "provenance": dict(provenance)}


def _random_unused(rng: random.Random, used: set[str]) -> dict[str, float]:
    """Sample an unused lattice point, with a deterministic exhaustion fallback."""
    for _ in range(1000):
        units = (rng.randint(0, RISK_AND_SHAPE_MAX), rng.randint(0, RISK_AND_SHAPE_MAX), rng.randint(0, EXPOSURE_MAX))
        weights = _weights(units)
        if _weight_key(weights) not in used:
            return weights
    for units in _all_unit_points():
        weights = _weights(units)
        if _weight_key(weights) not in used:
            return weights
    raise RuntimeError("The GA candidate lattice is exhausted")


def _initial_population(config: GAConfig) -> list[dict[str, Any]]:
    """Generate generation zero deterministically on the bounded 0.1 lattice."""
    rng = random.Random(config.seed)
    used: set[str] = set()
    population: list[dict[str, Any]] = []
    for slot in range(config.population_size):
        weights = _random_unused(rng, used)
        used.add(_weight_key(weights))
        population.append(_candidate(weights, {"method": "initial_random", "seed": config.seed, "slot": slot}))
    return population


def _tournament(rng: random.Random, ranked: Sequence[Mapping[str, Any]], size: int) -> Mapping[str, Any]:
    participants = [ranked[rng.randrange(len(ranked))] for _ in range(size)]
    return min(participants, key=lambda candidate: int(candidate.get("rank", ranked.index(candidate) + 1)))


def _mutate(rng: random.Random, units: tuple[int, int, int], radius: int) -> tuple[int, int, int]:
    coordinate = rng.randrange(3)
    delta = rng.randint(-radius, radius)
    if delta == 0:
        delta = 1 if rng.random() < 0.5 else -1
    changed = list(units)
    changed[coordinate] += delta
    return _clamp_units((changed[0], changed[1], changed[2]))


def _next_generation(
    config: GAConfig,
    generation: int,
    ranked: Sequence[Mapping[str, Any]],
    used: set[str],
) -> list[dict[str, Any]]:
    """Create one new, unique population with elite-parent mutation and crossover.

    Candidates are never re-evaluated.  Elitism therefore preserves the best
    candidates in the parent pool and forces two one-parent mutations, rather
    than spending evaluations on duplicate elite copies.
    """
    if len(ranked) != config.population_size:
        raise ValueError("A complete ranked population is required to make the next GA generation")
    rng = random.Random(config.seed + generation * 1_000_003)
    next_population: list[dict[str, Any]] = []
    generation_used = set(used)
    for slot in range(config.population_size):
        if slot < config.elite_count:
            parent = ranked[slot]
            parent_weights = parent["weights"]
            proposed = _weights(_mutate(rng, _units(parent_weights), config.mutation_radius_tenths))
            provenance: dict[str, Any] = {
                "method": "elite_mutation",
                "generation": generation,
                "elite_rank": int(parent.get("rank", ranked.index(parent) + 1)),
                "parent_keys": [str(parent["candidate_key"])],
            }
        else:
            parent_a = _tournament(rng, ranked, config.tournament_size)
            parent_b = _tournament(rng, ranked, config.tournament_size)
            units_a, units_b = _units(parent_a["weights"]), _units(parent_b["weights"])
            crossed = tuple(units_a[index] if rng.random() < 0.5 else units_b[index] for index in range(3))
            proposed = _weights(_mutate(rng, crossed, config.mutation_radius_tenths))
            provenance = {
                "method": "tournament_crossover_mutation",
                "generation": generation,
                "parent_keys": [str(parent_a["candidate_key"]), str(parent_b["candidate_key"])],
                "parent_ranks": [
                    int(parent_a.get("rank", ranked.index(parent_a) + 1)),
                    int(parent_b.get("rank", ranked.index(parent_b) + 1)),
                ],
            }
        if _weight_key(proposed) in generation_used:
            proposed = _random_unused(rng, generation_used)
            provenance["duplicate_resolution"] = "random_unused_lattice_point"
        key = _weight_key(proposed)
        generation_used.add(key)
        next_population.append(_candidate(proposed, {**provenance, "slot": slot}))
    return next_population


def _local_candidates(winner: Mapping[str, float]) -> list[dict[str, float]]:
    """Build the deduplicated ±0.5, 0.1 local lattice around the GA winner."""
    risk, shape, exposure = _units(winner)
    candidates: list[dict[str, float]] = []
    seen: set[str] = set()
    for candidate_risk in range(max(0, risk - 5), min(RISK_AND_SHAPE_MAX, risk + 5) + 1):
        for candidate_shape in range(max(0, shape - 5), min(RISK_AND_SHAPE_MAX, shape + 5) + 1):
            for candidate_exposure in range(max(0, exposure - 5), min(EXPOSURE_MAX, exposure + 5) + 1):
                weights = _weights((candidate_risk, candidate_shape, candidate_exposure))
                key = _weight_key(weights)
                if key not in seen:
                    seen.add(key)
                    candidates.append(weights)
    return candidates


def _rank(summary: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["eligible_hard_failures"]),
        float(summary["mean_objective"]) if summary["mean_objective"] is not None else float("inf"),
        float(summary["p90_objective"]) if summary["p90_objective"] is not None else float("inf"),
        float(summary["mean_planning_time_ms"]) if summary["mean_planning_time_ms"] is not None else float("inf"),
    )


def _complete_candidate(payload: Any, weights: Mapping[str, float], expected_runs: int) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("weights") == dict(weights)
        and isinstance(payload.get("records"), list)
        and len(payload["records"]) == expected_runs
    )


def _candidate_result(
    candidate_dir: Path,
    candidate: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, object]],
    baseline_index: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_runs: int,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    weights = candidate["weights"]
    key = str(candidate["candidate_key"])
    candidate_path = candidate_dir / f"{key}.json"
    payload = _load_json(candidate_path) if resume and candidate_path.exists() else None
    if _complete_candidate(payload, weights, expected_runs):
        records = payload["records"]
        reused = True
    else:
        records = _run_records(scenarios, weights, workers)
        _write_json(candidate_path, {
            "weights": weights,
            "candidate_key": key,
            "first_evaluation": dict(candidate),
            "records": records,
        })
        reused = False
    summary = _aggregate(records, baseline_index)
    return {**dict(candidate), "summary": summary, "candidate_file": f"candidates/{key}.json", "reused": reused}


def _state_results(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for generation in state.get("generations", []):
        results.extend(generation.get("results", []))
    results.extend(state.get("local_search", {}).get("results", []))
    return results


def _checkpoint(output_dir: Path, state: MutableMapping[str, Any]) -> None:
    _write_json(output_dir / "ga_state.json", state)
    _write_json(output_dir / "evaluated_candidates.json", {"candidates": _state_results(state)})


def _load_pilot_scenarios(output_dir: Path, scenario_count: int, base_seed: int, resume: bool) -> Mapping[str, Any]:
    """Copy the existing Pilot-100 payload when available; otherwise regenerate it identically."""
    local_path = output_dir / "scenarios.json"
    pilot_path = CURRENT_DIR / "results" / "pilot_100" / "scenarios.json"
    if resume and local_path.exists():
        payload = _load_json(local_path)
        if payload.get("scenario_count") == scenario_count and payload.get("base_seed") == base_seed:
            return payload
    if pilot_path.exists():
        payload = _load_json(pilot_path)
        if payload.get("scenario_count") == scenario_count and payload.get("base_seed") == base_seed:
            _write_json(local_path, payload)
            return payload
    return _ensure_scenarios(local_path, scenario_count, base_seed, resume=False)


def _selected_paths(
    output_dir: Path,
    weights: Mapping[str, float],
    scenarios: Sequence[Mapping[str, object]],
    expected_runs: int,
    workers: int,
    resume: bool,
) -> int:
    path = output_dir / "selected_paths.json"
    payload = _load_json(path) if resume and path.exists() else None
    has_events = isinstance(payload, dict) and isinstance(payload.get("records"), list) and all(
        "result" in record for record in payload["records"]
    )
    if not (_complete_candidate(payload, weights, expected_runs) and has_events):
        _write_json(path, {"weights": dict(weights), "records": _run_records(scenarios, weights, workers, include_events=True)})
    return expected_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA screening and local 0.1 Pilot-100 four-weight search")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "results" / "four_weight_search_ga_local")
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--scenario-count", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=PILOT_BASE_SEED)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-local-search", action="store_true", help="Test-only: stop after the mandatory 160-candidate GA screen")
    args = parser.parse_args()
    if args.scenario_count != 100:
        raise SystemExit("Formal search requires exactly --scenario-count 100")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    output_dir = args.output_dir.resolve()
    pilot_output = (CURRENT_DIR / "results" / "pilot_100").resolve()
    if output_dir == pilot_output:
        raise SystemExit("--output-dir must not overwrite results/pilot_100")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    resume = not args.no_resume
    config = GAConfig()
    scenarios_payload = _load_pilot_scenarios(output_dir, args.scenario_count, args.base_seed, resume)
    scenarios: Sequence[Mapping[str, object]] = scenarios_payload["scenarios"]
    expected_runs = args.scenario_count * 3
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_name": "pilot_100_ga_then_local_0.1_four_weight_search",
        "scenario_count": args.scenario_count,
        "starts_per_scenario": 3,
        "workers": args.workers,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "baseline_weights": BASELINE_WEIGHTS,
        "fixed_distance_weight": 1.0,
        "ga": {**asdict(config), "total_unique_candidates": config.total_candidates},
        "bounds": {"vessel_risk": [0.0, 4.0], "shape": [0.0, 4.0], "exposure": [0.0, 3.0]},
        "local_search": {"radius": 0.5, "step": 0.1, "default": "run_after_ga"},
        "selection_rule": ["fewest eligible hard failures", "lowest mean objective", "lowest p90 objective", "lowest mean planning time"],
    }
    _write_json(output_dir / "metadata.json", metadata)

    baseline_path = output_dir / "baseline.json"
    baseline_payload = _load_json(baseline_path) if resume and baseline_path.exists() else None
    if _complete_candidate(baseline_payload, BASELINE_WEIGHTS, expected_runs):
        baseline_records = baseline_payload["records"]
        print(f"Reusing baseline: {len(baseline_records)} scenario-start runs", flush=True)
    else:
        print("Running distance-only baseline…", flush=True)
        baseline_records = _run_records(scenarios, BASELINE_WEIGHTS, args.workers)
        _write_json(baseline_path, {"weights": BASELINE_WEIGHTS, "records": baseline_records})
    baseline_index = _record_index(baseline_records)

    state_path = output_dir / "ga_state.json"
    if resume and state_path.exists():
        state: dict[str, Any] = _load_json(state_path)
        if state.get("config") != asdict(config):
            raise SystemExit("Existing GA state has a different configuration; choose a new --output-dir or use --no-resume")
    else:
        state = {"config": asdict(config), "generations": [], "local_search": {}}

    for generation_number in range(config.generations):
        while len(state["generations"]) <= generation_number:
            if generation_number == 0:
                population = _initial_population(config)
            else:
                previous = state["generations"][generation_number - 1]
                if len(previous.get("results", [])) != config.population_size:
                    raise RuntimeError("Cannot schedule a generation before the previous population is complete")
                ranked_previous = sorted(previous["results"], key=lambda item: _rank(item["summary"]))
                for rank, item in enumerate(ranked_previous, start=1):
                    item["rank"] = rank
                used_keys = {item["candidate_key"] for item in _state_results(state)}
                population = _next_generation(config, generation_number, ranked_previous, used_keys)
            state["generations"].append({"generation": generation_number, "population": population, "results": []})
            _checkpoint(output_dir, state)
        generation = state["generations"][generation_number]
        existing_keys = {item["candidate_key"] for item in generation["results"]}
        for candidate in generation["population"]:
            if candidate["candidate_key"] in existing_keys:
                continue
            print(f"[GA {generation_number + 1}/{config.generations}] {candidate['candidate_key']}", flush=True)
            result = _candidate_result(candidate_dir, candidate, scenarios, baseline_index, expected_runs, args.workers, resume)
            generation["results"].append(result)
            _checkpoint(output_dir, state)

    ga_results = [result for generation in state["generations"] for result in generation["results"]]
    if len(ga_results) != config.total_candidates or len({item["candidate_key"] for item in ga_results}) != config.total_candidates:
        raise RuntimeError("GA did not produce the required 160 unique candidate evaluations")
    ranked_ga = sorted(ga_results, key=lambda item: _rank(item["summary"]))
    ga_winner = ranked_ga[0]

    final_results = ga_results
    if not args.skip_local_search:
        local = state["local_search"]
        if local and local.get("winner_key") != ga_winner["candidate_key"]:
            raise RuntimeError("Existing local search belongs to a different GA winner")
        if not local:
            local = {
                "winner_key": ga_winner["candidate_key"],
                "winner_weights": ga_winner["weights"],
                "population": _local_candidates(ga_winner["weights"]),
                "results": [],
            }
            state["local_search"] = local
            _checkpoint(output_dir, state)
        local_done = {item["candidate_key"] for item in local["results"]}
        for weights in local["population"]:
            key = _weight_key(weights)
            if key in local_done:
                continue
            candidate = _candidate(weights, {"method": "local_0.1_grid", "ga_winner": ga_winner["candidate_key"]})
            print(f"[local {len(local['results']) + 1}/{len(local['population'])}] {key}", flush=True)
            result = _candidate_result(candidate_dir, candidate, scenarios, baseline_index, expected_runs, args.workers, resume)
            local["results"].append(result)
            _checkpoint(output_dir, state)
        final_results = local["results"]

    ranked_final = sorted(final_results, key=lambda item: _rank(item["summary"]))
    selected = ranked_final[0]
    selected_count = _selected_paths(output_dir, selected["weights"], scenarios, expected_runs, args.workers, resume)
    _write_json(output_dir / "summary.json", {
        "metadata": metadata,
        "scenario_file": "scenarios.json",
        "baseline_file": "baseline.json",
        "candidate_directory": "candidates",
        "ga_candidate_count": len(ga_results),
        "ga_winner": ga_winner,
        "local_search_ran": not args.skip_local_search,
        "local_candidate_count": len(state["local_search"].get("results", [])),
        "selected_candidate": selected,
        "all_ga_candidates_ranked": ranked_ga,
        "all_final_candidates_ranked": ranked_final,
        "selected_paths_file": "selected_paths.json",
        "selected_path_run_count": selected_count,
        "selection_rule": metadata["selection_rule"],
    })
    print(f"GA + local four-weight search complete: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

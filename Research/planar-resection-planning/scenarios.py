"""Deterministic scenario generation for planar-simulator experiments."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from planner import Cell, boundary_cells, generate_domain, neighbors4

SCENARIO_GENERATOR_NAME = "planar_resection_scenario"
SCENARIO_GENERATOR_VERSION = 1
PILOT_BASE_SEED = 2026072801
EXPERIMENT_SPLIT_VERSION = 1
EXPERIMENT_BASE_SEEDS = {
    "train": 2026072901,
    "validation": 2026072902,
    "test": 2026072903,
    "stress": 2026072904,
}


def _cell_list(cells: Iterable[Cell]) -> List[List[int]]:
    return [[r, c] for r, c in sorted(cells)]


def _internal_cells(domain: Set[Cell]) -> Set[Cell]:
    return domain - boundary_cells(domain)


def _grow_component(
    rng: random.Random,
    allowed: Set[Cell],
    size: int,
    *,
    elongated: bool = False,
    blocked: Set[Cell] | None = None,
) -> Set[Cell]:
    """Grow one four-connected obstacle component inside ``allowed``."""
    blocked = blocked or set()
    candidates = sorted(allowed - blocked)
    if not candidates:
        return set()
    for _ in range(80):
        component = {rng.choice(candidates)}
        while len(component) < size:
            boundary = sorted({
                nxt
                for cell in component
                for nxt in neighbors4(cell)
                if nxt in allowed and nxt not in blocked and nxt not in component
            })
            if not boundary:
                break
            if elongated:
                tail = max(component)
                tail_neighbors = [nxt for nxt in neighbors4(tail) if nxt in boundary]
                chosen = rng.choice(tail_neighbors or boundary)
            else:
                chosen = rng.choice(boundary)
            component.add(chosen)
        if len(component) == size:
            return component
    return set()


def _ring_obstacle(domain: Set[Cell], internal: Set[Cell], rng: random.Random) -> Set[Cell]:
    """Create an intentionally non-releasable 3x3 vessel ring when possible."""
    centers = []
    for r, c in internal:
        square = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
        if square <= domain and (square - {(r, c)}) <= internal:
            centers.append((r, c))
    if not centers:
        return set()
    center = rng.choice(sorted(centers))
    return {
        (r, c)
        for r in range(center[0] - 1, center[0] + 2)
        for c in range(center[1] - 1, center[1] + 2)
        if (r, c) != center
    }


def _separated(component: Set[Cell], existing: Set[Cell]) -> bool:
    return all(
        max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
        for a in component for b in existing
    )


def _obstacles_for_type(
    scenario_type: str,
    domain: Set[Cell],
    rng: random.Random,
) -> Tuple[Set[Cell], bool]:
    internal = _internal_cells(domain)
    if scenario_type == "isolated":
        return _grow_component(rng, internal, 1), True
    if scenario_type == "compact":
        return _grow_component(rng, internal, rng.randint(3, 6)), True
    if scenario_type == "elongated":
        return _grow_component(rng, internal, rng.randint(4, 8), elongated=True), True
    if scenario_type == "multiple":
        obstacles: Set[Cell] = set()
        for _ in range(2):
            component: Set[Cell] = set()
            for _ in range(60):
                component = _grow_component(rng, internal, rng.randint(2, 4), blocked=obstacles)
                if component and _separated(component, obstacles):
                    break
            obstacles |= component
        return obstacles, True
    if scenario_type == "stress_ring":
        ring = _ring_obstacle(domain, internal, rng)
        if ring:
            return ring, False
        return _grow_component(rng, internal, rng.randint(4, 7), elongated=True), False
    raise ValueError(f"Unknown scenario type: {scenario_type}")


def _distance_to_obstacles(cell: Cell, obstacles: Set[Cell]) -> float:
    if not obstacles:
        return float("inf")
    return min(math.hypot(cell[0] - obstacle[0], cell[1] - obstacle[1]) for obstacle in obstacles)


def _starts(domain: Set[Cell], obstacles: Set[Cell]) -> List[Dict[str, object]]:
    boundary = sorted(boundary_cells(domain) - obstacles)
    if len(boundary) < 3:
        raise ValueError("Scenario has fewer than three legal boundary starts")
    distances = {cell: _distance_to_obstacles(cell, obstacles) for cell in boundary}
    near = min(boundary, key=lambda cell: (distances[cell], cell))
    far = max(boundary, key=lambda cell: (distances[cell], tuple(-value for value in cell)))
    remaining = [cell for cell in boundary if cell not in {near, far}]
    if not remaining:
        remaining = [cell for cell in boundary if cell != near]
    constricted = min(
        remaining,
        key=lambda cell: (sum(nxt in domain for nxt in neighbors4(cell)), distances[cell], cell),
    )
    selected = [("near_vessel", near), ("far_vessel", far), ("constricted_boundary", constricted)]
    seen: Set[Cell] = set()
    result = []
    for label, cell in selected:
        if cell in seen:
            fallback = next(candidate for candidate in boundary if candidate not in seen)
            cell = fallback
        seen.add(cell)
        result.append({"label": label, "cell": list(cell), "distance_to_obstacle": distances[cell]})
    return result


def _pilot_types(count: int) -> List[str]:
    proportions = [
        ("isolated", 0.20),
        ("compact", 0.25),
        ("elongated", 0.20),
        ("multiple", 0.20),
        ("stress_ring", 0.15),
    ]
    counts = {name: int(count * fraction) for name, fraction in proportions}
    remainder = count - sum(counts.values())
    for name, _ in proportions[:remainder]:
        counts[name] += 1
    return [name for name, _ in proportions for _ in range(counts[name])]


def generate_pilot_scenarios(count: int = 100, base_seed: int = PILOT_BASE_SEED) -> Dict[str, object]:
    """Create the fixed 100-scenario Pilot set described in the training plan."""
    scenarios = []
    for index, scenario_type in enumerate(_pilot_types(count)):
        seed = int(base_seed) + index * 7919
        rng = random.Random(seed ^ 0x5F3759DF)
        generated = generate_domain(seed=seed)
        domain = {tuple(cell) for cell in generated["domain_cells"]}
        obstacles, scoring_eligible = _obstacles_for_type(scenario_type, domain, rng)
        if not obstacles:
            raise RuntimeError(f"Could not place obstacles for scenario {index} ({scenario_type})")
        if not obstacles <= _internal_cells(domain):
            raise RuntimeError(f"Invalid boundary obstacle in scenario {index}")
        scenarios.append({
            "scenario_id": f"pilot-{index:03d}",
            "seed": seed,
            "scenario_type": scenario_type,
            "scoring_eligible": scoring_eligible,
            "expected_status": "ok" if scoring_eligible else "partial",
            "rows": generated["rows"],
            "cols": generated["cols"],
            "domain_cells": generated["domain_cells"],
            "obstacle_cells": _cell_list(obstacles),
            "starts": _starts(domain, obstacles),
        })
    return {
        "generator_name": SCENARIO_GENERATOR_NAME,
        "generator_version": SCENARIO_GENERATOR_VERSION,
        "base_seed": base_seed,
        "scenario_count": count,
        "scenarios": scenarios,
    }


def generate_experiment_splits(
    *,
    train_count: int = 500,
    validation_count: int = 150,
    test_count: int = 200,
    stress_count: int = 100,
) -> Dict[str, object]:
    """Generate the seed-disjoint, versioned splits required for RL experiments.

    The returned object is deterministic and JSON serializable.  Callers must
    write it once and retain that file with every training run; regenerating it
    is only valid after an environment-semantics or reward-version change.
    """
    requested = {
        "train": int(train_count), "validation": int(validation_count),
        "test": int(test_count), "stress": int(stress_count),
    }
    if any(count <= 0 for count in requested.values()):
        raise ValueError("Every experiment split must contain at least one scenario")
    splits: Dict[str, object] = {}
    all_seeds: Set[int] = set()
    for split_name, count in requested.items():
        generated = generate_pilot_scenarios(count=count, base_seed=EXPERIMENT_BASE_SEEDS[split_name])
        scenarios = generated["scenarios"]
        for index, scenario in enumerate(scenarios):
            scenario["scenario_id"] = f"{split_name}-{index:04d}"
            scenario["split"] = split_name
            seed = int(scenario["seed"])
            if seed in all_seeds:
                raise RuntimeError("Experiment split seeds unexpectedly overlap")
            all_seeds.add(seed)
        splits[split_name] = scenarios
    return {
        "split_version": EXPERIMENT_SPLIT_VERSION,
        "generator_name": SCENARIO_GENERATOR_NAME,
        "generator_version": SCENARIO_GENERATOR_VERSION,
        "base_seeds": dict(EXPERIMENT_BASE_SEEDS),
        "counts": requested,
        "splits": splits,
    }


def write_experiment_splits(path: str | Path, **counts: int) -> Dict[str, object]:
    """Persist one frozen split set; deliberately never overwrites an existing file."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split file: {destination}")
    payload = generate_experiment_splits(**counts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload

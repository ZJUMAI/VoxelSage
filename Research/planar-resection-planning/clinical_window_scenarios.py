"""Versioned scenario generation for clinical-window PPO experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from planner import boundary_cells, is_connected, neighbors4, neighbors8
from variable_scenarios import CURRICULUM_RANGES, make_scenario


CLINICAL_SCENARIO_GENERATOR = "clinical_window_planar_resection"
CLINICAL_SCENARIO_VERSION = 1
CLINICAL_SPLIT_VERSION = 1
CLINICAL_SPLIT_SEEDS = {
    "train": 2026080301,
    "validation": 2026080302,
    "test": 2026080303,
    "stress": 2026080304,
}
VESSEL_COUNT_RANGES = {
    "a": (1, 2),
    "b": (2, 4),
    "c": (3, 6),
    "d": (4, 8),
}
VESSEL_SIZE_RANGES = {
    "a": (1, 2),
    "b": (1, 3),
    "c": (1, 3),
    "d": (1, 4),
}


def _cell_list(cells: Iterable[tuple[int, int]]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(cells)]


def _grow_component(
    rng: random.Random,
    *,
    allowed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    size: int,
) -> set[tuple[int, int]]:
    starts = sorted(allowed - blocked)
    if not starts:
        return set()
    for _ in range(80):
        component = {rng.choice(starts)}
        while len(component) < size:
            frontier = sorted({
                neighbor
                for cell in component
                for neighbor in neighbors4(cell)
                if neighbor in allowed and neighbor not in blocked and neighbor not in component
            })
            if not frontier:
                break
            component.add(rng.choice(frontier))
        if len(component) == size:
            return component
    return set()


def _place_vessels(
    rng: random.Random,
    domain: set[tuple[int, int]],
    *,
    stage: str,
) -> set[tuple[int, int]]:
    internal = domain - boundary_cells(domain)
    low_count, high_count = VESSEL_COUNT_RANGES[stage]
    low_size, high_size = VESSEL_SIZE_RANGES[stage]
    for _ in range(160):
        vessels: set[tuple[int, int]] = set()
        blocked: set[tuple[int, int]] = set()
        target_count = rng.randint(low_count, high_count)
        components: list[set[tuple[int, int]]] = []
        for _component_index in range(target_count):
            size = rng.randint(low_size, high_size)
            component = _grow_component(rng, allowed=internal, blocked=blocked, size=size)
            if not component:
                break
            components.append(component)
            vessels.update(component)
            blocked.update(component)
            blocked.update(neighbor for cell in component for neighbor in neighbors8(cell))
        if len(components) != target_count:
            continue
        if is_connected(domain - vessels):
            return vessels
    raise RuntimeError(f"Could not place separated clinical vessels for stage {stage}")


def make_clinical_scenario(*, stage: str, index: int, seed: int, split: str) -> dict[str, Any]:
    if stage not in CURRICULUM_RANGES:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    base = make_scenario(stage=stage, index=index, seed=seed, split=split)
    domain = {tuple(cell) for cell in base["domain_cells"]}
    rng = random.Random(seed ^ 0xC11A1CA1)
    vessels = _place_vessels(rng, domain, stage=stage)
    result = dict(base)
    result.update({
        "scenario_id": f"clinical-{stage}-{split}-{index:04d}",
        "generator_name": CLINICAL_SCENARIO_GENERATOR,
        "generator_version": CLINICAL_SCENARIO_VERSION,
        "obstacle_cells": _cell_list(vessels),
    })
    return result


def generate_clinical_stage_pool(
    *, stage: str, count: int, seed: int, split: str,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [
        make_clinical_scenario(
            stage=stage,
            index=index,
            seed=seed + index * 7919,
            split=split,
        )
        for index in range(count)
    ]


def generate_clinical_curriculum_train_pool(
    *, stage: str, count: int, seed: int,
) -> list[dict[str, Any]]:
    order = tuple(CURRICULUM_RANGES)
    if stage not in order:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    eligible = order[:order.index(stage) + 1]
    return [
        make_clinical_scenario(
            stage=eligible[index % len(eligible)],
            index=index,
            seed=seed + index * 7919,
            split="train",
        )
        for index in range(count)
    ]


def generate_clinical_splits(
    *,
    train_count: int = 256,
    validation_count: int = 96,
    test_count: int = 120,
    stress_count: int = 80,
    stage: str = "d",
) -> dict[str, Any]:
    counts = {
        "train": int(train_count),
        "validation": int(validation_count),
        "test": int(test_count),
        "stress": int(stress_count),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Every split count must be positive")
    splits = {
        name: generate_clinical_stage_pool(
            stage=stage,
            count=count,
            seed=CLINICAL_SPLIT_SEEDS[name],
            split=name,
        )
        for name, count in counts.items()
    }
    seeds = [int(item["seed"]) for items in splits.values() for item in items]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Clinical split seeds unexpectedly overlap")
    return {
        "split_version": CLINICAL_SPLIT_VERSION,
        "generator_name": CLINICAL_SCENARIO_GENERATOR,
        "generator_version": CLINICAL_SCENARIO_VERSION,
        "stage": stage,
        "cell_size_mm": 4.0,
        "counts": counts,
        "base_seeds": dict(CLINICAL_SPLIT_SEEDS),
        "splits": splits,
    }


def write_clinical_splits(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split file: {destination}")
    payload = generate_clinical_splits(**kwargs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=tuple(CURRICULUM_RANGES), default="d")
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=96)
    parser.add_argument("--test-count", type=int, default=120)
    parser.add_argument("--stress-count", type=int, default=80)
    args = parser.parse_args()
    payload = write_clinical_splits(
        args.output,
        stage=args.stage,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        stress_count=args.stress_count,
    )
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


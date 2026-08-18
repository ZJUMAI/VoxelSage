"""Seeded variable-size scenario pools for the 4 mm, 30x40 curriculum."""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

from planner import boundary_cells, generate_domain, is_connected, neighbors4


MAX_ROWS = 30
MAX_COLS = 40
CURRICULUM_RANGES = {
    "a": ((5, 8), (5, 8)),
    "b": ((9, 16), (9, 16)),
    "c": ((17, 24), (17, 32)),
    "d": ((20, 30), (24, 40)),
}


def _cell_list(cells: Iterable[tuple[int, int]]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(cells)]


def _small_domain(rng: random.Random, rows: int, cols: int) -> set[tuple[int, int]]:
    """Make a connected, hole-free small domain by trimming exposed cells."""
    domain = {(row, col) for row in range(rows) for col in range(cols)}
    target_removals = rng.randint(1, max(1, (rows * cols) // 7))
    for _ in range(target_removals * 10):
        exposed = [
            cell for cell in sorted(boundary_cells(domain))
            if sum(nxt in domain for nxt in neighbors4(cell)) <= 2
        ]
        if not exposed:
            break
        candidate = rng.choice(exposed)
        reduced = domain - {candidate}
        if len(reduced) >= max(12, rows * cols // 2) and is_connected(reduced):
            domain = reduced
            target_removals -= 1
            if target_removals == 0:
                break
    return domain


def _choose_obstacles(
    rng: random.Random, domain: set[tuple[int, int]], *, stage: str,
) -> set[tuple[int, int]]:
    internal = sorted(domain - boundary_cells(domain))
    if not internal:
        return set()
    max_count = {"a": 1, "b": 3, "c": 6, "d": 10}[stage]
    count = rng.randint(1, min(max_count, len(internal)))
    for _ in range(100):
        obstacles = set(rng.sample(internal, count))
        if is_connected(domain - obstacles):
            return obstacles
    return set()


def make_scenario(*, stage: str, index: int, seed: int, split: str) -> dict[str, Any]:
    if stage not in CURRICULUM_RANGES:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    (row_low, row_high), (col_low, col_high) = CURRICULUM_RANGES[stage]
    rng = random.Random(seed)
    rows, cols = rng.randint(row_low, row_high), rng.randint(col_low, col_high)
    if rows >= 10 and cols >= 10:
        generated = generate_domain(seed=seed, rows=rows, cols=cols)
        domain = {tuple(cell) for cell in generated["domain_cells"]}
    else:
        domain = _small_domain(rng, rows, cols)
    obstacles = _choose_obstacles(rng, domain, stage=stage)
    starts = sorted(boundary_cells(domain) - obstacles)
    if not starts:
        raise RuntimeError("Variable-size generator produced no legal start")
    return {
        "scenario_id": f"variable-{stage}-{split}-{index:04d}",
        "split": split,
        "stage": stage,
        "seed": seed,
        "cell_size_mm": 4.0,
        "rows": rows,
        "cols": cols,
        "domain_cells": _cell_list(domain),
        "obstacle_cells": _cell_list(obstacles),
        "start_cell": list(rng.choice(starts)),
    }


def generate_stage_pool(
    *, stage: str, count: int, seed: int, split: str,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [
        make_scenario(stage=stage, index=index, seed=seed + index * 7919, split=split)
        for index in range(count)
    ]


def generate_curriculum_train_pool(*, stage: str, count: int, seed: int) -> list[dict[str, Any]]:
    """Mix the current stage with prior stages to avoid small-case forgetting."""
    order = tuple(CURRICULUM_RANGES)
    if stage not in order:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    eligible = order[:order.index(stage) + 1]
    return [
        make_scenario(
            stage=eligible[index % len(eligible)], index=index,
            seed=seed + index * 7919, split="train",
        )
        for index in range(count)
    ]

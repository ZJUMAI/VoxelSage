"""Freeze leak-safe v10.2 splits.

Seven splits are regenerated from the same Stage D generator distribution
(train/oracle_dev/probe/tuning/validation/test) plus a genuinely perturbed
Stress split.  All IDs and seeds are unique across every split.

Stress is not a re-seeded Stage D: it uses harder frozen perturbation
parameters (more vessel components, larger cross-sections, no separation
margin) written into the split metadata before training and never changed.

Scale calibration (time / blood / ischemia) uses the FULL Train-512 with the
mechanical 15/5 baseline.  Distribution consistency is checked over the full
Train split with pre-defined thresholds; any violation exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from clinical_window_evaluation import (
    rollout_clinical_policy,
    serpentine_hierarchical_policy,
)
from clinical_window_scenarios import (
    _cell_list,
    _grow_component,
    make_clinical_scenario,
)
from planner import boundary_cells, is_connected, neighbors8
from variable_scenarios import CURRICULUM_RANGES, make_scenario


V102_SPLIT_COUNTS = {
    "train": 512,
    "oracle_dev": 64,
    "probe": 64,
    "tuning": 64,
    "validation": 128,
    "test": 128,
    "stress": 128,
}
V102_SPLIT_SEEDS = {
    "train": 2026090101,
    "oracle_dev": 2026090102,
    "probe": 2026090103,
    "tuning": 2026090104,
    "validation": 2026090105,
    "test": 2026090106,
    "stress": 2026090107,
}
V102_SPLIT_USES = {
    "train": "BC, oracle data, PPO gradients",
    "oracle_dev": "oracle model selection (scene-isolated)",
    "probe": "fixed 2k-5k diagnostics + early stop",
    "tuning": "future limited hyper-parameter search only",
    "validation": "3-seed candidate confirmation + final model selection",
    "test": "one-shot evaluation after full freeze",
    "stress": "true OOD parameter perturbation, never used for selection",
}

# Stress perturbation: more components, larger cross-sections, no separation.
V102_STRESS_VESSEL_COUNT_RANGE = (9, 14)
V102_STRESS_VESSEL_SIZE_RANGE = (2, 6)


def _place_stress_vessels(
    rng: random.Random,
    domain: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Place vessels with Stress perturbation parameters (separation margin 0)."""
    internal = domain - boundary_cells(domain)
    low_count, high_count = V102_STRESS_VESSEL_COUNT_RANGE
    low_size, high_size = V102_STRESS_VESSEL_SIZE_RANGE
    for _ in range(320):
        vessels: set[tuple[int, int]] = set()
        blocked: set[tuple[int, int]] = set()
        target_count = rng.randint(low_count, high_count)
        components: list[set[tuple[int, int]]] = []
        for _component_index in range(target_count):
            size = rng.randint(low_size, high_size)
            component = _grow_component(
                rng, allowed=internal, blocked=blocked, size=size
            )
            if not component:
                break
            components.append(component)
            vessels.update(component)
            blocked.update(component)
            # separation_margin = 0: only block the component cells themselves,
            # so components may be 8-adjacent (more dense adjacency).
        if len(components) != target_count:
            continue
        if is_connected(domain - vessels):
            return vessels
    raise RuntimeError("Could not place v10.2 Stress vessels")


def make_stress_scenario(*, index: int, seed: int, split: str) -> dict[str, Any]:
    """Stage D domain geometry + Stress vessel perturbation."""
    base = make_scenario(stage="d", index=index, seed=seed, split=split)
    domain = {tuple(cell) for cell in base["domain_cells"]}
    rng = random.Random(seed ^ 0x5E77E55)
    vessels = _place_stress_vessels(rng, domain)
    result = dict(base)
    result.update({
        "scenario_id": f"clinical-d-{split}-{index:04d}",
        "obstacle_cells": _cell_list(vessels),
        "perturbation": {
            "vessel_count_range": list(V102_STRESS_VESSEL_COUNT_RANGE),
            "vessel_size_range": list(V102_STRESS_VESSEL_SIZE_RANGE),
            "separation_margin": 0,
            "reason": "OOD vascular density / cross-section / adjacency",
        },
    })
    return result


def _percentiles(values: list[float]) -> list[float]:
    ordered = sorted(values)
    n = len(ordered)
    def q(frac: float) -> float:
        pos = (n - 1) * frac
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])
    return [q(0.25), q(0.5), q(0.75)]


def _vessel_component_count(cells: list[list[int]]) -> int:
    from planner import neighbors4

    cell_set = {tuple(c) for c in cells}
    seen: set[tuple[int, int]] = set()
    components = 0
    for cell in cell_set:
        if cell in seen:
            continue
        components += 1
        queue: deque[tuple[int, int]] = deque([cell])
        seen.add(cell)
        while queue:
            current = queue.popleft()
            for neighbor in neighbors4(current):
                if neighbor in cell_set and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return components


def _calibrate_scales(train_scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    clinical_config = {
        "time_scale_minutes": 60.0,
        "blood_scale_ml": 100.0,
        "weight_kg": 70.0,
        "bleeding_probability": 1.0,
        "early_end_mode": "disabled",
        "early_end_minutes": 0.0,
    }
    reward_config = {
        "time_cost": 1.0,
        "blood_cost": 1.0,
        "progress_bonus": 0.0,
        "seal_progress_bonus": 0.0,
        "front_tension_cost": 0.0,
        "organ_energy_cost": 0.0,
        "vessel_strain_cost": 0.0,
    }
    records = [
        rollout_clinical_policy(
            scenario,
            serpentine_hierarchical_policy,
            clinical_config=clinical_config,
            reward_config=reward_config,
            control_mode="hierarchical",
        )
        for scenario in train_scenarios
    ]
    completed = [record for record in records if record["completion"]]
    if not completed:
        raise RuntimeError("v10.2 scale calibration produced no completed episodes")
    times = [float(record["elapsed_minutes"]) for record in completed]
    bloods = [
        float(record["expected_blood_loss_ml"])
        for record in completed
        if record["expected_blood_loss_ml"] > 0
    ]
    clamps = [
        float(record["total_clamped_minutes"])
        for record in completed
        if record["total_clamped_minutes"] > 0
    ]
    if not bloods:
        raise RuntimeError("v10.2 scale calibration produced no non-zero blood loss")
    if not clamps:
        raise RuntimeError("v10.2 scale calibration produced no non-zero clamp time")
    return {
        "calibration_version": 2,
        "source_split": "train",
        "episode_count": len(records),
        "completed_episode_count": len(completed),
        "time_scale_minutes": float(median(times)),
        "blood_scale_ml": float(max(median(bloods), 100.0)),
        "ischemia_scale_minutes": float(median(clamps)),
        "weight_kg": 70.0,
    }


def _check_split_distribution(payload: dict[str, Any]) -> dict[str, Any]:
    """Full-train distribution gate with pre-defined thresholds (non-zero exit)."""
    train = payload["splits"]["train"]
    train_cells = [len(s["obstacle_cells"]) for s in train]
    train_comp = [_vessel_component_count(s["obstacle_cells"]) for s in train]
    train_ref = {
        "vessel_cells": _percentiles(train_cells),
        "vessel_components": _percentiles(train_comp),
    }
    result: dict[str, Any] = {
        "thresholds": {
            "vessel_cells_q50_rel_diff": 0.20,
            "vessel_components_q50_rel_diff": 0.25,
        },
        "train_reference": train_ref,
        "splits": {},
    }
    ok = True
    for name, scenarios in payload["splits"].items():
        cells = [len(s["obstacle_cells"]) for s in scenarios]
        comps = [_vessel_component_count(s["obstacle_cells"]) for s in scenarios]
        cells_q = _percentiles(cells)
        comps_q = _percentiles(comps)
        cells_rel = abs(cells_q[1] - train_ref["vessel_cells"][1]) / max(
            1.0, train_ref["vessel_cells"][1]
        )
        comps_rel = abs(comps_q[1] - train_ref["vessel_components"][1]) / max(
            1.0, train_ref["vessel_components"][1]
        )
        name_ok = True
        if name != "stress":
            if cells_rel > result["thresholds"]["vessel_cells_q50_rel_diff"]:
                name_ok = False
            if comps_rel > result["thresholds"]["vessel_components_q50_rel_diff"]:
                name_ok = False
        else:
            # Stress is expected to differ; require it to be strictly harder:
            if cells_q[1] <= train_ref["vessel_cells"][1]:
                name_ok = False
            if comps_q[1] <= train_ref["vessel_components"][1]:
                name_ok = False
        if not name_ok:
            ok = False
        result["splits"][name] = {
            "vessel_cells_q25_q50_q75": cells_q,
            "vessel_components_q25_q50_q75": comps_q,
            "vessel_cells_q50_rel_diff": cells_rel,
            "vessel_components_q50_rel_diff": comps_rel,
            "within_threshold": name_ok,
        }
    result["overall_ok"] = bool(ok)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split file: {output}")

    splits: dict[str, list[dict[str, Any]]] = {}
    for name, count in V102_SPLIT_COUNTS.items():
        if name == "stress":
            splits[name] = [
                make_stress_scenario(
                    index=index,
                    seed=V102_SPLIT_SEEDS[name] + index * 7919,
                    split=f"v10.2-{name}",
                )
                for index in range(count)
            ]
        else:
            splits[name] = [
                make_clinical_scenario(
                    stage="d",
                    index=index,
                    seed=V102_SPLIT_SEEDS[name] + index * 7919,
                    split=f"v10.2-{name}",
                )
                for index in range(count)
            ]

    scenario_ids = [item["scenario_id"] for values in splits.values() for item in values]
    scenario_seeds = [int(item["seed"]) for values in splits.values() for item in values]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise RuntimeError("v10.2 scenario IDs overlap")
    if len(scenario_seeds) != len(set(scenario_seeds)):
        raise RuntimeError("v10.2 scenario seeds overlap")
    for name, values in splits.items():
        local_ids = [item["scenario_id"] for item in values]
        local_seeds = [int(item["seed"]) for item in values]
        if len(local_ids) != len(set(local_ids)) or len(local_seeds) != len(set(local_seeds)):
            raise RuntimeError(f"v10.2 split {name!r} has overlapping IDs/seeds internally")

    scales = _calibrate_scales(splits["train"])
    dist = _check_split_distribution({
        "splits": splits,
        "counts": {name: len(values) for name, values in splits.items()},
    })

    payload = {
        "version": "clinical-v102-splits-v1",
        "frozen": True,
        "stage": "d",
        "generator": "clinical_window_planar_resection v1 (Stage D) + v10.2 Stress perturbation",
        "vessel_count_range": "4-8 (train), 9-14 (stress)",
        "vessel_size_range": "1-4 cells (train), 2-6 cells (stress)",
        "counts": {name: len(values) for name, values in splits.items()},
        "base_seeds": V102_SPLIT_SEEDS,
        "uses": V102_SPLIT_USES,
        "note": "Stress uses frozen harder perturbation, not a re-seeded Stage D.",
        "splits": splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scales_path = output.parent / "scales_v10_2.json"
    scales_path.write_text(json.dumps(scales, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dist_path = output.parent / "split_distribution.json"
    dist_path.write_text(json.dumps(dist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sha_path = output.parent / "SHA256SUMS"
    lines = []
    for path in sorted(output.parent.glob("*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "counts": payload["counts"],
        "total_scenarios": len(scenario_ids),
        "unique_ids": len(set(scenario_ids)),
        "unique_seeds": len(set(scenario_seeds)),
        "scales": scales,
        "distribution_ok": dist["overall_ok"],
        "decision": "GO" if dist["overall_ok"] else "NO-GO",
    }, ensure_ascii=False))
    if not dist["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Freeze leak-safe v10.4 formal Stage-D data (guide Section 4.2).

Five fresh Stage-D splits are generated from the same generator as v10.2 but
with their own fixed seeds (all IDs and seeds unique across every split):

    Train-512     teacher ranking labels, BC/DAgger/PPO gradients
    Tuning-64     the only hyper-parameter search set
    Validation-128  3-seed candidate confirmation + final selection
    Test-128      one-shot evaluation after everything is frozen
    Stress-128    true OOD parameter perturbation, never used for selection

The formal Train-512 is internally reordered (fixed seed) into
``policy_train=448`` (teacher labels / BC/DAgger/PPO gradients / normalizer
scales) and ``policy_internal_dev=64`` (rollout admission + early stop for
Gate B/C).  Scale calibration (time/blood) uses ONLY policy_train, with the
mechanical macro 15/5 S-scan baseline under identical transfer/billing/blood
code.  Distribution audit covers vessel cells / components / baseline time /
baseline blood / clamp cycles (Q25/Q50/Q75) for every split.

Frozen directory (``results/clinical_window_v10_4_target_order/frozen/``):
    splits_v10_4.json  scales_v10_4.json  split_distribution.json
    DATA_USAGE.md      SHA256SUMS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_macro_environment import ClinicalMacroResectionEnv  # noqa: E402
from clinical_window_evaluation import (  # noqa: E402
    rollout_clinical_policy,
    serpentine_macro_target_policy,
)
from clinical_window_scenarios import _cell_list, _grow_component, make_clinical_scenario  # noqa: E402
from planner import boundary_cells, is_connected  # noqa: E402

V104_SPLIT_COUNTS = {
    "train": 512,
    "tuning": 64,
    "validation": 128,
    "test": 128,
    "stress": 128,
}
V104_SPLIT_SEEDS = {
    "train": 2026081201,
    "tuning": 2026081202,
    "validation": 2026081203,
    "test": 2026081204,
    "stress": 2026081205,
}
V104_SPLIT_USES = {
    "train": "teacher ranking labels; BC/DAgger/PPO gradients (policy_train only)",
    "tuning": "the ONLY hyper-parameter search set (Optuna)",
    "validation": "3-seed candidate confirmation + final model selection",
    "test": "one-shot evaluation after full freeze",
    "stress": "true OOD parameter perturbation, never used for selection",
}
V104_INTERNAL_SEED = 20260812
V104_POLICY_TRAIN = 448
V104_POLICY_INTERNAL_DEV = 64

# Stress perturbation (frozen, harder vascular density/cross-section/adjacency).
V104_STRESS_VESSEL_COUNT_RANGE = (9, 14)
V104_STRESS_VESSEL_SIZE_RANGE = (2, 6)

_GATE_CFG = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}
_GATE_REWARD = {
    "time_cost": 1.0,
    "blood_cost": 1.0,
    "progress_bonus": 0.0,
    "seal_progress_bonus": 0.0,
    "front_tension_cost": 0.0,
    "organ_energy_cost": 0.0,
    "vessel_strain_cost": 0.0,
}


def _place_stress_vessels(rng: random.Random, domain: set[tuple[int, int]]) -> set[tuple[int, int]]:
    internal = domain - boundary_cells(domain)
    low_count, high_count = V104_STRESS_VESSEL_COUNT_RANGE
    low_size, high_size = V104_STRESS_VESSEL_SIZE_RANGE
    for _ in range(320):
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
            blocked.update(component)  # separation_margin = 0
        if len(components) != target_count:
            continue
        if is_connected(domain - vessels):
            return vessels
    raise RuntimeError("Could not place v10.4 Stress vessels")


def make_stress_scenario(*, index: int, seed: int, split: str) -> dict[str, Any]:
    base = make_clinical_scenario(stage="d", index=index, seed=seed, split=split)
    domain = {tuple(cell) for cell in base["domain_cells"]}
    rng = random.Random(seed ^ 0x5E77E55)
    vessels = _place_stress_vessels(rng, domain)
    result = dict(base)
    result.update({
        "scenario_id": f"clinical-d-{split}-{index:04d}",
        "obstacle_cells": _cell_list(vessels),
        "perturbation": {
            "vessel_count_range": list(V104_STRESS_VESSEL_COUNT_RANGE),
            "vessel_size_range": list(V104_STRESS_VESSEL_SIZE_RANGE),
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


def _baseline_record(scenario: dict[str, Any]) -> dict[str, Any]:
    return rollout_clinical_policy(
        scenario,
        serpentine_macro_target_policy,
        clinical_config=_GATE_CFG,
        reward_config=_GATE_REWARD,
        mechanics_update_interval=0,
        control_mode="macro",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=SIM / "results/clinical_window_v10_4_target_order/frozen")
    parser.add_argument("--limit", type=int, default=None,
                        help="only generate first N per split (smoke)")
    args = parser.parse_args()
    out_dir = args.output_dir
    if (out_dir / "splits_v10_4.json").exists():
        raise FileExistsError("Refusing to overwrite frozen split file")

    splits: dict[str, list[dict[str, Any]]] = {}
    for name, count in V104_SPLIT_COUNTS.items():
        if args.limit:
            count = min(count, args.limit)
        if name == "stress":
            splits[name] = [
                make_stress_scenario(
                    index=i, seed=V104_SPLIT_SEEDS[name] + i * 7919,
                    split=f"v10.4-{name}",
                )
                for i in range(count)
            ]
        else:
            splits[name] = [
                make_clinical_scenario(
                    stage="d", index=i, seed=V104_SPLIT_SEEDS[name] + i * 7919,
                    split=f"v10.4-{name}",
                )
                for i in range(count)
            ]

    ids = [item["scenario_id"] for values in splits.values() for item in values]
    seeds = [int(item["seed"]) for values in splits.values() for item in values]
    if len(ids) != len(set(ids)):
        raise RuntimeError("v10.4 scenario IDs overlap")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("v10.4 scenario seeds overlap")
    for name, values in splits.items():
        if len({item["scenario_id"] for item in values}) != len(values):
            raise RuntimeError(f"v10.4 split {name!r} has overlapping IDs internally")

    # Internal Train split: fixed seed reorder -> policy_train / policy_internal_dev.
    # Full run uses exactly 448/64; a --limit smoke run scales proportionally.
    ordered = sorted(enumerate(splits["train"]), key=lambda pair: pair[1]["scenario_id"])
    rng = random.Random(V104_INTERNAL_SEED)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    if len(splits["train"]) >= V104_POLICY_TRAIN + V104_POLICY_INTERNAL_DEV:
        n_train, n_dev = V104_POLICY_TRAIN, V104_POLICY_INTERNAL_DEV
    else:
        n_train = int(len(splits["train"]) * 0.8)
        n_dev = len(splits["train"]) - n_train
    policy_train_idx = [i for i, _ in shuffled[:n_train]]
    policy_internal_dev_idx = [i for i, _ in shuffled[n_train:n_train + n_dev]]
    policy_train = [splits["train"][i] for i in policy_train_idx]
    policy_internal_dev = [splits["train"][i] for i in policy_internal_dev_idx]
    internal = {
        "policy_train": {
            "count": len(policy_train),
            "scenario_ids": sorted(s["scenario_id"] for s in policy_train),
        },
        "policy_internal_dev": {
            "count": len(policy_internal_dev),
            "scenario_ids": sorted(s["scenario_id"] for s in policy_internal_dev),
        },
    }
    if len(internal["policy_train"]["scenario_ids"]) != n_train:
        raise RuntimeError("policy_train count mismatch")

    # Scale calibration on policy_train ONLY (guide 4.2).
    train_records = {s["scenario_id"]: _baseline_record(s) for s in policy_train}
    completed = [r for r in train_records.values() if r["completion"]]
    if not completed:
        raise RuntimeError("v10.4 scale calibration produced no completed episodes")
    times = [float(r["elapsed_minutes"]) for r in completed]
    bloods = [float(r["expected_blood_loss_ml"]) for r in completed
              if r["expected_blood_loss_ml"] > 0]
    scales = {
        "calibration_version": 1,
        "source_split": "train:policy_train",
        "episode_count": len(policy_train),
        "completed_episode_count": len(completed),
        "time_scale_minutes": float(median(times)),
        "blood_scale_ml": float(max(median(bloods), 100.0)),
        "weight_kg": 70.0,
        "margin_fraction": 0.05,
    }
    print(f"scales: time_scale={scales['time_scale_minutes']:.1f} min, "
          f"blood_scale={scales['blood_scale_ml']:.1f} mL (policy_train={len(policy_train)})",
          flush=True)

    # Distribution audit over every split (vessel geometry + baseline T/B/clamps).
    dist: dict[str, Any] = {"splits": {}}
    for name, scenarios in splits.items():
        cells = [len(s["obstacle_cells"]) for s in scenarios]
        comps = [_vessel_component_count(s["obstacle_cells"]) for s in scenarios]
        base_recs = {s["scenario_id"]: r for s, r in zip(
            scenarios, [_baseline_record(s) for s in scenarios])}
        times_b = [float(r["elapsed_minutes"]) for r in base_recs.values()]
        bloods_b = [float(r["expected_blood_loss_ml"]) for r in base_recs.values()]
        clamps_b = [float(r["clamp_cycle_count"]) for r in base_recs.values()]
        dist["splits"][name] = {
            "vessel_cells_q25_q50_q75": _percentiles(cells),
            "vessel_components_q25_q50_q75": _percentiles(comps),
            "baseline_elapsed_minutes_q25_q50_q75": _percentiles(times_b),
            "baseline_blood_ml_q25_q50_q75": _percentiles(bloods_b),
            "baseline_clamp_cycles_q25_q50_q75": _percentiles(clamps_b),
        }
        print(f"  {name:10s} vessel_cells_q50={_percentiles(cells)[1]:.0f} "
              f"baseline_T_q50={_percentiles(times_b)[1]:.1f} "
              f"baseline_B_q50={_percentiles(bloods_b)[1]:.1f} "
              f"clamps_q50={_percentiles(clamps_b)[1]:.1f}", flush=True)

    payload = {
        "version": "clinical-v104-splits-v1",
        "frozen": True,
        "stage": "d",
        "generator": "clinical_window_planar_resection v1 (Stage D) + v10.4 Stress perturbation",
        "vessel_count_range": "4-8 (train), 9-14 (stress)",
        "vessel_size_range": "1-4 cells (train), 2-6 cells (stress)",
        "counts": {name: len(values) for name, values in splits.items()},
        "base_seeds": V104_SPLIT_SEEDS,
        "uses": V104_SPLIT_USES,
        "internal_train": internal,
        "note": (
            "Formal v10.4 data. Gate B/C teacher labels, BC/DAgger/PPO gradients "
            "and normalizer scales use policy_train only; policy_internal_dev is "
            "for rollout admission and early stop. Stress is frozen harder "
            "perturbation, never used for selection. Test is one-shot after full "
            "freeze. Do NOT reuse v9 D-16 scenes as Validation/Test."
        ),
        "splits": splits,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = out_dir / "splits_v10_4.json"
    splits_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "scales_v10_4.json").write_text(
        json.dumps(scales, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dist["overall_note"] = "audit values only; gate thresholds applied downstream"
    (out_dir / "split_distribution.json").write_text(
        json.dumps(dist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    usage = f"""# v10.4 Data Usage

Frozen formal Stage-D data. Access rules (guide Section 4.2):

- **Train-512** split into:
  - `policy_train` (448): teacher ranking labels, BC/DAgger gradients, PPO gradients,
    normalizer scales. Everything learnable lives here.
  - `policy_internal_dev` (64): rollout admission + early stop for Gate B/C only.
- **Tuning-64**: the ONLY set for hyper-parameter search (Optuna), Gate C GO after.
- **Validation-128**: 3-seed candidate confirmation + final model selection.
- **Test-128**: ONE-SHOT evaluation after full freeze. No tuning after.
- **Stress-128**: true OOD perturbation; never used for selection or tuning.

Hashes: see SHA256SUMS. Gate A development data lives in pilot_gate_a/.
"""
    (out_dir / "DATA_USAGE.md").write_text(usage, encoding="utf-8")

    sha_lines = []
    for path in sorted(out_dir.glob("*.json")) + sorted(out_dir.glob("*.md")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sha_lines.append(f"{digest}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(splits_path),
        "counts": payload["counts"],
        "total_scenarios": len(ids),
        "unique_ids": len(set(ids)),
        "unique_seeds": len(set(seeds)),
        "scales": scales,
        "internal_train": {k: v["count"] for k, v in internal.items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

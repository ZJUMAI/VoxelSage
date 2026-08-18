"""Freeze leak-safe formal v10.6 Stage-D data and its experiment manifest.

This is a new dataset. It does not parse or copy the v10.4 formal split.  The
only historical scenario IDs read are the already-used v10.5 planner-gate IDs,
solely to prove that the new IDs do not overlap them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_window_evaluation import rollout_clinical_policy, serpentine_macro_target_policy
from clinical_window_scenarios import _cell_list, _grow_component, make_clinical_scenario
from planner import boundary_cells, is_connected, neighbors4

VERSION = "clinical-v106-splits-v1"
MASTER_SEED = 2026081206
SPLIT_COUNTS = {
    "policy_train": 448,
    "policy_internal_dev": 64,
    "tuning": 64,
    "validation": 128,
    "test": 128,
    "stress": 128,
}
SPLIT_SEEDS = {
    name: MASTER_SEED * 100 + i * 1_000_003
    for i, name in enumerate(SPLIT_COUNTS, start=1)
}
SPLIT_USES = {
    "policy_train": "teacher labels, BC gradients, at most one DAgger round, all learned scales",
    "policy_internal_dev": "early stop and Gate T/L rollout only; never gradients",
    "tuning": "at most 12 predeclared BC configurations",
    "validation": "at most 3 candidates x 3 frozen seeds and final selection",
    "test": "one-shot in-distribution evaluation after full freeze",
    "stress": "one-shot OOD audit after Test; never selection or retraining",
}
TRAINING_SEEDS = [2026081601, 2026081602, 2026081603]
BOOTSTRAP_SEED = 2026081604
STRESS_VESSEL_COUNT_RANGE = (9, 14)
STRESS_VESSEL_SIZE_RANGE = (2, 6)
CFG = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}
REWARD = {
    "time_cost": 1.0,
    "blood_cost": 1.0,
    "progress_bonus": 0.0,
    "seal_progress_bonus": 0.0,
    "front_tension_cost": 0.0,
    "organ_energy_cost": 0.0,
    "vessel_strain_cost": 0.0,
}
HISTORICAL_GATE = (
    SIM / "results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _place_stress_vessels(rng: random.Random, domain: set[tuple[int, int]]) -> set[tuple[int, int]]:
    internal = domain - boundary_cells(domain)
    for _ in range(320):
        vessels: set[tuple[int, int]] = set()
        blocked: set[tuple[int, int]] = set()
        count = rng.randint(*STRESS_VESSEL_COUNT_RANGE)
        components = []
        for _ in range(count):
            comp = _grow_component(
                rng, allowed=internal, blocked=blocked,
                size=rng.randint(*STRESS_VESSEL_SIZE_RANGE),
            )
            if not comp:
                break
            components.append(comp)
            vessels.update(comp)
            blocked.update(comp)
        if len(components) == count and is_connected(domain - vessels):
            return vessels
    raise RuntimeError("Could not place v10.6 Stress vessels")


def make_scenario(name: str, index: int) -> dict[str, Any]:
    seed = SPLIT_SEEDS[name] + index * 7919
    scenario = make_clinical_scenario(
        stage="d", index=index, seed=seed, split=f"v10.6-{name}"
    )
    if name != "stress":
        return scenario
    domain = {tuple(cell) for cell in scenario["domain_cells"]}
    vessels = _place_stress_vessels(random.Random(seed ^ 0x1065AFE), domain)
    result = dict(scenario)
    result.update({
        "obstacle_cells": _cell_list(vessels),
        "perturbation": {
            "vessel_count_range": list(STRESS_VESSEL_COUNT_RANGE),
            "vessel_size_range": list(STRESS_VESSEL_SIZE_RANGE),
            "separation_margin": 0,
            "reason": "frozen OOD vascular density/cross-section/adjacency",
        },
    })
    return result


def baseline(scenario: dict[str, Any]) -> dict[str, Any]:
    return rollout_clinical_policy(
        scenario, serpentine_macro_target_policy,
        clinical_config=CFG, reward_config=REWARD,
        mechanics_update_interval=0, control_mode="macro",
    )


def percentiles(values: list[float]) -> list[float]:
    ordered = sorted(float(x) for x in values)
    def q(frac: float) -> float:
        pos = (len(ordered) - 1) * frac
        lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
        return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])
    return [q(0.25), q(0.5), q(0.75)]


def component_count(cells: list[list[int]]) -> int:
    cells_set = {tuple(c) for c in cells}
    seen: set[tuple[int, int]] = set()
    count = 0
    for cell in cells_set:
        if cell in seen:
            continue
        count += 1
        queue: deque[tuple[int, int]] = deque([cell])
        seen.add(cell)
        while queue:
            for nxt in neighbors4(queue.popleft()):
                if nxt in cells_set and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return count


def historical_gate_ids() -> set[str]:
    if not HISTORICAL_GATE.is_file():
        raise FileNotFoundError(f"Missing v10.5 input snapshot: {HISTORICAL_GATE}")
    payload = json.loads(HISTORICAL_GATE.read_text(encoding="utf-8"))
    return set(payload["splits"]["planner_gate"]["scenario_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=SIM / "results/clinical_window_v10_6_shielded_learning/frozen",
    )
    parser.add_argument("--limit", type=int, default=None, help="smoke only; creates a non-formal manifest")
    parser.add_argument("--workers", type=int, default=24, help="parallel S-baseline workers")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty frozen directory: {out}")

    counts = {
        name: min(count, args.limit) if args.limit else count
        for name, count in SPLIT_COUNTS.items()
    }
    splits = {
        name: [make_scenario(name, i) for i in range(count)]
        for name, count in counts.items()
    }
    ids_by_split = {name: {s["scenario_id"] for s in scenes} for name, scenes in splits.items()}
    seeds = [int(s["seed"]) for scenes in splits.values() for s in scenes]
    ids = [s["scenario_id"] for scenes in splits.values() for s in scenes]
    if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise RuntimeError("v10.6 IDs or seeds overlap")
    names = list(ids_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if ids_by_split[left] & ids_by_split[right]:
                raise RuntimeError(f"v10.6 split overlap: {left}/{right}")
    overlap = set(ids) & historical_gate_ids()
    if overlap:
        raise RuntimeError(f"v10.6 overlaps v10.5 planner_gate: {sorted(overlap)[:5]}")

    baseline_records: dict[str, dict[str, Any]] = {}
    dist: dict[str, Any] = {"version": VERSION, "splits": {}}
    flat_scenes = [scene for scenes in splits.values() for scene in scenes]
    if args.workers > 1 and len(flat_scenes) > 1:
        with mp.get_context("fork").Pool(min(args.workers, len(flat_scenes))) as pool:
            flat_records = pool.map(baseline, flat_scenes)
    else:
        flat_records = [baseline(scene) for scene in flat_scenes]
    record_by_id = {
        scene["scenario_id"]: record for scene, record in zip(flat_scenes, flat_records)
    }
    for name, scenes in splits.items():
        records = []
        for scene in scenes:
            rec = record_by_id[scene["scenario_id"]]
            if not rec["completion"] or rec["legal_action_rate"] != 1.0:
                raise RuntimeError(f"baseline failed for {scene['scenario_id']}: {rec['failure_reason']}")
            baseline_records[scene["scenario_id"]] = {
                "elapsed_minutes": float(rec["elapsed_minutes"]),
                "expected_blood_loss_ml": float(rec["expected_blood_loss_ml"]),
                "clamp_cycle_count": int(rec["clamp_cycle_count"]),
            }
            records.append(rec)
        dist["splits"][name] = {
            "count": len(scenes),
            "vessel_cells_q25_q50_q75": percentiles([len(s["obstacle_cells"]) for s in scenes]),
            "vessel_components_q25_q50_q75": percentiles([component_count(s["obstacle_cells"]) for s in scenes]),
            "baseline_elapsed_minutes_q25_q50_q75": percentiles([r["elapsed_minutes"] for r in records]),
            "baseline_blood_ml_q25_q50_q75": percentiles([r["expected_blood_loss_ml"] for r in records]),
            "baseline_clamp_cycles_q25_q50_q75": percentiles([r["clamp_cycle_count"] for r in records]),
        }
        print(f"{name}: {len(scenes)} baselines complete", flush=True)

    train_ids = sorted(ids_by_split["policy_train"])
    train_recs = [baseline_records[i] for i in train_ids]
    train_blood = [r["expected_blood_loss_ml"] for r in train_recs]
    positive_blood = [x for x in train_blood if x > 0]
    scales = {
        "version": "v10.6-scales-v1",
        "source_split": "policy_train",
        "episode_count": len(train_recs),
        "time_scale_minutes": float(median(r["elapsed_minutes"] for r in train_recs)),
        "blood_scale_ml": float(max(median(positive_blood), 100.0)),
        "margin_fraction": 0.05,
        "margin_ml": float(0.05 * sum(train_blood) / len(train_blood)),
    }
    formal = args.limit is None
    manifest = {
        "version": VERSION,
        "formal": formal,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "d",
        "master_seed": MASTER_SEED,
        "split_seeds": SPLIT_SEEDS,
        "counts": counts,
        "uses": SPLIT_USES,
        "training_seeds": TRAINING_SEEDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "candidate_count": 6,
        "max_bc_configs": 12,
        "max_dagger_rounds": 1,
        "stress_parameters": {
            "vessel_count_range": list(STRESS_VESSEL_COUNT_RANGE),
            "vessel_size_range": list(STRESS_VESSEL_SIZE_RANGE),
        },
        "historical_v105_planner_gate_file": str(HISTORICAL_GATE.relative_to(SIM)),
        "historical_v105_planner_gate_sha256": sha256(HISTORICAL_GATE),
        "historical_overlap_count": 0,
    }
    payload = {
        "version": VERSION,
        "formal": formal,
        "frozen": True,
        "counts": counts,
        "uses": SPLIT_USES,
        "split_files": {name: f"split_{name}.json" for name in splits},
        "scenario_ids": {name: sorted(ids_by_split[name]) for name in splits},
    }
    out.mkdir(parents=True, exist_ok=False)
    json_files = {
        "experiment_manifest.json": manifest,
        "splits_v10_6.json": payload,
        "scales_v10_6.json": scales,
        "split_distribution.json": dist,
    }
    for name, value in json_files.items():
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, scenes in splits.items():
        (out / f"split_{name}.json").write_text(
            json.dumps({
                "version": VERSION,
                "split": name,
                "count": len(scenes),
                "use": SPLIT_USES[name],
                "scenarios": scenes,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out / f"baseline_{name}.json").write_text(
            json.dumps({
                "version": VERSION,
                "split": name,
                "count": len(scenes),
                "records": {scene["scenario_id"]: baseline_records[scene["scenario_id"]]
                            for scene in scenes},
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    usage = """# v10.6 Data Usage

- `policy_train`: teacher labels, BC gradients, one optional DAgger round, learned scales.
- `policy_internal_dev`: teacher and shielded-policy gates/early stop only; no gradients.
- `tuning`: at most 12 predeclared BC configurations; locked until Gate L GO.
- `validation`: at most 3 candidates x 3 frozen seeds; locked until Tuning selection.
- `test`: one-shot after code/checkpoint/scales/shield/report freeze.
- `stress`: one-shot OOD audit after Test; never retraining or selection.

All splits and their baseline records are generated into separate `split_<name>.json` and
`baseline_<name>.json` files and byte-hashed now.
`splits_v10_6.json` contains only file names and scenario IDs, never scenario payloads.
Downstream programs must open only their explicitly authorized split file and enforce phase
locks; no training collector imports or parses held-out splits.
The v10.5 planner-gate file is parsed only for scenario-ID non-overlap and its SHA256 is
recorded in the experiment manifest. No v10.4 formal split or teacher data is read.
"""
    (out / "DATA_USAGE.md").write_text(usage, encoding="utf-8")
    # Strong phase-isolation audit: an authorized payload/baseline file may not
    # contain even one scenario ID from another split. The master file is the
    # only cross-split ID index and contains no scenario or metric payloads.
    for name in splits:
        own_ids = ids_by_split[name]
        foreign_ids = set(ids) - own_ids
        for prefix in ("split", "baseline"):
            text = (out / f"{prefix}_{name}.json").read_text(encoding="utf-8")
            leaked = [sid for sid in foreign_ids if sid in text]
            if leaked:
                raise RuntimeError(
                    f"phase isolation failed for {prefix}_{name}.json: {leaked[:3]}"
                )
    paths = sorted(p for p in out.iterdir() if p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in paths), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(out), "formal": formal, "counts": counts,
        "unique_ids": len(set(ids)), "unique_seeds": len(set(seeds)),
        "historical_overlap": 0, "margin_ml": scales["margin_ml"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

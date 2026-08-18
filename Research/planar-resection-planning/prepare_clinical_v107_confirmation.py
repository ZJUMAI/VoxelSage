"""Freeze fresh v10.7 confirmatory Stage-D data and the experiment manifest.

This is a completely new dataset for the publication-confirmation experiment.
It does not parse or copy any v10.4/v10.6 formal split or teacher data.  The
only historical scenario IDs read are (a) the v10.5 planner-gate IDs and
(b) the v10.6 master `splits_v10_6.json` index, solely to prove non-overlap.

Splits (pure Stage-D, master seed 2026081707):

  dev_smoke         32   code/controller/statistics debugging; never in results
  replication      256   one-shot primary confirmatory experiment
  sensitivity_base 128   geometric scenes shared by five frozen conditions

Content hashes intentionally ignore ``scenario_id`` so that renaming a scene
alone cannot create a duplicate.
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

VERSION = "clinical-v107-confirmation-splits-v1"
MASTER_SEED = 2026081707
SPLIT_COUNTS = {
    "dev_smoke": 32,
    "replication": 256,
    "sensitivity_base": 128,
}
SPLIT_SEEDS = {
    "dev_smoke": 202608170701,
    "replication": 202608170702,
    "sensitivity_base": 202608170703,
}
BOOTSTRAP_SEED = 202608170704
SPLIT_USES = {
    "dev_smoke": "code/controller/statistics debugging only; never enters formal results",
    "replication": "one-shot primary confirmatory experiment; never retraining",
    "sensitivity_base": "geometric scenes shared by five frozen sensitivity conditions",
}
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
V106_MASTER = SIM / "results/clinical_window_v10_6_shielded_learning/frozen/splits_v10_6.json"
V106_MARGIN_ML = 16.07054347826075

# Five frozen sensitivity conditions sharing the same 128 geometric scenes.
SENSITIVITY_CONDITIONS = {
    "S0": {
        "id": "S0", "max_clamp_minutes": 15.0, "unclamp_minutes": 5.0,
        "bleeding_probability": 1.00, "explanation": "main condition replication",
    },
    "S1": {
        "id": "S1", "max_clamp_minutes": 12.0, "unclamp_minutes": 5.0,
        "bleeding_probability": 1.00, "explanation": "shorter clamp ceiling",
    },
    "S2": {
        "id": "S2", "max_clamp_minutes": 10.0, "unclamp_minutes": 5.0,
        "bleeding_probability": 1.00, "explanation": "even shorter clamp ceiling",
    },
    "S3": {
        "id": "S3", "max_clamp_minutes": 15.0, "unclamp_minutes": 5.0,
        "bleeding_probability": 0.50, "explanation": "moderate uniform bleeding coefficient",
    },
    "S4": {
        "id": "S4", "max_clamp_minutes": 15.0, "unclamp_minutes": 5.0,
        "bleeding_probability": 0.25, "explanation": "lower uniform bleeding coefficient",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(scenario: dict[str, Any]) -> str:
    """Canonical content hash ignoring scenario_id (rename-safe)."""
    payload = {
        "rows": scenario["rows"],
        "cols": scenario["cols"],
        "cell_size_mm": scenario["cell_size_mm"],
        "domain_cells": sorted(tuple(c) for c in scenario["domain_cells"]),
        "obstacle_cells": sorted(tuple(c) for c in scenario["obstacle_cells"]),
        "start_cell": list(scenario["start_cell"]),
        "generator_name": scenario.get("generator_name"),
        "generator_version": scenario.get("generator_version"),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def make_scenario(name: str, index: int) -> dict[str, Any]:
    seed = SPLIT_SEEDS[name] + index * 7919
    return make_clinical_scenario(
        stage="d", index=index, seed=seed, split=f"v10.7-{name}"
    )


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


def v106_ids() -> set[str]:
    if not V106_MASTER.is_file():
        raise FileNotFoundError(f"Missing v10.6 master index: {V106_MASTER}")
    payload = json.loads(V106_MASTER.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for name, scene_ids in payload["scenario_ids"].items():
        ids.update(scene_ids)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=SIM / "results/clinical_window_v10_7_confirmation/frozen",
    )
    parser.add_argument("--limit", type=int, default=None, help="smoke only; non-formal manifest")
    parser.add_argument("--workers", type=int, default=24, help="parallel S-baseline workers")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty frozen directory: {out}")
        out.rmdir()  # allow re-creating an empty (aborted) directory
    out.mkdir(parents=True, exist_ok=False)

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
        raise RuntimeError("v10.7 IDs or seeds overlap")
    names = list(ids_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if ids_by_split[left] & ids_by_split[right]:
                raise RuntimeError(f"v10.7 split overlap: {left}/{right}")
    # Content-hash isolation (rename-safe): no two scenarios may have the same
    # canonical geometry across the whole formal set.
    hash_seen: dict[str, str] = {}
    for name, scenes in splits.items():
        for scene in scenes:
            h = content_hash(scene)
            if h in hash_seen:
                raise RuntimeError(
                    f"content-hash duplicate: {scene['scenario_id']} vs {hash_seen[h]}"
                )
            hash_seen[h] = scene["scenario_id"]
    all_ids = set(ids)
    overlap_v105 = all_ids & historical_gate_ids()
    overlap_v106 = all_ids & v106_ids()
    if overlap_v105 or overlap_v106:
        raise RuntimeError(
            f"v10.7 overlaps historical: v105={sorted(overlap_v105)[:3]} v106={sorted(overlap_v106)[:3]}"
        )

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
                raise RuntimeError(
                    f"S-baseline failed for {scene['scenario_id']}: {rec['failure_reason']}"
                )
            baseline_records[scene["scenario_id"]] = {
                "elapsed_minutes": float(rec["elapsed_minutes"]),
                "expected_blood_loss_ml": float(rec["expected_blood_loss_ml"]),
                "clamp_cycle_count": int(rec["clamp_cycle_count"]),
            }
            records.append(rec)
        dist["splits"][name] = {
            "count": len(scenes),
            "vessel_cells_q25_q50_q75": percentiles([len(s["obstacle_cells"]) for s in scenes]),
            "vessel_components_q25_q50_q75": percentiles(
                [component_count(s["obstacle_cells"]) for s in scenes]
            ),
            "baseline_elapsed_minutes_q25_q50_q75": percentiles(
                [r["elapsed_minutes"] for r in records]
            ),
            "baseline_blood_ml_q25_q50_q75": percentiles(
                [r["expected_blood_loss_ml"] for r in records]
            ),
            "baseline_clamp_cycles_q25_q50_q75": percentiles(
                [r["clamp_cycle_count"] for r in records]
            ),
        }
        print(f"{name}: {len(scenes)} baselines complete", flush=True)

    formal = args.limit is None
    manifest = {
        "version": VERSION,
        "formal": formal,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": MASTER_SEED,
        "split_seeds": SPLIT_SEEDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "counts": counts,
        "uses": SPLIT_USES,
        "margin_ml": V106_MARGIN_ML,
        "margin_source": "frozen v10.6 margin; reused verbatim, never recomputed",
        "clinical_config": {
            "max_clamp_minutes": 15.0, "unclamp_minutes": 5.0,
            "bleeding_probability": 1.0, "early_end_mode": "disabled",
            "early_end_minutes": 0.0, "large_vessel_min_cells": 2.0,
            "large_vessel_time_multiplier": 3.0,
        },
        "sensitivity_conditions": SENSITIVITY_CONDITIONS,
        "historical_v105_planner_gate_file": str(HISTORICAL_GATE.relative_to(SIM)),
        "historical_v105_planner_gate_sha256": sha256(HISTORICAL_GATE),
        "v106_master_file": str(V106_MASTER.relative_to(SIM)),
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
        "content_hashes": {name: {scene["scenario_id"]: content_hash(scene)
                                  for scene in splits[name]} for name in splits},
    }
    json_files = {
        "experiment_manifest.json": manifest,
        "splits_v10_7.json": payload,
        "sensitivity_conditions.json": SENSITIVITY_CONDITIONS,
        "split_distribution.json": dist,
    }
    for name, value in json_files.items():
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, scenes in splits.items():
        (out / f"split_{name}.json").write_text(
            json.dumps({
                "version": VERSION, "split": name, "count": len(scenes),
                "use": SPLIT_USES[name], "scenarios": scenes,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out / f"baseline_{name}.json").write_text(
            json.dumps({
                "version": VERSION, "split": name, "count": len(scenes),
                "records": {scene["scenario_id"]: baseline_records[scene["scenario_id"]]
                            for scene in scenes},
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    usage = """# v10.7 Data Usage

- `dev_smoke`: code/controller/statistics debugging only; never enters formal results.
- `replication`: one-shot primary confirmatory experiment; never retraining.
- `sensitivity_base`: geometric scenes shared by five frozen sensitivity conditions.

All splits and their baseline records are generated into separate
`split_<name>.json` and `baseline_<name>.json` files and byte-hashed now.
`splits_v10_7.json` contains only file names, scenario IDs and rename-safe
content hashes, never scenario payloads.

Phase locks: only `dev_smoke` is parsed before Gate A/B freeze.  `replication`
is parsed once, after the formal candidate manifest is written.  `sensitivity_base`
is parsed only after Replication results are written to disk.  No v10.4/v10.6
formal split or teacher data is read by any generator here.
"""
    (out / "DATA_USAGE.md").write_text(usage, encoding="utf-8")
    # Strong phase isolation: an authorized payload/baseline file may not contain
    # even one scenario ID from another split.  The master file is the only
    # cross-split ID index and contains no scenario or metric payloads.
    for name in splits:
        own_ids = ids_by_split[name]
        foreign_ids = all_ids - own_ids
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
    # Overlap audit artifact.
    audit = {
        "version": "v10.7-split-overlap-audit-v1",
        "unique_ids": len(all_ids),
        "unique_seeds": len(set(seeds)),
        "unique_content_hashes": len(hash_seen),
        "content_hash_duplicates": len(hash_seen) - len(all_ids),
        "historical_v105_overlap": sorted(all_ids & historical_gate_ids()),
        "v106_overlap": sorted(all_ids & v106_ids()),
        "margin_ml": V106_MARGIN_ML,
    }
    (SIM / "results/clinical_window_v10_7_confirmation/audit/split_overlap_audit.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (SIM / "results/clinical_window_v10_7_confirmation/audit/split_overlap_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(out), "formal": formal, "counts": counts,
        "unique_ids": len(all_ids), "unique_seeds": len(set(seeds)),
        "unique_content_hashes": len(hash_seen),
        "historical_overlap": 0, "margin_ml": V106_MARGIN_ML,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Freeze the v10.7.1 condition-specific-baseline sensitivity correction.

This supplement fixes one and only one v10.7 defect: S1--S4 were evaluated
against the S0 blood-loss baseline.  v10.7.1 generates fresh Stage-D geometry
and freezes an independent direct-serpentine (C0) baseline under each clinical
condition before C2/C4 are evaluated.  It never trains or changes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_window_scenarios import make_clinical_scenario
from confirmation_controllers_v107 import rollout_controller
from plan_target_order_v105 import DEFAULT_GATE_CLINICAL_CONFIG

VERSION = "clinical-v1071-sensitivity-correction-v1"
MASTER_SEED = 20260817071
SPLIT_SEED = 2026081707101
BOOTSTRAP_SEED = 2026081707102
COUNT = 128
MARGIN_ML = 16.07054347826075
BASE = SIM / "results/clinical_window_v10_7_1_sensitivity_correction"
V107_FROZEN = SIM / "results/clinical_window_v10_7_confirmation/frozen"
V106_MASTER = SIM / "results/clinical_window_v10_6_shielded_learning/frozen/splits_v10_6.json"
CHECKPOINT = (
    SIM / "results/clinical_window_v10_6_shielded_learning/runs/bc/"
    "config_05_seed_2026081603/epoch_05.pt"
)

CONDITIONS = {
    "S0": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.00},
    "S1": {"max_clamp_minutes": 12.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.00},
    "S2": {"max_clamp_minutes": 10.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.00},
    "S3": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.50},
    "S4": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.25},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(scene: dict[str, Any]) -> str:
    payload = {
        "rows": scene["rows"], "cols": scene["cols"],
        "cell_size_mm": scene["cell_size_mm"],
        "domain_cells": sorted(tuple(x) for x in scene["domain_cells"]),
        "obstacle_cells": sorted(tuple(x) for x in scene["obstacle_cells"]),
        "start_cell": list(scene["start_cell"]),
        "generator_name": scene.get("generator_name"),
        "generator_version": scene.get("generator_version"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def scenario_hash(scene: dict[str, Any]) -> str:
    raw = json.dumps({k: v for k, v in scene.items() if k != "scenario_id"},
                     sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def make_scene(index: int) -> dict[str, Any]:
    return make_clinical_scenario(
        stage="d", index=index, seed=SPLIT_SEED + index * 7919,
        split="v10.7.1-sensitivity-correction",
    )


def clinical_config(condition: str) -> dict[str, Any]:
    cfg = dict(DEFAULT_GATE_CLINICAL_CONFIG)
    cfg.update(CONDITIONS[condition])
    cfg.update({"early_end_mode": "disabled", "early_end_minutes": 0.0})
    return cfg


def config_hash(condition: str) -> str:
    raw = json.dumps(clinical_config(condition), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _baseline_task(task: tuple[str, dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    condition, scene = task
    row = rollout_controller(
        "C0", scene, baseline_blood=0.0, margin_ml=MARGIN_ML,
        cfg=clinical_config(condition), checkpoint_path=CHECKPOINT,
    )
    if not row["completion"] or row["legal_action_rate"] != 1.0:
        raise RuntimeError(f"C0 failed: {condition}/{scene['scenario_id']}: {row['failure_reason']}")
    return condition, scene["scenario_id"], row


def _historical_ids() -> set[str]:
    ids: set[str] = set()
    for path in (V106_MASTER, V107_FROZEN / "splits_v10_7.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for values in payload["scenario_ids"].values():
            ids.update(values)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "frozen")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="non-formal smoke generation")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty frozen directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    count = min(COUNT, args.limit) if args.limit else COUNT
    scenes = [make_scene(i) for i in range(count)]
    ids = [scene["scenario_id"] for scene in scenes]
    seeds = [int(scene["seed"]) for scene in scenes]
    hashes = [content_hash(scene) for scene in scenes]
    if len(set(ids)) != count or len(set(seeds)) != count or len(set(hashes)) != count:
        raise RuntimeError("new sensitivity split is not internally unique")
    overlap_ids = set(ids) & _historical_ids()
    v107 = json.loads((V107_FROZEN / "splits_v10_7.json").read_text(encoding="utf-8"))
    old_hashes = {value for mapping in v107["content_hashes"].values() for value in mapping.values()}
    overlap_hashes = set(hashes) & old_hashes
    if overlap_ids or overlap_hashes:
        raise RuntimeError(f"historical overlap: ids={len(overlap_ids)} content={len(overlap_hashes)}")

    tasks = [(condition, scene) for condition in CONDITIONS for scene in scenes]
    if args.workers > 1:
        with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
            rows = pool.map(_baseline_task, tasks)
    else:
        rows = [_baseline_task(task) for task in tasks]
    by_condition: dict[str, dict[str, dict[str, Any]]] = {condition: {} for condition in CONDITIONS}
    for condition, sid, row in rows:
        by_condition[condition][sid] = row

    split_payload = {
        "version": VERSION, "split": "sensitivity_correction", "count": count,
        "use": "one-shot v10.7.1 condition-specific-baseline sensitivity correction",
        "scenarios": scenes,
    }
    (out / "split_sensitivity_correction.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for condition in CONDITIONS:
        records = {}
        for scene in scenes:
            sid = scene["scenario_id"]
            row = by_condition[condition][sid]
            records[sid] = {
                "condition": condition,
                "condition_config_sha256": config_hash(condition),
                "elapsed_minutes": float(row["elapsed_minutes"]),
                "expected_blood_loss_ml": float(row["realized_episode_B_ml"]),
                "action_sequence_hash": row["action_sequence_hash"],
                "completion": bool(row["completion"]),
                "legal_action_rate": float(row["legal_action_rate"]),
                "clamp_cycle_count": int(row["clamp_cycle_count"]),
            }
        payload = {
            "version": VERSION, "controller": "C0", "condition": condition,
            "condition_config": clinical_config(condition),
            "condition_config_sha256": config_hash(condition),
            "count": count, "records": records,
        }
        (out / f"baseline_{condition}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    code_files = (
        "prepare_sensitivity_v1071.py", "evaluate_sensitivity_v1071.py",
        "aggregate_sensitivity_v1071.py", "report_sensitivity_v1071.py",
        "confirmation_controllers_v107.py", "clinical_target_order_features_v106.py",
        "clinical_target_order_policy_v106.py", "clinical_safety_shield_v106.py",
        "clinical_macro_environment.py", "clinical_window_environment.py",
        "plan_target_order_v104.py", "plan_target_order_v105.py",
    )
    manifest = {
        "version": VERSION, "formal": args.limit is None, "frozen": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": MASTER_SEED, "split_seed": SPLIT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED, "count": count,
        "conditions": CONDITIONS, "condition_config_sha256": {c: config_hash(c) for c in CONDITIONS},
        "margin_ml": MARGIN_ML,
        "margin_source": "v10.6 frozen margin, reused without recomputation",
        "checkpoint": str(CHECKPOINT.relative_to(SIM)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "code_sha256": {name: sha256(SIM / name) for name in code_files},
        "scenario_ids": ids,
        "scenario_hashes": {scene["scenario_id"]: scenario_hash(scene) for scene in scenes},
        "content_hashes": {scene["scenario_id"]: content_hash(scene) for scene in scenes},
        "historical_id_overlap": 0, "historical_v107_content_overlap": 0,
        "protocol": [
            "generate fresh Stage-D geometry",
            "freeze C0 separately under each S0-S4 condition",
            "set each per-scene budget to same-condition C0 blood plus frozen margin",
            "evaluate frozen C2 and C4 once; no training, tuning, or threshold changes",
        ],
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = sorted(path for path in out.iterdir() if path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in paths), encoding="utf-8"
    )
    print(json.dumps({"output": str(out), "count": count, "formal": args.limit is None,
                      "baselines": len(CONDITIONS) * count, "overlap": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()

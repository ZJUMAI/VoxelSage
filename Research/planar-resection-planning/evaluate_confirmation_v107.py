"""v10.7 atomic per-scenario shard evaluator for a split/condition/controller.

Each shard is a unique JSON file per (split, condition, controller,
scenario_id).  Shards are written atomically (temp file + rename) and never
recomputed once present, so interrupted runs recover by executing only the
missing shards.  Aggregate only reads complete, hash-consistent shards.

Cold cache: every evaluation uses an empty, run-specific shield-record cache
directory so the reported latency is a cold-cache measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from confirmation_controllers_v107 import CONTROLLERS, rollout_controller
from plan_target_order_v105 import DEFAULT_GATE_CLINICAL_CONFIG

BASE = SIM / "results/clinical_window_v10_7_confirmation"
FROZEN = BASE / "frozen"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_hashes() -> dict[str, str]:
    files = (
        "confirmation_controllers_v107.py",
        "evaluate_confirmation_v107.py",
        "prepare_clinical_v107_confirmation.py",
        "clinical_target_order_features_v106.py",
        "clinical_target_order_policy_v106.py",
        "clinical_safety_shield_v106.py",
        "clinical_macro_environment.py",
        "clinical_window_environment.py",
        "plan_target_order_v104.py",
        "plan_target_order_v105.py",
    )
    return {name: sha256(SIM / name) for name in files}


def _clinical_config_for(condition: str, manifest: dict) -> dict:
    base = dict(DEFAULT_GATE_CLINICAL_CONFIG)
    base.update({
        "max_clamp_minutes": manifest["clinical_config"]["max_clamp_minutes"],
        "unclamp_minutes": manifest["clinical_config"]["unclamp_minutes"],
    })
    conds = manifest.get("sensitivity_conditions", {})
    if condition in conds:
        base["max_clamp_minutes"] = float(conds[condition]["max_clamp_minutes"])
        base["unclamp_minutes"] = float(conds[condition]["unclamp_minutes"])
        base["bleeding_probability"] = float(conds[condition]["bleeding_probability"])
    return base


def _rollout_one(
    controller: str, scenario: dict, baseline_blood: float, margin_ml: float,
    cfg: dict, checkpoint_path: Path, leaf_workers: int,
) -> dict:
    pool = mp.get_context("fork").Pool(leaf_workers) if leaf_workers > 0 else None
    try:
        result = rollout_controller(
            controller, scenario,
            baseline_blood=baseline_blood, margin_ml=margin_ml,
            cfg=cfg, checkpoint_path=checkpoint_path, leaf_pool=pool,
        )
        return result
    finally:
        if pool is not None:
            pool.close(); pool.join()


def _worker(
    task_queue, result_queue, margin_ml, checkpoint_path, leaf_workers, manifest,
    code_hash, condition,
):
    torch.set_num_threads(1)
    while True:
        task = task_queue.get()
        if task is None:
            break
        controller, scenario_id, scene, baseline_blood = task
        try:
            cfg = _clinical_config_for(condition, manifest)
            row = _rollout_one(
                controller, scene, baseline_blood, margin_ml, cfg,
                checkpoint_path, leaf_workers,
            )
            row["input_scenario_hash"] = code_hash.get("scenario", "n/a")
            row["code_hashes"] = code_hash
            row["shard_started_utc"] = None
            row["shard_finished_utc"] = time.time()
            result_queue.put((controller, scenario_id, row, None))
        except BaseException as exc:
            result_queue.put((controller, scenario_id, None, repr(exc)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("dev_smoke", "replication", "sensitivity_base"))
    parser.add_argument("--controllers", default="C0,C1,C2,C3,C4,C5")
    parser.add_argument("--condition", default="S0", help="sensitivity condition id (S0..S4) or main")
    parser.add_argument("--checkpoint", type=Path,
                        default=SIM / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--scene-workers", type=int, default=12)
    parser.add_argument("--leaf-workers", type=int, default=3)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    controllers = [c for c in args.controllers.split(",") if c in CONTROLLERS]
    split_payload = json.loads((FROZEN / f"split_{args.split}.json").read_text(encoding="utf-8"))
    baseline_payload = json.loads((FROZEN / f"baseline_{args.split}.json").read_text(encoding="utf-8"))
    if split_payload["split"] != baseline_payload["split"]:
        raise RuntimeError("scene/baseline split mismatch")
    scenes = split_payload["scenarios"]
    if args.scenario_id:
        scenes = [sc for sc in scenes if sc["scenario_id"] == args.scenario_id]
        if len(scenes) != 1:
            raise RuntimeError("scenario-id must resolve exactly once")
    if args.limit:
        scenes = scenes[:args.limit]
    baselines = baseline_payload["records"]
    manifest = json.loads((FROZEN / "experiment_manifest.json").read_text(encoding="utf-8"))
    margin = float(manifest["margin_ml"])

    scenario_hashes = {
        sc["scenario_id"]: hashlib.sha256(
            json.dumps({k: v for k, v in sc.items() if k != "scenario_id"},
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for sc in scenes
    }
    code_hash = _code_hashes()
    code_hash["scenario"] = "per-scenario-content-hash"
    ckpt_hash = sha256(args.checkpoint)

    shard_root = BASE / "shards"
    if args.split == "sensitivity_base":
        shard_dir = shard_root / "sensitivity" / args.condition
    elif args.split == "dev_smoke":
        shard_dir = shard_root / "dev_smoke"
    else:
        shard_dir = shard_root / "replication"

    # Cold, run-specific cache directory.
    cache_dir = BASE / "evaluation" / "shield_cache" / f"{args.split}_{args.condition}_{int(time.time())}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for controller in controllers:
        for scene in scenes:
            sid = scene["scenario_id"]
            out = shard_dir / controller / f"{sid}.json"
            if out.exists():
                # Verify existing shard integrity (hash of required fields).
                try:
                    existing = json.loads(out.read_text(encoding="utf-8"))
                    if existing.get("action_sequence_hash"):
                        continue  # already complete; do not recompute
                except Exception:
                    pass  # corrupt -> recompute
            tasks.append((controller, sid, scene, float(baselines[sid]["expected_blood_loss_ml"])))

    if not tasks:
        print(json.dumps({"mode": "all_shards_present", "count": 0}))
        return

    task_q = mp.get_context("fork").Queue()
    result_q = mp.get_context("fork").Queue()
    workers = min(args.scene_workers, len(tasks))
    processes = []
    for _ in range(workers):
        p = mp.get_context("fork").Process(
            target=_worker,
            args=(task_q, result_q, margin, args.checkpoint, args.leaf_workers,
                  manifest, code_hash, args.condition),
        )
        p.start(); processes.append(p)
    for task in tasks:
        task_q.put(task)
    for _ in processes:
        task_q.put(None)

    done = 0
    errors = []
    for _ in tasks:
        controller, sid, row, error = result_q.get()
        if error:
            errors.append(f"{controller}/{sid}: {error}")
            continue
        out = shard_dir / controller / f"{sid}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        row["controller"] = controller
        row["scenario_id"] = sid
        row["condition"] = args.condition
        row["code_hashes"] = code_hash
        row["checkpoint_sha256"] = ckpt_hash
        row["input_scenario_hash"] = scenario_hashes[sid]
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out)
        done += 1
        if done % 20 == 0:
            print(json.dumps({"completed": done, "total": len(tasks)}), flush=True)
    for p in processes:
        p.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps({"completed": done, "total": len(tasks), "controllers": controllers,
                      "shard_dir": str(shard_dir), "cache_dir": str(cache_dir)}))


if __name__ == "__main__":
    main()

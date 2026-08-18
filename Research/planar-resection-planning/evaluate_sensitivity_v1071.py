"""Evaluate frozen C2/C4 on v10.7.1 with condition-specific C0 budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from confirmation_controllers_v107 import rollout_controller
from prepare_sensitivity_v1071 import BASE, CHECKPOINT, CONDITIONS, clinical_config, config_hash, sha256

FROZEN = BASE / "frozen"
CONTROLLERS = ("C2", "C4")


def _worker(task_queue, result_queue, condition, margin, checkpoint, leaf_workers):
    torch.set_num_threads(1)
    cfg = clinical_config(condition)
    while True:
        task = task_queue.get()
        if task is None:
            return
        controller, scene, baseline_blood = task
        sid = scene["scenario_id"]
        pool = mp.get_context("fork").Pool(leaf_workers) if leaf_workers > 0 else None
        try:
            row = rollout_controller(
                controller, scene, baseline_blood=baseline_blood, margin_ml=margin,
                cfg=cfg, checkpoint_path=checkpoint, leaf_pool=pool,
            )
            result_queue.put((controller, sid, row, None))
        except BaseException as exc:
            result_queue.put((controller, sid, None, repr(exc)))
        finally:
            if pool is not None:
                pool.close(); pool.join()


def _scenario_hash(scene):
    raw = json.dumps({k: v for k, v in scene.items() if k != "scenario_id"},
                     sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _verify_frozen(manifest: dict) -> None:
    if not manifest["formal"] or not manifest["frozen"]:
        raise RuntimeError("v10.7.1 formal freeze is required")
    if sha256(CHECKPOINT) != manifest["checkpoint_sha256"]:
        raise RuntimeError("checkpoint drift")
    for name, expected in manifest["code_sha256"].items():
        if sha256(SIM / name) != expected:
            raise RuntimeError(f"post-freeze code drift: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--scene-workers", type=int, default=12)
    parser.add_argument("--leaf-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    controllers = [c for c in args.controllers.split(",") if c in CONTROLLERS]
    if not controllers:
        raise ValueError("at least one of C2,C4 is required")

    manifest = json.loads((FROZEN / "experiment_manifest.json").read_text(encoding="utf-8"))
    _verify_frozen(manifest)
    split = json.loads((FROZEN / "split_sensitivity_correction.json").read_text(encoding="utf-8"))
    baseline_path = FROZEN / f"baseline_{args.condition}.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_cfg_hash = config_hash(args.condition)
    if baseline["condition"] != args.condition:
        raise RuntimeError("condition/baseline mismatch")
    if baseline["condition_config_sha256"] != expected_cfg_hash:
        raise RuntimeError("condition-specific baseline config mismatch")
    if manifest["condition_config_sha256"][args.condition] != expected_cfg_hash:
        raise RuntimeError("condition config differs from frozen manifest")
    scenes = split["scenarios"][:args.limit] if args.limit else split["scenarios"]
    records = baseline["records"]
    margin = float(manifest["margin_ml"])
    baseline_file_hash = sha256(baseline_path)

    shard_root = BASE / "shards" / args.condition
    tasks = []
    for controller in controllers:
        for scene in scenes:
            sid = scene["scenario_id"]
            if sid not in records or records[sid]["condition"] != args.condition:
                raise RuntimeError(f"missing or mismatched baseline record: {sid}")
            out = shard_root / controller / f"{sid}.json"
            if out.exists():
                row = json.loads(out.read_text(encoding="utf-8"))
                valid = (
                    row.get("condition") == args.condition
                    and row.get("controller") == controller
                    and row.get("input_scenario_hash") == manifest["scenario_hashes"][sid]
                    and row.get("baseline_file_sha256") == baseline_file_hash
                    and row.get("checkpoint_sha256") == manifest["checkpoint_sha256"]
                )
                if valid:
                    continue
                raise RuntimeError(f"refusing to overwrite incompatible shard: {out}")
            tasks.append((controller, scene, float(records[sid]["expected_blood_loss_ml"])))
    if not tasks:
        print(json.dumps({"condition": args.condition, "mode": "all_shards_present"}))
        return

    ctx = mp.get_context("fork")
    task_q, result_q = ctx.Queue(), ctx.Queue()
    processes = []
    for _ in range(min(args.scene_workers, len(tasks))):
        process = ctx.Process(target=_worker, args=(
            task_q, result_q, args.condition, margin, CHECKPOINT, args.leaf_workers,
        ))
        process.start(); processes.append(process)
    for task in tasks:
        task_q.put(task)
    for _ in processes:
        task_q.put(None)

    errors = []
    for index in range(len(tasks)):
        controller, sid, row, error = result_q.get()
        if error:
            errors.append(f"{controller}/{sid}: {error}")
            continue
        out = shard_root / controller / f"{sid}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        row.update({
            "version": "clinical-v1071-sensitivity-shard-v1",
            "condition": args.condition, "controller": controller,
            "input_scenario_hash": manifest["scenario_hashes"][sid],
            "condition_config_sha256": expected_cfg_hash,
            "baseline_file_sha256": baseline_file_hash,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "finished_unix": time.time(),
        })
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out)
        if (index + 1) % 20 == 0:
            print(json.dumps({"condition": args.condition, "completed": index + 1,
                              "total": len(tasks)}), flush=True)
    for process in processes:
        process.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps({"condition": args.condition, "completed": len(tasks),
                      "shard_root": str(shard_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

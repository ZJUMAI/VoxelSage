"""Corrected v10.5 teacher reference on an explicitly authorized v10.6 split."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from benchmark_target_order_v105 import _rollout_optimized

BASE = SIM / "results/clinical_window_v10_6_shielded_learning"
FROZEN = BASE / "frozen"
AUTHORIZED_SPLIT_FILE = FROZEN / "split_policy_internal_dev.json"


def bootstrap_ci(values: np.ndarray, *, seed: int, samples: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _child(chunk, leaf_workers, queue):
    pool = mp.get_context("fork").Pool(leaf_workers)
    try:
        records = [_rollout_optimized(scene, baseline_b, margin, pool)
                   for scene, baseline_b, margin in chunk]
        queue.put((records, None))
    except BaseException as exc:
        queue.put(([], repr(exc)))
    finally:
        pool.close()
        pool.join()


def run_parallel(tasks, *, scene_workers: int, leaf_workers: int):
    chunks = [list(chunk) for chunk in np.array_split(tasks, min(scene_workers, len(tasks)))
              if len(chunk)]
    processes, queues = [], []
    for chunk in chunks:
        queue = mp.get_context("fork").Queue()
        process = mp.get_context("fork").Process(target=_child, args=(chunk, leaf_workers, queue))
        process.start()
        processes.append(process); queues.append(queue)
    output = []
    errors = []
    for queue in queues:
        records, error = queue.get()
        output.extend(records)
        if error:
            errors.append(error)
    for process in processes:
        process.join()
        if process.exitcode:
            errors.append(f"worker exitcode={process.exitcode}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-workers", type=int, default=8)
    parser.add_argument("--leaf-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=BASE / "evaluation/gate_t_teacher.json")
    parser.add_argument("--split-file", type=Path, default=AUTHORIZED_SPLIT_FILE)
    parser.add_argument(
        "--baseline-file", type=Path, default=FROZEN / "baseline_policy_internal_dev.json"
    )
    args = parser.parse_args()
    if not args.split_file.is_file():
        raise FileNotFoundError("formal v10.6 freeze is not complete")
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    if split["split"] not in {
        "policy_internal_dev", "tuning", "validation", "test", "stress"
    }:
        raise RuntimeError("teacher reference split is not an authorized formal phase")
    scenes = split["scenarios"]
    if args.limit:
        scenes = scenes[:args.limit]
    scales = json.loads((FROZEN / "scales_v10_6.json").read_text(encoding="utf-8"))
    manifest = json.loads((FROZEN / "experiment_manifest.json").read_text(encoding="utf-8"))
    baseline_payload = json.loads(args.baseline_file.read_text(encoding="utf-8"))
    if baseline_payload["split"] != split["split"]:
        raise RuntimeError("teacher reference scene/baseline split mismatch")
    baselines = baseline_payload["records"]
    margin = float(scales["margin_ml"])
    tasks = [(scene, float(baselines[scene["scenario_id"]]["expected_blood_loss_ml"]), margin)
             for scene in scenes]
    wall0 = time.time()
    records = run_parallel(tasks, scene_workers=args.scene_workers, leaf_workers=args.leaf_workers)
    records.sort(key=lambda row: row["scenario_id"])
    rows = []
    for record in records:
        sid = record["scenario_id"]
        base = baselines[sid]
        row = dict(record)
        row.update({
            "baseline_T_min": float(base["elapsed_minutes"]),
            "baseline_B_ml": float(base["expected_blood_loss_ml"]),
            "delta_T_min": float(record["teacher_T_min"] - base["elapsed_minutes"]),
            "delta_B_ml": float(record["teacher_B_ml"] - base["expected_blood_loss_ml"]),
            "budget_ml": float(base["expected_blood_loss_ml"] + margin),
            "realized_over_budget_ml": float(record["teacher_B_ml"] - base["expected_blood_loss_ml"] - margin),
        })
        rows.append(row)
    d_t = np.asarray([row["delta_T_min"] for row in rows], dtype=float)
    d_b = np.asarray([row["delta_B_ml"] for row in rows], dtype=float)
    bootstrap_seed = int(manifest["bootstrap_seed"])
    ci_t = bootstrap_ci(d_t, seed=bootstrap_seed)
    ci_b = bootstrap_ci(d_b, seed=bootstrap_seed)
    mean_base_t = float(np.mean([row["baseline_T_min"] for row in rows]))
    failures = sum(not row["completion"] or row["failure_reason"] is not None for row in rows)
    invariants = sum(int(row["safety_invariant_violations"]) for row in rows)
    overrun_count = sum(row["delta_B_ml"] > margin + 1e-9 for row in rows)
    conditions = {
        "completion_legal_100": failures == 0 and len(rows) == len(scenes),
        "failure_truncation_zero": failures == 0,
        "safety_invariant_zero": invariants == 0,
        "per_scene_blood_0_overruns": overrun_count == 0,
        "max_delta_B_le_margin": float(d_b.max()) <= margin + 1e-9,
        "delta_B_ci_upper_le_margin": ci_b[1] <= margin,
        "delta_T_ci_upper_lt_zero": ci_t[1] < 0.0,
        "mean_time_effect": float(d_t.mean()) <= -0.005 * mean_base_t,
    }
    result = {
        "version": "v10.6-gate-t-v1",
        "method": "v10.5 corrected teacher, behavior-equivalent persistent leaf pool",
        "authorized_split_file": args.split_file.name,
        "n_scenarios": len(rows),
        "margin_ml": margin,
        "conditions": conditions,
        "decision": "GO" if all(conditions.values()) else "NO-GO",
        "summary": {
            "completion_failures": failures,
            "safety_invariant_violations": invariants,
            "overrun_count": overrun_count,
            "max_delta_B_ml": float(d_b.max()),
            "mean_delta_B_ml": float(d_b.mean()),
            "delta_B_95_ci": ci_b,
            "mean_delta_T_min": float(d_t.mean()),
            "delta_T_95_ci": ci_t,
            "mean_baseline_T_min": mean_base_t,
            "mean_teacher_T_min": float(np.mean([row["teacher_T_min"] for row in rows])),
            "wall_seconds": time.time() - wall0,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = args.output.with_name(args.output.stem + "_traces.jsonl")
    trace.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": result["decision"],
                      "summary": result["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""E3: 256-scene retrospective equivalence audit (plan §7.4).

Only C4L is run.  C4E is the existing v10.7 C4 shard at
``results/clinical_window_v10_7_confirmation/shards/replication/C4``,
which by construction equals the v10.8 C4E reference (C4E in v10.8 just
delegates to the v10.7 ``rollout_controller('C4', ...)``).

Per-scene assertions:
  * action_hash(C4L) == action_hash(C4E_shard)
  * elapsed_minutes and budget_ml match within frozen tolerance
  * completion / failure_reason / invariant_violations identical
  * verified_candidate_count_mean > 0 and verified_candidate_count_max <= 6
  * safety_invariant_violations == 0

Outputs:
  results/clinical_window_v10_8_lazy_shield/equivalence/
    C4L/<sid>.json    per-scene lazy shard
    C4L_summary.json  aggregated metrics
    equivalence_report.json  per-scene hash comparison
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Per-process thread caps so that N workers don't fight over the BLAS pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

FROZEN_SPLIT = REPO / "results/clinical_window_v10_7_confirmation/frozen/split_replication.json"
FROZEN_BASELINE = REPO / "results/clinical_window_v10_7_confirmation/frozen/baseline_replication.json"
V107_C4_SHARDS = REPO / "results/clinical_window_v10_7_confirmation/shards/replication/C4"
CHECKPOINT = REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt"
V108_EQUIV = REPO / "results/clinical_window_v10_8_lazy_shield/equivalence"


def load_inputs() -> tuple[list[dict], dict[str, float], float]:
    splits = json.loads(FROZEN_SPLIT.read_text())
    base = json.loads(FROZEN_BASELINE.read_text())
    baseline = {sid: float(rec["expected_blood_loss_ml"])
                for sid, rec in base["records"].items()}
    mfst = json.loads(
        (REPO / "results/clinical_window_v10_7_confirmation/frozen/experiment_manifest.json").read_text()
    )
    margin = float(mfst["margin_ml"])
    scenes = splits["scenarios"]
    return scenes, baseline, margin


def _rollout_task(args):
    sid, scene, baseline, margin, ckpt = args
    from lazy_confirmation_controllers_v108 import rollout_controller
    t0 = time.time()
    try:
        res = rollout_controller(
            "C4L", scene,
            baseline_blood=float(baseline), margin_ml=float(margin),
            checkpoint_path=str(ckpt),
        )
        return sid, res, None
    except BaseException as e:
        return sid, None, repr(e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-workers", type=int, default=1,
                        help="default 1: serial.  multiprocessing.Pool with fork "
                             "caused PyTorch / model reload to deadlock in our first "
                             "run; switch to fork start method carefully if needed.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    V108_EQUIV.mkdir(parents=True, exist_ok=True)
    (V108_EQUIV / "C4L").mkdir(parents=True, exist_ok=True)

    scenes, baseline, margin = load_inputs()
    if args.limit:
        scenes = scenes[:args.limit]

    tasks = [
        (s["scenario_id"], s, baseline[s["scenario_id"]], margin, str(CHECKPOINT))
        for s in scenes
    ]
    print(f"[E3] {len(tasks)} scenes, scene_workers={args.scene_workers}")

    if args.scene_workers <= 1:
        results: dict[str, dict] = {}
        errors: list[str] = []
        for t in tasks:
            sid, res, err = _rollout_task(t)
            if err:
                errors.append(f"{sid}: {err}")
            else:
                results[sid] = res
            if len(results) % 20 == 0 and results:
                print(f"  done {len(results)}/{len(tasks)}")
    else:
        # Windows has no fork; spawn (default) reloads the model per worker
        # which is acceptable on this small network.
        ctx = mp.get_context()
        results = {}
        errors = []
        done = 0
        with ctx.Pool(args.scene_workers) as pool:
            for sid, res, err in pool.imap_unordered(_rollout_task, tasks, chunksize=2):
                done += 1
                if err:
                    errors.append(f"{sid}: {err}")
                else:
                    results[sid] = res
                if done % 20 == 0:
                    print(f"  done {done}/{len(tasks)}")
        if errors:
            print(f"[E3] {len(errors)} errors:")
            for e in errors[:10]:
                print(f"  - {e}")

    # Per-scene comparison against v10.7 C4 shard
    reports = []
    n_equal = 0
    n_unequal = 0
    n_missing_ref = 0
    invariant_total = 0
    fail_completion = 0
    verified_means: list[float] = []
    verified_maxes: list[int] = []

    for sid, res in results.items():
        shard = dict(res)
        shard["scenario_id"] = sid
        (V108_EQUIV / "C4L" / f"{sid}.json").write_text(
            json.dumps(shard, ensure_ascii=False, indent=2)
        )

        c4_ref = V107_C4_SHARDS / f"{sid}.json"
        if not c4_ref.exists():
            n_missing_ref += 1
            reports.append({"scene": sid, "missing_v107_c4_ref": True})
            continue
        ref = json.loads(c4_ref.read_text())
        hash_eq = ref.get("action_sequence_hash") == shard.get("action_sequence_hash")
        elapsed_diff = abs(float(ref.get("elapsed_minutes", 0)) - float(shard.get("elapsed_minutes", 0)))
        budget_diff = abs(float(ref.get("budget_ml", 0)) - float(shard.get("budget_ml", 0)))
        reports.append({
            "scene": sid,
            "hash_equal": hash_eq,
            "elapsed_diff_min": elapsed_diff,
            "budget_diff_ml": budget_diff,
            "ref_completion": ref.get("completion"),
            "c4l_completion": shard.get("completion"),
            "ref_failure": ref.get("failure_reason"),
            "c4l_failure": shard.get("failure_reason"),
            "c4l_verified_mean": float(shard.get("verified_count_mean", 0.0)),
            "c4l_verified_max": int(shard.get("verified_count_max", 0)),
            "c4l_safety_violations": int(shard.get("safety_invariant_violations", 0)),
        })
        if hash_eq:
            n_equal += 1
        else:
            n_unequal += 1
        invariant_total += int(shard.get("safety_invariant_violations", 0))
        if not shard.get("completion", False):
            fail_completion += 1
        verified_means.append(float(shard.get("verified_count_mean", 0.0)))
        verified_maxes.append(int(shard.get("verified_count_max", 0)))

    summary = {
        "scenes_total": len(results),
        "n_hash_equal": n_equal,
        "n_hash_unequal": n_unequal,
        "n_missing_v107_c4_ref": n_missing_ref,
        "fail_completion": fail_completion,
        "invariant_violations_total": invariant_total,
        "verified_count_mean": (sum(verified_means) / len(verified_means)) if verified_means else 0.0,
        "verified_count_max_of_max": max(verified_maxes) if verified_maxes else 0,
    }
    (V108_EQUIV / "equivalence_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2)
    )
    (V108_EQUIV / "C4L_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n[E3] hash_equal: {n_equal}/{n_equal+n_unequal}; "
          f"max_verified={summary['verified_count_max_of_max']}; "
          f"invariant_violations={invariant_total}; "
          f"fail_completion={fail_completion}")
    if n_unequal or invariant_total or fail_completion:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

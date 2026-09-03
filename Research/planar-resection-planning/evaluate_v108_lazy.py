"""E6: New 256-scene confirmatory experiment on v10.8 lazy-shield split.

Runs C0, C2, C3, C4E, C4L, C5 on each of the 256 scenes produced by
``prepare_v108_lazy_split.py``.  Outputs atomic per-(controller, scene)
shards under ``results/clinical_window_v10_8_lazy_shield/shards/<C>/``.

Each shard contains the same per-scene fields that v10.7 shards use,
plus the v10.8 diagnostic fields (verified_count_mean/max, selected_rank,
fallback_used, fallback_reason) on C4L shards.

C4E and C4L are the v10.8 research controllers; the others are the
v10.7 reference controllers carried forward unchanged.

Usage:
  evaluate_v108_lazy.py --split-file ... --baseline-file ... \
                        --manifest ... --margin 16.07054347826075 \
                        --scene-workers 20 --controllers C0,C2,C3,C4E,C4L,C5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

V108_OUT = REPO / "results/clinical_window_v10_8_lazy_shield"
SHARDS_OUT = V108_OUT / "shards"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


def _task(args):
    controller, sid, scene, baseline, margin, ckpt = args
    from lazy_confirmation_controllers_v108 import rollout_controller
    t0 = time.time()
    try:
        res = rollout_controller(
            controller, scene,
            baseline_blood=float(baseline), margin_ml=float(margin),
            checkpoint_path=str(ckpt),
        )
        res["wall_seconds_pilot"] = time.time() - t0
        return sid, controller, res, None
    except BaseException as e:
        return sid, controller, None, repr(e)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path,
                        default=V108_OUT / "frozen/split_lazy_replication.json")
    parser.add_argument("--baseline-file", type=Path,
                        default=V108_OUT / "frozen/baseline_lazy_replication.json")
    parser.add_argument("--manifest", type=Path,
                        default=V108_OUT / "frozen/experiment_manifest_v108.json")
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--scene-workers", type=int, default=20)
    parser.add_argument("--controllers", default="C0,C2,C3,C4E,C4L,C5")
    args = parser.parse_args(argv)
    controllers = [c for c in args.controllers.split(",") if c]

    if not args.split_file.exists():
        print(f"[E6] missing split file: {args.split_file}")
        return 1
    if not args.baseline_file.exists():
        print(f"[E6] missing baseline file: {args.baseline_file}")
        return 1

    SHARDS_OUT.mkdir(parents=True, exist_ok=True)
    for c in controllers:
        (SHARDS_OUT / c).mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = split["scenarios"]
    baseline_records = base["records"]
    margin = float(args.margin)

    tasks = []
    for s in scenes:
        sid = s["scenario_id"]
        if sid not in baseline_records:
            continue
        # Interleave controllers per scene so imap_unordered does not
        # starve slow controllers behind a fast batch.
        for c in controllers:
            tasks.append((c, sid, s, float(baseline_records[sid]["expected_blood_loss_ml"]),
                          margin, str(args.checkpoint)))
    import random
    random.seed(0)
    random.shuffle(tasks)
    print(f"[E6] {len(tasks)} task tuples ({len(controllers)} controllers x {len(scenes)} scenes, interleaved+shuffled)")

    if args.scene_workers <= 1:
        results = []
        for t in tasks:
            sid, c, res, err = _task(t)
            if err:
                print(f"  ERR {c}/{sid}: {err}")
            else:
                results.append((c, sid, res))
                shard = dict(res)
                shard["scenario_id"] = sid
                shard["controller"] = c
                (SHARDS_OUT / c / f"{sid}.json").write_text(
                    json.dumps(shard, ensure_ascii=False, indent=2)
                )
                if len(results) % 50 == 0:
                    print(f"  done {len(results)}/{len(tasks)}")
    else:
        ctx = mp.get_context()
        results = []
        done = 0
        with ctx.Pool(args.scene_workers) as pool:
            for sid, c, res, err in pool.imap_unordered(_task, tasks, chunksize=2):
                done += 1
                if err:
                    print(f"  ERR {c}/{sid}: {err}")
                else:
                    results.append((c, sid, res))
                    shard = dict(res)
                    shard["scenario_id"] = sid
                    shard["controller"] = c
                    (SHARDS_OUT / c / f"{sid}.json").write_text(
                        json.dumps(shard, ensure_ascii=False, indent=2)
                    )
                if done % 50 == 0:
                    print(f"  done {done}/{len(tasks)}")

    print(f"[E6] wrote {len(results)} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

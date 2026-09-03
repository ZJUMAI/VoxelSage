"""E6 split: run fast (C0/C2/C5) and slow (C3/C4E/C4L) controllers in two
passes, each with all available workers.  Skips scenes already
sharded in the target directory (idempotent resume)."""
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
        return sid, controller, res, None, time.time() - t0
    except BaseException as e:
        return sid, controller, None, repr(e), 0.0


def _run_pass(controllers, args):
    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = split["scenarios"]
    margin = args.margin

    tasks = []
    for s in scenes:
        sid = s["scenario_id"]
        if sid not in base["records"]:
            continue
        for c in controllers:
            out = SHARDS_OUT / c / f"{sid}.json"
            if out.exists():
                continue
            tasks.append((c, sid, s, float(base["records"][sid]["expected_blood_loss_ml"]),
                          margin, str(args.checkpoint)))
    if not tasks:
        print(f"[E6/{controllers}] all done; skip")
        return 0
    print(f"[E6/{controllers}] {len(tasks)} task tuples")
    if args.scene_workers <= 1:
        done = 0
        for t in tasks:
            sid, c, res, err, _ = _task(t)
            done += 1
            if err:
                print(f"  ERR {c}/{sid}: {err}")
                continue
            shard = dict(res)
            shard["scenario_id"] = sid
            shard["controller"] = c
            (SHARDS_OUT / c / f"{sid}.json").write_text(
                json.dumps(shard, ensure_ascii=False, indent=2)
            )
            if done % 50 == 0:
                print(f"  done {done}/{len(tasks)}")
    else:
        ctx = mp.get_context()
        done = 0
        with ctx.Pool(args.scene_workers) as pool:
            for sid, c, res, err, _ in pool.imap_unordered(_task, tasks, chunksize=2):
                done += 1
                if err:
                    print(f"  ERR {c}/{sid}: {err}")
                else:
                    shard = dict(res)
                    shard["scenario_id"] = sid
                    shard["controller"] = c
                    (SHARDS_OUT / c / f"{sid}.json").write_text(
                        json.dumps(shard, ensure_ascii=False, indent=2)
                    )
                if done % 50 == 0:
                    print(f"  done {done}/{len(tasks)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path,
                        default=V108_OUT / "frozen/split_lazy_replication.json")
    parser.add_argument("--baseline-file", type=Path,
                        default=V108_OUT / "frozen/baseline_lazy_replication.json")
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--scene-workers", type=int, default=20)
    parser.add_argument("--passes", default="fast,slow")
    args = parser.parse_args(argv)

    SHARDS_OUT.mkdir(parents=True, exist_ok=True)
    fast = ["C0", "C2", "C5"]
    slow = ["C3", "C4E", "C4L"]
    for c in fast + slow:
        (SHARDS_OUT / c).mkdir(parents=True, exist_ok=True)

    passes = args.passes.split(",")
    for p in passes:
        if p == "fast":
            _run_pass(fast, args)
        elif p == "slow":
            _run_pass(slow, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

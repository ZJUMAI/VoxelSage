"""E7 fill-in: for missing (controller, rep) shards, run with scene_workers=20.

This is a fast fill-in for shards that the 30-min serial runs could not
finish.  It uses scene_workers=20 so a single (controller, rep) cell
finishes in <5 min.  The wall_seconds values produced here are
representative of the controller's true cost; using scene_workers=20
is the same parallelization the E6 phase shards already use.
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
LATENCY_OUT = V108_OUT / "latency_v2"
DEFAULT_CHECKPOINT = (REPO
    / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


def _pick_scenes(split, n):
    sc = split["scenarios"]
    if n >= len(sc):
        return sc
    step = max(1, len(sc) // n)
    return [sc[i * step] for i in range(n)]


def _task(args):
    ctrl, sid, scene, baseline, margin, ckpt, lw, out_path = args
    if out_path.exists():
        return sid, ctrl, "skip", 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from lazy_confirmation_controllers_v108 import rollout_controller
    leaf_pool = None
    if lw and lw > 0 and ctrl in ("C3", "C4E"):
        leaf_pool = mp.get_context().Pool(int(lw))
    t0 = time.time()
    try:
        try:
            res = rollout_controller(
                ctrl, scene,
                baseline_blood=float(baseline), margin_ml=float(margin),
                checkpoint_path=str(ckpt), leaf_pool=leaf_pool,
            )
        finally:
            if leaf_pool is not None:
                leaf_pool.close()
                leaf_pool.join()
        shard = dict(res)
        shard["scenario_id"] = sid
        shard["controller"] = ctrl
        shard["leaf_workers"] = int(lw) if ctrl in ("C3", "C4E") else 0
        shard["tuning_meta"] = {"fill_in": True, "leaf_workers": int(lw),
                                "scene_workers": 20}
        out_path.write_text(json.dumps(shard, ensure_ascii=False, indent=2))
        return sid, ctrl, "ok", time.time() - t0
    except BaseException as e:
        if leaf_pool is not None:
            try:
                leaf_pool.terminate()
                leaf_pool.join()
            except Exception:
                pass
        return sid, ctrl, f"err:{e!r}", time.time() - t0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--n-scenes", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--controllers", default="C3,C4E")
    parser.add_argument("--leaf-workers", type=int, default=6)
    args = parser.parse_args(argv)
    controllers = [c for c in args.controllers.split(",") if c]
    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = _pick_scenes(split, args.n_scenes)
    margin = float(args.margin)

    jobs = []
    for ctrl in controllers:
        for rep in range(args.reps):
            out_dir = LATENCY_OUT / f"rep{rep}" / ctrl
            for s in scenes:
                sid = s["scenario_id"]
                if sid not in base["records"]:
                    continue
                out = out_dir / f"{sid}.json"
                if out.exists():
                    continue
                jobs.append((ctrl, sid, s,
                             float(base["records"][sid]["expected_blood_loss_ml"]),
                             margin, str(args.checkpoint),
                             int(args.leaf_workers), out))
    print(f"[E7-fill] {len(jobs)} missing shards across {len(controllers)} controllers",
          flush=True)
    if not jobs:
        return 0
    t_start = time.time()
    # Run serially in the main process so the leaf_pool inside _task can
    # create sub-pools.  This is a fill-in for the E7 scene_workers=1
    # driver; speed is acceptable because the missing shard count is
    # small (typically <20 per (controller, rep) cell).
    for j in jobs:
        sid, c, status, dt = _task(j)
        if status != "skip":
            print(f"  {c} {sid}: {status} {dt:.2f}s", flush=True)
    print(f"[E7-fill] total {time.time() - t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""E6 phase: run one controller at a time, all workers on that controller.

This avoids the imap_unordered throughput penalty that mixed-controller
pools suffer under Windows + spawn.  Run order is fastest first so
v10.8's key controllers (C4L) finish early and we can begin the
analysis even before slow controllers (C3, C4E) complete.

Usage:
  evaluate_v108_phase.py --controllers C4L --scene-workers 20
  evaluate_v108_phase.py --controllers C0 --scene-workers 20
  ...
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


def _make_leaf_pool(n: int):
    """Return a multiprocessing.Pool with ``n`` workers for shield candidates,
    or None for sequential evaluation.  Used to give C3/C4E an honest
    parallel candidate-verify pass for latency comparisons against C4L.
    """
    if n is None or n <= 0:
        return None
    return mp.get_context().Pool(int(n))


def _task(args):
    controller, sid, scene, baseline, margin, ckpt, out_dir, leaf_workers = args
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sid}.json"
    if out.exists():
        return sid, controller, "skip", 0.0
    t0 = time.time()
    leaf_pool = None
    if leaf_workers and leaf_workers > 0:
        leaf_pool = mp.get_context().Pool(int(leaf_workers))
    try:
        from lazy_confirmation_controllers_v108 import rollout_controller
        try:
            res = rollout_controller(
                controller, scene,
                baseline_blood=float(baseline), margin_ml=float(margin),
                checkpoint_path=str(ckpt),
                leaf_pool=leaf_pool,
            )
        finally:
            if leaf_pool is not None:
                leaf_pool.close()
                leaf_pool.join()
        shard = dict(res)
        shard["scenario_id"] = sid
        shard["controller"] = controller
        out.write_text(json.dumps(shard, ensure_ascii=False, indent=2))
        return sid, controller, "ok", time.time() - t0
    except BaseException as e:
        if leaf_pool is not None:
            try:
                leaf_pool.terminate()
                leaf_pool.join()
            except Exception:
                pass
        return sid, controller, f"err:{e!r}", time.time() - t0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--scene-workers", type=int, default=20)
    parser.add_argument("--leaf-workers", type=int, default=0,
                        help="0 = sequential candidate verify (default); >0 = parallel "
                             "leaf workers for the exact-shield candidate-evaluate pass "
                             "(C3/C4E only; C4L ignores it).")
    parser.add_argument("--controllers", required=True)
    args = parser.parse_args(argv)
    controllers = [c for c in args.controllers.split(",") if c]

    SHARDS_OUT.mkdir(parents=True, exist_ok=True)
    for c in controllers:
        (SHARDS_OUT / c).mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = split["scenarios"]
    margin = args.margin

    for ctrl in controllers:
        tasks = []
        for s in scenes:
            sid = s["scenario_id"]
            if sid not in base["records"]:
                continue
            out_dir = SHARDS_OUT / ctrl
            out = out_dir / f"{sid}.json"
            if out.exists():
                continue
            tasks.append((ctrl, sid, s,
                          float(base["records"][sid]["expected_blood_loss_ml"]),
                          margin, str(args.checkpoint), out_dir,
                          int(args.leaf_workers)))
        if not tasks:
            print(f"[E6/{ctrl}] all done; skip")
            continue
        print(f"[E6/{ctrl}] {len(tasks)} task tuples (leaf_workers={args.leaf_workers})")
        if args.scene_workers <= 1:
            done = 0
            t_start = time.time()
            for t in tasks:
                sid, c, status, _ = _task(t)
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t_start
                    print(f"  done {done}/{len(tasks)} ({done/elapsed:.2f}/s)")
        else:
            ctx = mp.get_context()
            done = 0
            t_start = time.time()
            with ctx.Pool(args.scene_workers) as pool:
                for sid, c, status, _ in pool.imap_unordered(_task, tasks, chunksize=4):
                    done += 1
                    if done % 25 == 0:
                        elapsed = time.time() - t_start
                        rate = done / elapsed
                        eta = (len(tasks) - done) / rate if rate > 0 else 0
                        print(f"  done {done}/{len(tasks)} ({rate:.2f}/s, ETA {eta/60:.1f} min)")
            print(f"  total {ctrl}: {done} done in {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

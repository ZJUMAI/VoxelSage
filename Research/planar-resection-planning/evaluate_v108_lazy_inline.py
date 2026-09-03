"""E6 split: in-worker shard write, main process only tracks completion flag.

This eliminates IPC pickling of ~15 KB shards, which was the bottleneck
on Windows.  Workers write the shard file directly and return a 1-tuple
status; the main loop just tallies counts.
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
    controller, sid, scene, baseline, margin, ckpt, out_dir = args
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sid}.json"
    if out.exists():
        return sid, controller, "skip", 0.0
    t0 = time.time()
    try:
        from lazy_confirmation_controllers_v108 import rollout_controller
        res = rollout_controller(
            controller, scene,
            baseline_blood=float(baseline), margin_ml=float(margin),
            checkpoint_path=str(ckpt),
        )
        shard = dict(res)
        shard["scenario_id"] = sid
        shard["controller"] = controller
        out.write_text(json.dumps(shard, ensure_ascii=False, indent=2))
        return sid, controller, "ok", time.time() - t0
    except BaseException as e:
        return sid, controller, f"err:{e!r}", time.time() - t0


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
            tasks.append((c, sid, s, float(base["records"][sid]["expected_blood_loss_ml"]),
                          margin, str(args.checkpoint), SHARDS_OUT / c))
    if not tasks:
        print(f"[E6/{controllers}] all done; skip")
        return 0
    print(f"[E6/{controllers}] {len(tasks)} task tuples")
    if args.scene_workers <= 1:
        done = 0
        for t in tasks:
            sid, c, status, _ = _task(t)
            done += 1
            if done % 50 == 0:
                print(f"  done {done}/{len(tasks)}")
    else:
        ctx = mp.get_context()
        done = 0
        ok = err = skip = 0
        t_start = time.time()
        with ctx.Pool(args.scene_workers) as pool:
            for sid, c, status, _ in pool.imap_unordered(_task, tasks, chunksize=8):
                done += 1
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    err += 1
                if done % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = done / elapsed
                    eta = (len(tasks) - done) / rate
                    print(f"  done {done}/{len(tasks)} ({rate:.2f}/s, ETA {eta/60:.1f} min) ok={ok} skip={skip} err={err}")
        print(f"  total done {done}: ok={ok} skip={skip} err={err}")
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

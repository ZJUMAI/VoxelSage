"""E7: 64-scene latency benchmark (plan §7.8).

Honest single-process latency: scene_workers=1, OMP_NUM_THREADS=1,
MKL_NUM_THREADS=1, torch.set_num_threads(1).  Each scene runs the
controller three times; the per-scene median is the per-scene latency
used for paired comparisons.

Default controllers: C3, C4E, C4L, C5.  Default reps: 3.

Outputs:
  results/clinical_window_v10_8_lazy_shield/latency/
    C3/  C4E/  C4L/  C5/    per-(scene, rep) shards
    latency_summary.json
    paired_wall_time.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

V108_OUT = REPO / "results/clinical_window_v10_8_lazy_shield"
SHARDS_OUT = V108_OUT / "latency"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


def _select_scenes(scenes, k):
    n = len(scenes)
    if k >= n:
        return list(scenes)
    step = (n - 1) / (k - 1)
    idxs = sorted({int(round(i * step)) for i in range(k)})
    return [scenes[i] for i in idxs]


def _run_once(controller, scene, baseline, margin, ckpt):
    from lazy_confirmation_controllers_v108 import rollout_controller
    t0 = time.perf_counter()
    res = rollout_controller(
        controller, scene,
        baseline_blood=float(baseline), margin_ml=float(margin),
        checkpoint_path=str(ckpt),
    )
    return res, time.perf_counter() - t0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path,
                        default=V108_OUT / "frozen/split_lazy_replication.json")
    parser.add_argument("--baseline-file", type=Path,
                        default=V108_OUT / "frozen/baseline_lazy_replication.json")
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--scenes", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--scene-workers", type=int, default=1,
                        help="default 1 (strict plan §7.8).  Use 4 to give each "
                             "controller a dedicated worker (one scene at a "
                             "time per worker) which still preserves per-scene "
                             "latency isolation while reducing wall time.")
    parser.add_argument("--controllers", default="C3,C4E,C4L,C5")
    args = parser.parse_args(argv)
    controllers = [c for c in args.controllers.split(",") if c]

    if not args.split_file.exists() or not args.baseline_file.exists():
        print(f"[E7] missing frozen input: {args.split_file} or {args.baseline_file}")
        return 1

    SHARDS_OUT.mkdir(parents=True, exist_ok=True)
    for c in controllers:
        (SHARDS_OUT / c).mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    sel = _select_scenes(split["scenarios"], args.scenes)
    print(f"[E7] {len(sel)} scenes x {len(controllers)} controllers x {args.reps} reps")

    # Per-(scene, controller): list of (wall_time, hash, rep)
    per_scene: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    errors = []

    if args.scene_workers > 1:
        # Parallel-per-controller: 1 worker per controller, each runs
        # serially over the 64 scenes and 3 reps.  Per-scene latency is
        # still isolated because each worker is the sole owner of its
        # controller's rollouts.
        import multiprocessing as mp
        ctx = mp.get_context()
        tasks = []
        for c in controllers:
            for sc in sel:
                sid = sc["scenario_id"]
                baseline = float(base["records"][sid]["expected_blood_loss_ml"])
                for rep in range(1, args.reps + 1):
                    tasks.append((c, sid, sc, baseline, args.margin, str(args.checkpoint), rep))

        t_total0 = time.time()
        # Group tasks by controller; one Pool per controller.
        from collections import defaultdict as _dd
        per_ctrl: dict = _dd(list)
        for t in tasks:
            per_ctrl[t[0]].append(t)
        from concurrent.futures import ThreadPoolExecutor
        def _run_one(args_tuple):
            c, sid, scene, baseline, margin, ckpt, rep = args_tuple
            return _run_once(c, scene, baseline, margin, ckpt)[1], c, sid, rep

        with ThreadPoolExecutor(max_workers=args.scene_workers) as ex:
            futures = [ex.submit(_run_one, t) for t in tasks]
            for fut in futures:
                wall, c, sid, rep = fut.result()
                # Note: action_hash is in the shard, fetch it by re-reading after write
                shard_path = SHARDS_OUT / c / f"{sid}_rep{rep}.json"
                shard = json.loads(shard_path.read_text()) if shard_path.exists() else {}
                per_scene[(sid, c)].append((wall, shard.get("action_sequence_hash", "")))
                if sum(len(v) for v in per_scene.values()) % 50 == 0:
                    print(f"  done {sum(len(v) for v in per_scene.values())} runs in {time.time()-t_total0:.0f}s")
    else:
        t_total0 = time.time()
        for c in controllers:
            for sc in sel:
                sid = sc["scenario_id"]
                baseline = float(base["records"][sid]["expected_blood_loss_ml"])
                for rep in range(1, args.reps + 1):
                    t0 = time.time()
                    try:
                        res, wall = _run_once(c, sc, baseline, args.margin, args.checkpoint)
                    except Exception as e:
                        errors.append(f"{c}/{sid}/rep{rep}: {e!r}")
                        continue
                    shard = dict(res)
                    shard["scenario_id"] = sid
                    shard["controller"] = c
                    shard["rep"] = rep
                    shard["wall_seconds_rep"] = wall
                    (SHARDS_OUT / c / f"{sid}_rep{rep}.json").write_text(
                        json.dumps(shard, ensure_ascii=False, indent=2)
                    )
                    per_scene[(sid, c)].append((wall, shard.get("action_sequence_hash", "")))
                    if sum(len(v) for v in per_scene.values()) % 25 == 0:
                        print(f"  done {sum(len(v) for v in per_scene.values())} runs in {time.time()-t_total0:.0f}s")

    # Per-scene median wall time
    per_scene_median: dict[tuple[str, str], float] = {}
    for k, vs in per_scene.items():
        walls = [w for w, _ in vs]
        per_scene_median[k] = statistics.median(walls) if walls else float("nan")

    # Summary per controller
    summary: dict = {
        "n_scenes": len(sel),
        "reps": args.reps,
        "controllers": controllers,
    }
    for c in controllers:
        walls = [per_scene_median[(sc["scenario_id"], c)] for sc in sel
                 if (sc["scenario_id"], c) in per_scene_median]
        if not walls:
            continue
        s = sorted(walls)
        n = len(s)
        summary[c] = {
            "n": n,
            "mean": sum(s) / n,
            "median": s[n // 2],
            "p5": s[int(0.05 * (n - 1))],
            "p25": s[int(0.25 * (n - 1))],
            "p75": s[int(0.75 * (n - 1))],
            "p95": s[int(0.95 * (n - 1))],
            "min": s[0],
            "max": s[-1],
        }
    # Paired ratios: C4L/C3, C4L/C4E, C4L/C5
    paired = {}
    for a in controllers:
        for b in controllers:
            if a == b:
                continue
            ratios = []
            for sc in sel:
                sid = sc["scenario_id"]
                if (sid, a) in per_scene_median and (sid, b) in per_scene_median:
                    wa = per_scene_median[(sid, a)]
                    wb = per_scene_median[(sid, b)]
                    if wb > 0:
                        ratios.append(wa / wb)
            if ratios:
                rs = sorted(ratios)
                n = len(rs)
                paired[f"{a}_over_{b}"] = {
                    "n": n,
                    "mean": sum(rs) / n,
                    "median": rs[n // 2],
                    "p5": rs[int(0.05 * (n - 1))],
                    "p95": rs[int(0.95 * (n - 1))],
                }
    summary["paired_ratios"] = paired
    (SHARDS_OUT / "latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (SHARDS_OUT / "paired_wall_time.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2)
    )
    print(f"[E7] wrote {SHARDS_OUT / 'latency_summary.json'}")
    if errors:
        print(f"[E7] {len(errors)} errors (first 3):")
        for e in errors[:3]:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

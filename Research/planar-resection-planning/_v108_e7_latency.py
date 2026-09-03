"""v10.8 E7 latency experiment (rewrite, per Bryce 2026-09-04).

Spec:
  * 64 latency scenes (evenly-spaced from the 256 E5 split)
  * 4 controllers: C0, C3, C4E, C4L
  * 3 reps per (scene, controller)
  * scene_workers = 1 (serial within a controller)
  * Per-scene median wall time across 3 reps
  * Controller run order is balanced-randomized across the 3 reps
    (cyclic Latin square) so that no controller is always first.

Output:
  results/clinical_window_v10_8_lazy_shield/latency_v2/rep<N>/<ctrl>/<sid>.json
  results/clinical_window_v10_8_lazy_shield/latency_v2/latency_summary.json
  results/clinical_window_v10_8_lazy_shield/latency_v2/latency_per_scene.csv

``--leaf-workers`` is applied to C3/C4E (their eager candidate-verify
pass) and ignored by C4L.  ``--tuned-leaf-workers`` reads the value
recommended by ``_v108_worker_tune.py``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

V108_OUT = REPO / "results/clinical_window_v10_8_lazy_shield"
LATENCY_OUT = V108_OUT / "latency_v2"
DEFAULT_CHECKPOINT = (REPO
    / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")

CONTROLLERS = ["C0", "C3", "C4E", "C4L"]

# Cyclic Latin square: each controller takes each starting position
# once across the 3 reps (mod 4).  Rep 2 is a reverse so position-3
# is also exercised by every controller at least once.
REP_ORDER = [
    ["C0", "C3", "C4E", "C4L"],   # rep 0
    ["C4L", "C0", "C3", "C4E"],   # rep 1
    ["C4E", "C4L", "C0", "C3"],   # rep 2
]

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


def _pick_scenes(split: dict, n: int) -> list[dict]:
    sc = split["scenarios"]
    if n >= len(sc):
        return sc
    step = max(1, len(sc) // n)
    return [sc[i * step] for i in range(n)]


def _run(ctrl: str, scene: dict, baseline: float, margin: float,
         checkpoint: str, leaf_workers: int, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    from lazy_confirmation_controllers_v108 import rollout_controller
    leaf_pool = None
    if leaf_workers and leaf_workers > 0 and ctrl in ("C3", "C4E"):
        leaf_pool = mp.get_context().Pool(int(leaf_workers))
    t0 = time.time()
    try:
        try:
            res = rollout_controller(
                ctrl, scene,
                baseline_blood=float(baseline), margin_ml=float(margin),
                checkpoint_path=str(checkpoint),
                leaf_pool=leaf_pool,
            )
        finally:
            if leaf_pool is not None:
                leaf_pool.close()
                leaf_pool.join()
    except BaseException as e:
        if leaf_pool is not None:
            try:
                leaf_pool.terminate()
                leaf_pool.join()
            except Exception:
                pass
        err_shard = {"scenario_id": scene["scenario_id"], "controller": ctrl,
                     "error": repr(e), "wall_seconds": time.time() - t0,
                     "completion": False}
        out_path.write_text(json.dumps(err_shard, ensure_ascii=False, indent=2))
        return err_shard
    shard = dict(res)
    shard["scenario_id"] = scene["scenario_id"]
    shard["controller"] = ctrl
    shard["leaf_workers"] = int(leaf_workers) if ctrl in ("C3", "C4E") else 0
    shard["wall_seconds"] = float(res.get("wall_seconds", time.time() - t0))
    shard["tuning_meta"] = {"leaf_workers": int(leaf_workers), "controller": ctrl}
    out_path.write_text(json.dumps(shard, ensure_ascii=False, indent=2))
    return shard


def _pick_leaf(ctrl: str, leaf_workers: int, tuned: dict) -> int:
    if leaf_workers and leaf_workers > 0:
        return int(leaf_workers)
    if tuned:
        rec = tuned.get("best_per_controller", {}).get(ctrl)
        if rec:
            return int(rec["leaf_workers"])
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--n-scenes", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--leaf-workers", type=int, default=0,
                        help="Override; if 0, use the value recommended by "
                             "_v108_worker_tune.py summary for each controller.")
    parser.add_argument("--tuning-summary", type=Path,
                        default=LATENCY_OUT / "../tuning/worker_tuning_summary.json")
    parser.add_argument("--reps-to-run", default="0,1,2",
                        help="Comma-separated rep indices to actually run "
                             "(default all 3; useful for resuming).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; don't run any rollouts.")
    args = parser.parse_args(argv)

    controllers = [c for c in args.controllers.split(",") if c]
    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = _pick_scenes(split, args.n_scenes)
    margin = float(args.margin)

    tuned = {}
    if args.tuning_summary and args.tuning_summary.exists():
        tuned = json.loads(args.tuning_summary.read_text())
    leaf_per_ctrl = {c: _pick_leaf(c, args.leaf_workers, tuned) for c in controllers}

    LATENCY_OUT.mkdir(parents=True, exist_ok=True)
    for c in controllers:
        for r in range(args.reps):
            (LATENCY_OUT / f"rep{r}" / c).mkdir(parents=True, exist_ok=True)

    reps_to_run = [int(r) for r in args.reps_to_run.split(",") if r]
    plan_rows = []
    for rep in reps_to_run:
        order = REP_ORDER[rep % len(REP_ORDER)]
        for ctrl in order:
            if ctrl not in controllers:
                continue
            for scene in scenes:
                sid = scene["scenario_id"]
                if sid not in base["records"]:
                    continue
                plan_rows.append((rep, ctrl, scene, sid, leaf_per_ctrl[ctrl]))

    print(f"[E7] plan: {len(plan_rows)} rollouts "
          f"({len(scenes)} scenes x {len(controllers)} controllers x {len(reps_to_run)} reps)")
    print(f"[E7] leaf_workers per controller: {leaf_per_ctrl}")
    print(f"[E7] rep order: {[REP_ORDER[r % len(REP_ORDER)] for r in reps_to_run]}")

    if args.dry_run:
        return 0

    t_start = time.time()
    done = 0
    for rep, ctrl, scene, sid, lw in plan_rows:
        out_path = LATENCY_OUT / f"rep{rep}" / ctrl / f"{sid}.json"
        baseline = float(base["records"][sid]["expected_blood_loss_ml"])
        shard = _run(ctrl, scene, baseline, margin, str(args.checkpoint), lw, out_path)
        done += 1
        if done % 32 == 0 or "error" in shard:
            err = "ERR" if "error" in shard else "ok"
            print(f"  [{done}/{len(plan_rows)}] rep{rep} {ctrl} {sid}: {err} "
                  f"wall={shard.get('wall_seconds', 0):.2f}s")

    print(f"[E7] total: {done} rollouts in {time.time() - t_start:.0f}s")
    _aggregate(scenes, controllers, args.reps)
    return 0


def _aggregate(scenes, controllers, reps):
    rows = []
    per_ctrl: dict = {}
    for ctrl in controllers:
        per_scene: dict = {}
        for scene in scenes:
            sid = scene["scenario_id"]
            walls = []
            for rep in range(reps):
                p = LATENCY_OUT / f"rep{rep}" / ctrl / f"{sid}.json"
                if p.exists():
                    try:
                        j = json.loads(p.read_text())
                        walls.append(float(j.get("wall_seconds", 0.0)))
                    except Exception:
                        pass
            if walls:
                med = statistics.median(walls)
                per_scene[sid] = {"n": len(walls), "median": med, "walls": walls}
                rows.append({"controller": ctrl, "scenario_id": sid,
                             "n_reps": len(walls), "median_s": med})
        all_walls = [v["median"] for v in per_scene.values()]
        if all_walls:
            per_ctrl[ctrl] = {
                "n_scenes": len(all_walls),
                "median_s": statistics.median(all_walls),
                "mean_s": statistics.mean(all_walls),
                "p95_s": sorted(all_walls)[int(0.95 * (len(all_walls) - 1))] if len(all_walls) > 1 else all_walls[0],
                "max_s": max(all_walls),
            }

    summary = {
        "spec": {
            "n_scenes": len(scenes),
            "controllers": controllers,
            "reps": reps,
            "scene_workers": 1,
            "controller_order_per_rep": REP_ORDER[:reps],
        },
        "per_controller": per_ctrl,
    }
    (LATENCY_OUT / "latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    csv_lines = ["controller,scenario_id,n_reps,median_wall_s"]
    for r in rows:
        csv_lines.append(f"{r['controller']},{r['scenario_id']},{r['n_reps']},{r['median_s']:.4f}")
    (LATENCY_OUT / "latency_per_scene.csv").write_text("\n".join(csv_lines) + "\n")
    print()
    print("[E7] per-controller median wall time (median across 3 reps, mean across scenes):")
    for ctrl, info in per_ctrl.items():
        print(f"  {ctrl}: n_scenes={info['n_scenes']:3d}  median_s={info['median_s']:.2f}  "
              f"mean_s={info['mean_s']:.2f}  p95_s={info['p95_s']:.2f}  max_s={info['max_s']:.2f}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

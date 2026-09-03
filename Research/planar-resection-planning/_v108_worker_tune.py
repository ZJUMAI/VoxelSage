"""v10.8 C3/C4E worker tuning.

Sweeps ``leaf_workers`` in {1,2,3,6} on 8 representative scenes for the
two eager-verify controllers C3 and C4E, measures wall time per cell,
and writes a summary table that picks the best ``leaf_workers`` for the
E7 latency comparison against C4L.

Output:
  results/clinical_window_v10_8_lazy_shield/tuning/<ctrl>_lw<N>/<sid>.json
  results/clinical_window_v10_8_lazy_shield/tuning/worker_tuning_summary.json
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
TUNE_OUT = V108_OUT / "tuning"
DEFAULT_CHECKPOINT = (REPO
    / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


def _pick_scenes(split: dict, n: int) -> list[dict]:
    """Pick ``n`` evenly-spaced scenarios from the 256 E5 split."""
    sc = split["scenarios"]
    if n >= len(sc):
        return sc
    step = len(sc) // n
    return [sc[i * step] for i in range(n)]


def _run(controller: str, scene: dict, baseline: float, margin: float,
         checkpoint: str, leaf_workers: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = scene["scenario_id"]
    out = out_dir / f"{sid}.json"
    if out.exists():
        j = json.loads(out.read_text())
        j.setdefault("tuning_meta", {})
        j["tuning_meta"]["reused"] = True
        return j
    from lazy_confirmation_controllers_v108 import rollout_controller
    leaf_pool = None
    if leaf_workers and leaf_workers > 0:
        leaf_pool = mp.get_context().Pool(int(leaf_workers))
    t0 = time.time()
    try:
        try:
            res = rollout_controller(
                controller, scene,
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
        return {"scenario_id": sid, "controller": controller, "leaf_workers": leaf_workers,
                "error": repr(e), "wall_seconds": time.time() - t0}
    shard = dict(res)
    shard["scenario_id"] = sid
    shard["controller"] = controller
    shard["leaf_workers"] = int(leaf_workers)
    shard["wall_seconds"] = float(res.get("wall_seconds", time.time() - t0))
    shard["tuning_meta"] = {"leaf_workers": int(leaf_workers),
                            "controller": controller, "split_idx": "even_step"}
    out.write_text(json.dumps(shard, ensure_ascii=False, indent=2))
    return shard


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--n-scenes", type=int, default=8)
    parser.add_argument("--leaf-workers", type=int, nargs="+", default=[1, 2, 3, 6])
    parser.add_argument("--controllers", default="C3,C4E")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output shard already exists.")
    args = parser.parse_args(argv)

    if args.force and TUNE_OUT.exists():
        import shutil
        for child in TUNE_OUT.iterdir():
            if child.is_dir() and child.name.startswith(("C3_lw", "C4E_lw")):
                shutil.rmtree(child)

    controllers = [c for c in args.controllers.split(",") if c]
    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = _pick_scenes(split, args.n_scenes)
    margin = float(args.margin)

    jobs = []
    for scene in scenes:
        sid = scene["scenario_id"]
        if sid not in base["records"]:
            print(f"  skip {sid}: no baseline")
            continue
        baseline = float(base["records"][sid]["expected_blood_loss_ml"])
        for ctrl in controllers:
            for lw in args.leaf_workers:
                jobs.append((ctrl, scene, baseline, margin,
                             str(args.checkpoint), int(lw), TUNE_OUT / f"{ctrl}_lw{lw}"))
    print(f"[tune] {len(scenes)} scenes x {len(controllers)} controllers x "
          f"{len(args.leaf_workers)} leaf_values = {len(jobs)} rollouts")
    t_start = time.time()
    done = 0
    for j in jobs:
        shard = _run(*j)
        done += 1
        err = "ERR" if "error" in shard else "ok"
        print(f"  [{done}/{len(jobs)}] {shard.get('controller', j[0])} lw={j[5]} "
              f"{shard.get('scenario_id', j[1]['scenario_id'])}: {err} "
              f"wall={shard.get('wall_seconds', 0):.2f}s")
    print(f"  total tune: {done} rollouts in {time.time() - t_start:.0f}s")

    summary = {"per_cell": {}, "best_per_controller": {}}
    for ctrl in controllers:
        for lw in args.leaf_workers:
            out_dir = TUNE_OUT / f"{ctrl}_lw{lw}"
            walls = []
            for f in sorted(out_dir.glob("*.json")):
                try:
                    j = json.loads(f.read_text())
                    walls.append(float(j.get("wall_seconds", 0.0)))
                except Exception:
                    pass
            if walls:
                summary["per_cell"][f"{ctrl}_lw{lw}"] = {
                    "n": len(walls),
                    "mean": statistics.mean(walls),
                    "median": statistics.median(walls),
                    "min": min(walls),
                    "max": max(walls),
                }
    for ctrl in controllers:
        cells = {k: v for k, v in summary["per_cell"].items() if k.startswith(ctrl + "_lw")}
        if cells:
            best = min(cells.items(), key=lambda kv: kv[1]["mean"])
            summary["best_per_controller"][ctrl] = {
                "leaf_workers": int(best[0].split("_lw")[1]),
                "mean_seconds": best[1]["mean"],
                "n": best[1]["n"],
            }
    TUNE_OUT.mkdir(parents=True, exist_ok=True)
    (TUNE_OUT / "worker_tuning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("[tune] best per controller:")
    for ctrl, info in summary["best_per_controller"].items():
        print(f"  {ctrl}: leaf_workers={info['leaf_workers']}  mean={info['mean_seconds']:.2f}s  n={info['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

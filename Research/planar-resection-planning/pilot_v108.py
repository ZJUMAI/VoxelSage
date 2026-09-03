"""E4: 32-scene latency pilot (plan §7.5).

Goal: decide whether C4L is plausibly faster than C3 under fair
conditions, so we can stop early if it isn't.

Controllers: C3, C4E (re-run on these 32 scenes to keep latency
comparable), C4L, C5.  Old v10.7 C3/C4/C5 shards are kept for the
256-scene confirmatory set but the latency comparison must use
fresh runs under the chosen resource config to avoid cold-cache /
thread-affinity contamination.

Default mode is single-process (`--scene-workers 1`) with
`OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, torch.set_num_threads(1)`.
The 32 scene IDs are taken at evenly spaced indices from the v10.7
Replication split (deterministic, pre-registered).

Outputs:
  results/clinical_window_v10_8_lazy_shield/pilot/
    C3/  C4E/  C4L/  C5/    per-scene shards
    pilot_summary.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

FROZEN_SPLIT = REPO / "results/clinical_window_v10_7_confirmation/frozen/split_replication.json"
FROZEN_BASELINE = REPO / "results/clinical_window_v10_7_confirmation/frozen/baseline_replication.json"
CHECKPOINT = REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt"
V108_PILOT = REPO / "results/clinical_window_v10_8_lazy_shield/pilot"


def _select_scenes(scenes: list[dict], k: int) -> list[dict]:
    """Even-spaced index selection: indices = round(i*(N-1)/(k-1)) for i in [0,k-1]."""
    n = len(scenes)
    if k >= n:
        return list(scenes)
    step = (n - 1) / (k - 1)
    idxs = sorted({int(round(i * step)) for i in range(k)})
    return [scenes[i] for i in idxs]


def load_inputs() -> tuple[list[dict], dict[str, float], float]:
    splits = json.loads(FROZEN_SPLIT.read_text())
    base = json.loads(FROZEN_BASELINE.read_text())
    baseline = {sid: float(rec["expected_blood_loss_ml"])
                for sid, rec in base["records"].items()}
    mfst = json.loads(
        (REPO / "results/clinical_window_v10_7_confirmation/frozen/experiment_manifest.json").read_text()
    )
    return splits["scenarios"], baseline, float(mfst["margin_ml"])


def _setup_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)


def _task(args):
    _setup_threads()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-workers", type=int, default=1)
    parser.add_argument("--scenes", type=int, default=32)
    parser.add_argument("--controllers", default="C3,C4E,C4L,C5")
    args = parser.parse_args(argv)
    controllers = [c for c in args.controllers.split(",") if c]

    V108_PILOT.mkdir(parents=True, exist_ok=True)
    for c in controllers:
        (V108_PILOT / c).mkdir(parents=True, exist_ok=True)

    scenes, baseline, margin = load_inputs()
    sel = _select_scenes(scenes, args.scenes)
    print(f"[E4] {len(sel)} scenes x {len(controllers)} controllers, scene_workers={args.scene_workers}")

    tasks = []
    for c in controllers:
        for s in sel:
            tasks.append((c, s["scenario_id"], s, baseline[s["scenario_id"]], margin, str(CHECKPOINT)))

    if args.scene_workers <= 1:
        results: list = []
        for t in tasks:
            sid, c, res, err = _task(t)
            if err:
                print(f"  ERR {c}/{sid}: {err}")
            else:
                results.append((c, sid, res))
                shard = dict(res)
                shard["scenario_id"] = sid
                shard["controller"] = c
                (V108_PILOT / c / f"{sid}.json").write_text(
                    json.dumps(shard, ensure_ascii=False, indent=2)
                )
    else:
        # Windows has no fork; use the default (spawn).
        ctx = mp.get_context()
        results = []
        done = 0
        with ctx.Pool(args.scene_workers) as pool:
            for sid, c, res, err in pool.imap_unordered(_task, tasks, chunksize=1):
                done += 1
                if err:
                    print(f"  ERR {c}/{sid}: {err}")
                else:
                    results.append((c, sid, res))
                    shard = dict(res)
                    shard["scenario_id"] = sid
                    shard["controller"] = c
                    (V108_PILOT / c / f"{sid}.json").write_text(
                        json.dumps(shard, ensure_ascii=False, indent=2)
                    )
                if done % 10 == 0:
                    print(f"  done {done}/{len(tasks)}")

    # Aggregate
    by_controller: dict[str, list[float]] = {c: [] for c in controllers}
    for c, sid, res in results:
        by_controller[c].append(float(res.get("wall_seconds", res.get("wall_seconds_pilot", 0))))
    summary = {
        "n_scenes": len(sel),
        "scene_workers": args.scene_workers,
        "controllers": controllers,
        "per_controller": {
            c: {
                "n": len(v),
                "mean": (sum(v) / len(v)) if v else 0.0,
                "p50": (sorted(v)[len(v) // 2] if v else 0.0),
                "p95": (sorted(v)[int(0.95 * (len(v) - 1))] if v else 0.0),
                "max": (max(v) if v else 0.0),
            } for c, v in by_controller.items()
        },
        "ratios": {
            f"{a}_over_{b}": (
                (sum(by_controller[a]) / sum(by_controller[b]))
                if by_controller[a] and by_controller[b] else None
            ) for a in controllers for b in controllers if a != b
        },
    }
    (V108_PILOT / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n[E4] wrote {V108_PILOT / 'pilot_summary.json'}")
    print(json.dumps(summary["per_controller"], indent=2))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))

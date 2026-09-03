"""E8: 5-condition sensitivity on v10.8 lazy-shield base scenes (plan §7.9).

Reuses the v10.7.1 frozen sensitivity conditions (S0..S4) and the
v10.8 128-scene sensitivity_base split (or the 256-scene replication
split as fallback).  Runs C0, C3, C4E, C4L, C5 on each.

Outputs:
  results/clinical_window_v10_8_lazy_shield/sensitivity/
    S0/  S1/  S2/  S3/  S4/
      C0/  C3/  C4E/  C4L/  C5/
        <sid>.json
    sensitivity_summary.json
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
SENS_OUT = V108_OUT / "sensitivity"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


SENSITIVITY_CONDITIONS = {
    "S0": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S1": {"max_clamp_minutes": 12.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S2": {"max_clamp_minutes": 10.0, "unclamp_minutes": 5.0, "bleeding_probability": 1.0},
    "S3": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.5},
    "S4": {"max_clamp_minutes": 15.0, "unclamp_minutes": 5.0, "bleeding_probability": 0.25},
}


def _task(args):
    controller, condition, sid, scene, baseline, margin, ckpt, cfg_overrides = args
    from lazy_confirmation_controllers_v108 import rollout_controller
    try:
        res = rollout_controller(
            controller, scene,
            baseline_blood=float(baseline), margin_ml=float(margin),
            cfg=cfg_overrides,
            checkpoint_path=str(ckpt),
        )
        return condition, controller, sid, res, None
    except BaseException as e:
        return condition, controller, sid, None, repr(e)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path,
                        default=V108_OUT / "frozen/split_lazy_replication.json")
    parser.add_argument("--baseline-file", type=Path,
                        default=V108_OUT / "frozen/baseline_lazy_replication.json")
    parser.add_argument("--margin", type=float, default=16.07054347826075)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--conditions", default="S0,S1,S2,S3,S4")
    parser.add_argument("--controllers", default="C0,C3,C4E,C4L,C5")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt")
    parser.add_argument("--scene-workers", type=int, default=20)
    args = parser.parse_args(argv)
    conds = args.conditions.split(",")
    controllers = args.controllers.split(",")

    if not args.split_file.exists() or not args.baseline_file.exists():
        print(f"[E8] missing frozen input")
        return 1

    split = json.loads(args.split_file.read_text())
    base = json.loads(args.baseline_file.read_text())
    scenes = split["scenarios"][:args.limit]
    print(f"[E8] {len(scenes)} scenes x {len(conds)} conditions x {len(controllers)} controllers")

    SENS_OUT.mkdir(parents=True, exist_ok=True)
    for c in conds:
        (SENS_OUT / c).mkdir(parents=True, exist_ok=True)
        for ctrl in controllers:
            (SENS_OUT / c / ctrl).mkdir(parents=True, exist_ok=True)

    tasks = []
    for c in conds:
        cfg = SENSITIVITY_CONDITIONS[c]
        for sc in scenes:
            sid = sc["scenario_id"]
            if sid not in base["records"]:
                continue
            for ctrl in controllers:
                tasks.append((ctrl, c, sid, sc,
                              float(base["records"][sid]["expected_blood_loss_ml"]),
                              args.margin, str(args.checkpoint), cfg))

    print(f"[E8] {len(tasks)} task tuples")
    if args.scene_workers <= 1:
        results = []
        for t in tasks:
            cond, ctrl, sid, res, err = _task(t)
            if err:
                continue
            results.append((cond, ctrl, sid, res))
            shard = dict(res)
            shard["scenario_id"] = sid
            shard["controller"] = ctrl
            shard["condition"] = cond
            (SENS_OUT / cond / ctrl / f"{sid}.json").write_text(
                json.dumps(shard, ensure_ascii=False, indent=2)
            )
    else:
        ctx = mp.get_context()
        results = []
        done = 0
        with ctx.Pool(args.scene_workers) as pool:
            for cond, ctrl, sid, res, err in pool.imap_unordered(_task, tasks, chunksize=2):
                done += 1
                if err:
                    pass
                else:
                    results.append((cond, ctrl, sid, res))
                    shard = dict(res)
                    shard["scenario_id"] = sid
                    shard["controller"] = ctrl
                    shard["condition"] = cond
                    (SENS_OUT / cond / ctrl / f"{sid}.json").write_text(
                        json.dumps(shard, ensure_ascii=False, indent=2)
                    )
                if done % 50 == 0:
                    print(f"  done {done}/{len(tasks)}")

    print(f"[E8] wrote {len(results)} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

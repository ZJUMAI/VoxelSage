"""8-scene Gate A smoke: timing and T/B effect of the window-aware planner.

Runs serpentine baseline then the window-aware planner on a small planner_gate
subset, reports per-scene wall time, delta T and delta B. Used to decide whether
the planner is within the single-scene 10 min budget before expanding to the
32-scene resource Pilot.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_window_evaluation import serpentine_macro_target_policy
from plan_target_order_v104 import (
    SerpentineTail,
    WindowAwarePlanner,
    make_gate_rollout,
    rollout_planner,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--lookahead-depth", type=int, default=1)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--split", default="planner_gate")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--leaf-workers", type=int, default=24)
    args = parser.parse_args()

    data = json.load(open("results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"))
    scenarios = data["splits"][args.split]["scenarios"][args.offset:args.offset + args.limit]
    run_serp = make_gate_rollout(serpentine_macro_target_policy)

    baselines = []
    for sc in scenarios:
        t0 = time.time()
        r = run_serp(sc)
        baselines.append(r)
        print(f"serp  {sc['scenario_id']} T={r['elapsed_minutes']:.1f} B={r['expected_blood_loss_ml']:.1f} "
              f"comp={r['completion']} ({time.time()-t0:.1f}s)", flush=True)

    mean_b = sum(r["expected_blood_loss_ml"] for r in baselines) / len(baselines)
    margin = 0.05 * mean_b
    print(f"\nmean baseline B={mean_b:.1f} mL, margin M_B={margin:.1f} mL", flush=True)

    wall_times = []
    cfg = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}
    leaf_pool = None
    if args.leaf_workers > 0:
        leaf_pool = mp.get_context("fork").Pool(args.leaf_workers)
    try:
        for i, (sc, base) in enumerate(zip(scenarios, baselines)):
            tail = SerpentineTail(clinical_config=cfg)
            planner = WindowAwarePlanner(
                candidate_count=args.candidate_count,
                beam_width=args.beam_width,
                lookahead_depth=args.lookahead_depth,
                margin_blood_ml=margin,
                tail=tail,
                leaf_pool=leaf_pool,
                clinical_config=cfg,
            )
            t0 = time.time()
            rp = rollout_planner(
                sc, planner,
                baseline_blood=base["expected_blood_loss_ml"],
                replan_interval=args.replan_interval,
                clinical_config=cfg,
            )
            dt = time.time() - t0
            wall_times.append(dt)
            dT = rp["elapsed_minutes"] - base["elapsed_minutes"]
            dB = rp["expected_blood_loss_ml"] - base["expected_blood_loss_ml"]
            print(f"[{i+1}/{len(scenarios)}] plan {sc['scenario_id']} T={rp['elapsed_minutes']:.1f} "
                  f"B={rp['expected_blood_loss_ml']:.1f} comp={rp['completion']} dT={dT:+.2f} "
                  f"dB={dB:+.1f} ({dt:.1f}s trees={planner.tree_count} leaves={planner.leaf_count} "
                  f"cache={tail.cache_size})", flush=True)
    finally:
        if leaf_pool is not None:
            leaf_pool.close()
            leaf_pool.join()

    wall_times.sort()
    print(f"\nwall-time median={wall_times[len(wall_times)//2]:.1f}s max={max(wall_times):.1f}s "
          f"min={min(wall_times):.1f}s", flush=True)


if __name__ == "__main__":
    main()

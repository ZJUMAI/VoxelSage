"""Gate A evaluation for v10.4: serpentine / nearest / window-aware planner.

Workflow (guide Section 6):
  1. serpentine baseline over the chosen split -> per-scene (T, B);
     margin M_B = 0.05 * mean(B_baseline).
  2. nearest frontier branch (fair comparison).
  3. window-aware planner branch (depth-1 MPC with full S-tail evaluation,
     guide 6.2), each scene constrained by its own baseline blood + M_B.
  4. scene-paired bootstrap CI on Delta T / Delta B; GO/NO-GO table (guide 6.4).

Parallelism (guide 11): ``--scene-workers`` child processes, each with its own
``--leaf-workers`` process pool for the parallel candidate-tail evaluations.
Everything is deterministic.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_window_evaluation import serpentine_macro_target_policy
from plan_target_order_v104 import (
    SerpentineTail,
    WindowAwarePlanner,
    make_gate_rollout,
    nearest_frontier_macro_policy,
    rollout_planner,
)

OUT_DIR = SIM / "results/clinical_window_v10_4_target_order"
GATE_SPLITS = OUT_DIR / "pilot_gate_a/gate_a_splits_v104.json"


def _run_serpentine(args):
    scen, cfg = args
    run = make_gate_rollout(serpentine_macro_target_policy, clinical_config=cfg)
    rec = run(scen)
    rec["wall_seconds"] = 0.0
    return rec


def _run_nearest(args):
    scen, cfg = args
    run = make_gate_rollout(nearest_frontier_macro_policy, clinical_config=cfg)
    t0 = time.time()
    rec = run(scen)
    rec["wall_seconds"] = time.time() - t0
    return rec


def _run_planner(args):
    scen, baseline_blood, margin, cfg, params, leaf_pool = args
    tail = SerpentineTail(clinical_config=cfg)
    planner = WindowAwarePlanner(
        candidate_count=params["candidate_count"],
        beam_width=params["beam_width"],
        lookahead_depth=params["lookahead_depth"],
        margin_blood_ml=margin,
        tail=tail,
        leaf_pool=leaf_pool,
        clinical_config=cfg,
    )
    t0 = time.time()
    rec = rollout_planner(
        scen, planner,
        baseline_blood=baseline_blood,
        replan_interval=params["replan_interval"],
        clinical_config=cfg,
    )
    rec["wall_seconds"] = time.time() - t0
    rec["planner_trees"] = planner.tree_count
    rec["planner_leaves"] = planner.leaf_count
    rec["planner_nodes"] = planner.nodes_expanded
    rec["tail_cache_size"] = tail.cache_size
    return rec


def _child_worker(scenes, baseline_map, margin, cfg, params, leaf_workers, queue):
    """One child process: its own leaf pool, runs nearest + planner for its
    scene subset, returns aggregated records through a multiprocessing.Queue."""
    leaf_pool = mp.get_context("fork").Pool(leaf_workers)
    try:
        nearest = [_run_nearest((sc, cfg)) for sc in scenes]
        planner = [
            _run_planner((sc, baseline_map[sc["scenario_id"]]["expected_blood_loss_ml"],
                          margin, cfg, params, leaf_pool))
            for sc in scenes
        ]
    finally:
        leaf_pool.close()
        leaf_pool.join()
    queue.put({"nearest": nearest, "planner": planner})


def _paired_delta(records, baseline_records, field):
    diffs = np.asarray([
        float(rec[field]) - float(baseline_records[rec["scenario_id"]][field])
        for rec in records
    ])
    return diffs


def _bootstrap_ci(diffs, *, samples=10_000, seed=20260811):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(samples, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("planner_tune", "planner_gate"), default="planner_gate")
    parser.add_argument("--limit", type=int, default=None, help="run first N scenes only")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--scene-workers", type=int, default=8)
    parser.add_argument("--leaf-workers", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--lookahead-depth", type=int, default=1)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path,
                        default=OUT_DIR / "pilot_gate_a/gate_a_evaluation.json")
    args = parser.parse_args()

    gate = json.loads(GATE_SPLITS.read_text(encoding="utf-8"))
    scenarios = list(gate["splits"][args.split]["scenarios"])
    if args.offset:
        scenarios = scenarios[args.offset:]
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit("no scenarios selected")
    print(f"split={args.split} scenarios={len(scenarios)} "
          f"scene_workers={args.scene_workers} leaf_workers={args.leaf_workers}", flush=True)

    cfg = {
        "early_end_mode": "disabled",
        "early_end_minutes": 0.0,
        "bleeding_probability": 1.0,
        "max_steps_multiplier": 8.0,
    }

    # Phase 1: serpentine baseline (sequential; needed to compute margin).
    t0 = time.time()
    baseline_records = {}
    for scen in scenarios:
        rec = _run_serpentine((scen, cfg))
        baseline_records[scen["scenario_id"]] = rec
    mean_b = float(np.mean([r["expected_blood_loss_ml"] for r in baseline_records.values()]))
    margin = 0.05 * mean_b
    print(f"baseline done ({time.time()-t0:.1f}s): mean_B={mean_b:.1f} mL, M_B={margin:.1f} mL",
          flush=True)

    params = {
        "candidate_count": args.candidate_count,
        "beam_width": args.beam_width,
        "lookahead_depth": args.lookahead_depth,
        "replan_interval": args.replan_interval,
    }

    # Phase 2+3: nearest + planner across child processes (each with a leaf pool).
    if args.scene_workers <= 1:
        nearest_records = [_run_nearest((sc, cfg)) for sc in scenarios]
        leaf_pool = mp.get_context("fork").Pool(args.leaf_workers)
        try:
            planner_records = [
                _run_planner((sc, baseline_records[sc["scenario_id"]]["expected_blood_loss_ml"],
                              margin, cfg, params, leaf_pool))
                for sc in scenarios
            ]
        finally:
            leaf_pool.close()
            leaf_pool.join()
    else:
        workers = min(args.scene_workers, len(scenarios))
        chunk_size = (len(scenarios) + workers - 1) // workers
        chunks = [scenarios[i:i + chunk_size] for i in range(0, len(scenarios), chunk_size)]
        processes = []
        queues = []
        for chunk in chunks:
            q = mp.get_context("fork").Queue()
            p = mp.get_context("fork").Process(
                target=_child_worker,
                args=(chunk, baseline_records, margin, cfg, params, args.leaf_workers, q),
            )
            p.start()
            processes.append(p)
            queues.append(q)
        collected = [q.get() for q in queues]
        for p in processes:
            p.join()
        nearest_records = [r for c in collected for r in c["nearest"]]
        planner_records = [r for c in collected for r in c["planner"]]

    def summarize(name, records):
        dT = _paired_delta(records, baseline_records, "elapsed_minutes")
        dB = _paired_delta(records, baseline_records, "expected_blood_loss_ml")
        ciT = _bootstrap_ci(dT, samples=args.bootstrap_samples)
        ciB = _bootstrap_ci(dB, samples=args.bootstrap_samples)
        return {
            "name": name,
            "n": len(records),
            "completion_rate": float(np.mean([r["completion"] for r in records])),
            "legal_action_rate": float(np.mean([r["legal_action_rate"] for r in records])),
            "end_count": int(sum(r["early_end_count"] for r in records)),
            "failure_count": int(sum(r["status"] != "ok" for r in records)),
            "stagnation_failure": int(sum(bool(r["stagnation_failure"]) for r in records)),
            "loop_failure": int(sum(bool(r["two_cell_loop_failure"]) for r in records)),
            "mean_T": float(np.mean([r["elapsed_minutes"] for r in records])),
            "mean_B": float(np.mean([r["expected_blood_loss_ml"] for r in records])),
            "mean_dT": float(dT.mean()),
            "dT_95_ci": [float(ciT[0]), float(ciT[1])],
            "mean_dB": float(dB.mean()),
            "dB_95_ci": [float(ciB[0]), float(ciB[1])],
            "median_wall_seconds": float(np.median([r["wall_seconds"] for r in records])),
        }

    summary = {
        "gate_version": "v10.4-gate-a-evaluation-v1",
        "split": args.split,
        "n_scenarios": len(scenarios),
        "margin_ml": margin,
        "params": params,
        "baseline": summarize("serpentine", list(baseline_records.values())),
        "nearest": summarize("nearest_frontier", nearest_records),
        "planner": summarize("window_aware_planner", planner_records),
        "go_no_go": None,
    }

    if args.split == "planner_gate" and len(scenarios) == 128:
        p = summary["planner"]
        b = summary["baseline"]
        go_conditions = {
            "completion_128_128": p["completion_rate"] == 1.0 and b["completion_rate"] == 1.0,
            "legal_rate_1_0": p["legal_action_rate"] == 1.0 and b["legal_action_rate"] == 1.0,
            "no_end_or_failure": (p["end_count"] == 0 and b["end_count"] == 0
                                  and p["failure_count"] == 0 and b["failure_count"] == 0),
            "blood_ci_upper_le_margin": p["dB_95_ci"][1] <= margin,
            "time_ci_upper_lt_0": p["dT_95_ci"][1] < 0.0,
            "time_effect_size": p["mean_dT"] <= -0.005 * b["mean_T"],
        }
        summary["go_no_go"] = {
            "conditions": go_conditions,
            "decision": "GO" if all(go_conditions.values()) else "NO-GO",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    raw_path = args.output.with_name("gate_a_raw_records.json")
    raw = {
        "baseline": list(baseline_records.values()),
        "nearest": nearest_records,
        "planner": planner_records,
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {raw_path}")


if __name__ == "__main__":
    main()

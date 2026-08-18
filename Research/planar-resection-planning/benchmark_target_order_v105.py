"""v10.5 behaviour-equivalent optimisation + benchmark (guide 9).

Only runs after Gate R GO. Freezes the corrected reference (already written to
reference/reference_traces.jsonl), then evaluates an *implementation-optimised*
planner whose decision rule is identical (same candidate set, budget, safe
filter, fallback, tie-break) but whose candidate tails are executed on a
persistent leaf pool.

Outputs:
  optimized/runtime_benchmark.json   latency + throughput
  optimized/equivalence_audit.json   128/128 action-hash + T/B agreement
  optimized/optimized_traces.jsonl   per-scene optimized results

Speed gate (9.2): p50 speedup >=3x vs serial reference, OR p50 <= 20 s; p95 <= 60 s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import _scan_rank
from plan_target_order_v105 import (
    CorrectedPlannerV105,
    SerpentineTailV105,
    _env_state_payload_v105,
    _step_macro_target,
    compute_margin_ml,
    scene_budget,
)
from plan_target_order_v105 import _candidate_sources_v105  # noqa: F401
from plan_target_order_v104 import serpentine_target_of  # noqa: F401

SIM = Path(__file__).resolve().parent
GATE_A_FILE = SIM / "results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"
BASE = SIM / "results/clinical_window_v10_5_safe_planner"
REF_TRACES = BASE / "reference" / "reference_traces.jsonl"
CFG = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}


def _tail_worker_v105(payload):
    """From a serialised state, execute target + frozen S tail to episode end.
    Returns (dt, db, completion, failure_reason) total from that state."""
    scenario, state, target, cfg = payload
    e = ClinicalMacroResectionEnv(scenario=scenario, clinical_config=cfg)
    e.reset()
    e.__dict__.update(state)
    e.events = []
    t0 = e.elapsed_minutes
    b0 = e.expected_blood_loss_ml
    _step_macro_target(e, target)
    if e.terminated or e.truncated:
        completion = bool(e.terminated and e.failure_reason is None)
        return (e.elapsed_minutes - t0, e.expected_blood_loss_ml - b0, completion, e.failure_reason)
    while not e.terminated and not e.truncated:
        legal = e._frontier()
        if not legal:
            e.terminated = True
            e.failure_reason = "serpentine tail lost all legal targets"
            break
        _step_macro_target(e, min(legal, key=lambda cell: _scan_rank(e, cell)))
    completion = bool(e.terminated and e.failure_reason is None)
    return (e.elapsed_minutes - t0, e.expected_blood_loss_ml - b0, completion, e.failure_reason)


class OptimizedPlannerV105:
    """Same decision rule as CorrectedPlannerV105; candidate tails run on a
    persistent leaf pool (behaviour-identical, faster wall clock)."""

    def __init__(self, *, candidate_count=6, margin_ml=None, leaf_pool=None,
                 clinical_config=None):
        self.candidate_count = int(candidate_count)
        self.margin_ml = margin_ml
        self.leaf_pool = leaf_pool
        self.clinical_config_used = dict(CFG)
        if clinical_config:
            self.clinical_config_used.update(clinical_config)
        self.fallback_count = 0
        self.plan_count = 0

    def plan(self, env, baseline_blood, budget):
        self.plan_count += 1
        budget_ml = float(budget) if budget is not None else float("inf")
        b_past = float(env.expected_blood_loss_ml)
        t_past = float(env.elapsed_minutes)
        counts = env._transfer_counts()
        frontier = env._frontier()
        sourced = _candidate_sources_v105(env, count=self.candidate_count, transfer_counts=counts)
        targets = [t for t, _s in sourced if t in frontier]

        if self.leaf_pool is not None and len(targets) > 1:
            state = _env_state_payload_v105(env)
            payloads = [(env.scenario, state, tuple(target), dict(self.clinical_config_used))
                        for target in targets]
            results = self.leaf_pool.map(_tail_worker_v105, payloads)
            leaves = [(target, dt, db, completion, reason)
                      for target, (dt, db, completion, reason) in zip(targets, results)]
        else:
            leaves = []
            tail = SerpentineTailV105(clinical_config=self.clinical_config_used)
            for target in targets:
                e2 = _clone_env_for_tail(env)
                t0, b0 = e2.elapsed_minutes, e2.expected_blood_loss_ml
                _step_macro_target(e2, target)
                dt = e2.elapsed_minutes - t0
                db = e2.expected_blood_loss_ml - b0
                if e2.terminated or e2.truncated:
                    completion = bool(e2.terminated and e2.failure_reason is None)
                    leaves.append((target, dt, db, completion, e2.failure_reason))
                else:
                    tdt, tdb, completion, reason = tail.tail(e2)
                    leaves.append((target, dt + tdt, db + tdb, completion, reason))

        safe = []
        max_b_total = b_past
        for target, dt, db, completion, reason in leaves:
            b_total = b_past + db
            max_b_total = max(max_b_total, b_total)
            if completion and reason is None and b_total <= budget_ml + 1e-9:
                safe.append((target, t_past + dt, b_total))
        if not safe:
            self.fallback_count += 1
            return [serpentine_target_of(env)], {"safety_invariant_violation": True,
                                                 "safe_candidate_count": 0,
                                                 "max_B_total_ml": float(max_b_total)}
        # Same quantised tie-break as CorrectedPlannerV105.
        best = min(safe, key=lambda item: (round(item[1], 6), round(item[2], 6), item[0]))
        return [best[0]], {"safety_invariant_violation": False,
                           "safe_candidate_count": len(safe),
                           "max_B_total_ml": float(max_b_total)}


def _clone_env_for_tail(env):
    from plan_target_order_v104 import _clone_env
    return _clone_env(env)


def _rollout_optimized(sc, baseline_blood, margin_ml, leaf_pool):
    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=CFG)
    env.reset()
    planner = OptimizedPlannerV105(candidate_count=6, margin_ml=margin_ml,
                                   leaf_pool=leaf_pool, clinical_config=CFG)
    budget = scene_budget(baseline_blood, margin_ml)
    actions = []
    inv = 0
    max_b = 0.0
    wall0 = time.time()
    while not env.terminated and not env.truncated:
        traj, info = planner.plan(env, baseline_blood, budget)
        max_b = max(max_b, float(info["max_B_total_ml"]))
        if info["safety_invariant_violation"]:
            inv += 1
        target = traj[0]
        if target not in env._frontier():
            from plan_target_order_v104 import serpentine_target_of
            target = serpentine_target_of(env)
        actions.append((int(target[0]), int(target[1])))
        _step_macro_target(env, target)
    return {
        "scenario_id": sc["scenario_id"],
        "teacher_T_min": float(env.elapsed_minutes),
        "teacher_B_ml": float(env.expected_blood_loss_ml),
        "completion": bool(env.terminated and env.failure_reason is None),
        "failure_reason": env.failure_reason,
        "max_B_total_ml": float(max_b),
        "safety_invariant_violations": inv,
        "action_sequence_hash": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode()).hexdigest(),
        "macro_action_count": env.step_count,
        "wall_seconds": time.time() - wall0,
        "fallback_count": planner.fallback_count,
    }


def _child_worker(scene_args, leaf_workers, queue):
    """Non-daemon scene worker: owns its leaf pool (candidate-tail parallel)."""
    leaf_pool = mp.get_context("fork").Pool(leaf_workers)
    try:
        out = []
        for sc, baseline_blood, margin_ml in scene_args:
            out.append(_rollout_optimized(sc, baseline_blood, margin_ml, leaf_pool))
    finally:
        leaf_pool.close()
        leaf_pool.join()
    queue.put(out)


def _run_batch(scene_args, leaf_workers):
    """One non-daemon Process handling one batch of scenes."""
    q = mp.get_context("fork").Queue()
    p = mp.get_context("fork").Process(
        target=_child_worker, args=(scene_args, leaf_workers, q))
    p.start()
    out = q.get()
    p.join()
    return out


def _run_batch_parallel(scene_args, leaf_workers, scene_workers):
    """scene_workers non-daemon Processes in parallel, each with its own leaf pool."""
    scene_args = list(scene_args)
    chunks = [list(c) for c in np.array_split(scene_args, min(scene_workers, len(scene_args)))
              if len(c) > 0]
    procs, queues = [], []
    for chunk in chunks:
        q = mp.get_context("fork").Queue()
        p = mp.get_context("fork").Process(target=_child_worker, args=(chunk, leaf_workers, q))
        p.start()
        procs.append(p)
        queues.append(q)
    results = []
    for q in queues:
        results.extend(q.get())
    for p in procs:
        p.join()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-workers", type=int, default=8)
    parser.add_argument("--leaf-workers", type=int, default=6)
    parser.add_argument("--latency-limit", type=int, default=32,
                        help="scenes to sample for scene_workers=1 latency mode")
    parser.add_argument("--latency-only", action="store_true")
    args = parser.parse_args()

    gate = json.loads(GATE_A_FILE.read_text(encoding="utf-8"))
    scenarios = gate["splits"]["planner_gate"]["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    n = len(scenarios)

    # Reference hashes for equivalence (from frozen reference traces).
    ref = {}
    for line in REF_TRACES.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ref[r["scenario_id"]] = r
    print(f"reference traces loaded: {len(ref)}", flush=True)

    # M_B from frozen Gate R evaluation (identical 128 scenes).
    eval_ = json.loads((BASE / "reference" / "gate_r_evaluation.json").read_text(encoding="utf-8"))
    margin_ml = float(eval_["margin_ml"])
    baseline_B = {r["scenario_id"]: r["baseline_B_ml"] for r in eval_["rows"]}
    print(f"M_B = {margin_ml:.2f} mL", flush=True)

    # ---- Latency (scene_workers=1, leaf pool active, sampled) ----
    lat_scenes = scenarios[: args.latency_limit]
    print(f"latency mode: 1 scene at a time, leaf pool = {args.leaf_workers}, "
          f"sampled scenes = {len(lat_scenes)}", flush=True)
    lat_walls = []
    lat_recs = []
    for sc in lat_scenes:
        rec = _run_batch([(sc, baseline_B[sc["scenario_id"]], margin_ml)], args.leaf_workers)[0]
        lat_recs.append(rec)
        lat_walls.append(rec["wall_seconds"])
    lat_walls = np.asarray(lat_walls)
    eval_rows = {r["scenario_id"]: r for r in eval_["rows"]}
    ref_walls = [eval_rows[rec["scenario_id"]]["wall_seconds"] for rec in lat_recs]
    latency = {
        "mode": f"scene_workers=1, leaf_workers={args.leaf_workers}, sampled={len(lat_recs)}",
        "reference_p50": float(np.quantile(ref_walls, 0.5)),
        "reference_p95": float(np.quantile(ref_walls, 0.95)),
        "optimized_p50": float(np.quantile(lat_walls, 0.5)),
        "optimized_p95": float(np.quantile(lat_walls, 0.95)),
    }
    if latency["optimized_p50"] > 0:
        latency["speedup"] = float(latency["reference_p50"] / latency["optimized_p50"])
    print(f"latency: ref p50={latency['reference_p50']:.1f}s p95={latency['reference_p95']:.1f}s | "
          f"opt p50={latency['optimized_p50']:.1f}s p95={latency['optimized_p95']:.1f}s "
          f"speedup={latency.get('speedup', 0):.1f}x", flush=True)

    if args.latency_only:
        print("latency-only done")
        return

    # ---- Throughput (fixed scene + leaf workers) ----
    t0 = time.time()
    tasks = [(sc, baseline_B[sc["scenario_id"]], margin_ml) for sc in scenarios]
    opt_recs = _run_batch_parallel(tasks, args.leaf_workers, args.scene_workers)
    wall = time.time() - t0
    throughput = {
        "scene_workers": args.scene_workers,
        "leaf_workers": args.leaf_workers,
        "n_scenes": n,
        "wall_seconds": wall,
        "scenes_per_hour": float(n / max(wall, 1e-9) * 3600),
    }
    print(f"throughput: {n} scenes in {wall:.0f}s -> {throughput['scenes_per_hour']:.0f} scenes/h "
          f"({args.scene_workers}x{args.leaf_workers})", flush=True)

    # ---- Equivalence audit vs reference ----
    max_dt = max_db = 0.0
    hash_mismatch = []
    inv_total = 0
    for rec in opt_recs:
        r = ref[rec["scenario_id"]]
        if rec["action_sequence_hash"] != r["action_sequence_hash"]:
            hash_mismatch.append(rec["scenario_id"])
        max_dt = max(max_dt, abs(rec["teacher_T_min"] - r["teacher_T_min"]))
        max_db = max(max_db, abs(rec["teacher_B_ml"] - r["teacher_B_ml"]))
        inv_total += rec["safety_invariant_violations"]
    equivalence = {
        "action_hash_match_128_128": len(hash_mismatch) == 0,
        "hash_mismatch_scenarios": hash_mismatch[:10],
        "hash_mismatch_count": len(hash_mismatch),
        "max_abs_diff_T_min": max_dt,
        "max_abs_diff_B_ml": max_db,
        "invariant_violations_optimized": inv_total,
    }
    print(f"equivalence: hash match {len(opt_recs)-len(hash_mismatch)}/{len(opt_recs)} "
          f"| max |ΔT|={max_dt:.2e} |ΔB|={max_db:.2e} | inv={inv_total}", flush=True)

    bench = {"version": "v10.5-runtime-benchmark-v1", "latency": latency, "throughput": throughput,
             "cpu_note": "see report; fixed topology = planner_gate 128"}
    (BASE / "optimized" / "runtime_benchmark.json").write_text(
        json.dumps(bench, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BASE / "optimized" / "equivalence_audit.json").write_text(
        json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BASE / "optimized" / "optimized_traces.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in opt_recs) + "\n", encoding="utf-8")
    print("wrote runtime_benchmark.json / equivalence_audit.json / optimized_traces.jsonl")


if __name__ == "__main__":
    main()

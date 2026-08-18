"""Collect v10.4 teacher ranking data on Train policy_train (guide 4.2/7.4).

For every decision state of a deterministic depth-1 MPC rollout (the Gate A
strong planner), record:
  - the shared global context vector,
  - for each legal candidate: the explicit candidate features AND the teacher
    full-episode cost (S-scan tail), i.e. the complete candidate ranking signal.

The teacher cost is used ONLY as the supervised ranking/regression target; the
candidate features never contain it (guide 7.1).  The teacher's frozen branch
rule (feasibility -> blood margin -> shortest time -> less blood) defines the
ranking labels.  Continuous feature scales are computed on this Train data.

Parallelism: ``--scene-workers`` child processes, each with its own
``--leaf-workers`` pool for the parallel S-tail evaluations.
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

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import _scan_rank, serpentine_macro_target_policy
from clinical_target_order_features import (
    CANDIDATE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    candidate_features,
    compute_feature_scales,
    global_context,
)
from plan_target_order_v104 import (
    _env_state_payload,
    _step_macro_target,
    candidate_targets,
    make_gate_rollout,
    serpentine_target_of,
)

FROZEN_DIR = SIM / "results/clinical_window_v10_4_target_order/frozen"
TEACHER_DIR = SIM / "results/clinical_window_v10_4_target_order/teacher"
GATE_CFG = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}


def _tail_from_state_worker(payload):
    """S-scan tail from an env-after-target state; returns (dT, dB, completion,
    failure_reason) measured from that state to episode end."""
    scenario, state, cfg = payload
    e = ClinicalMacroResectionEnv(scenario=scenario, clinical_config=cfg)
    e.reset()
    e.__dict__.update(state)
    e.events = []
    t0 = e.elapsed_minutes
    b0 = e.expected_blood_loss_ml
    while not e.terminated and not e.truncated:
        legal = e._frontier()
        if not legal:
            e.terminated = True
            e.failure_reason = "tail lost all legal targets"
            break
        _step_macro_target(e, min(legal, key=lambda cell: _scan_rank(e, cell)))
    return (e.elapsed_minutes - t0, e.expected_blood_loss_ml - b0,
            bool(e.terminated and e.failure_reason is None), e.failure_reason)


def _collect_scene(scene, baseline_blood, margin, cand_count, leaf_pool):
    """One scene's decision-state samples. Returns list of
    ``{"global": ndarray, "candidates": [record, ...]}``."""
    samples = []
    env = ClinicalMacroResectionEnv(scenario=scene, clinical_config=GATE_CFG)
    env.reset()
    while not env.terminated and not env.truncated:
        targets = candidate_targets(env, count=cand_count)
        if not targets:
            break
        cand_records = []
        after_states = []
        for t in targets:
            feat, e_after, dt, db = candidate_features(env, t)
            cand_records.append({
                "target": [int(t[0]), int(t[1])],
                "feature": feat,
                "dt": float(dt),
                "db": float(db),
            })
            after_states.append(_env_state_payload(e_after))
        if leaf_pool is not None:
            payloads = [(scene, st, GATE_CFG) for st in after_states]
            tails = leaf_pool.map(_tail_from_state_worker, payloads)
        else:
            tails = [_tail_from_state_worker((scene, st, GATE_CFG)) for st in after_states]
        for rec, (tdt, tdb, comp, reason) in zip(cand_records, tails):
            rec["cost_T"] = float(rec["dt"] + tdt)
            rec["cost_B"] = float(rec["db"] + tdb)
            rec["completion"] = bool(comp)
        samples.append({
            "global": global_context(env),
            "candidates": cand_records,
            "n_candidates": len(cand_records),
            "safe_threshold": (baseline_blood + margin) if baseline_blood is not None else None,
        })
        # Teacher frozen branch rule -> execute the first target.
        feasible = [r for r in cand_records if r["completion"]]
        if baseline_blood is not None and margin is not None:
            within = [r for r in feasible if r["cost_B"] <= baseline_blood + margin]
            feasible = within if within else feasible
        chosen = min(feasible, key=lambda r: (r["cost_T"], r["cost_B"])) if feasible else None
        target = tuple(chosen["target"]) if chosen else serpentine_target_of(env)
        if target not in env._frontier():
            target = serpentine_target_of(env)
        _step_macro_target(env, target)
    return samples


def _child_worker(scenes, baselines, margin, cand_count, leaf_workers, queue):
    leaf_pool = mp.get_context("fork").Pool(leaf_workers)
    try:
        out = []
        for sc in scenes:
            bb = baselines[sc["scenario_id"]]["expected_blood_loss_ml"]
            out.extend(_collect_scene(sc, bb, margin, cand_count, leaf_pool))
    finally:
        leaf_pool.close()
        leaf_pool.join()
    queue.put(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--scene-workers", type=int, default=8)
    parser.add_argument("--leaf-workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=TEACHER_DIR / "teacher_rankings.npz")
    parser.add_argument("--scales-output", type=Path, default=TEACHER_DIR / "feature_scales.json")
    parser.add_argument("--manifest-output", type=Path, default=TEACHER_DIR / "teacher_manifest.json")
    args = parser.parse_args()

    payload = json.loads((FROZEN_DIR / "splits_v10_4.json").read_text(encoding="utf-8"))
    if not payload.get("frozen"):
        raise RuntimeError("v10.4 splits must be frozen")
    internal = payload["internal_train"]
    train_by_id = {s["scenario_id"]: s for s in payload["splits"]["train"]}
    policy_train = [train_by_id[i] for i in internal["policy_train"]["scenario_ids"]]
    if args.limit:
        policy_train = policy_train[: args.limit]
    print(f"policy_train scenes: {len(policy_train)}", flush=True)

    # Serpentine baseline per scene (for the teacher's blood margin).
    run_serp = make_gate_rollout(serpentine_macro_target_policy, clinical_config=GATE_CFG)
    baselines = {}
    for sc in policy_train:
        baselines[sc["scenario_id"]] = run_serp(sc)
    mean_b = float(np.mean([r["expected_blood_loss_ml"] for r in baselines.values()]))
    margin = 0.05 * mean_b
    print(f"mean baseline B={mean_b:.1f} mL, M_B={margin:.1f} mL", flush=True)

    # Collect across child processes.
    t0 = time.time()
    if args.scene_workers <= 1:
        all_samples = []
        for sc in policy_train:
            all_samples.extend(_collect_scene(
                sc, baselines[sc["scenario_id"]]["expected_blood_loss_ml"], margin,
                args.candidate_count, None))
    else:
        workers = min(args.scene_workers, len(policy_train))
        chunk = (len(policy_train) + workers - 1) // workers
        chunks = [policy_train[i:i + chunk] for i in range(0, len(policy_train), chunk)]
        processes, queues = [], []
        for c in chunks:
            q = mp.get_context("fork").Queue()
            p = mp.get_context("fork").Process(
                target=_child_worker,
                args=(c, baselines, margin, args.candidate_count, args.leaf_workers, q))
            p.start()
            processes.append(p)
            queues.append(q)
        collected = [q.get() for q in queues]
        for p in processes:
            p.join()
        all_samples = [s for c in collected for s in c]
    print(f"collected {len(all_samples)} decision states in {time.time()-t0:.0f}s", flush=True)

    # Flatten into arrays.
    n_states = len(all_samples)
    max_k = max(s["n_candidates"] for s in all_samples) if n_states else 0
    feat = np.zeros((n_states, max_k, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    glob = np.zeros((n_states, GLOBAL_FEATURE_DIM), dtype=np.float32)
    cost_T = np.full((n_states, max_k), np.nan, dtype=np.float32)
    cost_B = np.full((n_states, max_k), np.nan, dtype=np.float32)
    valid = np.zeros((n_states, max_k), dtype=bool)
    comp = np.zeros((n_states, max_k), dtype=bool)
    targets = np.full((n_states, max_k, 2), -1, dtype=np.int32)
    safe_thr = np.full(n_states, np.nan, dtype=np.float32)
    for si, s in enumerate(all_samples):
        glob[si] = s["global"]
        safe_thr[si] = s["safe_threshold"] if s["safe_threshold"] is not None else np.nan
        for ki, c in enumerate(s["candidates"]):
            feat[si, ki] = c["feature"]
            cost_T[si, ki] = c["cost_T"]
            cost_B[si, ki] = c["cost_B"]
            valid[si, ki] = True
            comp[si, ki] = c["completion"]
            targets[si, ki] = c["target"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=feat, global_context=glob, cost_T=cost_T, cost_B=cost_B,
        valid=valid, completion=comp, targets=targets, safe_threshold=safe_thr,
        margin_ml=float(margin), mean_baseline_blood_ml=float(mean_b),
        candidate_count=args.candidate_count,
    )
    print(f"wrote {args.output} ({n_states} states, max_k={max_k})")

    # Feature scales on this Train data (guide 7.1: continuous scales frozen).
    flat = feat[valid]
    scales = compute_feature_scales(list(flat), name="v104_candidate_features")
    args.scales_output.write_text(json.dumps(scales, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print(f"wrote {args.scales_output}")

    manifest = {
        "version": "v10.4-teacher-v1",
        "source_split": "train:policy_train",
        "n_scenes": len(policy_train),
        "n_states": n_states,
        "max_candidates": max_k,
        "candidate_count": args.candidate_count,
        "margin_ml": float(margin),
        "mean_baseline_blood_ml": float(mean_b),
        "scenario_ids": [s["scenario_id"] for s in policy_train],
    }
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"wrote {args.manifest_output}")


if __name__ == "__main__":
    main()

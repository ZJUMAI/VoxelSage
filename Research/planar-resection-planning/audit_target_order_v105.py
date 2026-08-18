"""v10.5 semantic audit + Gate R evaluation (guide 5/8).

Phase A - semantic audit (subsample):
  - candidate_source distribution of candidate_targets_v105
  - v10.4 vs v10.5 candidate-set diff (Jaccard, S-presence, near-hidden entry)
  - confirmation that corrected teacher keeps max B_total <= budget everywhere

Phase B - Gate R on the already-used planner_gate 128 scenes:
  - paired S baseline + corrected teacher per scene
  - per-scene delta-B / delta-T / budget / max B_total / safe counts /
    fallback / invariant violations
  - Gate R conditions from guide 8.3, hard per-scene overage first.

Reads ONLY pilot_gate_a/gate_a_splits_v104.json (already-used dev data, hashed
in frozen_inputs/INPUT_SHA256SUMS). Never parses formal splits_v10_4.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from clinical_window_evaluation import rollout_clinical_policy, serpentine_macro_target_policy
from plan_target_order_v105 import (
    CorrectedPlannerV105,
    _candidate_sources_v105,
    compute_margin_ml,
    rollout_teacher_v105,
    scene_budget,
)

SIM = Path(__file__).resolve().parent
GATE_A_FILE = SIM / "results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"
OUT_DIR = SIM / "results/clinical_window_v10_5_safe_planner"
CFG = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}


def _baseline_worker(sc):
    return rollout_clinical_policy(
        sc, serpentine_macro_target_policy, clinical_config=CFG,
        mechanics_update_interval=0, control_mode="macro",
    )


def _teacher_worker(args):
    sc, baseline_blood, margin_ml = args
    rec = rollout_teacher_v105(
        sc, baseline_blood=baseline_blood, margin_ml=margin_ml,
        budget=scene_budget(baseline_blood, margin_ml),
        clinical_config=CFG, candidate_count=6,
    )
    return rec


def _paired_bootstrap(diffs, samples=10000, seed=20260812):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(samples, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _candidate_source_stats(sc, n_states=400):
    from clinical_macro_environment import ClinicalMacroResectionEnv
    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=CFG)
    env.reset()
    counts = {}
    seen = 0
    while not env.terminated and not env.truncated and seen < n_states:
        for _t, source in _candidate_sources_v105(env, count=6):
            counts[source] = counts.get(source, 0) + 1
        seen += 1
        from plan_target_order_v105 import _step_macro_target
        from plan_target_order_v104 import serpentine_target_of
        _step_macro_target(env, serpentine_target_of(env))
    return counts


def _candidate_diff(sc, n_states=120):
    from clinical_macro_environment import ClinicalMacroResectionEnv
    from plan_target_order_v104 import candidate_targets
    from plan_target_order_v105 import _step_macro_target
    from plan_target_order_v104 import serpentine_target_of
    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=CFG)
    env.reset()
    total = {"new_has_s": 0, "old_has_s": 0, "states": 0}
    jacs = []
    near_entries = 0
    while not env.terminated and not env.truncated and total["states"] < n_states:
        old = candidate_targets(env, count=6)
        new = [t for t, _s in _candidate_sources_v105(env, count=6)]
        if old and new:
            inter = len(set(old) & set(new))
            jacs.append(inter / len(set(old) | set(new)))
        from plan_target_order_v104 import serpentine_target_of as _s
        total["new_has_s"] += int(_s(env) in new)
        total["old_has_s"] += int(_s(env) in old)
        total["states"] += 1
        _step_macro_target(env, _s(env))
    return {"total_states": total["states"], "new_has_s": total["new_has_s"],
            "old_has_s": total["old_has_s"],
            "mean_jaccard": float(np.mean(jacs)) if jacs else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-workers", type=int, default=8)
    parser.add_argument("--skip-teacher", action="store_true", help="baseline+audit only")
    args = parser.parse_args()

    gate = json.loads(GATE_A_FILE.read_text(encoding="utf-8"))
    scenarios = gate["splits"]["planner_gate"]["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    n = len(scenarios)
    print(f"planner_gate scenes: {n} (seed={gate.get('seed')})", flush=True)

    # ---- Phase A: semantic audit on a small subsample ----
    audit_scenes = scenarios[: min(6, n)]
    source_counts: dict[str, int] = {}
    for sc in audit_scenes:
        for k, v in _candidate_source_stats(sc).items():
            source_counts[k] = source_counts.get(k, 0) + v
    diff = _candidate_diff(audit_scenes[0]) if audit_scenes else None
    semantic_audit = {
        "candidate_source_counts": source_counts,
        "candidate_diff_v104_vs_v105": diff,
        "note": "subsample of planner_gate; full per-scene safety in gate_r_evaluation.json",
    }
    (OUT_DIR / "audit" / "semantic_audit.json").write_text(
        json.dumps(semantic_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"semantic audit: sources={source_counts}", flush=True)
    print(f"candidate diff (v104 vs v105): {diff}", flush=True)

    # ---- Phase B: paired baseline + corrected teacher ----
    t0 = time.time()
    with mp.get_context("fork").Pool(args.scene_workers) as pool:
        base_recs = pool.map(_baseline_worker, scenarios)
    print(f"baseline done ({time.time()-t0:.0f}s): mean_B="
          f"{np.mean([r['expected_blood_loss_ml'] for r in base_recs]):.1f}", flush=True)

    margin_ml = compute_margin_ml([r["expected_blood_loss_ml"] for r in base_recs])
    print(f"M_B = 0.05 x mean_B = {margin_ml:.2f} mL", flush=True)

    if args.skip_teacher:
        print("skipped teacher (--skip-teacher)")
        return

    t0 = time.time()
    tasks = [(sc, base["expected_blood_loss_ml"], margin_ml)
             for sc, base in zip(scenarios, base_recs)]
    with mp.get_context("fork").Pool(args.scene_workers) as pool:
        teach_recs = pool.map(_teacher_worker, tasks)
    print(f"teacher done ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for sc, base, teach in zip(scenarios, base_recs, teach_recs):
        b_base = float(base["expected_blood_loss_ml"])
        b_teach = float(teach["teacher_B_ml"])
        budget = scene_budget(b_base, margin_ml)
        rows.append({
            "scenario_id": sc["scenario_id"],
            "baseline_T_min": float(base["elapsed_minutes"]),
            "baseline_B_ml": b_base,
            "teacher_T_min": teach["teacher_T_min"],
            "teacher_B_ml": b_teach,
            "delta_T_min": teach["teacher_T_min"] - float(base["elapsed_minutes"]),
            "delta_B_ml": b_teach - b_base,
            "budget_ml": float(budget),
            "max_B_total_ml": float(teach["max_B_total_ml"]),
            "safe_candidate_count_min": int(teach["safe_candidate_count_min"]),
            "safe_candidate_count_median": float(teach["safe_candidate_count_median"]),
            "fallback_count": int(teach["fallback_count"]),
            "safety_invariant_violations": int(teach["safety_invariant_violations"]),
            "completion": bool(teach["completion"]),
            "legal_action_rate": float(teach["legal_action_rate"]),
            "failure_reason": teach["failure_reason"],
            "macro_action_count": int(teach["macro_action_count"]),
            "clamp_cycle_count": int(teach["clamp_cycle_count"]),
            "action_sequence_hash": teach["action_sequence_hash"],
            "wall_seconds": float(teach["wall_seconds"]),
            "tail_cache_hits": int(teach["tail_cache_hits"]),
            "tail_cache_misses": int(teach["tail_cache_misses"]),
        })

    # ---- Gate R conditions (guide 8.3) ----
    dT = np.asarray([r["delta_T_min"] for r in rows])
    dB = np.asarray([r["delta_B_ml"] for r in rows])
    inv = sum(1 for r in rows if r["safety_invariant_violations"] > 0)
    fail = sum(1 for r in rows if not r["completion"] or r["failure_reason"])
    legal_bad = sum(1 for r in rows if r["legal_action_rate"] < 1.0 - 1e-9)
    over = [r for r in rows if r["delta_B_ml"] > margin_ml + 1e-9]
    max_dB = float(np.max(dB)) if len(dB) else 0.0
    db_ci = _paired_bootstrap(dB.tolist())
    dt_ci = _paired_bootstrap(dT.tolist())
    mean_Ts = float(np.mean([r["baseline_T_min"] for r in rows]))
    mean_dT = float(np.mean(dT))
    time_effect_ok = mean_dT <= -0.005 * mean_Ts

    conditions = {
        "completion_100": fail == 0,
        "legal_rate_1_0": legal_bad == 0,
        "no_failure_or_truncation": fail == 0,
        "no_safety_invariant_violation": inv == 0,
        "per_scene_dB_le_margin_128_128": len(over) == 0,
        "max_dB_le_margin": max_dB <= margin_ml + 1e-9,
        "db_ci_upper_le_margin": db_ci[1] <= margin_ml,
        "dt_ci_upper_lt_0": dt_ci[1] < 0.0,
        "mean_time_effect": time_effect_ok,
    }
    decision = "GO" if all(conditions.values()) else "NO-GO"

    evaluation = {
        "version": "v10.5-gate-r-v1",
        "split": "planner_gate (already-used Gate A dev data)",
        "n_scenarios": n,
        "margin_ml": float(margin_ml),
        "M_B": float(margin_ml),
        "conditions": conditions,
        "decision": decision,
        "summary": {
            "completion": f"{fail} failure / {n}",
            "legal_bad": legal_bad,
            "invariant_violations": inv,
            "max_delta_B_ml": max_dB,
            "over_margin_count": len(over),
            "delta_B_mean": float(np.mean(dB)),
            "delta_B_ci": list(db_ci),
            "delta_T_mean": mean_dT,
            "delta_T_ci": list(dt_ci),
            "mean_baseline_T_min": mean_Ts,
            "mean_teacher_T_min": float(np.mean([r["teacher_T_min"] for r in rows])),
            "mean_teacher_B_ml": float(np.mean([r["teacher_B_ml"] for r in rows])),
        },
        "rows": rows,
    }
    (OUT_DIR / "reference" / "gate_r_evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nGate R: {decision}")
    for k, v in conditions.items():
        print(f"  {k:34s} {v}")
    print(f"  max_delta_B = {max_dB:.2f} vs M_B={margin_ml:.2f} ({len(over)} over)")
    print(f"  delta_B 95% CI [{db_ci[0]:.2f}, {db_ci[1]:.2f}]")
    print(f"  delta_T 95% CI [{dt_ci[0]:.2f}, {dt_ci[1]:.2f}], mean {mean_dT:.2f}")


if __name__ == "__main__":
    main()

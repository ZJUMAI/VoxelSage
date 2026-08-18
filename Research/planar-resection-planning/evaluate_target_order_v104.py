"""Gate B admission evaluation (guide Section 7.5).

On the frozen ``policy_internal_dev`` scenes (64, Train-internal), run:
  - serpentine baseline (same shared transfer/blood code),
  - teacher = Gate A depth-1 MPC strong planner (its full-episode S-tail eval),
  - model  = BC global candidate scorer (deterministic argmax).

Then compute paired bootstrap CI on Delta T / Delta B and the teacher-benefit
retention ratio

    R_T = (T_base - T_model) / (T_base - T_teacher)

Gate B GO requires: completion/legal 100%, END/failure 0,
CI_upper(Delta B) <= M_B (0.05 * mean baseline blood), CI_upper(Delta T) < 0,
and R_T >= 0.50.  A denominator below 0.5% of baseline time means Gate A's
effect size is too small to enter PPO with an unstable ratio.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_target_order_features import global_context  # noqa: E402
from clinical_target_order_policy import TargetOrderScorer, make_selector  # noqa: E402
from clinical_window_evaluation import (  # noqa: E402
    rollout_clinical_policy,
    serpentine_macro_target_policy,
)
from plan_target_order_v104 import (  # noqa: E402
    SerpentineTail,
    WindowAwarePlanner,
    make_gate_rollout,
    rollout_planner,
)

FROZEN_DIR = SIM / "results/clinical_window_v10_4_target_order/frozen"
RUNS_DIR = SIM / "results/clinical_window_v10_4_target_order/runs"
GATE_CFG = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}


def _bootstrap_ci(diffs, samples=10_000, seed=20260812):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(samples, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _run_teacher_worker(args):
    scen, baseline_blood, margin, cfg, params = args
    tail = SerpentineTail(clinical_config=cfg)
    planner = WindowAwarePlanner(
        candidate_count=params["candidate_count"], beam_width=1, lookahead_depth=1,
        margin_blood_ml=margin, tail=tail, clinical_config=cfg)
    t0 = time.time()
    rec = rollout_planner(scen, planner, baseline_blood=baseline_blood,
                          replan_interval=1, clinical_config=cfg)
    rec["wall_seconds"] = time.time() - t0
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "target_order_bc.pt")
    parser.add_argument("--scales", type=Path,
                        default=RUNS_DIR.parent / "teacher/feature_scales.json")
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--leaf-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=RUNS_DIR / "gate_b_evaluation.json")
    args = parser.parse_args()

    payload = json.loads((FROZEN_DIR / "splits_v10_4.json").read_text(encoding="utf-8"))
    internal = payload["internal_train"]
    train_by_id = {s["scenario_id"]: s for s in payload["splits"]["train"]}
    scenes = [train_by_id[i] for i in internal["policy_internal_dev"]["scenario_ids"]]
    if args.limit:
        scenes = scenes[: args.limit]
    print(f"policy_internal_dev scenes: {len(scenes)}", flush=True)

    # Baseline.
    run_serp = make_gate_rollout(serpentine_macro_target_policy, clinical_config=GATE_CFG)
    t0 = time.time()
    baseline_recs = {}
    for sc in scenes:
        wall = time.time()
        rec = run_serp(sc)
        rec["wall_seconds"] = time.time() - wall
        baseline_recs[sc["scenario_id"]] = rec
    mean_b = float(np.mean([r["expected_blood_loss_ml"] for r in baseline_recs.values()]))
    margin = 0.05 * mean_b
    print(f"baseline done ({time.time()-t0:.0f}s): mean_B={mean_b:.1f} mL, M_B={margin:.1f} mL",
          flush=True)

    # Teacher (depth-1 MPC) with a shared leaf pool.
    leaf_pool = mp.get_context("fork").Pool(args.leaf_workers)
    teacher_args = []
    try:
        for sc in scenes:
            bb = baseline_recs[sc["scenario_id"]]["expected_blood_loss_ml"]
            teacher_args.append((sc, bb, margin, GATE_CFG,
                                 {"candidate_count": args.candidate_count}))
        t0 = time.time()
        teacher_recs = leaf_pool.map(_run_teacher_worker, teacher_args)
        print(f"teacher done ({time.time()-t0:.0f}s)", flush=True)
    finally:
        leaf_pool.close()
        leaf_pool.join()

    # Model (BC scorer) rollout.
    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    model = TargetOrderScorer()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    selector = make_selector(model, scales, candidate_count=args.candidate_count)
    t0 = time.time()
    model_recs = []
    for sc in scenes:
        wall = time.time()
        rec = rollout_clinical_policy(sc, selector, clinical_config=GATE_CFG,
                                      mechanics_update_interval=0, control_mode="macro")
        rec["wall_seconds"] = time.time() - wall
        model_recs.append(rec)
    print(f"model done ({time.time()-t0:.0f}s)", flush=True)

    def deltas(recs, field):
        return np.asarray([float(r[field]) - float(baseline_recs[r["scenario_id"]][field])
                           for r in recs])

    dT_t, dT_m = deltas(teacher_recs, "elapsed_minutes"), deltas(model_recs, "elapsed_minutes")
    dB_t, dB_m = deltas(teacher_recs, "expected_blood_loss_ml"), deltas(model_recs, "expected_blood_loss_ml")
    ciT_t, ciB_t = _bootstrap_ci(dT_t), _bootstrap_ci(dB_t)
    ciT_m, ciB_m = _bootstrap_ci(dT_m), _bootstrap_ci(dB_m)

    mean_T_base = float(np.mean([r["elapsed_minutes"] for r in baseline_recs.values()]))
    mean_T_teacher = float(np.mean([r["elapsed_minutes"] for r in teacher_recs]))
    mean_T_model = float(np.mean([r["elapsed_minutes"] for r in model_recs]))
    denom = mean_T_base - mean_T_teacher
    r_t = (mean_T_base - mean_T_model) / denom if abs(denom) > 1e-9 else float("nan")
    denom_ratio = abs(denom) / max(mean_T_base, 1e-9)

    def summarize(name, recs, ciT, ciB, dT, dB):
        return {
            "name": name, "n": len(recs),
            "completion_rate": float(np.mean([r["completion"] for r in recs])),
            "legal_action_rate": float(np.mean([r["legal_action_rate"] for r in recs])),
            "end_count": int(sum(r["early_end_count"] for r in recs)),
            "failure_count": int(sum(r["status"] != "ok" for r in recs)),
            "mean_T": float(np.mean([r["elapsed_minutes"] for r in recs])),
            "mean_B": float(np.mean([r["expected_blood_loss_ml"] for r in recs])),
            "mean_dT": float(dT.mean()),
            "dT_95_ci": [ciT[0], ciT[1]],
            "mean_dB": float(dB.mean()),
            "dB_95_ci": [ciB[0], ciB[1]],
            "median_wall_seconds": float(np.median([r["wall_seconds"] for r in recs])),
        }

    summary = {
        "version": "v10.4-gate-b-evaluation-v1",
        "split": "train:policy_internal_dev",
        "n_scenarios": len(scenes),
        "margin_ml": margin,
        "baseline": summarize("serpentine", list(baseline_recs.values()), [0, 0], [0, 0],
                              np.zeros(len(scenes)), np.zeros(len(scenes))),
        "teacher": summarize("depth1_mpc", teacher_recs, ciT_t, ciB_t, dT_t, dB_t),
        "model": summarize("bc_scorer", model_recs, ciT_m, ciB_m, dT_m, dB_m),
        "r_t": r_t,
        "denominator_fraction_of_baseline": denom_ratio,
    }

    p = summary["model"]
    conditions = {
        "completion_100": p["completion_rate"] == 1.0 and summary["baseline"]["completion_rate"] == 1.0,
        "legal_1_0": p["legal_action_rate"] == 1.0 and summary["baseline"]["legal_action_rate"] == 1.0,
        "no_end_or_failure": p["end_count"] == 0 and p["failure_count"] == 0,
        "blood_ci_upper_le_margin": p["dB_95_ci"][1] <= margin,
        "time_ci_upper_lt_0": p["dT_95_ci"][1] < 0.0,
        "teacher_benefit_retention_ge_050": r_t >= 0.50,
    }
    summary["go_no_go"] = {
        "conditions": conditions,
        "decision": "GO" if all(conditions.values()) else "NO-GO",
    }
    if denom_ratio < 0.005:
        summary["go_no_go"]["decision"] = "NO-GO"
        summary["go_no_go"]["reason"] = ("teacher effect size below 0.5% of baseline; "
                                         "unstable ratio, do not enter PPO (guide 7.5)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

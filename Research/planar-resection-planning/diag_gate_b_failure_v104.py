"""Per-scene failure diagnosis for Gate B NO-GO (informational).

Baseline vs BC-model per-scene delta-blood on frozen policy_internal_dev.
Shows whether the model's blood overage is systemic (most scenes) or a few
outliers, and characterises the worst scenes by vessel geometry.

Reads frozen splits + BC checkpoint only; no selection, no tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_target_order_policy import TargetOrderScorer, make_selector  # noqa: E402
from clinical_window_evaluation import (  # noqa: E402
    rollout_clinical_policy,
    serpentine_macro_target_policy,
)
from plan_target_order_v104 import make_gate_rollout  # noqa: E402
from prepare_clinical_v104_splits import _vessel_component_count  # noqa: E402

FROZEN_DIR = SIM / "results/clinical_window_v10_4_target_order/frozen"
RUNS_DIR = SIM / "results/clinical_window_v10_4_target_order/runs"
TEACHER_DIR = SIM / "results/clinical_window_v10_4_target_order/teacher"
GATE_CFG = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "target_order_bc.pt")
    parser.add_argument("--scales", type=Path, default=TEACHER_DIR / "feature_scales.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "gate_b_failure_diag.json")
    args = parser.parse_args()

    payload = json.loads((FROZEN_DIR / "splits_v10_4.json").read_text(encoding="utf-8"))
    internal = payload["internal_train"]
    train_by_id = {s["scenario_id"]: s for s in payload["splits"]["train"]}
    scenes = [train_by_id[i] for i in internal["policy_internal_dev"]["scenario_ids"]]
    if args.limit:
        scenes = scenes[: args.limit]

    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    model = TargetOrderScorer()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    selector = make_selector(model, scales, candidate_count=6)

    run_serp = make_gate_rollout(serpentine_macro_target_policy, clinical_config=GATE_CFG)

    rows = []
    for sc in scenes:
        s = run_serp(sc)
        m = rollout_clinical_policy(sc, selector, clinical_config=GATE_CFG,
                                    mechanics_update_interval=0, control_mode="macro")
        rows.append({
            "scenario_id": sc["scenario_id"],
            "vessel_cells": len(sc["obstacle_cells"]),
            "vessel_components": _vessel_component_count(sc["obstacle_cells"]),
            "serp_T": float(s["elapsed_minutes"]),
            "serp_B": float(s["expected_blood_loss_ml"]),
            "model_T": float(m["elapsed_minutes"]),
            "model_B": float(m["expected_blood_loss_ml"]),
            "model_completion": bool(m["completion"]),
        })
    for r in rows:
        r["dB"] = r["model_B"] - r["serp_B"]
        r["dT"] = r["model_T"] - r["serp_T"]

    n = len(rows)
    db = [r["dB"] for r in rows]
    worst = sorted(rows, key=lambda r: -r["dB"])
    margin = 0.05 * float(np.mean([r["serp_B"] for r in rows]))

    over_margin = sum(1 for r in rows if r["dB"] > margin)
    bad = sum(1 for r in rows if r["model_B"] > r["serp_B"] * 1.5)
    summary = {
        "n_scenarios": n,
        "margin_ml": margin,
        "dB_mean": float(np.mean(db)),
        "dB_median": float(np.median(db)),
        "dB_p90": float(np.quantile(db, 0.90)),
        "dB_max": float(np.max(db)),
        "n_over_margin": over_margin,
        "n_bad_gt_1p5x_serp_blood": bad,
        "n_completion_fail": sum(1 for r in rows if not r["model_completion"]),
        "worst_10": worst[:10],
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"dB: mean={summary['dB_mean']:.1f} median={summary['dB_median']:.1f} "
          f"p90={summary['dB_p90']:.1f} max={summary['dB_max']:.1f} (M_B={margin:.1f})")
    print(f"over margin: {over_margin}/{n}, >1.5x serp blood: {bad}/{n}, "
          f"completion fail: {summary['n_completion_fail']}/{n}")


if __name__ == "__main__":
    main()

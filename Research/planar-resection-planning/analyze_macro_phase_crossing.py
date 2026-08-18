#!/usr/bin/env python
"""统计每个宏动作是否跨越 clamp/unclamp 时相边界。

对给定模型（ppo 或 serpentine）在 eval 场景上 replay，记录每个宏动作
执行前/后的 phase，统计跨边界宏动作数。输出每 episode 汇总 + 全局汇总。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import (
    aggregate_clinical_records,
    make_ppo_selector,
    serpentine_macro_target_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--algorithm", choices=("serpentine", "ppo"), default="serpentine")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--early-end-mode", default="disabled")
    parser.add_argument("--early-end-minutes", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scenarios = list(split_payload["splits"][args.split])[: args.limit]
    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    clinical_config = {
        "max_steps_multiplier": 8.0,
        "bleeding_probability": 1.0,
        "time_scale_minutes": float(scales["time_scale_minutes"]),
        "blood_scale_ml": float(scales["blood_scale_ml"]),
        "weight_kg": float(scales.get("weight_kg", 70.0)),
        "early_end_mode": args.early_end_mode,
        "early_end_minutes": args.early_end_minutes,
        "stagnation_soft_start_steps": 40,
        "stagnation_penalty_ramp_steps": 24,
        "stagnation_limit_steps": 96,
        "two_cell_loop_soft_start_traversals": 6,
        "two_cell_loop_limit_traversals": 12,
    }
    reward_config = {
        "progress_bonus": 5.0,
        "seal_progress_bonus": 2.0,
        "stagnation_penalty_cap": 0.05,
        "two_cell_loop_penalty": 0.25,
        "clinical_cost_cap": 10.0,
    }

    if args.algorithm == "serpentine":
        selector = serpentine_macro_target_policy
    else:
        import torch

        torch.set_num_threads(args.torch_threads)
        from sb3_contrib import MaskablePPO

        import clinical_macro_policy  # noqa: F401

        if args.model is None:
            parser.error("--model required for --algorithm ppo")
        model = MaskablePPO.load(str(args.model), device="auto")
        selector = make_ppo_selector(model)

    records = []
    for scenario in scenarios:
        env = ClinicalMacroResectionEnv(
            scenario=scenario,
            clinical_config=clinical_config,
            reward_config=reward_config,
            mechanics_update_interval=0,
        )
        env.reset()
        macro_count = 0
        crossing_count = 0
        while not env.terminated and not env.truncated:
            action = int(selector(env))
            phase_before = env.phase
            _, _, _, _, info = env.step(action)
            phase_after = env.phase
            macro_count += 1
            if phase_before != phase_after:
                crossing_count += 1
        records.append({
            "scenario_id": scenario.get("scenario_id"),
            "macro_action_count": macro_count,
            "crossing_phase_boundary_count": crossing_count,
            "crossing_ratio": crossing_count / max(1, macro_count),
            "elapsed_minutes": env.elapsed_minutes,
            "expected_blood_loss_ml": env.expected_blood_loss_ml,
            "early_end_count": env.early_end_count,
            "phase_changes": sum(
                1 for e in env.events if e.get("action") == "phase_change"
            ),
            "total_clamped_minutes": env.total_clamped_minutes,
            "total_unclamped_minutes": env.total_unclamped_minutes,
        })

    n = len(records)
    summary = {
        "algorithm": args.algorithm,
        "control_mode": "macro",
        "model": str(args.model) if args.model else "serpentine",
        "episode_count": n,
        "mean_macro_action_count": sum(r["macro_action_count"] for r in records) / n,
        "mean_crossing_phase_boundary_count": sum(r["crossing_phase_boundary_count"] for r in records) / n,
        "mean_crossing_ratio": sum(r["crossing_ratio"] for r in records) / n,
        "total_crossing_phase_boundary_count": sum(r["crossing_phase_boundary_count"] for r in records),
        "max_crossing_phase_boundary_count": max(r["crossing_phase_boundary_count"] for r in records),
        "mean_phase_changes_per_episode": sum(r["phase_changes"] for r in records) / n,
        "mean_elapsed_minutes": sum(r["elapsed_minutes"] for r in records) / n,
        "mean_expected_blood_loss_ml": sum(r["expected_blood_loss_ml"] for r in records) / n,
        "mean_early_end_count": sum(r["early_end_count"] for r in records) / n,
    }
    payload = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

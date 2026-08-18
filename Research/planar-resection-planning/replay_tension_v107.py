"""Replay frozen v10.7 controller trajectories with mechanics_update_interval=1.

For Replication C0/C2/C4, replay the frozen action sequence in a mechanics-
enabled environment.  The actions, completion, T and B must match the frozen
main trajectory per scene.  Reports uncalibrated tension/energy/strain
descriptives.  These never enter the main Gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from multiprocessing import get_context
from pathlib import Path
from statistics import mean

import numpy as np

from clinical_macro_environment import ClinicalMacroResectionEnv
from mechanics import DEFAULT_MECHANICS
from plan_target_order_v104 import _step_macro_target
from plan_target_order_v105 import DEFAULT_GATE_CLINICAL_CONFIG

SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_7_confirmation"
FROZEN = BASE / "frozen"
SAFE_STRAIN = float(DEFAULT_MECHANICS["safe_vessel_strain"])
TEAR_STRAIN = float(DEFAULT_MECHANICS["tear_vessel_strain"])


def worst_tenth(values: list[float]) -> float:
    count = max(1, ceil(0.10 * len(values)))
    return float(mean(sorted(values, reverse=True)[:count])) if values else 0.0


def replay_one(task: tuple[dict, dict]) -> dict:
    scene, frozen = task
    actions = [tuple(map(int, action)) for action in frozen["actions"]]
    action_hash = hashlib.sha256(
        json.dumps(actions, separators=(",", ":")).encode()
    ).hexdigest()
    if action_hash != frozen["action_sequence_hash"]:
        raise RuntimeError(f"stored action hash mismatch: {scene['scenario_id']}")
    env = ClinicalMacroResectionEnv(
        scenario=scene, clinical_config=DEFAULT_GATE_CLINICAL_CONFIG, mechanics_update_interval=1
    )
    env.reset()
    front: list[float] = []
    organ: list[float] = []
    strain: list[float] = []
    for row, col in actions:
        _step_macro_target(env, (row, col))
        env._update_mechanics(force=True)
        front.append(float(env.mechanics["peak_front_tension"]))
        organ.append(float(env.mechanics["peak_organ_energy"]))
        strain.append(float(env.mechanics["peak_vessel_strain"]))
    checks = {
        "completion_same": bool(env.terminated and env.failure_reason is None) == bool(frozen["completion"]),
        "failure_same": env.failure_reason == frozen["failure_reason"],
        "action_count_same": len(actions) == int(frozen["macro_action_count"]),
        "time_same": abs(float(env.elapsed_minutes) - float(frozen["elapsed_minutes"])) <= 1e-9,
        "blood_same": abs(float(env.expected_blood_loss_ml) - float(frozen["realized_episode_B_ml"])) <= 1e-9,
    }
    if not all(checks.values()):
        raise RuntimeError(f"mechanics replay changed frozen trajectory: {scene['scenario_id']} {checks}")
    return {
        "scenario_id": scene["scenario_id"], "action_sequence_hash": action_hash,
        "macro_action_count": len(actions), "elapsed_minutes": float(env.elapsed_minutes),
        "realized_episode_B_ml": float(env.expected_blood_loss_ml), "checks": checks,
        "mean_front_tension": float(mean(front)) if front else 0.0,
        "p95_front_tension": float(np.quantile(front, 0.95)) if front else 0.0,
        "max_front_tension": max(front, default=0.0),
        "mean_organ_energy": float(mean(organ)) if organ else 0.0,
        "p95_organ_energy": float(np.quantile(organ, 0.95)) if organ else 0.0,
        "max_organ_energy": max(organ, default=0.0),
        "mean_vessel_strain": float(mean(strain)) if strain else 0.0,
        "cumulative_vessel_strain": float(sum(strain)),
        "worst_10pct_vessel_strain": worst_tenth(strain),
        "max_vessel_strain": max(strain, default=0.0),
        "fraction_steps_above_safe": float(mean(v > SAFE_STRAIN for v in strain)) if strain else 0.0,
        "fraction_steps_above_tear": float(mean(v > TEAR_STRAIN for v in strain)) if strain else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", default="C0,C2,C4")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    controllers = [c for c in args.controllers.split(",") if c in ("C0", "C2", "C4")]
    split_payload = json.loads((FROZEN / "split_replication.json").read_text(encoding="utf-8"))
    scenes = {sc["scenario_id"]: sc for sc in split_payload["scenarios"]}
    rows_by_controller = {}
    for controller in controllers:
        shard_dir = BASE / "shards" / "replication" / controller
        frozen = {}
        for shard in shard_dir.glob("*.json"):
            row = json.loads(shard.read_text(encoding="utf-8"))
            frozen[row["scenario_id"]] = row
        if set(scenes) != set(frozen):
            raise RuntimeError(f"{controller}: split and shard scenario IDs differ")
        tasks = [(scenes[sid], frozen[sid]) for sid in sorted(scenes)]
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(tasks)), mp_context=get_context("fork")
        ) as executor:
            rows = list(executor.map(replay_one, tasks))
        rows_by_controller[controller] = rows

    fields = (
        "mean_front_tension", "p95_front_tension", "max_front_tension",
        "mean_organ_energy", "p95_organ_energy", "max_organ_energy",
        "mean_vessel_strain", "cumulative_vessel_strain",
        "worst_10pct_vessel_strain", "max_vessel_strain",
        "fraction_steps_above_safe", "fraction_steps_above_tear",
    )
    result = {
        "version": "v10.7-frozen-trajectory-mechanics-replay-v1",
        "split": "replication", "mechanics_update_interval": 1,
        "trajectory_equivalence": "PASS",
        "controllers": {},
    }
    for controller, rows in rows_by_controller.items():
        result["controllers"][controller] = {
            "n_scenarios": len(rows),
            "summary": {field: float(mean(row[field] for row in rows)) for field in fields},
        }
    out = BASE / "evaluation" / "tension_replay.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "decision": "PASS",
                      "controllers": {c: rows_by_controller[c][0]["macro_action_count"] for c in controllers}}))


if __name__ == "__main__":
    main()

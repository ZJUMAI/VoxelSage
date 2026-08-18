"""Verify _step_macro_target matches env.step exactly on physical state.

Drives two identically-reset envs through the SAME sequence of target cells
(serpentine order), one via env.step(action) and one via _step_macro_target.
After each macro action it compares elapsed time, blood loss, cut set, current,
phase, transfer counters and completion. Run: python check_step_consistency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import _scan_rank
from plan_target_order_v104 import _step_macro_target


def run(scenario, *, use_fast: bool) -> dict:
    cfg = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}
    env = ClinicalMacroResectionEnv(scenario=scenario, clinical_config=cfg)
    env.reset()
    targets = []
    while not env.terminated and not env.truncated:
        legal = sorted(env._frontier())
        if not legal:
            break
        t = min(legal, key=lambda c: _scan_rank(env, c))
        targets.append(t)
        if use_fast:
            _step_macro_target(env, t)
        else:
            env.step(t[0] * env.max_cols + t[1])
    return {
        "elapsed": env.elapsed_minutes,
        "blood": env.expected_blood_loss_ml,
        "cut": frozenset(env.cut),
        "current": env.current,
        "phase": env.phase,
        "phase_elapsed": env.phase_elapsed_minutes,
        "transfer": env.transfer_count,
        "direction": env.direction_action_count,
        "terminated": env.terminated,
        "truncated": env.truncated,
        "failure": env.failure_reason,
        "clamp_cycles": env.clamp_cycle_count,
        "exposed": frozenset(env.exposed_ids),
        "sealed": frozenset(env.sealed_ids),
        "n_targets": len(targets),
    }


def main() -> None:
    d = json.load(open("results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"))
    scenarios = d["splits"]["planner_gate"]["scenarios"][:6]
    mismatches = 0
    for sc in scenarios:
        slow = run(sc, use_fast=False)
        fast = run(sc, use_fast=True)
        same = (
            abs(slow["elapsed"] - fast["elapsed"]) < 1e-9
            and abs(slow["blood"] - fast["blood"]) < 1e-9
            and slow["cut"] == fast["cut"]
            and slow["current"] == fast["current"]
            and slow["phase"] == fast["phase"]
            and abs(slow["phase_elapsed"] - fast["phase_elapsed"]) < 1e-9
            and slow["transfer"] == fast["transfer"]
            and slow["direction"] == fast["direction"]
            and slow["terminated"] == fast["terminated"]
            and slow["truncated"] == fast["truncated"]
            and slow["failure"] == fast["failure"]
            and slow["clamp_cycles"] == fast["clamp_cycles"]
            and slow["exposed"] == fast["exposed"]
            and slow["sealed"] == fast["sealed"]
            and slow["n_targets"] == fast["n_targets"]
        )
        status = "OK " if same else "MISMATCH"
        if not same:
            mismatches += 1
            print(f"{status} {sc['scenario_id']}")
            for k in ("elapsed", "blood", "current", "phase", "phase_elapsed", "transfer",
                      "direction", "terminated", "truncated", "failure", "clamp_cycles",
                      "n_targets"):
                if slow[k] != fast[k]:
                    print(f"    {k}: slow={slow[k]!r} fast={fast[k]!r}")
            print(f"    cut slow={len(slow['cut'])} fast={len(fast['cut'])}")
            print(f"    exposed slow={slow['exposed']} fast={fast['exposed']}")
            print(f"    sealed slow={slow['sealed']} fast={fast['sealed']}")
        else:
            print(f"{status} {sc['scenario_id']}: T={slow['elapsed']:.2f} B={slow['blood']:.1f} "
                  f"n={slow['n_targets']} transfer={slow['transfer']}")
    print(f"\n{6 - mismatches}/6 scenarios consistent")


if __name__ == "__main__":
    main()

"""Diagnose receding-horizon length: same scene, different replan_interval.

Compares window-aware planner T/B across replan_interval H in {2,4,8} (same
beam/depth/candidates) against serpentine baseline and seal-first heuristic.
Small beam so the test completes quickly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import _scan_rank, serpentine_macro_target_policy
from plan_target_order_v104 import SerpentineTail, WindowAwarePlanner, _step_macro_target, rollout_planner

SCENE_INDEX = 0
CAND, BEAM, DEPTH = 4, 2, 8


def seal_first(env, legal):
    exposed = env._exposed_cells()
    sealable = [c for c in legal if c in exposed]
    counts = env._transfer_counts()
    if sealable:
        return min(sealable, key=lambda c: (counts.get(c, 1e9), _scan_rank(env, c)))
    return min(legal, key=lambda c: (counts.get(c, 1e9), _scan_rank(env, c)))


def rollout(env, selector):
    while not env.terminated and not env.truncated:
        legal = sorted(env._frontier())
        if not legal:
            break
        _step_macro_target(env, selector(env, legal))
    return env


def main() -> None:
    d = json.load(open("results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"))
    sc = d["splits"]["planner_gate"]["scenarios"][SCENE_INDEX]
    cfg = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}

    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=cfg)
    env.reset()
    rs = rollout(env, lambda e, legal: min(legal, key=lambda c: _scan_rank(e, c)))
    rn = rollout(ClinicalMacroResectionEnv(scenario=sc, clinical_config=cfg), seal_first)
    print(f"scene {sc['scenario_id']}")
    print(f"  serpentine T={rs.elapsed_minutes:.2f} B={rs.expected_blood_loss_ml:.1f}")
    print(f"  seal_first T={rn.elapsed_minutes:.2f} B={rn.expected_blood_loss_ml:.1f} "
          f"dT={rn.elapsed_minutes-rs.elapsed_minutes:+.2f}")
    margin = 0.05 * rs.expected_blood_loss_ml
    print(f"  margin M_B={margin:.2f} mL, threshold={rs.expected_blood_loss_ml+margin:.2f}")
    for H in (2, 4, 8):
        tail = SerpentineTail(clinical_config=cfg)
        planner = WindowAwarePlanner(candidate_count=CAND, beam_width=BEAM,
                                     lookahead_depth=DEPTH, margin_blood_ml=margin, tail=tail)
        t0 = time.time()
        rp = rollout_planner(sc, planner, baseline_blood=rs.expected_blood_loss_ml,
                             replan_interval=H, clinical_config=cfg)
        dt = time.time() - t0
        print(f"  planner H={H}: T={rp['elapsed_minutes']:.2f} B={rp['expected_blood_loss_ml']:.1f} "
              f"dT={rp['elapsed_minutes']-rs.elapsed_minutes:+.2f} dB={rp['expected_blood_loss_ml']-rs.expected_blood_loss_ml:+.1f} "
              f"({dt:.0f}s comp={rp['completion']})", flush=True)


if __name__ == "__main__":
    main()

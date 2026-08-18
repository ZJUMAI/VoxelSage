"""Compare planner's per-step target choices against the seal-first heuristic.

Traces the first N macro targets of (a) the window-aware planner and
(b) seal_first_nearest on one scene, so we can see where the planner diverges
from the fast-and-safe heuristic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_window_evaluation import _scan_rank
from plan_target_order_v104 import (
    SerpentineTail,
    WindowAwarePlanner,
    _step_macro_target,
    rollout_planner,
)
from clinical_window_evaluation import serpentine_macro_target_policy


def seal_first(env, legal):
    exposed = env._exposed_cells()
    sealable = [c for c in legal if c in exposed]
    counts = env._transfer_counts()
    if sealable:
        return min(sealable, key=lambda c: (counts.get(c, 1e9), _scan_rank(env, c)))
    return min(legal, key=lambda c: (counts.get(c, 1e9), _scan_rank(env, c)))


def trace(env, selector, n=20):
    out = []
    while len(out) < n and not env.terminated and not env.truncated:
        legal = sorted(env._frontier())
        if not legal:
            break
        t = selector(env, legal)
        out.append((len(env.cut), t, env.phase, round(env.phase_elapsed_minutes, 1),
                    round(env.elapsed_minutes, 1)))
        _step_macro_target(env, t)
    return out


def main() -> None:
    d = json.load(open("results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"))
    sc = d["splits"]["planner_gate"]["scenarios"][0]
    cfg = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}
    run = rollout_planner

    # Planner trace (H=1 so every step is a fresh plan decision).
    tail = SerpentineTail(clinical_config=cfg)
    planner = WindowAwarePlanner(candidate_count=6, beam_width=4, lookahead_depth=8,
                                 margin_blood_ml=0.05 * 463.6, tail=tail)
    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=cfg)
    env.reset()
    seq = []
    for _ in range(20):
        if env.terminated or env.truncated:
            break
        plan = planner.plan_sequence(env, 1, baseline_blood=463.6)
        t = plan[0]
        seq.append((len(env.cut), t, env.phase, round(env.phase_elapsed_minutes, 1)))
        _step_macro_target(env, t)

    env2 = ClinicalMacroResectionEnv(scenario=sc, clinical_config=cfg)
    env2.reset()
    seq2 = trace(env2, seal_first, 20)

    env3 = ClinicalMacroResectionEnv(scenario=sc, clinical_config=cfg)
    env3.reset()
    seq3 = trace(env3, lambda e, legal: min(legal, key=lambda c: _scan_rank(e, c)), 20)

    print(f"{'cut':>4} {'planner':>10} {'seal1st':>10} {'serp':>10} {'phase_el'}")
    for i in range(20):
        a = seq[i] if i < len(seq) else None
        b = seq2[i] if i < len(seq2) else None
        c = seq3[i] if i < len(seq3) else None
        print(f"{a[0]:>4} {(str(a[1]) if a else '-'):>10} {(str(b[1]) if b else '-'):>10} "
              f"{(str(c[1]) if c else '-'):>10} {(str(a[3]) if a else '')}")


if __name__ == "__main__":
    main()

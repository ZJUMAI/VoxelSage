"""Quick divergence diagnosis: worker-tail vs clone+tail on a mid-episode state."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_macro_environment import ClinicalMacroResectionEnv  # noqa: E402
from clinical_window_evaluation import _scan_rank  # noqa: E402
from plan_target_order_v104 import _clone_env, _step_macro_target  # noqa: E402
from plan_target_order_v105 import (  # noqa: E402
    CorrectedPlannerV105,
    SerpentineTailV105,
    _candidate_sources_v105,
    _env_state_payload_v105,
    scene_budget,
)
from benchmark_target_order_v105 import _tail_worker_v105  # noqa: E402

CFG = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}
BB = 296.0
MARGIN = 14.82


def main() -> None:
    gate = json.loads(Path(
        "results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"
    ).read_text(encoding="utf-8"))
    sid = sys.argv[1] if len(sys.argv) > 1 else "clinical-d-v10.2-train-0089"
    sc = next(s for s in gate["splits"]["planner_gate"]["scenarios"] if s["scenario_id"] == sid)

    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=CFG)
    env.reset()
    planner = CorrectedPlannerV105(candidate_count=6, margin_ml=MARGIN, clinical_config=CFG)
    budget = scene_budget(BB, MARGIN)

    # Advance a fixed number of steps with the serial planner (cheap due to cache).
    for _ in range(40):
        if env.terminated or env.truncated:
            break
        traj, _info = planner.plan(env, BB, budget)
        t = traj[0]
        if t not in env._frontier():
            t = _serp(env)
        _step_macro_target(env, t)
    print(f"state after {env.step_count} steps: elapsed={env.elapsed_minutes:.2f} "
          f"blood={env.expected_blood_loss_ml:.2f}", flush=True)

    sourced = _candidate_sources_v105(env, count=6)
    targets = [t for t, _s in sourced]
    print(f"candidates ({len(targets)}): {targets}", flush=True)

    # Clone+tail per candidate (serial).
    serial_results = []
    for t in targets:
        e2 = _clone_env(env)
        t0, b0 = e2.elapsed_minutes, e2.expected_blood_loss_ml
        _step_macro_target(e2, t)
        dt, db = e2.elapsed_minutes - t0, e2.expected_blood_loss_ml - b0
        if e2.terminated or e2.truncated:
            serial_results.append((t, dt, db, bool(e2.terminated and e2.failure_reason is None),
                                   e2.failure_reason))
        else:
            tdt, tdb, comp, reason = SerpentineTailV105(clinical_config=CFG).tail(e2)
            serial_results.append((t, dt + tdt, db + tdb, comp, reason))

    # Worker-tail per candidate (parallel).
    state = _env_state_payload_v105(env)
    with mp.get_context("fork").Pool(8) as pool:
        worker_results = pool.map(
            _tail_worker_v105, [(sc, state, tuple(t), CFG) for t in targets])

    all_equal = True
    for (t, dt1, db1, c1, r1), (dt2, db2, c2, r2) in zip(serial_results, worker_results):
        ok = abs(dt1 - dt2) < 1e-9 and abs(db1 - db2) < 1e-9 and c1 == c2
        all_equal &= ok
        print(f"  {t}: clone=({dt1:.4f},{db1:.4f},c={c1}) worker=({dt2:.4f},{db2:.4f},c={c2}) "
              f"{'EQUAL' if ok else 'DIFF'}")
    print("ALL EQUAL:", all_equal)


def _serp(env):
    from plan_target_order_v104 import serpentine_target_of
    return serpentine_target_of(env)


if __name__ == "__main__":
    main()

"""Find the first decision divergence between Corrected (serial cache) and
Optimized (parallel no-cache) planners, then test whether the Corrected tail
cache returned a stale/wrong value at that state."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clinical_macro_environment import ClinicalMacroResectionEnv  # noqa: E402
from plan_target_order_v104 import _step_macro_target  # noqa: E402
from plan_target_order_v105 import (  # noqa: E402
    CorrectedPlannerV105,
    SerpentineTailV105,
    _candidate_sources_v105,
    scene_budget,
)
from benchmark_target_order_v105 import OptimizedPlannerV105  # noqa: E402

CFG = {"early_end_mode": "disabled", "bleeding_probability": 1.0, "max_steps_multiplier": 8.0}
MARGIN = 14.82


def _baseline_of(sid: str) -> float:
    d = json.loads(Path(
        "results/clinical_window_v10_5_safe_planner/reference/gate_r_evaluation.json"
    ).read_text(encoding="utf-8"))
    for r in d["rows"]:
        if r["scenario_id"] == sid:
            return float(r["baseline_B_ml"])
    raise KeyError(sid)


def main() -> None:
    gate = json.loads(Path(
        "results/clinical_window_v10_4_target_order/pilot_gate_a/gate_a_splits_v104.json"
    ).read_text(encoding="utf-8"))
    sid = sys.argv[1] if len(sys.argv) > 1 else "clinical-d-v10.2-train-0089"
    sc = next(s for s in gate["splits"]["planner_gate"]["scenarios"] if s["scenario_id"] == sid)
    budget = scene_budget(BB, MARGIN)

    serial = CorrectedPlannerV105(candidate_count=6, margin_ml=MARGIN, clinical_config=CFG)
    pool = mp.get_context("fork").Pool(8)
    optim = OptimizedPlannerV105(candidate_count=6, margin_ml=MARGIN, leaf_pool=pool,
                                 clinical_config=CFG)

    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=CFG)
    env.reset()
    divergence_state = None
    for step in range(900):
        if env.terminated or env.truncated:
            break
        t1, _i1 = serial.plan(env, BB, budget)
        t2, _i2 = optim.plan(env, BB, budget)
        if t1[0] != t2[0]:
            print(f"DIVERGENCE at step {step}: serial={t1[0]} optim={t2[0]} "
                  f"elapsed={env.elapsed_minutes:.2f} blood={env.expected_blood_loss_ml:.2f} "
                  f"cache_size={serial.tail.cache_size}", flush=True)
            divergence_state = env
            break
        _step_macro_target(env, t1[0])
    else:
        print(f"no divergence in 900 steps (scene {sid})", flush=True)
        pool.close()
        pool.join()
        return

    # At the divergence state, check each candidate's tail: cached (serial.tail)
    # vs fresh recompute vs worker.
    sourced = _candidate_sources_v105(divergence_state, count=6)
    targets = [t for t, _s in sourced]
    print("candidates:", targets, flush=True)
    for t in targets:
        from plan_target_order_v104 import _clone_env
        e2 = _clone_env(divergence_state)
        t0, b0 = e2.elapsed_minutes, e2.expected_blood_loss_ml
        _step_macro_target(e2, t)
        dt, db = e2.elapsed_minutes - t0, e2.expected_blood_loss_ml - b0
        cached = None
        fresh = None
        if not (e2.terminated or e2.truncated):
            key = serial.tail._state_key(e2)
            in_cache = key in serial.tail._cache
            cached = serial.tail.tail(e2)          # may hit the shared cache
            fresh = SerpentineTailV105(clinical_config=CFG).tail(
                _clone_env(e2))                     # guaranteed recompute
            flag = "CACHE-HIT" if in_cache else "miss"
            same = cached == fresh
            print(f"  {t}: {flag} cached_tail={cached[:2]} fresh={fresh[:2]} "
                  f"{'OK' if same else '*** CACHE MISMATCH ***'}", flush=True)
        else:
            print(f"  {t}: terminated after step, dt={dt:.3f} db={db:.3f}", flush=True)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()

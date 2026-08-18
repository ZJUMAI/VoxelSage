"""v10.3 pilot look-ahead features.

Forward-simulates the RELEASE branch from a decision state over one 15/5
window and extracts the decision-maker's look-ahead feature list as
window-bounded, explainable physics quantities:

  1. time to next vessel exposure
  2. exposure -> seal time
  3. unsealed area-time integral over the window
  4. max bleeding rate and cumulative blood loss
  5. large-vessel count, handling duration, whether the window crosses a
     clamp -> unclamp boundary

These are INTERMEDIATE quantities from a bounded forward simulation; the
oracle's full counterfactual Delta-B / Delta-I (which require running both
branches to termination) are never used here.  They are the supervised
regression targets, computed separately in the collector.

The future macro-target sequence ``targets`` is recorded from the frozen,
clamp-blind BC rollout (the clamp schedule does not change the target
sequence), so the deep-copied sim can be driven by a sequence selector.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_target_conditioned_environment import (  # noqa: E402
    CLAMP_CONTINUE,
    TargetConditionedClampEnv,
)
from train_target_conditioned_clamp_oracle import _bind_sequence_selector  # noqa: E402


def lookahead_features(
    env: TargetConditionedClampEnv,
    targets: list[int],
    *,
    window_minutes: float | None = None,
) -> dict[str, float]:
    """Compute window-bounded look-ahead features at the current decision state."""
    if window_minutes is None:
        window_minutes = (
            float(env.clinical_config["max_clamp_minutes"])
            + float(env.clinical_config["unclamp_minutes"])
        )
    # Capture current-state context from the ORIGINAL env (not the sim).
    cur_phase_limit = float(
        env.clinical_config["max_clamp_minutes"] if env.phase == "clamped"
        else env.clinical_config["unclamp_minutes"]
    )
    cur_context = {
        "cur_clamped": float(env.phase == "clamped"),
        "cur_phase_elapsed_frac": float(env.phase_elapsed_minutes / max(cur_phase_limit, 1e-9)),
        "cur_remaining_phase_frac": float(max(0.0, (cur_phase_limit - env.phase_elapsed_minutes) / max(cur_phase_limit, 1e-9))),
        "cur_total_clamped_frac": float(env.total_clamped_minutes / max(float(env.clinical_config["max_episode_minutes"]), 1e-9)),
        "cur_elapsed_frac": float(env.elapsed_minutes / max(float(env.clinical_config["max_episode_minutes"]), 1e-9)),
        "cur_cut_frac": float(len(env.cut) / max(len(env.domain), 1)),
        "cur_expected_bleed_rate": float(env._expected_bleeding_rate()),
        "cur_route_len": float(len(env.planned_route_cells)),
        "cur_planned_macro_duration": float(env.planned_macro_duration_minutes),
    }

    sim = copy.deepcopy(env)
    _bind_sequence_selector(sim, targets)
    # Release branch: lift the clamp now and run the planned future forward.
    if sim.phase == "clamped":
        sim._switch_phase("unclamped", [], reason="pilot_lookahead_release")
    start_elapsed = float(sim.elapsed_minutes)
    start_blood = float(sim.expected_blood_loss_ml)
    end_elapsed = start_elapsed + float(window_minutes)

    exposure_times: dict[int, float] = {}
    exposure_area: dict[int, float] = {}
    seal_times: dict[int, float] = {}
    all_exposed: list[int] = []
    large_exposed: int = 0
    peak_rate = 0.0
    handled_minutes = 0.0
    cells_cut_before = len(sim.cut)
    crossed_boundary = False
    last_phase = sim.phase
    n_steps = 0
    window_len = 0.0

    while sim.elapsed_minutes < end_elapsed - 1e-9 and not sim.terminated and not sim.truncated:
        before_exposed = set(sim.exposed_ids)
        before_blood = float(sim.expected_blood_loss_ml)
        before_elapsed = float(sim.elapsed_minutes)
        _, _, _, _, info = sim.step(CLAMP_CONTINUE, build_obs=False)
        n_steps += 1
        dt = float(sim.elapsed_minutes) - before_elapsed
        dblood = float(sim.expected_blood_loss_ml) - before_blood
        if dblood > 1e-9 and dt > 1e-9:
            peak_rate = max(peak_rate, dblood / dt)
        if sim.phase != last_phase:
            crossed_boundary = True
            last_phase = sim.phase
        for cid in sim.exposed_ids - before_exposed:
            comp = sim._component(cid)
            exposure_times[cid] = float(sim.elapsed_minutes) - start_elapsed
            exposure_area[cid] = float(comp["area_mm2"])
            all_exposed.append(int(cid))
            if bool(comp["is_large"]):
                large_exposed += 1
        for cid in before_exposed - sim.exposed_ids:
            if cid not in seal_times:
                seal_times[cid] = float(sim.elapsed_minutes) - start_elapsed
        for ev in info.get("events", []):
            if ev.get("action") in ("cut", "seal_and_cut_vessel"):
                handled_minutes += float(ev.get("duration_minutes", 0.0))
        if sim.elapsed_minutes >= end_elapsed - 1e-9 or sim.terminated or sim.truncated:
            window_len = float(sim.elapsed_minutes) - start_elapsed
            break
    if window_len <= 0.0:
        window_len = min(float(sim.elapsed_minutes) - start_elapsed, float(window_minutes))

    window_blood = float(sim.expected_blood_loss_ml) - start_blood
    cells_cut = len(sim.cut) - cells_cut_before

    # area-time integral over the window
    area_time = 0.0
    for cid in all_exposed:
        expose_t = exposure_times[cid]
        seal_t = seal_times.get(cid, window_len)
        area_time += exposure_area[cid] * max(0.0, seal_t - expose_t)

    time_to_first = None
    first_to_seal = None
    if all_exposed:
        first_cid = all_exposed[0]
        time_to_first = exposure_times[first_cid]
        seal_t = seal_times.get(first_cid)
        first_to_seal = (
            seal_t - time_to_first if seal_t is not None else window_len - time_to_first
        )

    return {
        # ---- decision-maker feature list ----
        "rw_time_to_first_exposure": float(time_to_first) if time_to_first is not None else float(window_len),
        "rw_first_exposure_to_seal": float(first_to_seal) if first_to_seal is not None else 0.0,
        "rw_unsealed_area_time": float(area_time),
        "rw_max_bleed_rate": float(min(peak_rate, float(sim.reference_flow_ml_per_min))),
        "rw_blood_loss": float(window_blood),
        "rw_large_exposed_count": float(large_exposed),
        "rw_exposed_count": float(len(all_exposed)),
        "rw_handling_duration": float(handled_minutes),
        "rw_crossed_boundary": float(crossed_boundary),
        "rw_unsealed_at_window_end": float(len(sim.exposed_ids) > 0),
        "rw_window_len": float(window_len),
        # ---- current-state context (from the original env) ----
        **cur_context,
        "fut_cells_cut": float(cells_cut),
        "fut_steps": float(n_steps),
    }


FEATURE_NAMES: list[str] = [
    "rw_time_to_first_exposure", "rw_first_exposure_to_seal", "rw_unsealed_area_time",
    "rw_max_bleed_rate", "rw_blood_loss", "rw_large_exposed_count", "rw_exposed_count",
    "rw_handling_duration", "rw_crossed_boundary", "rw_unsealed_at_window_end", "rw_window_len",
    "cur_clamped", "cur_phase_elapsed_frac", "cur_remaining_phase_frac",
    "cur_total_clamped_frac", "cur_elapsed_frac", "cur_cut_frac", "cur_expected_bleed_rate",
    "cur_route_len", "cur_planned_macro_duration", "fut_cells_cut", "fut_steps",
]

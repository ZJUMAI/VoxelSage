"""Auditable candidate features for the v10.4 target-order scorer (guide 7.1).

Every feature is a function of the *current observable state* and the
*deterministic next macro action* (transfer path + cut/seal under the shared
15/5 + blood code).  No teacher end-of-episode cost or full counterfactual is
leaked into the features: the one-step ``delta_T``/``delta_B`` are the guide's
own "precise next-macro-action" features, and the S-scan tail cost is only the
supervised regression/ranking target.

The vector ordering is fixed by ``FEATURE_NAMES``; continuous features are
normalised with Train-only scales computed by
``clinical_target_order_features.compute_feature_scales``.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from clinical_macro_environment import ClinicalMacroResectionEnv
from plan_target_order_v104 import _clone_env, _step_macro_target

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
CANDIDATE_FEATURE_NAMES = (
    "transfer_cells_norm",        # shortest transfer steps through cut region
    "transfer_time_norm",         # transfer duration / time_scale
    "cut_time_norm",              # target action duration (3x large vessel)
    "is_large_vessel",            # 0/1 target is a large-vessel cell
    "vessel_cells_norm",          # component cell count / domain size
    "vessel_area_norm",           # component area mm2 / 100
    "target_exposed_vessel",      # 0/1 sealing an exposed vessel this step
    "target_near_hidden_vessel",  # 0/1 target 4-adjacent to a hidden vessel
    "delta_T_norm",               # precise next-macro T increment / time_scale
    "delta_B_norm",               # precise next-macro B increment / blood_scale
    "arrival_phase",              # 0 clamped / 1 unclamped after this macro
    "phase_remaining_norm",       # time to next phase boundary / phase length
    "crosses_boundary",           # 0/1 macro crosses a 15/5 boundary
    "new_sealed_area_norm",       # area sealed this step / 100
    "new_exposed_area_norm",      # area newly exposed this step / 100
    "unsealed_area_after_norm",   # unsealed area after this step / 100
    "bleed_rate_after_norm",      # expected bleeding rate / reference flow
    "current_cut_frac",           # global cut fraction before this step
    "transfer_overhead",          # transfer_count / direction_action_count
    "total_clamped_frac",         # total clamped minutes / elapsed
)
CANDIDATE_FEATURE_DIM = len(CANDIDATE_FEATURE_NAMES)

GLOBAL_FEATURE_NAMES = (
    "phase",                      # 0 clamped / 1 unclamped
    "phase_elapsed_frac",         # elapsed within current phase / phase length
    "remaining_phase_frac",
    "elapsed_frac",               # elapsed / max_episode_minutes
    "cut_frac",                   # |cut| / |domain|
    "expected_bleed_rate_norm",   # current bleeding rate / reference flow
    "exposed_count",              # exposed vessel component count
    "hidden_count",               # hidden vessel component count
    "sealed_count",               # sealed vessel component count
    "unsealed_area_norm",         # unsealed area / 100
    "total_clamped_frac",         # total clamped / elapsed
)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

# Geometric-only feature names whose scales are computed on Train (policy_train).
SCALED_FEATURES = (
    "transfer_cells_norm",
    "transfer_time_norm",
    "cut_time_norm",
    "vessel_cells_norm",
    "vessel_area_norm",
    "new_sealed_area_norm",
    "new_exposed_area_norm",
    "unsealed_area_after_norm",
)


def _phase_length_minutes(env: ClinicalMacroResectionEnv) -> float:
    if env.phase == "clamped":
        return float(env.clinical_config["max_clamp_minutes"])
    return float(env.clinical_config["unclamp_minutes"])


def _component_of(env: ClinicalMacroResectionEnv, target: tuple[int, int]):
    cid = env.component_by_cell.get(target)
    return env._component(cid) if cid is not None else None


def global_context(env: ClinicalMacroResectionEnv) -> np.ndarray:
    """Shared global scalar context of the current decision state."""
    phase_len = _phase_length_minutes(env)
    exposed_area = sum(float(env._component(cid)["area_mm2"]) for cid in env.exposed_ids)
    hidden_area = sum(float(env._component(cid)["area_mm2"]) for cid in env.hidden_ids)
    ref = float(env.reference_flow_ml_per_min)
    elapsed = max(env.elapsed_minutes, 1e-9)
    vec = [
        float(env.phase == "unclamped"),
        env.phase_elapsed_minutes / max(phase_len, 1e-9),
        max(0.0, phase_len - env.phase_elapsed_minutes) / max(phase_len, 1e-9),
        env.elapsed_minutes / float(env.clinical_config["max_episode_minutes"]),
        len(env.cut) / max(1, len(env.domain)),
        env._expected_bleeding_rate() / max(ref, 1e-9),
        float(len(env.exposed_ids)),
        float(len(env.hidden_ids)),
        float(len(env.sealed_ids)),
        (exposed_area + hidden_area) / 100.0,
        env.total_clamped_minutes / max(elapsed, 1e-9),
    ]
    return np.asarray(vec, dtype=np.float32)


def candidate_features(
    env: ClinicalMacroResectionEnv,
    target: tuple[int, int],
) -> tuple[np.ndarray, ClinicalMacroResectionEnv, float, float]:
    """Extract the candidate feature vector by cloning + executing the real
    macro action (no teacher cost).  Returns ``(feat, env_after, dt, db)``."""
    counts = env._transfer_counts()
    transfer_cells = float(counts.get(target, 0.0))
    base = env.base_action_minutes
    transfer_time = transfer_cells * base
    cut_time = env._action_duration(target)
    phase_len = _phase_length_minutes(env)
    crossing_time = (phase_len - env.phase_elapsed_minutes)
    crosses_boundary = float((transfer_time + cut_time) > crossing_time + 1e-9)

    component = _component_of(env, target)
    vessel_cells = float(len(component["cells"])) if component else 0.0
    vessel_area = float(component["area_mm2"]) if component else 0.0
    is_large = float(component["is_large"]) if component else 0.0
    target_exposed = float(target in env._exposed_cells())
    target_hidden = float(any(n in env._hidden_cells() for n in neighbors4(target)))

    e = _clone_env(env)
    t0 = e.elapsed_minutes
    b0 = e.expected_blood_loss_ml
    sealed_before = {cid for cid in e.sealed_ids}
    exposed_before = {cid for cid in e.exposed_ids}
    _step_macro_target(e, target)
    dt = e.elapsed_minutes - t0
    db = e.expected_blood_loss_ml - b0

    new_sealed = e.sealed_ids - sealed_before
    new_exposed = e.exposed_ids - exposed_before
    new_sealed_area = sum(float(e._component(cid)["area_mm2"]) for cid in new_sealed)
    new_exposed_area = sum(float(e._component(cid)["area_mm2"]) for cid in new_exposed)
    unsealed_after = sum(
        float(e._component(cid)["area_mm2"])
        for cid in (e.hidden_ids | e.exposed_ids)
    )
    ref = max(float(e.reference_flow_ml_per_min), 1e-9)
    time_scale = float(e.clinical_config["time_scale_minutes"])
    blood_scale = float(e.clinical_config["blood_scale_ml"])
    elapsed = max(e.elapsed_minutes, 1e-9)

    vec = [
        transfer_cells / max(1.0, float(e.max_rows + e.max_cols)),
        transfer_time / time_scale,
        cut_time / time_scale,
        is_large,
        vessel_cells / max(1.0, float(len(e.domain))),
        vessel_area / 100.0,
        target_exposed,
        target_hidden,
        dt / time_scale,
        db / max(blood_scale, 1e-9),
        float(e.phase == "unclamped"),
        max(0.0, phase_len - e.phase_elapsed_minutes) / max(phase_len, 1e-9),
        crosses_boundary,
        new_sealed_area / 100.0,
        new_exposed_area / 100.0,
        unsealed_after / 100.0,
        e._expected_bleeding_rate() / ref,
        len(env.cut) / max(1, len(env.domain)),
        e.transfer_count / max(1, e.direction_action_count),
        e.total_clamped_minutes / max(elapsed, 1e-9),
    ]
    feat = np.asarray(vec, dtype=np.float32)
    if feat.shape[0] != CANDIDATE_FEATURE_DIM:
        raise RuntimeError(f"candidate feature dim mismatch {feat.shape[0]} != {CANDIDATE_FEATURE_DIM}")
    return feat, e, dt, db


def compute_feature_scales(
    samples: list[np.ndarray],
    *,
    name: str = "candidate_features",
) -> dict[str, Any]:
    """Per-feature mean/std over the geometric subset, computed on policy_train
    only (guide 4.2/7.1).  Returns a dict usable by ``normalize``."""
    mat = np.vstack(samples).astype(np.float32)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    idx = [CANDIDATE_FEATURE_NAMES.index(n) for n in SCALED_FEATURES]
    return {
        "name": name,
        "n_samples": len(samples),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "scaled_indices": idx,
    }


def normalize_features(
    feat: np.ndarray,
    scales: Mapping[str, Any],
) -> np.ndarray:
    mean = np.asarray(scales["mean"], dtype=np.float32)
    std = np.asarray(scales["std"], dtype=np.float32)
    out = feat.astype(np.float32).copy()
    for i in scales["scaled_indices"]:
        out[i] = (out[i] - mean[i]) / std[i]
    return out


def neighbors4(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))

"""v10.6 budget-aware candidate observations.

Teacher S-tail costs are deliberately absent from the model input.  They are
supervision targets only.  All input fields are current state, frozen scene
budget, candidate provenance, or the deterministic next macro action.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from clinical_macro_environment import ClinicalMacroResectionEnv
from clinical_target_order_features import (
    CANDIDATE_FEATURE_NAMES as V104_CANDIDATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES as V104_GLOBAL_FEATURE_NAMES,
    candidate_features as candidate_features_v104,
    global_context as global_context_v104,
)
from plan_target_order_v104 import serpentine_target_of
from plan_target_order_v105 import _candidate_sources_v105

SOURCE_NAMES = ("exposed", "near_hidden", "nearest", "fill")
CANDIDATE_FEATURE_NAMES = V104_CANDIDATE_FEATURE_NAMES + (
    "is_serpentine_fallback",
    "source_exposed",
    "source_near_hidden",
    "source_nearest",
    "source_fill",
)
GLOBAL_FEATURE_NAMES = V104_GLOBAL_FEATURE_NAMES + (
    "B_past_norm",
    "B_baseline_scene_norm",
    "M_B_norm",
    "B_budget_total_norm",
    "B_remaining_norm",
    "B_past_budget_ratio",
    "B_remaining_budget_ratio",
    "budget_negative",
)
CANDIDATE_FEATURE_DIM = len(CANDIDATE_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)


def budget_values(
    env: ClinicalMacroResectionEnv,
    *,
    baseline_blood_ml: float,
    margin_ml: float,
) -> dict[str, float]:
    past = float(env.expected_blood_loss_ml)
    baseline = float(baseline_blood_ml)
    margin = float(margin_ml)
    total = baseline + margin
    remaining = total - past
    return {
        "B_past_ml": past,
        "B_baseline_scene_ml": baseline,
        "M_B_ml": margin,
        "B_budget_total_ml": total,
        "B_remaining_ml": remaining,
    }


def global_context_v106(
    env: ClinicalMacroResectionEnv,
    *,
    baseline_blood_ml: float,
    margin_ml: float,
    blood_scale_ml: float,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = budget_values(env, baseline_blood_ml=baseline_blood_ml, margin_ml=margin_ml)
    scale = max(float(blood_scale_ml), 1e-9)
    budget = max(raw["B_budget_total_ml"], 1e-9)
    extra = np.asarray([
        raw["B_past_ml"] / scale,
        raw["B_baseline_scene_ml"] / scale,
        raw["M_B_ml"] / scale,
        raw["B_budget_total_ml"] / scale,
        raw["B_remaining_ml"] / scale,
        raw["B_past_ml"] / budget,
        raw["B_remaining_ml"] / budget,
        float(raw["B_remaining_ml"] < -1e-9),
    ], dtype=np.float32)
    return np.concatenate([global_context_v104(env), extra]), raw


def candidate_features_v106(
    env: ClinicalMacroResectionEnv,
    target: tuple[int, int],
    *,
    source: str | None = None,
) -> tuple[np.ndarray, ClinicalMacroResectionEnv, float, float, dict[str, Any]]:
    if source is None:
        source_map = dict(_candidate_sources_v105(env, count=6))
        source = source_map.get(target, "fill")
    feat, after, dt, db = candidate_features_v104(env, target)
    is_s = float(target == serpentine_target_of(env))
    canonical_source = "fill" if source == "s_target" else source
    extra = np.asarray(
        [is_s] + [float(canonical_source == name) for name in SOURCE_NAMES],
        dtype=np.float32,
    )
    result = np.concatenate([feat, extra])
    meta = {
        "is_serpentine_fallback": bool(is_s),
        "candidate_source": str(source),
        "delta_T_action": float(dt),
        "delta_B_action": float(db),
    }
    return result, after, float(dt), float(db), meta


def compute_feature_scales(samples: list[np.ndarray] | np.ndarray) -> dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("feature scale calibration needs samples")
    mat = np.vstack(samples).astype(np.float32)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    # Binary provenance fields are intentionally not standardized.
    scaled = list(range(len(V104_CANDIDATE_FEATURE_NAMES)))
    return {
        "version": "v10.6-candidate-feature-scales-v1",
        "n_samples": len(samples),
        "feature_names": list(CANDIDATE_FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "scaled_indices": scaled,
    }


def normalize_features(feat: np.ndarray, scales: Mapping[str, Any]) -> np.ndarray:
    out = feat.astype(np.float32).copy()
    mean = np.asarray(scales["mean"], dtype=np.float32)
    std = np.asarray(scales["std"], dtype=np.float32)
    for i in scales["scaled_indices"]:
        out[i] = (out[i] - mean[i]) / std[i]
    return out

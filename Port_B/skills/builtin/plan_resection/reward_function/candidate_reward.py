#!/usr/bin/env python3
"""Pareto/knee candidate reward for rule-based CRLM surface planning.

This module is intentionally independent of the doctor resection mask.  It
scores a bank of Bezier surface candidates from computable anatomy only:

- tumor recall is the first gate: removed tumor voxels / all tumor voxels;
- the final choice is made on a compact Pareto front rather than by matching a
  single target resection ratio;
- liver PCA / lobar axes / portal and hepatic vessel directions / tumor-surface
  corridor candidates receive anatomy-dependent support;
- stability is measured by robust tumor recall and scale consistency under
  origin/normal perturbations, not by raw mask disagreement alone.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScaleBand:
    lower: float
    soft_upper: float
    hard_upper: float
    compact_weight: float
    under_weight: float


SCALE_BANDS: dict[str, ScaleBand] = {
    "local": ScaleBand(0.000, 0.090, 0.220, 1.35, 0.00),
    "expanded_local": ScaleBand(0.150, 0.240, 0.380, 0.55, 1.00),
    "intermediate_local": ScaleBand(0.020, 0.260, 0.460, 0.75, 0.30),
    "segmental": ScaleBand(0.055, 0.430, 0.680, 0.45, 0.70),
    "major": ScaleBand(0.140, 0.720, 0.900, 0.20, 1.15),
    "none": ScaleBand(0.000, 0.080, 0.180, 1.40, 0.00),
}

def unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        return vec / norm
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return unit(fallback, np.array([0.0, 1.0, 0.0], dtype=np.float64))


def _bernstein3(t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return np.stack([u**3, 3.0 * u**2 * t, 3.0 * u * t**2, t**3], axis=1)


def eval_surface(points: np.ndarray, surface: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ref = surface["reference_plane"]
    origin = np.asarray(ref["origin_mm"], dtype=np.float64)
    normal = np.asarray(ref["normal_world"], dtype=np.float64)
    u_axis = np.asarray(ref["u_axis_world"], dtype=np.float64)
    v_axis = np.asarray(ref["v_axis_world"], dtype=np.float64)
    u0, u1 = [float(v) for v in ref["u_range_mm"]]
    v0, v1 = [float(v) for v in ref["v_range_mm"]]
    grid = np.asarray(surface["height_control_4x4_mm"], dtype=np.float64)

    vec = points.astype(np.float64) - origin
    uu = vec @ u_axis
    vv = vec @ v_axis
    nn = vec @ normal
    us = (uu - u0) / max(u1 - u0, 1e-8)
    vs = (vv - v0) / max(v1 - v0, 1e-8)
    inside = (us >= 0.0) & (us <= 1.0) & (vs >= 0.0) & (vs <= 1.0)
    bu = _bernstein3(np.clip(us, 0.0, 1.0))
    bv = _bernstein3(np.clip(vs, 0.0, 1.0))
    height = np.einsum("ni,ij,nj->n", bu, grid, bv)
    return inside, nn - height <= 0.0


def eval_surfaces(points: np.ndarray, surfaces: list[dict[str, Any]]) -> np.ndarray:
    pred = np.zeros(len(points), dtype=bool)
    for surface in surfaces:
        inside, neg = eval_surface(points, surface)
        pred[inside] |= neg[inside]
    return pred


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def _shifted_surfaces(surfaces: list[dict[str, Any]], shift_mm: float) -> list[dict[str, Any]]:
    out = copy.deepcopy(surfaces)
    for surface in out:
        ref = surface["reference_plane"]
        origin = np.asarray(ref["origin_mm"], dtype=np.float64)
        normal = unit(np.asarray(ref["normal_world"], dtype=np.float64))
        ref["origin_mm"] = (origin + shift_mm * normal).astype(float).tolist()
    return out


def _tilted_surfaces(surfaces: list[dict[str, Any]], angle_deg: float, axis_name: str) -> list[dict[str, Any]]:
    out = copy.deepcopy(surfaces)
    angle = math.radians(angle_deg)
    for surface in out:
        ref = surface["reference_plane"]
        normal = unit(np.asarray(ref["normal_world"], dtype=np.float64))
        u_axis = unit(np.asarray(ref["u_axis_world"], dtype=np.float64))
        v_axis = unit(np.asarray(ref["v_axis_world"], dtype=np.float64))
        tilt_axis = u_axis if axis_name == "u" else v_axis
        new_normal = unit(math.cos(angle) * normal + math.sin(angle) * tilt_axis, normal)
        if axis_name == "u":
            new_u = unit(u_axis - float(np.dot(u_axis, new_normal)) * new_normal, v_axis)
            new_v = unit(np.cross(new_normal, new_u), v_axis)
        else:
            new_v = unit(v_axis - float(np.dot(v_axis, new_normal)) * new_normal, u_axis)
            new_u = unit(np.cross(new_v, new_normal), u_axis)
        ref["normal_world"] = new_normal.astype(float).tolist()
        ref["u_axis_world"] = new_u.astype(float).tolist()
        ref["v_axis_world"] = new_v.astype(float).tolist()
    return out


def candidate_family(name: str) -> str:
    if name.startswith("legacy_local"):
        return "legacy_local"
    if name.startswith("local"):
        return "local"
    if "portal" in name:
        return "portal_vessel"
    if "hepatic" in name:
        return "hepatic_vessel"
    if "liver_pca" in name:
        return "liver_pca"
    if "tumor_to_surface" in name or "tumor_to_liver_center" in name:
        return "corridor"
    if name.startswith("plane_x") or name.startswith("plane_y") or name.startswith("plane_z"):
        return "lobar_axis"
    if name.startswith("plane"):
        return "plane"
    return "other"


def build_case_context(
    liver_sample_xyz: np.ndarray,
    tumor_xyz: np.ndarray,
    hepatic_vein_xyz: np.ndarray | None,
    portal_vein_xyz: np.ndarray | None,
    predicted_scale: str,
    n_tumor_components: int,
    tumor_liver_ratio: float = 0.0,
    tumor_total_volume_mm3: float = 0.0,
    tumor_max_volume_mm3: float = 0.0,
    tumor_max_radius_mm: float = 0.0,
) -> dict[str, Any]:
    liver_center = liver_sample_xyz.mean(axis=0) if len(liver_sample_xyz) else np.zeros(3, dtype=np.float64)
    tumor_center = tumor_xyz.mean(axis=0) if len(tumor_xyz) else liver_center
    bbox = np.ptp(liver_sample_xyz, axis=0) if len(liver_sample_xyz) else np.ones(3, dtype=np.float64)
    bbox = np.where(bbox < 1e-6, 1.0, bbox)
    rel = np.abs((tumor_center - liver_center) / bbox)
    centrality = float(1.0 - min(1.0, np.max(rel) * 2.0))
    spread = float(np.linalg.norm(np.ptp(tumor_xyz, axis=0))) if len(tumor_xyz) else 0.0
    max_dim = float(np.max(np.ptp(tumor_xyz, axis=0))) if len(tumor_xyz) else 0.0

    def vessel_near(vessel_xyz: np.ndarray | None, threshold: float) -> bool:
        if vessel_xyz is None or len(vessel_xyz) == 0 or len(tumor_xyz) == 0:
            return False
        v = _subsample(vessel_xyz, 5000)
        t = _subsample(tumor_xyz, 2000)
        d = np.min(np.linalg.norm(t[:, None, :] - v[None, :, :], axis=2), axis=1)
        return float(np.percentile(d, 10)) < threshold

    portal_near = vessel_near(portal_vein_xyz, 13.0)
    hepatic_near = vessel_near(hepatic_vein_xyz, 15.0)
    side_x = float(np.sign(tumor_center[0] - liver_center[0]))
    extent_signal = 0.0
    if n_tumor_components >= 2:
        extent_signal += 0.30
    if n_tumor_components >= 4:
        extent_signal += 0.25
    if spread > 45.0:
        extent_signal += 0.25
    if spread > 80.0:
        extent_signal += 0.25
    if max_dim > 25.0:
        extent_signal += 0.15
    if max_dim > 45.0:
        extent_signal += 0.20
    if max_dim > 65.0:
        extent_signal += 0.15
    vessel_signal = (0.20 if portal_near else 0.0) + (0.20 if hepatic_near else 0.0)
    burden_signal = 0.0
    if n_tumor_components >= 4:
        burden_signal += 0.35
    if spread > 80.0:
        burden_signal += 0.30
    if max_dim > 55.0:
        burden_signal += 0.20
    if centrality > 0.45:
        burden_signal += 0.15
    if portal_near or hepatic_near:
        burden_signal += 0.15

    return {
        "liver_center": liver_center,
        "tumor_center": tumor_center,
        "liver_bbox_mm": bbox,
        "centrality": centrality,
        "tumor_spread_mm": spread,
        "tumor_max_dim_mm": max_dim,
        "portal_near": portal_near,
        "hepatic_near": hepatic_near,
        "side_x": side_x,
        "predicted_scale": predicted_scale,
        "n_tumor_components": int(n_tumor_components),
        "tumor_liver_ratio": float(tumor_liver_ratio),
        "tumor_total_volume_mm3": float(tumor_total_volume_mm3),
        "tumor_max_volume_mm3": float(tumor_max_volume_mm3),
        "tumor_max_radius_mm": float(tumor_max_radius_mm),
        "extent_signal": float(min(1.0, extent_signal)),
        "vessel_signal": float(min(1.0, vessel_signal)),
        "burden_signal": float(min(1.0, burden_signal)),
    }


def surgical_mode_gate(predicted_scale: str, context: dict[str, Any]) -> tuple[str, str]:
    """Route selection with computable case facts before candidate scoring."""
    centrality = float(context.get("centrality", 0.0))
    ratio = float(context.get("tumor_liver_ratio", 0.0))
    total_volume = float(context.get("tumor_total_volume_mm3", 0.0))
    max_volume = float(context.get("tumor_max_volume_mm3", 0.0))
    n_components = int(context.get("n_tumor_components", 1))
    spread = float(context.get("tumor_spread_mm", 0.0))
    vessel = float(context.get("vessel_signal", 0.0))

    if predicted_scale == "major":
        high_total_volume = total_volume >= 18000.0
        low_central_local = (
            centrality < 0.50
            and ratio < 0.012
            and n_components <= 3
            and vessel <= 0.25
            and not high_total_volume
        )
        small_sparse_local = (
            max_volume < 5000.0
            and ratio < 0.0045
            and n_components <= 3
            and not (centrality >= 0.65 and vessel >= 0.40)
        )
        peripheral_low_burden = centrality < 0.25 and ratio < 0.010 and not high_total_volume
        if low_central_local:
            return "local_protect", "major_low_central_low_burden"
        if small_sparse_local:
            return "local_protect", "major_small_sparse"
        if peripheral_low_burden:
            return "local_protect", "major_peripheral_low_burden"

    if ratio >= 0.018 or total_volume >= 28000.0 or max_volume >= 18000.0:
        return "high_burden_anatomic", "large_tumor_burden"
    if centrality >= 0.65 and (n_components >= 3 or spread >= 110.0 or vessel >= 0.20):
        return "central_anatomic", "central_multifocal_or_vessel"
    if predicted_scale in {"segmental", "major"}:
        return "regional_segmental", "scale_regional"
    return "compact", "scale_compact"


def anatomy_support(name: str, predicted_scale: str, context: dict[str, Any]) -> float:
    family = candidate_family(name)
    support = 0.0
    if family in {"local", "legacy_local", "corridor"}:
        if predicted_scale in {"local", "expanded_local"}:
            support += 0.28
        elif predicted_scale == "intermediate_local":
            support += 0.16
        elif predicted_scale == "segmental":
            support += 0.02
        else:
            support -= 0.04
    if family in {"liver_pca", "lobar_axis"}:
        if predicted_scale in {"segmental", "major"}:
            support += 0.20
        elif predicted_scale == "intermediate_local":
            support += 0.08
        else:
            support -= 0.02
    if family in {"portal_vessel", "hepatic_vessel"}:
        if context.get("portal_near") or context.get("hepatic_near"):
            support += 0.18
        if predicted_scale in {"segmental", "major", "intermediate_local"}:
            support += 0.08
    if family == "corridor" and context.get("centrality", 0.0) < 0.42:
        support += 0.08
    if family == "legacy_local":
        support += 0.04
    if name.startswith("plane_x") and abs(context.get("side_x", 0.0)) > 0.0:
        support += 0.03
    return float(np.clip(support, -0.10, 0.36))


def candidate_geometry_features(
    name: str,
    surfaces: list[dict[str, Any]],
    liver_sample_xyz: np.ndarray,
    pred_sample: np.ndarray,
    tumor_xyz: np.ndarray,
    context: dict[str, Any],
) -> dict[str, float]:
    family = candidate_family(name)
    liver_center = np.asarray(context.get("liver_center", np.zeros(3)), dtype=np.float64)
    tumor_center = np.asarray(context.get("tumor_center", liver_center), dtype=np.float64)
    bbox = np.asarray(context.get("liver_bbox_mm", np.ones(3)), dtype=np.float64)
    bbox = np.where(bbox < 1e-6, 1.0, bbox)
    tumor_axis = unit(tumor_center - liver_center, np.array([1.0, 0.0, 0.0]))

    if len(liver_sample_xyz) == 0 or len(pred_sample) == 0 or not np.any(pred_sample):
        return {
            "resection_centroid_alignment": 0.0,
            "resection_centroid_distance_mm": 1e6,
            "tumor_side_match_score": 0.5,
            "tumor_side_confidence": 0.0,
            "local_surface_tumor_fit": 0.5,
            "local_active_surface_count": 0.0,
            "local_expected_surface_count": 1.0,
            "local_focus_score": 0.0,
            "geometry_support": 0.0,
            "geometry_penalty": 0.5,
        }

    removed = liver_sample_xyz[pred_sample]
    removed_centroid = removed.mean(axis=0)
    removed_axis = unit(removed_centroid - liver_center, tumor_axis)
    centroid_alignment = float(np.clip(np.dot(removed_axis, tumor_axis), -1.0, 1.0))
    centroid_distance = float(np.linalg.norm(removed_centroid - tumor_center))

    tumor_dx = float(tumor_center[0] - liver_center[0])
    removed_dx = float(removed_centroid[0] - liver_center[0])
    side_confidence = float(min(1.0, abs(tumor_dx) / max(0.16 * bbox[0], 1e-6)))
    if side_confidence < 0.20 or abs(removed_dx) < 1e-6:
        side_match = 0.5
    else:
        side_match = 1.0 if np.sign(tumor_dx) == np.sign(removed_dx) else 0.0

    tumor_spread = float(context.get("tumor_spread_mm", 0.0))
    tumor_radius = float(context.get("tumor_max_radius_mm", 0.0))
    focus_scale = max(22.0, 0.35 * tumor_spread + 2.0 * tumor_radius)
    local_focus = float(math.exp(-centroid_distance / max(focus_scale, 1e-6)))

    expected_surfaces = 1
    active_count = 0
    local_fit = 0.5
    if family in {"local", "legacy_local"} and len(tumor_xyz) > 0 and surfaces:
        tumor_geom = _subsample(tumor_xyz, 2500)
        per_surface = []
        for surface in surfaces:
            inside, neg = eval_surface(tumor_geom, surface)
            per_surface.append(float(np.mean(inside & neg)))
        active = [v for v in per_surface if v >= 0.08]
        active_count = len(active)
        local_fit = (
            1.0
            if active_count <= 1
            else max(0.35, 1.0 - 0.35 * (active_count - 1))
        )
    elif family == "corridor":
        local_fit = 0.65 + 0.35 * max(0.0, centroid_alignment)
    else:
        local_fit = 0.5

    corridor_good = max(0.0, centroid_alignment)
    corridor_bad = max(0.0, -centroid_alignment)
    side_support = side_confidence * (side_match - 0.5) * 2.0

    if family in {"local", "legacy_local", "corridor"}:
        support = 0.34 * corridor_good + 0.22 * max(0.0, side_support) + 0.24 * local_fit + 0.20 * local_focus
        penalty = 0.32 * corridor_bad + 0.28 * max(0.0, -side_support) + 0.25 * (1.0 - local_fit)
    else:
        # Large anatomical planes can legitimately put the resection centroid
        # away from the tumor-side corridor.  Use side consistency as a weak
        # prior, and only penalize severe opposite-side cuts in non-major modes.
        support = 0.16 * max(0.0, side_support) + 0.10 * corridor_good
        non_major_weight = 0.45 if context.get("predicted_scale") != "major" else 0.15
        penalty = non_major_weight * max(0.0, -side_support) + 0.08 * corridor_bad

    return {
        "resection_centroid_alignment": float(centroid_alignment),
        "resection_centroid_distance_mm": float(centroid_distance),
        "tumor_side_match_score": float(side_match),
        "tumor_side_confidence": float(side_confidence),
        "local_surface_tumor_fit": float(local_fit),
        "local_active_surface_count": float(active_count),
        "local_expected_surface_count": float(expected_surfaces),
        "local_focus_score": float(local_focus),
        "geometry_support": float(np.clip(support, 0.0, 1.0)),
        "geometry_penalty": float(np.clip(penalty, 0.0, 1.0)),
    }


def detachment_features(
    liver_sample_xyz: np.ndarray,
    pred_sample: np.ndarray,
    tumor_xyz: np.ndarray,
    context: dict[str, Any],
) -> dict[str, float]:
    """Approximate whether the removed tissue is surgically detachable.

    A valid resection should be open toward the liver surface along the
    tumor-to-surface corridor.  Small internal caps can cover the tumor and look
    stable, but they are not actionable because they do not free a liver piece.
    """
    if len(liver_sample_xyz) == 0 or len(pred_sample) == 0 or not np.any(pred_sample):
        return {
            "surface_touch_score": 0.0,
            "resection_radial_span_score": 0.0,
            "corridor_open_score": 0.0,
            "detachable_support": 0.0,
            "detachable_penalty": 1.0,
        }

    liver_center = np.asarray(context.get("liver_center", np.zeros(3)), dtype=np.float64)
    tumor_center = np.asarray(context.get("tumor_center", liver_center), dtype=np.float64)
    axis = unit(tumor_center - liver_center, np.array([1.0, 0.0, 0.0]))
    removed = liver_sample_xyz[pred_sample]

    liver_proj = (liver_sample_xyz - liver_center) @ axis
    removed_proj = (removed - liver_center) @ axis
    liver_min = float(np.percentile(liver_proj, 0.5))
    liver_max = float(np.percentile(liver_proj, 99.5))
    removed_min = float(np.percentile(removed_proj, 1.0))
    removed_max = float(np.percentile(removed_proj, 99.0))
    liver_span = max(liver_max - liver_min, 1e-6)

    # Does the removed side reach the outer liver envelope?
    surface_gap = max(0.0, liver_max - removed_max)
    surface_tol = max(8.0, 0.12 * liver_span)
    surface_touch = 1.0 - min(1.0, surface_gap / surface_tol)

    # Does the removed side form a meaningful corridor instead of a tiny shell?
    removed_span = max(0.0, removed_max - removed_min)
    span_need = max(10.0, 0.20 * liver_span)
    radial_span = min(1.0, removed_span / span_need)

    tumor_proj = float((tumor_center - liver_center) @ axis)
    corridor_open = 1.0 if (removed_min <= tumor_proj + 8.0 and removed_max >= liver_max - surface_tol) else 0.0

    support = 0.58 * surface_touch + 0.32 * corridor_open + 0.10 * radial_span
    penalty = max(0.0, 0.72 - support) / 0.72
    return {
        "surface_touch_score": float(np.clip(surface_touch, 0.0, 1.0)),
        "resection_radial_span_score": float(np.clip(radial_span, 0.0, 1.0)),
        "corridor_open_score": float(np.clip(corridor_open, 0.0, 1.0)),
        "detachable_support": float(np.clip(support, 0.0, 1.0)),
        "detachable_penalty": float(np.clip(penalty, 0.0, 1.0)),
    }


def stability_penalty(
    surfaces: list[dict[str, Any]],
    liver_sample_xyz: np.ndarray,
    tumor_xyz: np.ndarray | None = None,
    max_points: int = 6000,
    shift_mm: float = 4.0,
    angle_deg: float = 5.0,
    predicted_scale: str = "intermediate_local",
) -> dict[str, float]:
    if max_points <= 0:
        return {
            "stability_penalty": 0.0,
            "shift_disagreement": float("nan"),
            "normal_disagreement": float("nan"),
            "ratio_perturb_delta": float("nan"),
            "robust_tumor_recall_min": float("nan"),
            "tumor_recall_drop": float("nan"),
            "r0_instability_penalty": float("nan"),
            "scale_instability_penalty": float("nan"),
            "boundary_instability_penalty": float("nan"),
        }
    if not surfaces or len(liver_sample_xyz) == 0:
        return {
            "stability_penalty": 1.0,
            "shift_disagreement": 1.0,
            "normal_disagreement": 1.0,
            "ratio_perturb_delta": 1.0,
            "robust_tumor_recall_min": 0.0,
            "tumor_recall_drop": 1.0,
            "r0_instability_penalty": 1.0,
            "scale_instability_penalty": 1.0,
            "boundary_instability_penalty": 1.0,
        }
    pts = _subsample(liver_sample_xyz, max_points)
    base = eval_surfaces(pts, surfaces)
    base_ratio = float(base.mean())
    variants = [
        _shifted_surfaces(surfaces, shift_mm),
        _shifted_surfaces(surfaces, -shift_mm),
        _tilted_surfaces(surfaces, angle_deg, "u"),
        _tilted_surfaces(surfaces, angle_deg, "v"),
    ]
    shifted_plus, shifted_minus, tilt_u, tilt_v = [eval_surfaces(pts, v) for v in variants]

    shift_dis = 0.5 * (float(np.mean(base != shifted_plus)) + float(np.mean(base != shifted_minus)))
    normal_dis = 0.5 * (float(np.mean(base != tilt_u)) + float(np.mean(base != tilt_v)))
    ratio_delta = float(
        np.mean(
            [
                abs(float(shifted_plus.mean()) - base_ratio),
                abs(float(shifted_minus.mean()) - base_ratio),
                abs(float(tilt_u.mean()) - base_ratio),
                abs(float(tilt_v.mean()) - base_ratio),
            ]
        )
    )

    robust_tumor_recall_min = 1.0
    tumor_recall_drop = 0.0
    if tumor_xyz is not None and len(tumor_xyz) > 0:
        tumor_pts = _subsample(tumor_xyz, max_points)
        base_tumor = eval_surfaces(tumor_pts, surfaces)
        base_recall = float(base_tumor.mean())
        variant_recalls = [float(eval_surfaces(tumor_pts, v).mean()) for v in variants]
        robust_tumor_recall_min = float(min([base_recall, *variant_recalls]))
        tumor_recall_drop = float(max(0.0, base_recall - robust_tumor_recall_min))

    # This is the clinically meaningful part of stability: after a small
    # perturbation, the candidate should still remove nearly all tumor.
    r0_instability = min(1.0, max(0.0, 0.985 - robust_tumor_recall_min) / 0.12)

    allowed_delta = {
        "local": 0.025,
        "expanded_local": 0.040,
        "intermediate_local": 0.060,
        "segmental": 0.090,
        "major": 0.120,
        "none": 0.025,
    }.get(predicted_scale, 0.060)
    scale_instability = min(1.0, max(0.0, ratio_delta - allowed_delta) / max(allowed_delta, 1e-6))
    boundary_instability = min(1.0, (0.5 * shift_dis + 0.5 * normal_dis) / 0.25)
    penalty = 0.72 * r0_instability + 0.23 * scale_instability + 0.05 * boundary_instability
    return {
        "stability_penalty": float(penalty),
        "shift_disagreement": float(shift_dis),
        "normal_disagreement": float(normal_dis),
        "ratio_perturb_delta": float(ratio_delta),
        "robust_tumor_recall_min": float(robust_tumor_recall_min),
        "tumor_recall_drop": float(tumor_recall_drop),
        "r0_instability_penalty": float(r0_instability),
        "scale_instability_penalty": float(scale_instability),
        "boundary_instability_penalty": float(boundary_instability),
    }


def extent_support(pred_ratio: float, predicted_scale: str, context: dict[str, Any]) -> dict[str, float]:
    need = (
        0.45 * float(context.get("burden_signal", 0.0))
        + 0.35 * float(context.get("extent_signal", 0.0))
        + 0.25 * float(context.get("vessel_signal", 0.0))
    )
    cap = {
        "local": 0.18,
        "expanded_local": 0.28,
        "intermediate_local": 0.50,
        "segmental": 0.72,
        "major": 0.78,
        "none": 0.12,
    }.get(predicted_scale, 0.50)
    cap = min(0.88, cap + 0.10 * min(1.0, need))
    support = min(1.0, pred_ratio / max(cap, 0.05))
    hard_cap = {
        "local": 0.38,
        "expanded_local": 0.50,
        "intermediate_local": 0.78,
        "segmental": 0.92,
        "major": 0.96,
        "none": 0.25,
    }.get(predicted_scale, 0.78)
    over = max(0.0, pred_ratio - hard_cap) / max(1.0 - hard_cap, 0.04)
    return {
        "extent_support": float(support),
        "extent_need": float(min(1.0, need)),
        "extent_support_cap": float(cap),
        "extent_overcut_penalty": float(over),
    }


def dominance_scale_floor(predicted_scale: str, context: dict[str, Any]) -> float:
    extent = float(context.get("extent_signal", 0.0))
    vessel = float(context.get("vessel_signal", 0.0))
    burden = float(context.get("burden_signal", 0.0))
    if predicted_scale == "local":
        return 0.0
    if predicted_scale == "expanded_local":
        return float(min(0.14, 0.055 + 0.045 * extent + 0.030 * vessel))
    if predicted_scale == "intermediate_local":
        return float(min(0.22, 0.070 + 0.085 * extent + 0.040 * vessel + 0.030 * burden))
    if predicted_scale == "segmental":
        return float(min(0.34, 0.135 + 0.095 * extent + 0.050 * burden + 0.030 * vessel))
    if predicted_scale == "major":
        return float(min(0.42, 0.180 + 0.120 * extent + 0.060 * burden + 0.040 * vessel))
    return 0.0


def volume_losses(pred_ratio: float, predicted_scale: str, context: dict[str, Any]) -> dict[str, float]:
    band = SCALE_BANDS.get(predicted_scale, SCALE_BANDS["intermediate_local"])
    burden = float(context.get("burden_signal", 0.0))
    lower = band.lower * (0.65 + 0.70 * burden)
    soft_upper = band.soft_upper * (0.85 + 0.35 * burden)
    hard_upper = band.hard_upper
    under = max(0.0, lower - pred_ratio) / max(lower, 0.04)
    soft_over = max(0.0, pred_ratio - soft_upper) / max(hard_upper - soft_upper, 0.08)
    hard_over = max(0.0, pred_ratio - hard_upper) / max(1.0 - hard_upper, 0.08)
    compact = pred_ratio / max(soft_upper, 0.05)
    return {
        "volume_under_loss": float(under),
        "volume_soft_over_loss": float(soft_over),
        "volume_hard_over_loss": float(hard_over),
        "compactness_loss": float(compact),
        "scale_lower": float(lower),
        "scale_soft_upper": float(soft_upper),
        "scale_hard_upper": float(hard_upper),
    }


def score_candidate(
    cand: dict[str, Any],
    liver_sample_xyz: np.ndarray,
    tumor_xyz: np.ndarray,
    hepatic_vein_xyz: np.ndarray | None,
    portal_vein_xyz: np.ndarray | None,
    target_ratio: float,
    predicted_scale: str,
    predicted_surface_count: int,
    context: dict[str, Any] | None = None,
    stability_points: int = 6000,
) -> dict[str, Any]:
    surfaces = cand["surfaces"]
    name = cand["name"]
    context = context or {}
    pred_sample = eval_surfaces(liver_sample_xyz, surfaces)
    pred_ratio = float(pred_sample.mean()) if len(pred_sample) else 0.0
    tumor_eval_xyz = _subsample(tumor_xyz, 20000)
    tumor_pred = eval_surfaces(tumor_eval_xyz, surfaces) if len(tumor_eval_xyz) else np.zeros(0, dtype=bool)
    # tumor_recall = removed tumor voxels / all tumor voxels.  This is not
    # resection purity, i.e. tumor voxels / all removed liver voxels.
    tumor_recall = float(tumor_pred.mean()) if len(tumor_pred) else 0.0
    geometry = candidate_geometry_features(name, surfaces, liver_sample_xyz, pred_sample, tumor_xyz, context)
    detachment = detachment_features(liver_sample_xyz, pred_sample, tumor_xyz, context)

    hepatic_ratio = 0.0
    portal_ratio = 0.0
    if hepatic_vein_xyz is not None and len(hepatic_vein_xyz) > 0:
        hepatic_eval = _subsample(hepatic_vein_xyz, 10000)
        hepatic_ratio = float(eval_surfaces(hepatic_eval, surfaces).mean())
    if portal_vein_xyz is not None and len(portal_vein_xyz) > 0:
        portal_eval = _subsample(portal_vein_xyz, 10000)
        portal_ratio = float(eval_surfaces(portal_eval, surfaces).mean())

    stability = stability_penalty(
        surfaces,
        liver_sample_xyz,
        tumor_xyz=tumor_xyz,
        max_points=stability_points,
        predicted_scale=predicted_scale,
    )
    anatomy = anatomy_support(name, predicted_scale, context)
    losses = volume_losses(pred_ratio, predicted_scale, context)
    extent = extent_support(pred_ratio, predicted_scale, context)
    count_penalty = 0.045 * abs(len(surfaces) - predicted_surface_count)
    band = SCALE_BANDS.get(predicted_scale, SCALE_BANDS["intermediate_local"])
    vessel_support = 0.30 * portal_ratio + 0.18 * hepatic_ratio
    vessel_penalty = 0.18 * portal_ratio + 0.10 * hepatic_ratio
    coverage_penalty = 8.0 * max(0.0, 0.985 - tumor_recall) ** 2
    volume_penalty = (
        band.under_weight * losses["volume_under_loss"]
        + 0.85 * losses["volume_soft_over_loss"]
        + 2.50 * losses["volume_hard_over_loss"]
        + band.compact_weight * 0.12 * losses["compactness_loss"]
    )
    score = (
        3.5 * tumor_recall
        + 0.75 * anatomy
        + 0.25 * detachment["detachable_support"]
        - vessel_penalty
        - coverage_penalty
        - volume_penalty
        - count_penalty
        - 0.45 * detachment["detachable_penalty"]
    )
    if candidate_family(name) in {"local", "legacy_local"} and detachment["surface_touch_score"] < 0.20:
        score -= 0.55
    if pred_ratio < 0.004:
        score -= 0.20
    if tumor_recall < 0.90:
        score -= 0.80 * (0.90 - tumor_recall)

    return {
        "score": float(score),
        "candidate_family": candidate_family(name),
        "pred_ratio_sample": pred_ratio,
        "tumor_recall_sample": tumor_recall,
        "tumor_coverage_sample": tumor_recall,
        "hepatic_vessel_ratio_sample": hepatic_ratio,
        "portal_vessel_ratio_sample": portal_ratio,
        "vessel_ratio_sample": float(0.5 * hepatic_ratio + 0.5 * portal_ratio),
        "vessel_support": float(vessel_support),
        "n_surfaces": len(surfaces),
        "anatomy_support": anatomy,
        "count_penalty": float(count_penalty),
        "target_ratio_reference": float(target_ratio),
        **geometry,
        **detachment,
        **stability,
        **losses,
        **extent,
    }


def _coverage_gate(scored: list[dict[str, Any]]) -> float:
    coverages = np.asarray([float(s["tumor_recall_sample"]) for s in scored], dtype=np.float64)
    for threshold in [0.995, 0.990, 0.985, 0.970, 0.950, 0.900]:
        if np.any(coverages >= threshold):
            return float(threshold)
    return float(max(0.0, np.max(coverages) - 1e-6)) if len(coverages) else 0.0


def _dominates(a: dict[str, Any], b: dict[str, Any], scale_lower: float) -> bool:
    a_pred = float(a["pred_ratio_sample"])
    b_pred = float(b["pred_ratio_sample"])
    if a_pred < scale_lower <= b_pred:
        return False
    better_or_equal = (
        float(a["tumor_recall_sample"]) >= float(b["tumor_recall_sample"]) - 0.003
        and a_pred <= b_pred + 0.010
        and float(a["vessel_ratio_sample"]) <= float(b["vessel_ratio_sample"]) + 0.020
        and float(a["anatomy_support"]) >= float(b["anatomy_support"]) - 0.030
    )
    strictly_better = (
        float(a["tumor_recall_sample"]) > float(b["tumor_recall_sample"]) + 0.010
        or a_pred < b_pred - 0.030
        or float(a["anatomy_support"]) > float(b["anatomy_support"]) + 0.060
    )
    return bool(better_or_equal and strictly_better)


def pareto_front_indices(indices: list[int], scored: list[dict[str, Any]], scale_lower: float) -> list[int]:
    front: list[int] = []
    for i in indices:
        if any(_dominates(scored[j], scored[i], scale_lower) for j in indices if j != i):
            continue
        front.append(i)
    return front


def select_candidate_index(
    scored: list[dict[str, Any]],
    predicted_scale: str,
    context: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    if not scored:
        raise ValueError("No scored candidates")
    context = context or {}
    coverage_gate = _coverage_gate(scored)
    eligible = [
        i
        for i, s in enumerate(scored)
        if float(s["tumor_recall_sample"]) >= coverage_gate and float(s["pred_ratio_sample"]) >= 0.002
    ]
    if not eligible:
        eligible = list(range(len(scored)))

    surgical_mode, surgical_mode_reason = surgical_mode_gate(predicted_scale, context)
    if surgical_mode == "local_protect":
        local_families = {"local", "legacy_local", "corridor"}
        total_volume = float(context.get("tumor_total_volume_mm3", 0.0))
        max_volume = float(context.get("tumor_max_volume_mm3", 0.0))
        ratio = float(context.get("tumor_liver_ratio", 0.0))
        n_components = int(context.get("n_tumor_components", 1))
        local_cap = 0.16
        if ratio >= 0.008 and total_volume >= 10000.0:
            local_cap = 0.34
        elif total_volume >= 6000.0 and max_volume < 9000.0:
            local_cap = 0.28
        elif n_components >= 3 and 1500.0 <= total_volume < 6000.0:
            local_cap = 0.28
        elif n_components >= 5 and total_volume >= 15000.0:
            local_cap = 0.34
        local_eligible = [
            i
            for i in eligible
            if scored[i]["candidate_family"] in local_families
            and float(scored[i]["pred_ratio_sample"]) <= local_cap
        ]
        if not local_eligible:
            local_eligible = [
                i
                for i, s in enumerate(scored)
                if s["candidate_family"] in local_families
                and float(s["pred_ratio_sample"]) <= local_cap
                and float(s["pred_ratio_sample"]) >= 0.002
            ]
        if local_eligible:
            best_idx = local_eligible[0]
            best_value = -1e9
            for i in local_eligible:
                s = scored[i]
                pred_ratio = float(s["pred_ratio_sample"])
                value = (
                    float(s["score"])
                    + 0.25 * min(1.0, pred_ratio / 0.12)
                    + 0.32 * float(s.get("local_surface_tumor_fit", 0.5))
                    + 0.20 * float(s.get("local_focus_score", 0.0))
                    + 0.18 * float(s.get("geometry_support", 0.0))
                    - 0.22 * float(s.get("geometry_penalty", 0.0))
                    - 0.25 * max(0.0, pred_ratio - 0.22) / 0.10
                )
                if value > best_value:
                    best_value = float(value)
                    best_idx = i
            details = {
                "selection_policy": "surgical_mode_gate_local_protect_v1",
                "surgical_mode": surgical_mode,
                "surgical_mode_reason": surgical_mode_reason,
                "coverage_gate": float(coverage_gate),
                "dominance_scale_floor": 0.0,
                "eligible_count": int(len(eligible)),
                "pareto_front_count": int(len(local_eligible)),
                "selected_knee_value": float(best_value),
                "local_protect_cap": float(local_cap),
            }
            return best_idx, details

    if surgical_mode in {"central_anatomic", "high_burden_anatomic", "regional_segmental"}:
        anatomic_families = {"corridor", "liver_pca", "lobar_axis", "portal_vessel", "hepatic_vessel", "plane", "other"}
        centrality = float(context.get("centrality", 0.0))
        extent = float(context.get("extent_signal", 0.0))
        vessel = float(context.get("vessel_signal", 0.0))
        burden = float(context.get("burden_signal", 0.0))
        scale_lower = dominance_scale_floor(predicted_scale, context)
        large_pressure = 0.0
        if surgical_mode == "central_anatomic" and centrality >= 0.60 and vessel >= 0.20:
            large_pressure = 1.0
        elif surgical_mode == "high_burden_anatomic" and centrality >= 0.45 and vessel >= 0.20:
            large_pressure = 0.55
        elif surgical_mode == "regional_segmental" and centrality >= 0.55 and vessel >= 0.20:
            large_pressure = 0.35
        if surgical_mode == "high_burden_anatomic":
            if large_pressure >= 0.50:
                mode_ratio_target = min(0.62, 0.34 + 0.10 * extent + 0.10 * burden + 0.08 * vessel)
            else:
                mode_ratio_target = min(0.38, 0.18 + 0.08 * extent + 0.08 * centrality + 0.06 * vessel)
        elif surgical_mode == "central_anatomic":
            if large_pressure >= 0.80:
                mode_ratio_target = min(0.72, 0.36 + 0.18 * centrality + 0.20 * vessel + 0.10 * burden + 0.08 * extent)
            else:
                mode_ratio_target = min(0.40, 0.18 + 0.12 * extent + 0.10 * centrality + 0.06 * vessel)
        else:
            if large_pressure >= 0.30:
                mode_ratio_target = min(0.48, 0.26 + 0.08 * extent + 0.08 * burden + 0.06 * vessel)
            else:
                mode_ratio_target = min(0.34, 0.16 + 0.08 * extent + 0.06 * centrality + 0.04 * vessel)
        if large_pressure >= 0.80:
            anatomic_floor = max(scale_lower, min(0.56, 0.55 * mode_ratio_target))
        elif large_pressure >= 0.30:
            anatomic_floor = max(min(scale_lower, 0.20), min(0.28, 0.40 * mode_ratio_target))
        else:
            anatomic_floor = min(0.16, max(0.035, 0.32 * mode_ratio_target))
        allowed_families = set(anatomic_families)
        if large_pressure < 0.50:
            allowed_families.update({"local", "legacy_local"})
        anatomic_eligible = [
            i
            for i in eligible
            if scored[i]["candidate_family"] in allowed_families
            and float(scored[i]["pred_ratio_sample"]) >= anatomic_floor
            and float(scored[i].get("detachable_support", 0.0)) >= 0.45
        ]
        if not anatomic_eligible:
            anatomic_eligible = [
                i
                for i in eligible
                if float(scored[i]["pred_ratio_sample"]) >= anatomic_floor
                and float(scored[i].get("detachable_support", 0.0)) >= 0.60
            ]
        if anatomic_eligible:
            best_idx = anatomic_eligible[0]
            best_value = -1e9
            for i in anatomic_eligible:
                s = scored[i]
                pred_ratio = float(s["pred_ratio_sample"])
                losses = volume_losses(pred_ratio, predicted_scale, context)
                ratio_support = min(1.0, pred_ratio / max(mode_ratio_target, 1e-6))
                ratio_under = max(0.0, mode_ratio_target - pred_ratio) / max(mode_ratio_target, 1e-6)
                over_grace = 0.04 + 0.14 * large_pressure
                ratio_over = max(0.0, pred_ratio - min(0.88, mode_ratio_target + over_grace)) / 0.10
                compact_over = max(0.0, pred_ratio - mode_ratio_target) / max(mode_ratio_target, 0.05)
                family = str(s.get("candidate_family", ""))
                local_detached_bonus = 0.20 if family in {"local", "legacy_local"} and large_pressure < 0.50 else 0.0
                value = (
                    0.25 * float(s["score"])
                    + 2.25 * float(s["tumor_recall_sample"])
                    + 0.45 * float(s.get("extent_support", 0.0))
                    + 0.85 * float(s.get("detachable_support", 0.0))
                    + 0.35 * float(s.get("anatomy_support", 0.0))
                    + 0.25 * float(s.get("vessel_support", 0.0))
                    + (0.35 + 0.80 * large_pressure) * ratio_support
                    - (0.25 + 1.00 * large_pressure) * ratio_under
                    - (0.70 - 0.25 * large_pressure) * ratio_over
                    - 0.22 * (1.0 - large_pressure) * compact_over
                    + local_detached_bonus
                    - 0.20 * float(s.get("geometry_penalty", 0.0))
                    - 0.10 * float(s.get("stability_penalty", 0.0))
                    - 0.12 * losses["volume_soft_over_loss"]
                    - 0.18 * losses["volume_hard_over_loss"]
                )
                if value > best_value:
                    best_value = float(value)
                    best_idx = i
            details = {
                "selection_policy": "surgical_mode_gate_anatomic_detachable_v1",
                "surgical_mode": surgical_mode,
                "surgical_mode_reason": surgical_mode_reason,
                "coverage_gate": float(coverage_gate),
                "dominance_scale_floor": float(scale_lower),
                "eligible_count": int(len(eligible)),
                "pareto_front_count": int(len(anatomic_eligible)),
                "selected_knee_value": float(best_value),
                "anatomic_detachable_floor": float(anatomic_floor),
                "anatomic_mode_ratio_target": float(mode_ratio_target),
                "anatomic_large_pressure": float(large_pressure),
                "anatomic_eligible_count": int(len(anatomic_eligible)),
            }
            return best_idx, details

    band = SCALE_BANDS.get(predicted_scale, SCALE_BANDS["intermediate_local"])
    non_extreme = [i for i in eligible if float(scored[i]["pred_ratio_sample"]) <= band.hard_upper]
    if non_extreme:
        eligible = non_extreme
    scale_lower = volume_losses(0.0, predicted_scale, context)["scale_lower"]
    front = pareto_front_indices(eligible, scored, scale_lower)
    if not front:
        front = eligible
    local_pred_ratios = [
        float(scored[i]["pred_ratio_sample"])
        for i in eligible
        if scored[i]["candidate_family"] in {"local", "legacy_local"}
        and float(scored[i]["tumor_recall_sample"]) >= coverage_gate
    ]
    min_local_pred_ratio = min(local_pred_ratios) if local_pred_ratios else float("inf")

    # Knee choice: among the coverage-qualified Pareto front, prefer the point
    # where extra removed liver volume no longer buys meaningful stability or
    # anatomy support.  This is deliberately not an absolute target-ratio match.
    best_idx = front[0]
    best_value = -1e9
    for i in front:
        s = scored[i]
        pred_ratio = float(s["pred_ratio_sample"])
        losses = volume_losses(pred_ratio, predicted_scale, context)
        geometry_support = 0.0
        geometry_penalty = 0.0
        allow_corridor_geometry = s.get("candidate_family") == "corridor" and predicted_scale != "local"
        vessel_signal = float(context.get("vessel_signal", 0.0))
        if predicted_scale == "expanded_local" and vessel_signal <= 0.05 and min_local_pred_ratio <= 0.14:
            allow_corridor_geometry = False
        if predicted_scale == "intermediate_local" and vessel_signal <= 0.05 and min_local_pred_ratio < 0.055:
            allow_corridor_geometry = False
        if allow_corridor_geometry:
            geometry_support = float(s.get("geometry_support", 0.0))
            geometry_penalty = float(s.get("geometry_penalty", 0.0))
        knee_value = (
            float(s["score"])
            + 0.35 * float(s["tumor_recall_sample"])
            + 0.22 * float(s["anatomy_support"])
            + 0.05 * geometry_support
            - 0.12 * geometry_penalty
            - 0.25 * losses["volume_under_loss"] * band.under_weight
            - 0.35 * losses["volume_soft_over_loss"]
            - 1.20 * losses["volume_hard_over_loss"]
        )
        if knee_value > best_value:
            best_value = float(knee_value)
            best_idx = i

    details = {
        "selection_policy": "surgical_mode_gate_v1",
        "surgical_mode": surgical_mode,
        "surgical_mode_reason": surgical_mode_reason,
        "coverage_gate": float(coverage_gate),
        "dominance_scale_floor": float(scale_lower),
        "eligible_count": int(len(eligible)),
        "pareto_front_count": int(len(front)),
        "selected_knee_value": float(best_value),
    }
    return best_idx, details


def score_and_select_candidates(
    candidates: list[dict[str, Any]],
    liver_sample_xyz: np.ndarray,
    tumor_xyz: np.ndarray,
    hepatic_vein_xyz: np.ndarray | None,
    portal_vein_xyz: np.ndarray | None,
    target_ratio: float,
    predicted_scale: str,
    predicted_surface_count: int,
    n_tumor_components: int,
    tumor_liver_ratio: float = 0.0,
    tumor_total_volume_mm3: float = 0.0,
    tumor_max_volume_mm3: float = 0.0,
    tumor_max_radius_mm: float = 0.0,
    stability_points: int = 6000,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    context = build_case_context(
        liver_sample_xyz,
        tumor_xyz,
        hepatic_vein_xyz,
        portal_vein_xyz,
        predicted_scale=predicted_scale,
        n_tumor_components=n_tumor_components,
        tumor_liver_ratio=tumor_liver_ratio,
        tumor_total_volume_mm3=tumor_total_volume_mm3,
        tumor_max_volume_mm3=tumor_max_volume_mm3,
        tumor_max_radius_mm=tumor_max_radius_mm,
    )
    scored: list[dict[str, Any]] = []
    for cand in candidates:
        scored.append(
            score_candidate(
                cand,
                liver_sample_xyz,
                tumor_xyz,
                hepatic_vein_xyz,
                portal_vein_xyz,
                target_ratio=target_ratio,
                predicted_scale=predicted_scale,
                predicted_surface_count=predicted_surface_count,
                context=context,
                stability_points=stability_points,
            )
        )
    best_idx, selection = select_candidate_index(scored, predicted_scale, context)
    selection.update(
        {
            "context_centrality": float(context.get("centrality", 0.0)),
            "context_tumor_spread_mm": float(context.get("tumor_spread_mm", 0.0)),
            "context_tumor_max_dim_mm": float(context.get("tumor_max_dim_mm", 0.0)),
            "context_portal_near": bool(context.get("portal_near", False)),
            "context_hepatic_near": bool(context.get("hepatic_near", False)),
            "context_extent_signal": float(context.get("extent_signal", 0.0)),
            "context_vessel_signal": float(context.get("vessel_signal", 0.0)),
            "context_burden_signal": float(context.get("burden_signal", 0.0)),
            "context_tumor_liver_ratio": float(context.get("tumor_liver_ratio", 0.0)),
            "context_tumor_total_volume_mm3": float(context.get("tumor_total_volume_mm3", 0.0)),
            "context_tumor_max_volume_mm3": float(context.get("tumor_max_volume_mm3", 0.0)),
            "context_tumor_max_radius_mm": float(context.get("tumor_max_radius_mm", 0.0)),
        }
    )
    return scored, best_idx, selection

#!/usr/bin/env python3
"""Final no-training rule planner with scale prediction and candidate reward.

This script does not train a model and does not use doctor labels to select the
final plan. It generates multiple Bezier candidate surface sets, scores them
with computable anatomy features, selects by a coverage-first Pareto/knee
reward, and then evaluates that selected candidate against the cleaned doctor
mask.

For diagnosis only, it also reports a sample-GT oracle over the same candidates
so we can separate candidate-set limitations from reward-selection limitations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


# The CLI entry point below defaults to local scratch directories that only
# exist on the original author's machine.  Keep the public source free of
# private absolute paths: these values are only used as argparse defaults for
# the standalone runner, never by the skill's function-call path.  Override
# them with PLAN_SURFACES_FILTER_ROOT / PLAN_SURFACES_OUTPUT_DIR when the
# standalone CLI is used.
_FILTER_ROOT_ENV = os.environ.get("PLAN_SURFACES_FILTER_ROOT", "").strip()
_OUTPUT_DIR_ENV = os.environ.get("PLAN_SURFACES_OUTPUT_DIR", "").strip()
FILTER_ROOT = Path(_FILTER_ROOT_ENV) if _FILTER_ROOT_ENV else Path("data/crlm_filter")
OUTPUT_DIR = Path(_OUTPUT_DIR_ENV) if _OUTPUT_DIR_ENV else Path("output/paper_submission_outputs")

STAGE2_CODE_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_CODE_DIR))

from reward_function.candidate_reward import SCALE_BANDS, score_and_select_candidates  # noqa: E402
from curved_refinement import (  # noqa: E402
    candidate_clearance,
    refine_candidate,
    tumor_boundary_points,
)
from surface_metrics import candidate_curvature_metrics  # noqa: E402


def unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        return vec / norm
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return unit(fallback, np.array([0.0, 1.0, 0.0], dtype=np.float64))


def ijk_to_xyz(ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    if len(ijk) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return nib.affines.apply_affine(affine, ijk.astype(np.float64)).astype(np.float32)


def dice_vec(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    denom = int(pred.sum()) + int(gt.sum())
    return 2.0 * inter / denom if denom else 0.0


def _bernstein3(t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return np.stack([u**3, 3 * u**2 * t, 3 * u * t**2, t**3], axis=1)


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


def choose_axes(normal: np.ndarray, points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    normal = unit(normal)
    if points is not None and len(points) > 4:
        centered = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
        centered = centered - (centered @ normal)[:, None] * normal[None, :]
        if float(np.linalg.norm(centered)) > 1e-6:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            u_axis = unit(vh[0] - np.dot(vh[0], normal) * normal, np.array([1.0, 0.0, 0.0]))
            v_axis = unit(np.cross(normal, u_axis), np.array([0.0, 1.0, 0.0]))
            return u_axis, v_axis
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, normal))) > 0.92:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_axis = unit(np.cross(normal, up), np.array([1.0, 0.0, 0.0]))
    v_axis = unit(np.cross(normal, u_axis), np.array([0.0, 1.0, 0.0]))
    return u_axis, v_axis


def make_surface(
    origin: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    half_u: float,
    half_v: float,
    grid_amp: float = 0.0,
    name: str = "",
) -> dict[str, Any]:
    grid = np.zeros((4, 4), dtype=np.float64)
    if grid_amp:
        # A gentle center bulge keeps the Bezier schema active while retaining
        # a mostly planar cut for large resections.
        for i in range(4):
            for j in range(4):
                du = (i / 3.0) - 0.5
                dv = (j / 3.0) - 0.5
                grid[i, j] = float(grid_amp * math.exp(-(du * du + dv * dv) / 0.18))
    return {
        "type": "heightfield_levelset_surface",
        "parameterization": "reference_plane_plus_bicubic_heightfield",
        "reference_plane": {
            "coordinate_system": "patient_mm",
            "origin_mm": np.asarray(origin, dtype=float).tolist(),
            "normal_world": unit(normal).astype(float).tolist(),
            "u_axis_world": unit(u_axis).astype(float).tolist(),
            "v_axis_world": unit(v_axis).astype(float).tolist(),
            "u_range_mm": [-float(half_u), float(half_u)],
            "v_range_mm": [-float(half_v), float(half_v)],
        },
        "height_decoder": {
            "type": "bicubic_bezier",
            "control_grid_size": [4, 4],
            "control_points_are_interpolated": False,
            "height_unit": "mm",
        },
        "height_control_4x4_mm": grid.tolist(),
        "semantics": {"positive_side": "remnant", "negative_side": "resection"},
        "rule_metadata": {"candidate_name": name},
    }


def full_extent(points: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, margin: float) -> tuple[float, float]:
    vec = points.astype(np.float64) - origin
    uu = vec @ u_axis
    vv = vec @ v_axis
    return float(np.max(np.abs(uu)) + margin), float(np.max(np.abs(vv)) + margin)


def local_extent(points: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, margin: float, min_half: float) -> tuple[float, float]:
    if len(points) == 0:
        return min_half, min_half
    vec = points.astype(np.float64) - origin
    uu = vec @ u_axis
    vv = vec @ v_axis
    return max(min_half, float(np.max(np.abs(uu)) + margin)), max(min_half, float(np.max(np.abs(vv)) + margin))


def predict_scale(
    liver_xyz_sample: np.ndarray,
    tumor_xyz: np.ndarray,
    vessel_xyz: np.ndarray,
    n_tumor_components: int,
    tumor_liver_ratio: float,
) -> tuple[str, float, str]:
    if len(tumor_xyz) == 0:
        return "none", 0.0, "no_tumor"
    liver_center = liver_xyz_sample.mean(axis=0)
    tumor_center = tumor_xyz.mean(axis=0)
    bbox = np.ptp(liver_xyz_sample, axis=0)
    bbox = np.where(bbox < 1e-6, 1.0, bbox)
    rel = np.abs((tumor_center - liver_center) / bbox)
    centrality = float(1.0 - min(1.0, np.max(rel) * 2.0))
    spread = float(np.linalg.norm(np.ptp(tumor_xyz, axis=0)))
    max_dim = float(np.max(np.ptp(tumor_xyz, axis=0)))
    vessel_near = False
    if len(vessel_xyz) > 0:
        # Fast approximate vessel distance with a deterministic subsample.
        v = vessel_xyz[:: max(1, len(vessel_xyz) // 5000)]
        t = tumor_xyz[:: max(1, len(tumor_xyz) // 2000)]
        d = np.min(np.linalg.norm(t[:, None, :] - v[None, :, :], axis=2), axis=1)
        vessel_near = float(np.percentile(d, 10)) < 12.0
    reasons = []
    if tumor_liver_ratio > 0.018:
        reasons.append("high_tumor_burden")
    if n_tumor_components >= 4:
        reasons.append("many_tumors")
    if spread > 80 or max_dim > 65:
        reasons.append("wide_tumor_spread")
    if centrality > 0.45:
        reasons.append("central_tumor")
    if vessel_near:
        reasons.append("near_major_vessels")

    if n_tumor_components >= 6 or spread > 125:
        return "major", 0.55, "+".join(reasons) or "major_default"
    if tumor_liver_ratio > 0.020 or n_tumor_components >= 4 or spread > 80:
        return "segmental", 0.32, "+".join(reasons) or "segmental_default"
    if n_tumor_components >= 2 or tumor_liver_ratio > 0.006 or spread > 45 or (centrality > 0.55 and vessel_near):
        return "intermediate_local", 0.24, "+".join(reasons) or "intermediate_default"
    if centrality > 0.45 or vessel_near:
        return "expanded_local", 0.18, "+".join(reasons) or "central_or_vessel_single_tumor"
    return "local", 0.12, "+".join(reasons) or "peripheral_low_burden"


def plane_candidate(
    liver_sample: np.ndarray,
    tumor_xyz: np.ndarray,
    normal: np.ndarray,
    target_ratio: float,
    name: str,
    margin: float = 12.0,
) -> dict[str, Any]:
    normal = unit(normal)
    proj_l = liver_sample @ normal
    threshold = float(np.quantile(proj_l, np.clip(target_ratio, 0.01, 0.90)))
    if len(tumor_xyz) > 0:
        threshold = max(threshold, float(np.percentile(tumor_xyz @ normal, 99.0)) + 2.0)
    origin = normal * threshold
    u_axis, v_axis = choose_axes(normal, tumor_xyz)
    hu, hv = full_extent(liver_sample, origin, u_axis, v_axis, margin=margin)
    return {"name": name, "surfaces": [make_surface(origin, normal, u_axis, v_axis, hu, hv, 0.0, name)]}


def local_candidate(
    liver_sample: np.ndarray,
    cluster_points: list[np.ndarray],
    liver_center: np.ndarray,
    margin_depth: float,
    lateral_margin: float,
    min_half: float,
    name: str,
) -> dict[str, Any]:
    surfaces: list[dict[str, Any]] = []
    for pts in cluster_points:
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 90)) if len(pts) > 4 else 5.0
        normal = unit(liver_center - center, np.array([0.0, 0.0, 1.0]))
        u_axis, v_axis = choose_axes(normal, pts)
        origin = center + normal * (radius + margin_depth)
        hu, hv = local_extent(pts, origin, u_axis, v_axis, margin=lateral_margin, min_half=min_half)
        surfaces.append(make_surface(origin, normal, u_axis, v_axis, hu, hv, grid_amp=min(10.0, 0.25 * radius), name=name))
    return {"name": name, "surfaces": surfaces}


def legacy_component_candidate(
    tumor_components: list[dict[str, Any]],
    liver_center: np.ndarray,
    margin_depth: float,
    lateral_margin: float,
    min_half: float,
    name: str,
) -> dict[str, Any]:
    if not tumor_components:
        return {"name": name, "surfaces": []}
    centers = np.asarray([c["centroid_world"] for c in tumor_components], dtype=np.float64)
    weights = np.asarray(
        [max(float(c.get("volume_mm3", 1.0)), 1.0) for c in tumor_components],
        dtype=np.float64,
    )
    radii = np.asarray(
        [max(float(c.get("radius_mm", 3.0)), 1.0) for c in tumor_components],
        dtype=np.float64,
    )
    center = np.average(centers, axis=0, weights=weights)
    radius = float(np.max(radii))
    normal = unit(liver_center - center, np.array([0.0, 0.0, 1.0]))
    u_axis, v_axis = choose_axes(normal, centers)
    origin = center + normal * (radius + margin_depth)
    vec = centers - origin
    uu = vec @ u_axis
    vv = vec @ v_axis
    hu = max(min_half, float(np.max(np.abs(uu) + radii + lateral_margin)))
    hv = max(min_half, float(np.max(np.abs(vv) + radii + lateral_margin)))
    surface = make_surface(
        origin,
        normal,
        u_axis,
        v_axis,
        hu,
        hv,
        grid_amp=min(10.0, 0.35 * radius),
        name=name,
    )
    return {"name": name, "surfaces": [surface]}


def append_vessel_normals(
    normals: list[tuple[str, np.ndarray]],
    label: str,
    vessel_xyz: np.ndarray,
    tumor_center: np.ndarray,
) -> None:
    if len(vessel_xyz) < 16:
        return
    step = max(1, len(vessel_xyz) // 8000)
    pts = vessel_xyz[::step].astype(np.float64)
    center = pts.mean(axis=0)
    to_tumor = unit(tumor_center - center, np.array([0.0, 0.0, 1.0]))
    normals.append((f"{label}_to_tumor_pos", to_tumor))
    normals.append((f"{label}_to_tumor_neg", -to_tumor))
    try:
        centered = pts - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        for k in range(min(2, len(vh))):
            normals.append((f"{label}_pca{k}_pos", vh[k]))
            normals.append((f"{label}_pca{k}_neg", -vh[k]))
    except np.linalg.LinAlgError:
        return


def build_candidates(
    liver_sample: np.ndarray,
    tumor_xyz: np.ndarray,
    hepatic_vein_xyz: np.ndarray,
    portal_vein_xyz: np.ndarray,
    n_tumor_components: int,
    target_ratio: float,
    tumor_components: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    liver_center = liver_sample.mean(axis=0)
    tumor_center = tumor_xyz.mean(axis=0) if len(tumor_xyz) else liver_center
    candidates: list[dict[str, Any]] = []
    # The online workflow intentionally treats the tumor mask as one target and
    # exports one editable surface.  Keep every active candidate single-patch so
    # scoring, browser editing, persistence, and sequence planning share the
    # same plan semantics.
    for depth in [5.0, 10.0, 18.0, 28.0]:
        for lat in [12.0, 24.0, 42.0]:
            candidates.append(
                local_candidate(
                    liver_sample,
                    [tumor_xyz],
                    liver_center,
                    margin_depth=depth,
                    lateral_margin=lat,
                    min_half=max(16.0, lat),
                    name=f"local_n1_d{depth:g}_lat{lat:g}",
                )
            )

    if tumor_components:
        for depth in [5.0, 10.0, 18.0]:
            for lat in [12.0, 24.0, 42.0]:
                candidates.append(
                    legacy_component_candidate(
                        tumor_components,
                        liver_center,
                        margin_depth=depth,
                        lateral_margin=lat,
                        min_half=max(16.0, lat),
                        name=f"legacy_local_s1_d{depth:g}_lat{lat:g}",
                    )
                )

    normals: list[tuple[str, np.ndarray]] = []
    base = unit(liver_center - tumor_center, np.array([0.0, 0.0, 1.0]))
    normals.extend(
        [
            ("tumor_to_surface_corridor", base),
            ("tumor_to_liver_center", base),
            ("reverse_tumor_to_liver_center", -base),
        ]
    )
    for axis_name, axis in [
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ]:
        normals.append((f"{axis_name}_pos", axis))
        normals.append((f"{axis_name}_neg", -axis))
    centered = liver_sample - liver_center
    try:
        _, _, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
        for k in range(min(3, len(vh))):
            normals.append((f"liver_pca{k}_pos", vh[k]))
            normals.append((f"liver_pca{k}_neg", -vh[k]))
    except np.linalg.LinAlgError:
        pass

    append_vessel_normals(normals, "hepatic_vessel", hepatic_vein_xyz, tumor_center)
    append_vessel_normals(normals, "portal_vessel", portal_vein_xyz, tumor_center)

    target_bank = sorted(set([0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.24, 0.32, 0.45, 0.58, 0.72, target_ratio]))
    for normal_name, normal in normals:
        for ratio in target_bank:
            candidates.append(plane_candidate(liver_sample, tumor_xyz, normal, ratio, name=f"plane_{normal_name}_r{ratio:.2f}"))

    # Deduplicate empty or identical names conservatively.
    seen = set()
    out = []
    for cand in candidates:
        if not cand["surfaces"]:
            continue
        if len(cand["surfaces"]) != 1:
            raise RuntimeError(
                f"Single-surface planning invariant violated by {cand['name']}: "
                f"found {len(cand['surfaces'])} surfaces"
            )
        name = cand["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(cand)
    return out


def sample_gt_mask(sample_ijk: np.ndarray, clean_resection_ijk: np.ndarray) -> np.ndarray:
    clean_set = {tuple(map(int, x)) for x in clean_resection_ijk.tolist()}
    return np.array([tuple(map(int, x)) in clean_set for x in sample_ijk], dtype=bool)


def evaluate_full(
    liver_ijk: np.ndarray,
    affine: np.ndarray,
    clean_resection_ijk: np.ndarray,
    clean_resection_points: np.ndarray,
    tumor_xyz: np.ndarray,
    surfaces: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, float]:
    clean_set = {tuple(map(int, x)) for x in clean_resection_ijk.tolist()}
    pred_all = np.zeros(len(liver_ijk), dtype=bool)
    gt_all = np.zeros(len(liver_ijk), dtype=bool)
    for start in range(0, len(liver_ijk), batch_size):
        end = min(start + batch_size, len(liver_ijk))
        batch_ijk = liver_ijk[start:end]
        batch_xyz = ijk_to_xyz(batch_ijk, affine)
        pred_all[start:end] = eval_surfaces(batch_xyz, surfaces)
        gt_all[start:end] = np.array([tuple(map(int, x)) in clean_set for x in batch_ijk.tolist()], dtype=bool)
    tumor_cov = float(eval_surfaces(tumor_xyz, surfaces).mean()) if len(tumor_xyz) else 0.0
    return {
        "resection_dice": dice_vec(pred_all, gt_all),
        "pred_resection_ratio": float(pred_all.mean()) if len(pred_all) else 0.0,
        "gt_resection_ratio": float(len(clean_resection_points) / max(len(liver_ijk), 1)),
        "tumor_coverage": tumor_cov,
    }


def predict_full_resection_ratio(
    liver_ijk: np.ndarray,
    affine: np.ndarray,
    surfaces: list[dict[str, Any]],
    batch_size: int,
) -> float:
    resection_count = 0
    for start in range(0, len(liver_ijk), batch_size):
        end = min(start + batch_size, len(liver_ijk))
        points = ijk_to_xyz(liver_ijk[start:end], affine)
        resection_count += int(eval_surfaces(points, surfaces).sum())
    return float(resection_count / max(len(liver_ijk), 1))


def write_surface_json(case_id: str, cand: dict[str, Any], row: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces = cand["surfaces"]
    record = {
        "schema": "margin_constrained_rule_candidate_bezier_v2",
        "case_id": case_id,
        "planner": "rule_candidate_parent_selection_then_margin_constrained_bezier_refinement",
        "selected_candidate": cand["name"],
        "selection_metrics": row,
        "llm_surface": surfaces[0] if surfaces else None,
        "llm_surface_set": {
            "type": "boolean_heightfield_levelset_surface_set",
            "coordinate_system": "patient_mm",
            "component_count": len(surfaces),
            "surface_count": len(surfaces),
            "aggregation": "union_of_components",
            "components": [{"surfaces": [s]} for s in surfaces],
            "surfaces": surfaces,
            "semantics": {
                "positive_side": "remnant",
                "negative_side": "resection",
                "resection": "union(components)",
            },
        },
    }
    (out_dir / f"{case_id}_canonical_world.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def process_case(case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    case_id = case_dir.name
    t0 = time.time()
    z = np.load(case_dir / "raw_pointclouds.npz")
    affine = z["affine"]
    liver_ijk = z["liver_ijk"]
    clean_resection_ijk = z["resection_clean_ijk"]
    clean_resection_xyz = z["resection_clean_xyz_mm"]
    tumor_ijk = z["tumor_all_ijk"]
    tumor_xyz = z["tumor_all_xyz_mm"]
    tumor_boundary_xyz = tumor_boundary_points(tumor_ijk, tumor_xyz)
    hepatic_vein_xyz = z["hepatic_vein_xyz_mm"]
    portal_vein_xyz = z["portal_vein_xyz_mm"]
    vessel_xyz = np.concatenate([hepatic_vein_xyz, portal_vein_xyz], axis=0)

    rng = np.random.default_rng(zlib.crc32(case_id.encode("utf-8")))
    sample_n = min(args.reward_sample_points, len(liver_ijk))
    sample_idx = np.sort(rng.choice(len(liver_ijk), size=sample_n, replace=False)) if sample_n < len(liver_ijk) else np.arange(len(liver_ijk))
    liver_sample_ijk = liver_ijk[sample_idx]
    liver_sample_xyz = ijk_to_xyz(liver_sample_ijk, affine)
    sample_gt = sample_gt_mask(liver_sample_ijk, clean_resection_ijk)

    tumor_liver_ratio = float(len(tumor_xyz) / max(len(liver_ijk), 1))
    n_tumor_components = 1
    tumor_components: list[dict[str, Any]] = []
    tumor_total_volume_mm3 = 0.0
    tumor_max_volume_mm3 = 0.0
    tumor_max_radius_mm = 0.0
    meta_path = case_dir / "metadata_filter.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tumor_components = meta.get("tumors", [])
        n_tumor_components = len(tumor_components) or 1
        tumor_volumes = [float(t.get("volume_mm3", 0.0)) for t in tumor_components]
        tumor_radii = [float(t.get("radius_mm", 0.0)) for t in tumor_components]
        tumor_total_volume_mm3 = float(sum(tumor_volumes))
        tumor_max_volume_mm3 = float(max(tumor_volumes)) if tumor_volumes else 0.0
        tumor_max_radius_mm = float(max(tumor_radii)) if tumor_radii else 0.0
    scale, target_ratio, scale_reason = predict_scale(
        liver_sample_xyz,
        tumor_xyz,
        vessel_xyz,
        n_tumor_components=n_tumor_components,
        tumor_liver_ratio=tumor_liver_ratio,
    )
    predicted_surface_count = 1
    original_candidates = build_candidates(
        liver_sample_xyz,
        tumor_xyz,
        hepatic_vein_xyz,
        portal_vein_xyz,
        n_tumor_components,
        target_ratio,
        tumor_components=tumor_components,
    )
    _, parent_index, parent_selection_info = score_and_select_candidates(
        original_candidates,
        liver_sample_xyz,
        tumor_xyz,
        hepatic_vein_xyz,
        portal_vein_xyz,
        target_ratio=target_ratio,
        predicted_scale=scale,
        predicted_surface_count=predicted_surface_count,
        n_tumor_components=n_tumor_components,
        tumor_liver_ratio=tumor_liver_ratio,
        tumor_total_volume_mm3=tumor_total_volume_mm3,
        tumor_max_volume_mm3=tumor_max_volume_mm3,
        tumor_max_radius_mm=tumor_max_radius_mm,
        stability_points=args.stability_sample_points,
    )

    refined_candidates: list[dict[str, Any]] = []
    refinement_by_source: dict[str, dict[str, Any]] = {}
    refined_index_by_source: dict[str, int] = {}
    for candidate in original_candidates:
        refined, refinement = refine_candidate(
            candidate,
            tumor_boundary_xyz,
            margin_mm=args.tumor_margin_mm,
            lateral_padding_mm=args.curve_lateral_padding_mm,
            smoothness=args.curve_smoothness,
            bins=args.curve_constraint_bins,
            max_iterations=args.curve_refine_iterations,
        )
        refinement_by_source[candidate["name"]] = refinement
        if refinement["success"]:
            refined_index_by_source[candidate["name"]] = len(refined_candidates)
            refined_candidates.append(refined)
    if not refined_candidates:
        raise RuntimeError("No candidate satisfied the Bezier tumor-margin constraint")

    refined_scored, safe_index, safe_selection_info = score_and_select_candidates(
        refined_candidates,
        liver_sample_xyz,
        tumor_xyz,
        hepatic_vein_xyz,
        portal_vein_xyz,
        target_ratio=target_ratio,
        predicted_scale=scale,
        predicted_surface_count=predicted_surface_count,
        n_tumor_components=n_tumor_components,
        tumor_liver_ratio=tumor_liver_ratio,
        tumor_total_volume_mm3=tumor_total_volume_mm3,
        tumor_max_volume_mm3=tumor_max_volume_mm3,
        tumor_max_radius_mm=tumor_max_radius_mm,
        stability_points=args.stability_sample_points,
    )

    parent_name = original_candidates[parent_index]["name"]
    parent_refined_index = refined_index_by_source.get(parent_name)
    hard_upper = float(SCALE_BANDS[scale].hard_upper)
    if parent_refined_index is None:
        parent_ratio: float | None = None
        reward_index = safe_index
        selection_policy = "safe_reselection_after_parent_refinement_failure"
        selection_info = safe_selection_info
    else:
        parent_ratio = predict_full_resection_ratio(
            liver_ijk,
            affine,
            refined_candidates[parent_refined_index]["surfaces"],
            args.batch_size,
        )
        if parent_ratio > hard_upper:
            reward_index = safe_index
            selection_policy = "safe_reselection_after_scale_violation"
            selection_info = safe_selection_info
        else:
            reward_index = parent_refined_index
            selection_policy = "parent_preserving_bezier_refinement"
            selection_info = parent_selection_info

    for idx, (candidate, score) in enumerate(zip(refined_candidates, refined_scored)):
        pred_sample = eval_surfaces(liver_sample_xyz, candidate["surfaces"])
        score["sample_gt_dice"] = dice_vec(pred_sample, sample_gt)
        score["candidate_index"] = idx
    best_oracle_idx = max(
        range(len(refined_scored)),
        key=lambda index: refined_scored[index]["sample_gt_dice"],
    )
    reward_cand = refined_candidates[reward_index]
    reward_score = refined_scored[reward_index]
    oracle_cand = refined_candidates[best_oracle_idx]
    reward_full = evaluate_full(
        liver_ijk,
        affine,
        clean_resection_ijk,
        clean_resection_xyz,
        tumor_xyz,
        reward_cand["surfaces"],
        args.batch_size,
    )
    oracle_full = evaluate_full(
        liver_ijk,
        affine,
        clean_resection_ijk,
        clean_resection_xyz,
        tumor_xyz,
        oracle_cand["surfaces"],
        args.batch_size,
    )
    reward_clearance = candidate_clearance(tumor_boundary_xyz, reward_cand["surfaces"])
    reward_curvature = candidate_curvature_metrics(
        reward_cand["surfaces"],
        liver_ijk,
        affine,
        sample_step_mm=args.curvature_sample_step_mm,
    )
    reward_source_name = reward_cand["name"].split("__curve_", maxsplit=1)[0]
    reward_refinement = refinement_by_source[reward_source_name]
    refinement_details = reward_refinement["details"]
    control_delta_rms = max(
        (float(detail.get("control_delta_rms_mm", 0.0)) for detail in refinement_details),
        default=0.0,
    )
    control_delta_max = max(
        (float(detail.get("control_delta_max_mm", 0.0)) for detail in refinement_details),
        default=0.0,
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "elapsed_sec": round(time.time() - t0, 3),
        "candidate_count": len(refined_candidates),
        "candidate_count_original": len(original_candidates),
        "candidate_count_safe": len(refined_candidates),
        "predicted_scale": scale,
        "scale_reason": scale_reason,
        "target_ratio": target_ratio,
        "n_tumor_components": n_tumor_components,
        "tumor_liver_ratio": tumor_liver_ratio,
        "tumor_total_volume_mm3": tumor_total_volume_mm3,
        "tumor_max_volume_mm3": tumor_max_volume_mm3,
        "tumor_max_radius_mm": tumor_max_radius_mm,
        "predicted_surface_count": predicted_surface_count,
        "parent_candidate": parent_name,
        "parent_refined_candidate": (
            refined_candidates[parent_refined_index]["name"]
            if parent_refined_index is not None
            else ""
        ),
        "parent_refined_pred_ratio": parent_ratio if parent_ratio is not None else "",
        "selection_policy": selection_policy,
        "selection_scale_hard_upper": hard_upper,
        "selection_fallback": selection_policy != "parent_preserving_bezier_refinement",
        "tumor_margin_target_mm": args.tumor_margin_mm,
        "tumor_margin_min_mm": float(np.min(reward_clearance)),
        "tumor_margin_p05_mm": float(np.percentile(reward_clearance, 5)),
        "tumor_margin_success": float(np.min(reward_clearance)) >= args.tumor_margin_mm - 0.05,
        "curve_applied": control_delta_max > 1e-6,
        "curve_control_delta_rms_mm": control_delta_rms,
        "curve_control_delta_max_mm": control_delta_max,
        "mean_abs_curvature_mm_inv": reward_curvature["mean_abs_curvature_mm_inv"],
        "p95_abs_curvature_mm_inv": reward_curvature["p95_abs_curvature_mm_inv"],
        "surface_area_mm2": reward_curvature["surface_area_mm2"],
        "reward_candidate": reward_cand["name"],
        "reward_candidate_family": reward_score["candidate_family"],
        "reward_score": reward_score["score"],
        "reward_sample_gt_dice": reward_score["sample_gt_dice"],
        "reward_sample_pred_ratio": reward_score["pred_ratio_sample"],
        "reward_sample_tumor_recall": reward_score["tumor_recall_sample"],
        "reward_sample_tumor_coverage": reward_score["tumor_coverage_sample"],
        "reward_sample_vessel_ratio": reward_score["vessel_ratio_sample"],
        "reward_sample_portal_vessel_ratio": reward_score["portal_vessel_ratio_sample"],
        "reward_sample_hepatic_vessel_ratio": reward_score["hepatic_vessel_ratio_sample"],
        "reward_anatomy_support": reward_score["anatomy_support"],
        "reward_stability_penalty": reward_score["stability_penalty"],
        "reward_shift_disagreement": reward_score["shift_disagreement"],
        "reward_normal_disagreement": reward_score["normal_disagreement"],
        "reward_ratio_perturb_delta": reward_score["ratio_perturb_delta"],
        "reward_robust_tumor_recall_min": reward_score["robust_tumor_recall_min"],
        "reward_tumor_recall_drop": reward_score["tumor_recall_drop"],
        "reward_r0_instability_penalty": reward_score["r0_instability_penalty"],
        "reward_scale_instability_penalty": reward_score["scale_instability_penalty"],
        "reward_boundary_instability_penalty": reward_score["boundary_instability_penalty"],
        "reward_extent_support": reward_score["extent_support"],
        "reward_extent_need": reward_score["extent_need"],
        "reward_extent_overcut_penalty": reward_score["extent_overcut_penalty"],
        "reward_vessel_support": reward_score["vessel_support"],
        "reward_geometry_support": reward_score["geometry_support"],
        "reward_geometry_penalty": reward_score["geometry_penalty"],
        "reward_surface_touch_score": reward_score["surface_touch_score"],
        "reward_resection_radial_span_score": reward_score["resection_radial_span_score"],
        "reward_corridor_open_score": reward_score["corridor_open_score"],
        "reward_detachable_support": reward_score["detachable_support"],
        "reward_detachable_penalty": reward_score["detachable_penalty"],
        "reward_resection_centroid_alignment": reward_score["resection_centroid_alignment"],
        "reward_resection_centroid_distance_mm": reward_score["resection_centroid_distance_mm"],
        "reward_tumor_side_match_score": reward_score["tumor_side_match_score"],
        "reward_tumor_side_confidence": reward_score["tumor_side_confidence"],
        "reward_local_surface_tumor_fit": reward_score["local_surface_tumor_fit"],
        "reward_local_active_surface_count": reward_score["local_active_surface_count"],
        "reward_local_expected_surface_count": reward_score["local_expected_surface_count"],
        "reward_local_focus_score": reward_score["local_focus_score"],
        "reward_volume_under_loss": reward_score["volume_under_loss"],
        "reward_volume_soft_over_loss": reward_score["volume_soft_over_loss"],
        "reward_volume_hard_over_loss": reward_score["volume_hard_over_loss"],
        "reward_n_surfaces": len(reward_cand["surfaces"]),
        "reward_full_resection_dice": reward_full["resection_dice"],
        "reward_full_pred_ratio": reward_full["pred_resection_ratio"],
        "reward_full_gt_ratio": reward_full["gt_resection_ratio"],
        "reward_full_tumor_recall": reward_full["tumor_coverage"],
        "reward_full_tumor_coverage": reward_full["tumor_coverage"],
        "oracle_candidate": oracle_cand["name"],
        "oracle_sample_gt_dice": refined_scored[best_oracle_idx]["sample_gt_dice"],
        "oracle_n_surfaces": len(oracle_cand["surfaces"]),
        "oracle_full_resection_dice": oracle_full["resection_dice"],
        "oracle_full_pred_ratio": oracle_full["pred_resection_ratio"],
        "oracle_full_tumor_recall": oracle_full["tumor_coverage"],
        "oracle_full_tumor_coverage": oracle_full["tumor_coverage"],
    }
    for key, value in selection_info.items():
        row[f"selection_{key}"] = value
    for key, value in parent_selection_info.items():
        row[f"parent_selection_{key}"] = value
    for key, value in safe_selection_info.items():
        row[f"safe_selection_{key}"] = value
    write_surface_json(case_id, reward_cand, row, args.output_dir / "reward_selected_surfaces")
    write_surface_json(case_id, oracle_cand, row, args.output_dir / "sample_oracle_surfaces")
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> None:
    write_csv(rows, args.output_dir / "surface_planner_results.csv")
    vals = lambda k: np.array([float(r[k]) for r in rows if k in r and r[k] != ""], dtype=np.float64)
    reward_dice = vals("reward_full_resection_dice")
    oracle_dice = vals("oracle_full_resection_dice")
    summary = {
        "n_cases": len(rows),
        "elapsed_sec": elapsed,
        "reward_selected": {
            "dice_mean": float(np.mean(reward_dice)),
            "dice_median": float(np.median(reward_dice)),
            "dice_std": float(np.std(reward_dice, ddof=1)),
            "dice_min": float(np.min(reward_dice)),
            "dice_max": float(np.max(reward_dice)),
            "thresholds": {f"ge_{t:.2f}": int(np.sum(reward_dice >= t)) for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]},
            "pred_ratio_mean": float(np.mean(vals("reward_full_pred_ratio"))),
            "gt_ratio_mean": float(np.mean(vals("reward_full_gt_ratio"))),
            "tumor_coverage_mean": float(np.mean(vals("reward_full_tumor_coverage"))),
            "tumor_margin_target_mm": args.tumor_margin_mm,
            "tumor_margin_success_rate": float(np.mean(vals("tumor_margin_success"))),
            "tumor_margin_median_mm": float(np.median(vals("tumor_margin_min_mm"))),
            "mean_abs_curvature_mm_inv": float(
                np.mean(vals("mean_abs_curvature_mm_inv"))
            ),
            "curved_case_count": int(np.sum(vals("curve_applied") > 0.5)),
            "scale_fallback_count": int(np.sum(vals("selection_fallback") > 0.5)),
        },
        "sample_oracle": {
            "dice_mean": float(np.mean(oracle_dice)),
            "dice_median": float(np.median(oracle_dice)),
            "dice_std": float(np.std(oracle_dice, ddof=1)),
            "dice_min": float(np.min(oracle_dice)),
            "dice_max": float(np.max(oracle_dice)),
            "thresholds": {f"ge_{t:.2f}": int(np.sum(oracle_dice >= t)) for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]},
            "pred_ratio_mean": float(np.mean(vals("oracle_full_pred_ratio"))),
            "tumor_coverage_mean": float(np.mean(vals("oracle_full_tumor_coverage"))),
        },
        "scale_counts": dict(Counter(r["predicted_scale"] for r in rows)),
        "reward_surface_count": dict(Counter(int(r["reward_n_surfaces"]) for r in rows)),
        "oracle_surface_count": dict(Counter(int(r["oracle_n_surfaces"]) for r in rows)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Final rule candidate planner summary",
        "",
        f"Cases: {len(rows)}",
        f"Elapsed: {elapsed:.1f}s",
        "",
        "## Reward-selected, no GT selection",
        f"- Resection Dice mean/median: {summary['reward_selected']['dice_mean']:.4f} / {summary['reward_selected']['dice_median']:.4f}",
        f"- Resection Dice min/max: {summary['reward_selected']['dice_min']:.4f} / {summary['reward_selected']['dice_max']:.4f}",
        f"- Thresholds: {summary['reward_selected']['thresholds']}",
        f"- Pred/GT resection ratio mean: {summary['reward_selected']['pred_ratio_mean']:.4f} / {summary['reward_selected']['gt_ratio_mean']:.4f}",
        f"- Tumor coverage mean: {summary['reward_selected']['tumor_coverage_mean']:.4f}",
        f"- Tumor-margin pass: {summary['reward_selected']['tumor_margin_success_rate']:.4f} at {args.tumor_margin_mm:g} mm",
        f"- Tumor-margin median: {summary['reward_selected']['tumor_margin_median_mm']:.4f} mm",
        f"- Area-weighted mean |H|: {summary['reward_selected']['mean_abs_curvature_mm_inv']:.6f} mm^-1",
        f"- Nonzero Bezier refinements: {summary['reward_selected']['curved_case_count']} / {len(rows)}",
        f"- Scale fallbacks: {summary['reward_selected']['scale_fallback_count']} / {len(rows)}",
        "",
        "## Sample-GT oracle over same candidates",
        f"- Resection Dice mean/median: {summary['sample_oracle']['dice_mean']:.4f} / {summary['sample_oracle']['dice_median']:.4f}",
        f"- Resection Dice min/max: {summary['sample_oracle']['dice_min']:.4f} / {summary['sample_oracle']['dice_max']:.4f}",
        f"- Thresholds: {summary['sample_oracle']['thresholds']}",
        f"- Pred resection ratio mean: {summary['sample_oracle']['pred_ratio_mean']:.4f}",
        f"- Tumor coverage mean: {summary['sample_oracle']['tumor_coverage_mean']:.4f}",
        "",
        f"Scale counts: {summary['scale_counts']}",
        f"Reward selected surface counts: {summary['reward_surface_count']}",
        f"Oracle selected surface counts: {summary['oracle_surface_count']}",
        "",
        "Outputs:",
        f"- {args.output_dir / 'surface_planner_results.csv'}",
        f"- {args.output_dir / 'reward_selected_surfaces'}",
        f"- {args.output_dir / 'sample_oracle_surfaces'}",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter_root", type=Path, default=FILTER_ROOT)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--case_id", action="append", default=[])
    parser.add_argument("--cases_file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tumor_margin_mm", type=float, default=5.0)
    parser.add_argument("--curve_lateral_padding_mm", type=float, default=5.0)
    parser.add_argument("--curve_smoothness", type=float, default=3.0)
    parser.add_argument("--curve_constraint_bins", type=int, default=20)
    parser.add_argument("--curve_refine_iterations", type=int, default=4)
    parser.add_argument("--curvature_sample_step_mm", type=float, default=0.75)
    parser.add_argument("--reward_sample_points", type=int, default=12000)
    parser.add_argument("--stability_sample_points", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=250000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.case_id)
    if args.cases_file:
        requested.update(args.cases_file.read_text(encoding="utf-8").split())
    cases = sorted(
        path
        for path in args.filter_root.iterdir()
        if path.is_dir() and (path / "raw_pointclouds.npz").exists()
    )
    if requested:
        cases = [path for path in cases if path.name in requested]
    if args.limit:
        cases = cases[: args.limit]
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    print(f"Final rule planner cases: {len(cases)}")
    print(
        "Bezier refinement: "
        f"margin={args.tumor_margin_mm:g}mm "
        f"smoothness={args.curve_smoothness:g} "
        f"curvature_step={args.curvature_sample_step_mm:g}mm"
    )
    print(f"output_dir: {args.output_dir}")
    for idx, case_dir in enumerate(cases, start=1):
        try:
            row = process_case(case_dir, args)
        except Exception as exc:
            row = {"case_id": case_dir.name, "error": repr(exc)}
        rows.append(row)
        write_csv(rows, args.output_dir / "surface_planner_results.partial.csv")
        print(
            f"[{idx:03d}/{len(cases):03d}] {case_dir.name} "
            f"scale={row.get('predicted_scale', 'ERR')} "
            f"reward_dice={row.get('reward_full_resection_dice', 'ERR')} "
            f"margin={row.get('tumor_margin_min_mm', 'ERR')} "
            f"policy={row.get('selection_policy', 'ERR')} "
            f"oracle_dice={row.get('oracle_full_resection_dice', 'ERR')} "
            f"cand={row.get('candidate_count', 'NA')} "
            f"error={row.get('error', '')}",
            flush=True,
        )
    elapsed = time.time() - t0
    summarize([r for r in rows if "error" not in r], args, elapsed)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()

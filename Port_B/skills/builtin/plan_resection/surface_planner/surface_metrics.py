"""Paper-facing geometric metrics for Bezier resection surfaces."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from curved_refinement import surface_basis_1d, surface_control_grid


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = np.searchsorted(cumulative, quantile * cumulative[-1], side="left")
    return float(sorted_values[index])


def surface_curvature_metrics(
    surface: dict[str, Any],
    liver_volume: np.ndarray,
    inverse_affine: np.ndarray,
    sample_step_mm: float,
) -> dict[str, float]:
    control = surface_control_grid(surface)
    ref = surface["reference_plane"]
    u0, u1 = [float(value) for value in ref["u_range_mm"]]
    v0, v1 = [float(value) for value in ref["v_range_mm"]]
    count_u = max(41, int(math.ceil((u1 - u0) / sample_step_mm)) + 1)
    count_v = max(41, int(math.ceil((v1 - v0) / sample_step_mm)) + 1)
    parameter_u = np.linspace(0.0, 1.0, count_u)
    parameter_v = np.linspace(0.0, 1.0, count_v)
    basis_u = surface_basis_1d(surface, parameter_u)
    basis_v = surface_basis_1d(surface, parameter_v)
    height = np.einsum("ui,ij,vj->uv", basis_u, control, basis_v)
    u_mm = np.linspace(u0, u1, count_u)
    v_mm = np.linspace(v0, v1, count_v)

    h_u, h_v = np.gradient(height, u_mm, v_mm, edge_order=2)
    h_uu = np.gradient(h_u, u_mm, axis=0, edge_order=2)
    h_uv = np.gradient(h_u, v_mm, axis=1, edge_order=2)
    h_vv = np.gradient(h_v, v_mm, axis=1, edge_order=2)
    normalizer = np.sqrt(1.0 + h_u**2 + h_v**2)
    second_u = h_uu / normalizer
    mixed = h_uv / normalizer
    second_v = h_vv / normalizer
    first_u = 1.0 + h_u**2
    first_uv = h_u * h_v
    first_v = 1.0 + h_v**2
    mean_curvature = (
        second_u * first_v - 2.0 * mixed * first_uv + second_v * first_u
    ) / np.maximum(2.0 * (first_u * first_v - first_uv**2), 1e-12)

    origin = np.asarray(ref["origin_mm"], dtype=np.float64)
    axis_u = np.asarray(ref["u_axis_world"], dtype=np.float64)
    axis_v = np.asarray(ref["v_axis_world"], dtype=np.float64)
    normal = np.asarray(ref["normal_world"], dtype=np.float64)
    world_points = (
        origin[None, None, :]
        + u_mm[:, None, None] * axis_u[None, None, :]
        + v_mm[None, :, None] * axis_v[None, None, :]
        + height[:, :, None] * normal[None, None, :]
    )
    homogeneous = np.concatenate(
        [world_points.reshape(-1, 3), np.ones((world_points.size // 3, 1))], axis=1
    )
    ijk_world = np.einsum("ni,ji->nj", homogeneous, inverse_affine)
    ijk = np.rint(ijk_world[:, :3]).astype(np.int64)
    valid = np.all(ijk >= 0, axis=1) & np.all(
        ijk < np.asarray(liver_volume.shape)[None, :], axis=1
    )
    inside = np.zeros(len(ijk), dtype=bool)
    inside[valid] = liver_volume[tuple(ijk[valid].T)]
    inside = inside.reshape(height.shape)
    if not np.any(inside):
        return {
            "weighted_abs_curvature_sum": 0.0,
            "area_mm2": 0.0,
            "p95_abs_curvature_mm_inv": math.nan,
        }

    delta_u = (u1 - u0) / max(count_u - 1, 1)
    delta_v = (v1 - v0) / max(count_v - 1, 1)
    area_weights = normalizer * delta_u * delta_v
    internal_weights = area_weights * inside
    absolute_curvature = np.abs(mean_curvature)
    return {
        "weighted_abs_curvature_sum": float(
            np.sum(absolute_curvature * internal_weights)
        ),
        "area_mm2": float(np.sum(internal_weights)),
        "p95_abs_curvature_mm_inv": weighted_quantile(
            absolute_curvature[inside], internal_weights[inside], 0.95
        ),
    }


def candidate_curvature_metrics(
    surfaces: list[dict[str, Any]],
    liver_ijk: np.ndarray,
    affine: np.ndarray,
    sample_step_mm: float = 0.75,
) -> dict[str, float]:
    shape = tuple((np.max(liver_ijk.astype(np.int64), axis=0) + 1).astype(int))
    liver_volume = np.zeros(shape, dtype=bool)
    liver_volume[tuple(liver_ijk.T)] = True
    inverse_affine = np.linalg.inv(affine)
    per_surface = [
        surface_curvature_metrics(
            surface,
            liver_volume,
            inverse_affine,
            sample_step_mm,
        )
        for surface in surfaces
    ]
    total_area = sum(metric["area_mm2"] for metric in per_surface)
    finite_p95 = [
        metric["p95_abs_curvature_mm_inv"]
        for metric in per_surface
        if math.isfinite(metric["p95_abs_curvature_mm_inv"])
    ]
    return {
        "mean_abs_curvature_mm_inv": sum(
            metric["weighted_abs_curvature_sum"] for metric in per_surface
        )
        / max(total_area, 1e-12),
        "p95_abs_curvature_mm_inv": max(finite_p95) if finite_p95 else math.nan,
        "surface_area_mm2": total_area,
    }

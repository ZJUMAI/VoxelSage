"""Constraint-based bicubic Bezier refinement for rule-generated surfaces."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion


def bernstein3(t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return np.stack([u**3, 3.0 * u**2 * t, 3.0 * u * t**2, t**3], axis=1)


def bernstein3_derivative(t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return np.stack(
        [
            -3.0 * u**2,
            3.0 * u * (1.0 - 3.0 * t),
            3.0 * t * (2.0 - 3.0 * t),
            3.0 * t**2,
        ],
        axis=1,
    )


def surface_control_grid(surface: dict[str, Any]) -> np.ndarray:
    grid = np.asarray(surface["height_control_4x4_mm"], dtype=np.float64)
    if grid.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 Bezier height grid, found {grid.shape}")
    return grid


def set_surface_control_grid(surface: dict[str, Any], grid: np.ndarray) -> None:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 Bezier height grid, found {grid.shape}")
    surface["height_control_4x4_mm"] = grid.tolist()


def surface_basis_1d(
    surface: dict[str, Any],
    t: np.ndarray,
    derivative: int = 0,
) -> np.ndarray:
    surface_control_grid(surface)
    clipped = np.clip(t, 0.0, 1.0)
    if derivative == 0:
        return bernstein3(clipped)
    if derivative == 1:
        return bernstein3_derivative(clipped)
    raise ValueError("Only first derivatives are defined for bicubic Bezier refinement")


def project_surface(points: np.ndarray, surface: dict[str, Any]) -> dict[str, np.ndarray]:
    ref = surface["reference_plane"]
    origin = np.asarray(ref["origin_mm"], dtype=np.float64)
    normal = np.asarray(ref["normal_world"], dtype=np.float64)
    u_axis = np.asarray(ref["u_axis_world"], dtype=np.float64)
    v_axis = np.asarray(ref["v_axis_world"], dtype=np.float64)
    u0, u1 = [float(value) for value in ref["u_range_mm"]]
    v0, v1 = [float(value) for value in ref["v_range_mm"]]
    vector = points.astype(np.float64) - origin
    uu = vector @ u_axis
    vv = vector @ v_axis
    nn = vector @ normal
    du = max(u1 - u0, 1e-8)
    dv = max(v1 - v0, 1e-8)
    us = (uu - u0) / du
    vs = (vv - v0) / dv
    inside = (us >= 0.0) & (us <= 1.0) & (vs >= 0.0) & (vs <= 1.0)
    basis_u = surface_basis_1d(surface, us)
    basis_v = surface_basis_1d(surface, vs)
    derivative_u = surface_basis_1d(surface, us, derivative=1) / du
    derivative_v = surface_basis_1d(surface, vs, derivative=1) / dv
    return {
        "inside": inside,
        "uu": uu,
        "vv": vv,
        "nn": nn,
        "us": us,
        "vs": vs,
        "basis": np.einsum("ni,nj->nij", basis_u, basis_v).reshape(-1, 16),
        "basis_u": np.einsum("ni,nj->nij", derivative_u, basis_v).reshape(-1, 16),
        "basis_v": np.einsum("ni,nj->nij", basis_u, derivative_v).reshape(-1, 16),
    }


def surface_clearance(points: np.ndarray, surface: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    projection = project_surface(points, surface)
    control = surface_control_grid(surface).reshape(-1)
    vertical = projection["basis"] @ control - projection["nn"]
    gradient_u = projection["basis_u"] @ control
    gradient_v = projection["basis_v"] @ control
    vertical /= np.sqrt(1.0 + gradient_u**2 + gradient_v**2)
    clearance = np.full(len(points), -np.inf, dtype=np.float64)
    clearance[projection["inside"]] = vertical[projection["inside"]]
    return clearance, projection["inside"]


def candidate_clearance(points: np.ndarray, surfaces: list[dict[str, Any]]) -> np.ndarray:
    if not surfaces:
        return np.full(len(points), -np.inf, dtype=np.float64)
    values = [surface_clearance(points, surface)[0] for surface in surfaces]
    return np.max(np.stack(values, axis=1), axis=1)


def normalized_outside_distance(points: np.ndarray, surface: dict[str, Any]) -> np.ndarray:
    projection = project_surface(points, surface)
    delta_u = np.maximum(np.maximum(-projection["us"], projection["us"] - 1.0), 0.0)
    delta_v = np.maximum(np.maximum(-projection["vs"], projection["vs"] - 1.0), 0.0)
    return np.sqrt(delta_u**2 + delta_v**2)


def assign_points_to_surfaces(
    points: np.ndarray,
    surfaces: list[dict[str, Any]],
) -> np.ndarray:
    clearances = np.stack(
        [surface_clearance(points, surface)[0] for surface in surfaces], axis=1
    )
    assignment = np.argmax(clearances, axis=1)
    uncovered = ~np.isfinite(np.max(clearances, axis=1))
    if np.any(uncovered):
        outside = np.stack(
            [normalized_outside_distance(points[uncovered], surface) for surface in surfaces],
            axis=1,
        )
        assignment[uncovered] = np.argmin(outside, axis=1)
    return assignment


def expand_surface_extent(
    surface: dict[str, Any],
    points: np.ndarray,
    lateral_padding_mm: float,
) -> None:
    if len(points) == 0:
        return
    ref = surface["reference_plane"]
    origin = np.asarray(ref["origin_mm"], dtype=np.float64)
    u_axis = np.asarray(ref["u_axis_world"], dtype=np.float64)
    v_axis = np.asarray(ref["v_axis_world"], dtype=np.float64)
    vector = points.astype(np.float64) - origin
    uu = vector @ u_axis
    vv = vector @ v_axis
    u0, u1 = [float(value) for value in ref["u_range_mm"]]
    v0, v1 = [float(value) for value in ref["v_range_mm"]]
    ref["u_range_mm"] = [
        min(u0, float(np.min(uu)) - lateral_padding_mm),
        max(u1, float(np.max(uu)) + lateral_padding_mm),
    ]
    ref["v_range_mm"] = [
        min(v0, float(np.min(vv)) - lateral_padding_mm),
        max(v1, float(np.max(vv)) + lateral_padding_mm),
    ]


def second_difference_matrix() -> np.ndarray:
    rows: list[np.ndarray] = []
    for i in range(1, 3):
        for j in range(4):
            row = np.zeros(16, dtype=np.float64)
            row[(i - 1) * 4 + j] = 1.0
            row[i * 4 + j] = -2.0
            row[(i + 1) * 4 + j] = 1.0
            rows.append(row)
    for i in range(4):
        for j in range(1, 3):
            row = np.zeros(16, dtype=np.float64)
            row[i * 4 + j - 1] = 1.0
            row[i * 4 + j] = -2.0
            row[i * 4 + j + 1] = 1.0
            rows.append(row)
    return np.stack(rows, axis=0)


def control_weights() -> np.ndarray:
    weights = np.ones(16, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if i in {0, 3} and j in {0, 3}:
                weights[i * 4 + j] = 16.0
            elif i in {0, 3} or j in {0, 3}:
                weights[i * 4 + j] = 4.0
    return weights


def tumor_boundary_points(tumor_ijk: np.ndarray, tumor_xyz: np.ndarray) -> np.ndarray:
    if len(tumor_ijk) == 0:
        return tumor_xyz
    coordinates = tumor_ijk.astype(np.int64)
    low = np.min(coordinates, axis=0) - 1
    high = np.max(coordinates, axis=0) + 1
    shape = tuple((high - low + 1).tolist())
    if int(np.prod(shape, dtype=np.int64)) <= 100_000_000:
        local = coordinates - low
        mask = np.zeros(shape, dtype=bool)
        mask[tuple(local.T)] = True
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = True
        structure[0, 1, 1] = structure[2, 1, 1] = True
        structure[1, 0, 1] = structure[1, 2, 1] = True
        structure[1, 1, 0] = structure[1, 1, 2] = True
        eroded = binary_erosion(mask, structure=structure, border_value=0)
        return tumor_xyz[~eroded[tuple(local.T)]]

    occupied = {tuple(int(value) for value in coordinate) for coordinate in coordinates.tolist()}
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    keep = np.fromiter(
        (
            any(
                (int(point[0]) + di, int(point[1]) + dj, int(point[2]) + dk)
                not in occupied
                for di, dj, dk in offsets
            )
            for point in coordinates
        ),
        dtype=bool,
        count=len(coordinates),
    )
    return tumor_xyz[keep]


def compress_constraints(
    projection: dict[str, np.ndarray],
    target: np.ndarray,
    control: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    us = np.clip(projection["us"], 0.0, 1.0)
    vs = np.clip(projection["vs"], 0.0, 1.0)
    basis = projection["basis"]
    deficit = target - basis @ control
    bin_u = np.minimum((us * bins).astype(np.int64), bins - 1)
    bin_v = np.minimum((vs * bins).astype(np.int64), bins - 1)
    bucket = bin_u * bins + bin_v
    keep: list[int] = []
    for value in np.unique(bucket):
        indices = np.where(bucket == value)[0]
        keep.append(int(indices[np.argmax(deficit[indices])]))
    worst = np.argsort(deficit)[-min(64, len(deficit)) :]
    keep.extend(int(index) for index in worst)
    selected = sorted(set(keep))
    return basis[selected], target[selected]


def solve_surface_controls(
    surface: dict[str, Any],
    tumor_points: np.ndarray,
    margin_mm: float,
    smoothness: float,
    bins: int,
    max_iterations: int,
) -> dict[str, Any]:
    if len(tumor_points) == 0:
        return {"success": True, "n_points": 0, "iterations": 0}
    projection = project_surface(tumor_points, surface)
    if not bool(np.all(projection["inside"])):
        return {
            "success": False,
            "reason": "assigned_points_outside_extent",
            "n_points": len(tumor_points),
        }

    initial_control = surface_control_grid(surface).reshape(-1)
    initial_gradient_u = projection["basis_u"] @ initial_control
    initial_gradient_v = projection["basis_v"] @ initial_control
    initial_clearance = (
        projection["basis"] @ initial_control - projection["nn"]
    ) / np.sqrt(1.0 + initial_gradient_u**2 + initial_gradient_v**2)
    initial_minimum = float(np.min(initial_clearance))
    if initial_minimum >= margin_mm - 0.02:
        return {
            "success": True,
            "n_points": len(tumor_points),
            "iterations": 0,
            "projection_steps": 0,
            "min_clearance_mm": initial_minimum,
            "control_delta_rms_mm": 0.0,
            "control_delta_max_mm": 0.0,
        }

    control = initial_control.copy()
    difference = second_difference_matrix()
    bending_energy = np.einsum("ki,kj->ij", difference, difference)
    metric = np.diag(control_weights()) + smoothness * bending_energy
    metric += np.eye(16, dtype=np.float64) * 1e-6
    metric_inverse = np.linalg.inv(metric)
    projection_steps = 0
    iteration = 0
    for iteration in range(max(12, max_iterations * 12)):
        gradient_u = projection["basis_u"] @ control
        gradient_v = projection["basis_v"] @ control
        tangent_factor = np.sqrt(1.0 + gradient_u**2 + gradient_v**2)
        target = projection["nn"] + margin_mm * tangent_factor
        matrix, boundary = compress_constraints(projection, target, control, bins)
        violation = boundary - matrix @ control
        if float(np.max(violation)) <= 0.01:
            effective = (projection["basis"] @ control - projection["nn"]) / tangent_factor
            if float(np.min(effective)) >= margin_mm - 0.02:
                break
        for constraint_index in np.argsort(violation)[::-1][: min(96, len(violation))]:
            deficit = float(boundary[constraint_index] - matrix[constraint_index] @ control)
            if deficit <= 0.005:
                continue
            direction = metric_inverse @ matrix[constraint_index]
            denominator = float(matrix[constraint_index] @ direction)
            if denominator > 1e-12:
                control += (deficit / denominator) * direction
                projection_steps += 1
        if not np.all(np.isfinite(control)):
            control = initial_control.copy()
            break

    set_surface_control_grid(surface, control.reshape(4, 4))
    final_clearance, _ = surface_clearance(tumor_points, surface)
    minimum_clearance = float(np.min(final_clearance))
    if minimum_clearance < margin_mm - 0.02:
        set_surface_control_grid(
            surface,
            control.reshape(4, 4) + margin_mm - minimum_clearance + 0.02,
        )
        final_clearance, _ = surface_clearance(tumor_points, surface)
        minimum_clearance = float(np.min(final_clearance))

    final_control = surface_control_grid(surface).reshape(-1)
    return {
        "success": minimum_clearance >= margin_mm - 0.05,
        "n_points": len(tumor_points),
        "iterations": iteration + 1,
        "projection_steps": projection_steps,
        "min_clearance_mm": minimum_clearance,
        "control_delta_rms_mm": float(
            np.sqrt(np.mean((final_control - initial_control) ** 2))
        ),
        "control_delta_max_mm": float(np.max(np.abs(final_control - initial_control))),
    }


def refine_candidate(
    candidate: dict[str, Any],
    tumor_points: np.ndarray,
    margin_mm: float = 5.0,
    lateral_padding_mm: float = 5.0,
    smoothness: float = 3.0,
    bins: int = 20,
    max_iterations: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    refined = copy.deepcopy(candidate)
    surfaces = refined["surfaces"]
    assignment = assign_points_to_surfaces(tumor_points, surfaces)
    for surface_index, surface in enumerate(surfaces):
        expand_surface_extent(
            surface,
            tumor_points[assignment == surface_index],
            lateral_padding_mm,
        )

    details: list[dict[str, Any]] = []
    for round_index in range(2):
        assignment = assign_points_to_surfaces(tumor_points, surfaces)
        for surface_index, surface in enumerate(surfaces):
            detail = solve_surface_controls(
                surface,
                tumor_points[assignment == surface_index],
                margin_mm,
                smoothness,
                bins,
                max_iterations,
            )
            detail.update({"round": round_index, "surface_index": surface_index})
            details.append(detail)
        clearance = candidate_clearance(tumor_points, surfaces)
        if bool(np.all(np.isfinite(clearance))) and float(np.min(clearance)) >= margin_mm - 0.05:
            break

    clearance = candidate_clearance(tumor_points, surfaces)
    finite = clearance[np.isfinite(clearance)]
    success = len(finite) == len(clearance) and float(np.min(finite)) >= margin_mm - 0.05
    source_name = candidate["name"]
    refined["name"] = f"{source_name}__curve_m{margin_mm:g}_bezier4"
    for surface in surfaces:
        metadata = surface.setdefault("rule_metadata", {})
        metadata["candidate_name"] = refined["name"]
        metadata["source_candidate_name"] = source_name
        metadata["curved_initialization"] = {
            "method": "minimum_displacement_margin_constrained_bicubic_bezier",
            "basis": "bicubic_bezier",
            "control_grid_size": [4, 4],
            "tumor_margin_mm": margin_mm,
            "clearance": "tangent_corrected_local_distance",
            "lateral_padding_mm": lateral_padding_mm,
            "smoothness": smoothness,
        }
    return refined, {
        "success": success,
        "source_candidate_name": source_name,
        "min_clearance_mm": float(np.min(finite)) if len(finite) else -math.inf,
        "p05_clearance_mm": float(np.percentile(finite, 5)) if len(finite) else -math.inf,
        "details": details,
    }

from dataclasses import asdict, dataclass
import heapq
from itertools import product
import json
import os
from pathlib import Path
import uuid

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from Tool_Box.vessel_optimization_contract import (
    AUDIT_SCHEMA,
    AUDIT_SCHEMA_VERSION,
    MAX_ALLOWED_GAP_MM,
    sha256_file,
    validate_max_gap_mm,
)


OPTIMIZATION_PASSES = 3


@dataclass(frozen=True)
class VesselOptimizationConfig:
    max_gap_mm: float = MAX_ALLOWED_GAP_MM
    max_direction_angle_deg: float = 30.0
    min_liver_path_fraction: float = 0.95
    direction_trace_mm: float = 3.0
    avoid_tumor: bool = True
    avoid_other_vessel: bool = True
    min_tangent_points: int = 4
    min_tangent_trace_extent_mm: float = 2.0
    min_tangent_eigenvalue_ratio: float = 4.0
    max_tube_radius_mm: float = 1.5
    max_radius_mismatch_ratio: float = 0.5
    max_endpoint_contact_distance_mm: float = 2.0
    max_added_volume_mm3_per_connection: float = 24.0
    max_connection_growth_fraction: float = 0.20
    max_cumulative_growth_fraction: float = 0.20
    score_distance_weight: float = 0.25
    score_direction_a_weight: float = 0.25
    score_direction_b_weight: float = 0.25
    score_radius_mismatch_weight: float = 0.25


@dataclass
class VesselOptimizationResult:
    optimized_mask: np.ndarray
    accepted_connections: list[dict]
    rejected_candidate_counts: dict[str, int]
    components_before: int
    components_after: int
    warnings: list[str]


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    try:
        if first.exists() and second.exists() and os.path.samefile(first, second):
            return True
    except OSError:
        pass
    return os.path.normcase(str(first.resolve(strict=False))) == os.path.normcase(
        str(second.resolve(strict=False))
    )


def _validate_mask_geometry(
    image: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    *,
    label: str,
) -> None:
    if len(image.shape) != 3:
        raise ValueError(f"{label} mask must be 3D, got shape {image.shape}")
    if image.shape != reference.shape:
        raise ValueError(
            f"{label} mask shape {image.shape} does not match vessel shape "
            f"{reference.shape}"
        )
    if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-4):
        raise ValueError(f"{label} mask affine does not match vessel affine")
    spatial_unit = image.header.get_xyzt_units()[0]
    if spatial_unit != "mm":
        raise ValueError(
            f"{label} mask spatial units must be millimeters, got "
            f"{spatial_unit!r}"
        )


def _validate_vessel_affine(
    image: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, str]:
    """Validate a millimeter, orthogonal affine and return voxel sizes."""
    affine = np.asarray(image.affine, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("vessel affine must be a finite 4x4 matrix")
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError(f"vessel affine has invalid voxel sizes: {spacing}")
    normalized = linear / spacing
    gram = normalized.T @ normalized
    if not np.allclose(gram, np.eye(3), rtol=0.0, atol=1e-5):
        raise ValueError(
            "vessel affine axes must be orthogonal; sheared affines are "
            "not supported"
        )
    if abs(float(np.linalg.det(linear))) <= np.finfo(float).eps:
        raise ValueError("vessel affine must be invertible")
    spatial_unit = image.header.get_xyzt_units()[0]
    if spatial_unit != "mm":
        raise ValueError(
            "vessel spatial units must be millimeters; "
            f"got {spatial_unit!r}"
        )
    return spacing, spatial_unit


def _load_matching_mask(
    path: Path,
    reference: nib.spatialimages.SpatialImage,
    *,
    label: str,
) -> np.ndarray:
    image = nib.load(str(path))
    _validate_mask_geometry(image, reference, label=label)
    return np.asanyarray(image.dataobj) != 0


def _json_safe(value, *, warnings=None, path="$"):
    """Return a strict-JSON-safe value and report replaced non-finite floats."""
    if warnings is None:
        warnings = []
    if isinstance(value, dict):
        return {
            str(key): _json_safe(
                item,
                warnings=warnings,
                path=f"{path}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, warnings=warnings, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), warnings=warnings, path=path)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), warnings=warnings, path=path)
    if isinstance(value, float) and not np.isfinite(value):
        warning = f"non_finite_audit_value_replaced_with_null:{path}"
        if warning not in warnings:
            warnings.append(warning)
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_name(report_path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temp_path.read_text(encoding="utf-8"))
        os.replace(temp_path, report_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_optimized_nifti_atomic(
    optimized_mask: np.ndarray,
    source_image: nib.spatialimages.SpatialImage,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp.nii.gz"
    )
    header = source_image.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(
        np.asarray(optimized_mask, dtype=np.uint8),
        source_image.affine,
        header=header,
    )
    try:
        nib.save(image, str(temp_path))
        reloaded = nib.load(str(temp_path))
        if reloaded.shape != source_image.shape:
            raise ValueError(
                f"temporary output shape {reloaded.shape} does not match source "
                f"shape {source_image.shape}"
            )
        if not np.allclose(
            reloaded.affine, source_image.affine, rtol=0.0, atol=1e-4
        ):
            raise ValueError("temporary output affine does not match source affine")
        if np.dtype(reloaded.get_data_dtype()) != np.dtype(np.uint8):
            raise ValueError("temporary output is not uint8")
        values = np.unique(np.asanyarray(reloaded.dataobj))
        if not set(values.tolist()).issubset({0, 1}):
            raise ValueError("temporary output is not binary")
        raw = np.asanyarray(source_image.dataobj) != 0
        optimized = np.asanyarray(reloaded.dataobj) != 0
        if np.any(raw & ~optimized):
            raise ValueError("temporary output is not a raw-mask superset")
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


_NEIGHBOR_OFFSETS = tuple(
    np.asarray(offset, dtype=int)
    for offset in product((-1, 0, 1), repeat=3)
    if offset != (0, 0, 0)
)
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Return a boolean 3D skeleton in the input array's ijk coordinates."""
    return np.asarray(skeletonize(np.asarray(mask, dtype=bool), method="lee"), dtype=bool)


def _find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Return ijk coordinates of skeleton voxels with exactly one 26-neighbor."""
    skeleton = np.asarray(skeleton, dtype=bool)
    neighbor_count = ndimage.convolve(
        skeleton.astype(np.uint8), _CONNECTIVITY_26, mode="constant", cval=0
    )
    neighbor_count -= skeleton.astype(np.uint8)
    return np.argwhere(skeleton & (neighbor_count == 1))


def _trace_outward_tangent(
    endpoint,
    skeleton,
    spacing,
    trace_mm,
    *,
    min_points=4,
    min_trace_extent_mm=2.0,
    min_eigenvalue_ratio=4.0,
) -> np.ndarray | None:
    """Estimate a stable outward tangent by PCA in physical coordinates."""
    endpoint_ijk = np.asarray(endpoint, dtype=int)
    skeleton = np.asarray(skeleton, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    shape = np.asarray(skeleton.shape, dtype=int)
    start = tuple(int(value) for value in endpoint_ijk)
    distances = {start: 0.0}
    queue = [(0.0, start)]
    traced = [endpoint_ijk.astype(float)]

    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances[current]:
            continue
        if current != start:
            traced.append(np.asarray(current, dtype=float))
        current_ijk = np.asarray(current, dtype=int)
        skeleton_neighbor_count = 0
        for offset in _NEIGHBOR_OFFSETS:
            neighbor_ijk = current_ijk + offset
            if np.any(neighbor_ijk < 0) or np.any(neighbor_ijk >= shape):
                continue
            neighbor = tuple(int(value) for value in neighbor_ijk)
            if not skeleton[neighbor]:
                continue
            skeleton_neighbor_count += 1
            new_distance = distance + float(np.linalg.norm(offset * spacing))
            if new_distance > trace_mm:
                continue
            if new_distance < distances.get(neighbor, np.inf):
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))
        if current != start and skeleton_neighbor_count > 2:
            return None

    if len(traced) < int(min_points):
        return None
    points_mm = np.asarray(traced, dtype=float) * spacing
    endpoint_mm = endpoint_ijk.astype(float) * spacing
    trace_extent_mm = float(
        np.max(np.linalg.norm(points_mm - endpoint_mm, axis=1))
    )
    if trace_extent_mm < float(min_trace_extent_mm):
        return None
    centered = points_mm - np.mean(points_mm, axis=0)
    covariance = centered.T @ centered / max(len(points_mm) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    dominant = float(eigenvalues[order[-1]])
    secondary = float(eigenvalues[order[-2]])
    if dominant <= np.finfo(float).eps:
        return None
    dominance = dominant / max(secondary, np.finfo(float).eps)
    if dominance < float(min_eigenvalue_ratio):
        return None
    tangent = np.asarray(eigenvectors[:, order[-1]], dtype=float)
    interior_centroid_mm = np.mean(points_mm[1:], axis=0)
    outward_hint = endpoint_mm - interior_centroid_mm
    if float(np.dot(tangent, outward_hint)) < 0.0:
        tangent = -tangent
    norm = float(np.linalg.norm(tangent))
    return tangent / norm if norm > 0.0 else None


def _candidate_pairs(
    endpoints,
    spacing,
    max_gap_mm,
) -> list[tuple[int, int, float]]:
    """Return endpoint index pairs within max_gap_mm in physical coordinates."""
    endpoints = np.asarray(endpoints)
    if len(endpoints) < 2:
        return []
    physical_endpoints = endpoints.astype(float) * np.asarray(spacing, dtype=float)
    pairs = cKDTree(physical_endpoints).query_pairs(
        r=float(max_gap_mm), output_type="ndarray"
    )
    candidates = [
        (
            int(first),
            int(second),
            float(np.linalg.norm(physical_endpoints[first] - physical_endpoints[second])),
        )
        for first, second in pairs
    ]
    return sorted(candidates, key=lambda candidate: (candidate[2], candidate[0], candidate[1]))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Return the unsigned angle in degrees between physical-coordinate vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _sample_hermite(p0, p1, t0, t1, spacing) -> np.ndarray:
    """Sample a physical-coordinate Hermite centerline at sub-voxel spacing."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    distance = float(np.linalg.norm(p1 - p0))
    count = max(2, int(np.ceil(distance / (float(np.min(spacing)) * 0.5))) + 1)
    u = np.linspace(0.0, 1.0, count)[:, None]
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    scale = distance * 0.5
    return h00 * p0 + h10 * (t0 * scale) + h01 * p1 + h11 * (-t1 * scale)


def _curve_indices(curve_mm, spacing, shape) -> np.ndarray:
    """Return unique in-bounds ijk voxel centers nearest to a physical curve."""
    spacing = np.asarray(spacing, dtype=float)
    shape_array = np.asarray(shape, dtype=int)
    indices = np.rint(np.asarray(curve_mm, dtype=float) / spacing).astype(int)
    in_bounds = np.all((indices >= 0) & (indices < shape_array), axis=1)
    return np.unique(indices[in_bounds], axis=0)


def _curve_liver_fraction(curve_mm, spacing, liver_mask) -> float:
    """Count every out-of-bounds physical curve sample as outside the liver."""
    liver = np.asarray(liver_mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    indices = np.rint(np.asarray(curve_mm, dtype=float) / spacing).astype(int)
    shape_array = np.asarray(liver.shape, dtype=int)
    in_bounds = np.all((indices >= 0) & (indices < shape_array), axis=1)
    sample_is_in_liver = np.zeros(len(indices), dtype=bool)
    bounded_indices = indices[in_bounds]
    sample_is_in_liver[in_bounds] = liver[tuple(bounded_indices.T)]
    return float(np.mean(sample_is_in_liver))


def _rasterize_tube(curve_mm, radius_mm, spacing, shape) -> np.ndarray:
    """Return ijk indices of a conservative tube around a physical curve."""
    curve_mm = np.asarray(curve_mm, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape_array = np.asarray(shape, dtype=int)
    centerline_indices = _curve_indices(curve_mm, spacing, shape_array)

    radius_mm = float(radius_mm)
    if radius_mm < float(np.min(spacing)):
        return centerline_indices

    lower = np.floor((np.min(curve_mm, axis=0) - radius_mm) / spacing).astype(int)
    upper = np.ceil((np.max(curve_mm, axis=0) + radius_mm) / spacing).astype(int)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, shape_array - 1)
    ranges = [np.arange(start, stop + 1) for start, stop in zip(lower, upper)]
    nearby_indices = np.stack(
        np.meshgrid(*ranges, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    nearby_mm = nearby_indices.astype(float) * spacing
    distances, _ = cKDTree(curve_mm).query(nearby_mm, k=1)
    within_radius = distances <= radius_mm + np.finfo(float).eps * 16
    selected = nearby_indices[within_radius]
    return np.unique(np.concatenate((centerline_indices, selected), axis=0), axis=0)


_REJECTION_REASONS = (
    "distance",
    "same_component",
    "direction",
    "caliber_mismatch",
    "outside_liver",
    "tumor_collision",
    "other_vessel_collision",
    "growth_per_connection",
    "growth_cumulative",
    "third_component_collision",
    "short_loop",
    "conflict",
    "merge_failed",
    "postcondition_failed",
)


def _validate_array_contract(
    vessel: np.ndarray,
    spacing: np.ndarray,
    liver_mask,
    tumor_mask,
    other_vessel_mask,
    config: VesselOptimizationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_max_gap_mm(config.max_gap_mm)
    if vessel.ndim != 3:
        raise ValueError(f"vessel_mask must be 3D, got shape {vessel.shape}")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(
        spacing <= 0.0
    ):
        raise ValueError("spacing must contain three positive finite values")
    if liver_mask is None:
        raise ValueError("liver_mask is required for automatic reconnection")
    liver = np.asarray(liver_mask, dtype=bool)
    tumor = (
        np.zeros_like(vessel)
        if tumor_mask is None
        else np.asarray(tumor_mask, dtype=bool)
    )
    other_vessel = (
        np.zeros_like(vessel)
        if other_vessel_mask is None
        else np.asarray(other_vessel_mask, dtype=bool)
    )
    for label, mask in (
        ("liver_mask", liver),
        ("tumor_mask", tumor),
        ("other_vessel_mask", other_vessel),
    ):
        if mask.shape != vessel.shape:
            raise ValueError(
                f"{label} shape {mask.shape} does not match vessel_mask "
                f"shape {vessel.shape}"
            )
    if not 0.0 <= float(config.min_liver_path_fraction) <= 1.0:
        raise ValueError("min_liver_path_fraction must be within [0, 1]")
    positive_fields = {
        "max_direction_angle_deg": config.max_direction_angle_deg,
        "direction_trace_mm": config.direction_trace_mm,
        "min_tangent_trace_extent_mm": config.min_tangent_trace_extent_mm,
        "min_tangent_eigenvalue_ratio": config.min_tangent_eigenvalue_ratio,
        "max_tube_radius_mm": config.max_tube_radius_mm,
        "max_endpoint_contact_distance_mm": (
            config.max_endpoint_contact_distance_mm
        ),
        "max_added_volume_mm3_per_connection": (
            config.max_added_volume_mm3_per_connection
        ),
        "max_connection_growth_fraction": (
            config.max_connection_growth_fraction
        ),
        "max_cumulative_growth_fraction": (
            config.max_cumulative_growth_fraction
        ),
    }
    for label, value in positive_fields.items():
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{label} must be positive and finite")
    if int(config.min_tangent_points) < 3:
        raise ValueError("min_tangent_points must be at least 3")
    if not 0.0 <= float(config.max_radius_mismatch_ratio) <= 1.0:
        raise ValueError("max_radius_mismatch_ratio must be within [0, 1]")
    weights = np.asarray(
        [
            config.score_distance_weight,
            config.score_direction_a_weight,
            config.score_direction_b_weight,
            config.score_radius_mismatch_weight,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(
        weights > 0.0
    ):
        raise ValueError("candidate score weights must be finite and non-negative")
    return liver, tumor, other_vessel


def _working_roi_slices(
    vessel: np.ndarray,
    spacing: np.ndarray,
    config: VesselOptimizationConfig,
) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(vessel)
    margin_mm = (
        float(config.max_gap_mm)
        + float(config.max_tube_radius_mm)
        + float(np.max(spacing))
    )
    margin = np.ceil(margin_mm / spacing).astype(int) + 1
    lower = np.maximum(np.min(coordinates, axis=0) - margin, 0)
    upper = np.minimum(
        np.max(coordinates, axis=0) + margin + 1,
        np.asarray(vessel.shape, dtype=int),
    )
    return tuple(
        slice(int(start), int(stop)) for start, stop in zip(lower, upper)
    )


def _estimate_endpoint_radii(
    endpoints: np.ndarray,
    skeleton: np.ndarray,
    component_labels: np.ndarray,
    distance_map: np.ndarray,
    spacing: np.ndarray,
    trace_mm: float,
) -> np.ndarray:
    """Estimate caliber just inside each endpoint, not at its cut face."""
    if len(endpoints) == 0:
        return np.empty(0, dtype=float)
    half_voxel = 0.5 * float(np.min(spacing))
    radii = []
    margin = np.ceil(float(trace_mm) / spacing).astype(int)
    shape = np.asarray(skeleton.shape, dtype=int)
    for endpoint in endpoints:
        lower = np.maximum(endpoint - margin, 0)
        upper = np.minimum(endpoint + margin + 1, shape)
        slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(lower, upper)
        )
        local_skeleton = skeleton[slices]
        local_labels = component_labels[slices]
        coordinates = np.argwhere(
            local_skeleton
            & (local_labels == component_labels[tuple(endpoint)])
        )
        coordinates += lower
        physical_distances = np.linalg.norm(
            (coordinates - endpoint) * spacing,
            axis=1,
        )
        coordinates = coordinates[physical_distances <= trace_mm + 1e-8]
        local_radius = float(
            np.max(distance_map[tuple(coordinates.T)])
        )
        radii.append(max(0.0, local_radius - half_voxel))
    return np.asarray(radii, dtype=float)


def _neighbor_indices(indices: np.ndarray, shape) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 3), dtype=int)
    offsets = np.concatenate(
        (np.zeros((1, 3), dtype=int), np.asarray(_NEIGHBOR_OFFSETS)),
        axis=0,
    )
    neighbors = indices[:, None, :] + offsets[None, :, :]
    neighbors = neighbors.reshape(-1, 3)
    shape_array = np.asarray(shape, dtype=int)
    in_bounds = np.all(
        (neighbors >= 0) & (neighbors < shape_array),
        axis=1,
    )
    return np.unique(neighbors[in_bounds], axis=0)


def _normalized_candidate_score(
    *,
    distance_mm: float,
    angle_a: float,
    angle_b: float,
    radius_mismatch: float,
    config: VesselOptimizationConfig,
) -> tuple[float, dict[str, float]]:
    terms = {
        "distance": distance_mm / float(config.max_gap_mm),
        "direction_a": angle_a / float(config.max_direction_angle_deg),
        "direction_b": angle_b / float(config.max_direction_angle_deg),
        "radius_mismatch": radius_mismatch,
    }
    weights = {
        "distance": float(config.score_distance_weight),
        "direction_a": float(config.score_direction_a_weight),
        "direction_b": float(config.score_direction_b_weight),
        "radius_mismatch": float(config.score_radius_mismatch_weight),
    }
    total_weight = sum(weights.values())
    score = sum(terms[key] * weights[key] for key in terms) / total_weight
    return float(score), terms


def optimize_vessel_array(
    vessel_mask,
    spacing,
    *,
    liver_mask=None,
    tumor_mask=None,
    other_vessel_mask=None,
    config=VesselOptimizationConfig(),
) -> VesselOptimizationResult:
    """Reconnect only fully validated short gaps inside a tight working ROI."""
    vessel = np.asarray(vessel_mask, dtype=bool)
    spacing_array = np.asarray(spacing, dtype=float)
    liver, tumor, other_vessel = _validate_array_contract(
        vessel,
        spacing_array,
        liver_mask,
        tumor_mask,
        other_vessel_mask,
        config,
    )
    warnings = []
    if tumor_mask is None:
        warnings.append("tumor_constraint_unavailable")
    if other_vessel_mask is None:
        warnings.append("other_vessel_constraint_unavailable")
    rejected_counts = {reason: 0 for reason in _REJECTION_REASONS}
    if not np.any(vessel):
        return VesselOptimizationResult(
            optimized_mask=vessel.copy(),
            accepted_connections=[],
            rejected_candidate_counts=rejected_counts,
            components_before=0,
            components_after=0,
            warnings=warnings,
        )

    roi_slices = _working_roi_slices(vessel, spacing_array, config)
    roi_origin = np.asarray(
        [axis_slice.start for axis_slice in roi_slices],
        dtype=int,
    )
    work_vessel = vessel[roi_slices]
    work_liver = liver[roi_slices]
    work_tumor = tumor[roi_slices]
    work_other = other_vessel[roi_slices]
    skeleton = _skeletonize(work_vessel)
    endpoints = _find_endpoints(skeleton)
    component_labels, components_before = ndimage.label(
        work_vessel, _CONNECTIVITY_26
    )
    distance_map = ndimage.distance_transform_edt(
        work_vessel, sampling=spacing_array
    )
    endpoint_radii = _estimate_endpoint_radii(
        endpoints,
        skeleton,
        component_labels,
        distance_map,
        spacing_array,
        config.direction_trace_mm,
    )
    tangents = [
        _trace_outward_tangent(
            endpoint,
            skeleton,
            spacing_array,
            config.direction_trace_mm,
            min_points=config.min_tangent_points,
            min_trace_extent_mm=config.min_tangent_trace_extent_mm,
            min_eigenvalue_ratio=config.min_tangent_eigenvalue_ratio,
        )
        for endpoint in endpoints
    ]
    safe_candidates = []

    for first, second, distance_mm in _candidate_pairs(
        endpoints, spacing_array, config.max_gap_mm
    ):
        endpoint_a = endpoints[first]
        endpoint_b = endpoints[second]
        component_a = int(component_labels[tuple(endpoint_a)])
        component_b = int(component_labels[tuple(endpoint_b)])
        if component_a == component_b:
            rejected_counts["same_component"] += 1
            continue

        gap_vector = (endpoint_b - endpoint_a).astype(float) * spacing_array
        tangent_a = tangents[first]
        tangent_b = tangents[second]
        angle_a = (
            _angle_deg(tangent_a, gap_vector)
            if tangent_a is not None
            else np.nan
        )
        angle_b = (
            _angle_deg(tangent_b, -gap_vector)
            if tangent_b is not None
            else np.nan
        )
        if (
            not np.isfinite(angle_a)
            or not np.isfinite(angle_b)
            or angle_a > config.max_direction_angle_deg
            or angle_b > config.max_direction_angle_deg
        ):
            rejected_counts["direction"] += 1
            continue

        radius_a = float(endpoint_radii[first])
        radius_b = float(endpoint_radii[second])
        radius_denominator = max(radius_a, radius_b)
        radius_mismatch = (
            abs(radius_a - radius_b) / radius_denominator
            if radius_denominator > 0.0
            else 0.0
        )
        if radius_mismatch > config.max_radius_mismatch_ratio:
            rejected_counts["caliber_mismatch"] += 1
            continue
        radius_mm = min(
            radius_a,
            radius_b,
            float(config.max_tube_radius_mm),
        )
        endpoint_a_mm = endpoint_a.astype(float) * spacing_array
        endpoint_b_mm = endpoint_b.astype(float) * spacing_array
        curve_mm = _sample_hermite(
            endpoint_a_mm,
            endpoint_b_mm,
            tangent_a,
            tangent_b,
            spacing_array,
        )
        tube_indices = _rasterize_tube(
            curve_mm,
            radius_mm,
            spacing_array,
            work_vessel.shape,
        )
        liver_fraction = _curve_liver_fraction(
            curve_mm, spacing_array, work_liver
        )
        if liver_fraction < config.min_liver_path_fraction:
            rejected_counts["outside_liver"] += 1
            continue

        score, score_terms = _normalized_candidate_score(
            distance_mm=distance_mm,
            angle_a=angle_a,
            angle_b=angle_b,
            radius_mismatch=radius_mismatch,
            config=config,
        )
        safe_candidates.append(
            {
                "endpoint_indices": (first, second),
                "endpoints_ijk_local": (
                    tuple(endpoint_a.tolist()),
                    tuple(endpoint_b.tolist()),
                ),
                "endpoints_ijk": (
                    tuple((endpoint_a + roi_origin).tolist()),
                    tuple((endpoint_b + roi_origin).tolist()),
                ),
                "distance_mm": distance_mm,
                "direction_angles_deg": (angle_a, angle_b),
                "component_labels": (component_a, component_b),
                "endpoint_radii_mm": (radius_a, radius_b),
                "tube_radius_mm": radius_mm,
                "liver_path_fraction": liver_fraction,
                "radius_mismatch": radius_mismatch,
                "tube_indices": tube_indices,
                "score": score,
                "score_terms": score_terms,
            }
        )

    parent = np.arange(int(components_before) + 1, dtype=int)

    def find(label: int) -> int:
        label = int(label)
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = int(parent[label])
        return label

    def union(first_label: int, second_label: int) -> int:
        first_root = find(first_label)
        second_root = find(second_label)
        new_root = min(first_root, second_root)
        old_root = max(first_root, second_root)
        parent[old_root] = new_root
        return new_root

    optimized = work_vessel.copy()
    owner_labels = component_labels.copy()
    raw_volume_mm3 = (
        float(np.count_nonzero(work_vessel)) * float(np.prod(spacing_array))
    )
    cumulative_added_volume_mm3 = 0.0
    used_endpoints = set()
    accepted = []

    for candidate in sorted(
        safe_candidates,
        key=lambda item: (
            item["score"],
            item["distance_mm"],
            item["endpoint_indices"],
        ),
    ):
        first, second = candidate["endpoint_indices"]
        if first in used_endpoints or second in used_endpoints:
            rejected_counts["conflict"] += 1
            continue
        component_a, component_b = candidate["component_labels"]
        root_a, root_b = find(component_a), find(component_b)
        if root_a == root_b:
            rejected_counts["same_component"] += 1
            continue

        tube_array = candidate["tube_indices"]
        tube_tuple = tuple(tube_array.T)
        added_array = tube_array[~optimized[tube_tuple]]
        if len(added_array) == 0:
            rejected_counts["merge_failed"] += 1
            continue
        added_tuple = tuple(added_array.T)
        if np.any(~work_liver[added_tuple]):
            rejected_counts["outside_liver"] += 1
            continue
        if config.avoid_tumor and np.any(work_tumor[added_tuple]):
            rejected_counts["tumor_collision"] += 1
            continue
        if config.avoid_other_vessel and np.any(work_other[added_tuple]):
            rejected_counts["other_vessel_collision"] += 1
            continue

        added_volume_mm3 = len(added_array) * float(np.prod(spacing_array))
        connection_growth_fraction = added_volume_mm3 / raw_volume_mm3
        if (
            added_volume_mm3
            > config.max_added_volume_mm3_per_connection
            or connection_growth_fraction
            > config.max_connection_growth_fraction
        ):
            rejected_counts["growth_per_connection"] += 1
            continue
        cumulative_growth_fraction = (
            cumulative_added_volume_mm3 + added_volume_mm3
        ) / raw_volume_mm3
        if cumulative_growth_fraction > config.max_cumulative_growth_fraction:
            rejected_counts["growth_cumulative"] += 1
            continue

        contact_indices = _neighbor_indices(tube_array, optimized.shape)
        contact_tuple = tuple(contact_indices.T)
        foreground_contacts = optimized[contact_tuple]
        contact_indices = contact_indices[foreground_contacts]
        contact_owners = owner_labels[tuple(contact_indices.T)]
        contact_roots = {
            find(label) for label in np.unique(contact_owners) if label > 0
        }
        intended_roots = {root_a, root_b}
        if contact_roots - intended_roots:
            rejected_counts["third_component_collision"] += 1
            continue
        if contact_roots != intended_roots:
            rejected_counts["merge_failed"] += 1
            continue

        endpoints_local = candidate["endpoints_ijk_local"]
        max_contact_distance = max(
            float(config.max_endpoint_contact_distance_mm),
            1.5 * float(np.max(spacing_array)),
        )
        creates_loop = False
        for root, endpoint in zip(
            (root_a, root_b),
            endpoints_local,
        ):
            root_contacts = np.asarray(
                [
                    coordinate
                    for coordinate, owner in zip(
                        contact_indices, contact_owners
                    )
                    if find(owner) == root
                ],
                dtype=int,
            )
            distances = np.linalg.norm(
                (root_contacts - np.asarray(endpoint)) * spacing_array,
                axis=1,
            )
            if len(distances) == 0 or np.max(distances) > max_contact_distance:
                creates_loop = True
                break
        if creates_loop:
            rejected_counts["short_loop"] += 1
            continue

        new_root = union(root_a, root_b)
        optimized[added_tuple] = True
        owner_labels[added_tuple] = new_root
        cumulative_added_volume_mm3 += added_volume_mm3
        used_endpoints.update((first, second))
        accepted.append(
            {
                key: value
                for key, value in candidate.items()
                if key
                not in {
                    "endpoint_indices",
                    "endpoints_ijk_local",
                    "radius_mismatch",
                    "tube_indices",
                }
            }
            | {
                "added_voxels": int(len(added_array)),
                "added_volume_mm3": float(added_volume_mm3),
                "connection_growth_fraction": float(
                    connection_growth_fraction
                ),
                "cumulative_growth_fraction": float(
                    cumulative_added_volume_mm3 / raw_volume_mm3
                ),
                "components_after_addition": int(
                    components_before - len(accepted) - 1
                ),
                "merge_verified": True,
                "short_loop_excluded": True,
            }
        )

    _, measured_components_after = ndimage.label(
        optimized, _CONNECTIVITY_26
    )
    expected_components_after = int(components_before) - len(accepted)
    final_added = optimized & ~work_vessel
    postconditions_hold = (
        int(measured_components_after) == expected_components_after
        and not np.any(final_added & ~work_liver)
        and not (
            config.avoid_tumor and np.any(final_added & work_tumor)
        )
        and not (
            config.avoid_other_vessel and np.any(final_added & work_other)
        )
    )
    if not postconditions_hold:
        rejected_counts["postcondition_failed"] += max(len(accepted), 1)
        warnings.append("final_postcondition_failed_raw_returned")
        optimized = work_vessel.copy()
        accepted = []
        measured_components_after = components_before

    optimized_full = vessel.copy()
    optimized_full[roi_slices] = optimized
    return VesselOptimizationResult(
        optimized_mask=optimized_full,
        accepted_connections=accepted,
        rejected_candidate_counts=rejected_counts,
        components_before=int(components_before),
        components_after=int(measured_components_after),
        warnings=warnings,
    )


def optimize_vessel_mask(
    vessel_path,
    liver_path,
    tumor_paths,
    output_path,
    *,
    other_vessel_path=None,
    max_gap_mm=MAX_ALLOWED_GAP_MM,
    report_path=None,
    generation_id=None,
) -> dict:
    """Optimize a vessel NIfTI without modifying any source mask."""
    max_gap_mm = validate_max_gap_mm(max_gap_mm)
    vessel_path = Path(vessel_path)
    liver_path = Path(liver_path)
    tumor_paths = tuple(Path(path) for path in (tumor_paths or ()))
    output_path = Path(output_path)
    other_vessel_path = (
        None if other_vessel_path is None else Path(other_vessel_path)
    )
    report_path = None if report_path is None else Path(report_path)
    generation_id = generation_id or uuid.uuid4().hex

    if not output_path.name.endswith(".nii.gz"):
        raise ValueError("output_path must end with .nii.gz")

    input_paths = [vessel_path, liver_path, *tumor_paths]
    if other_vessel_path is not None:
        input_paths.append(other_vessel_path)
    if any(
        _paths_refer_to_same_file(output_path, path)
        for path in input_paths
    ):
        raise ValueError("output_path must not overwrite an input mask")
    if report_path is not None:
        report_temp_path = report_path.with_name(report_path.name + ".tmp")
        if any(
            _paths_refer_to_same_file(candidate, path)
            for candidate in (report_path, report_temp_path)
            for path in input_paths
        ):
            raise ValueError("report_path must not overwrite an input mask")
        if any(
            _paths_refer_to_same_file(report_candidate, output_path)
            for report_candidate in (report_path, report_temp_path)
        ):
            raise ValueError("report_path and output_path must be different")

    config = VesselOptimizationConfig(max_gap_mm=max_gap_mm)
    warnings = []
    report = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generation_id": generation_id,
        "status": "pending",
        "audit_state": "pending",
        "source_path": str(vessel_path),
        "raw_filename": vessel_path.name,
        "liver_path": str(liver_path),
        "tumor_paths": [str(path) for path in tumor_paths],
        "other_vessel_path": (
            None if other_vessel_path is None else str(other_vessel_path)
        ),
        "output_path": str(output_path),
        "optimized_filename": output_path.name,
        "report_path": None if report_path is None else str(report_path),
        "shape": None,
        "affine": None,
        "spacing_mm": None,
        "spatial_units": None,
        "geometry": None,
        "raw_sha256": None,
        "optimized_sha256": None,
        "voxel_count_before": None,
        "voxel_count_after": None,
        "added_voxels": None,
        "components_before": None,
        "components_after": None,
        "accepted_connections": [],
        "optimization_passes_requested": OPTIMIZATION_PASSES,
        "optimization_passes_completed": 0,
        "pass_summaries": [],
        "rejected_candidate_counts": {
            reason: 0 for reason in _REJECTION_REASONS
        },
        "warnings": warnings,
        "config": asdict(config),
    }

    def finish() -> dict:
        safe_report = _json_safe(report, warnings=warnings)
        if report_path is not None:
            _write_json_atomic(safe_report, report_path)
        return safe_report

    vessel_image = nib.load(str(vessel_path))
    if len(vessel_image.shape) != 3:
        raise ValueError(
            f"vessel mask must be 3D, got shape {vessel_image.shape}"
        )
    spacing, spatial_units = _validate_vessel_affine(vessel_image)
    vessel = np.asanyarray(vessel_image.dataobj) != 0
    raw_sha256 = sha256_file(vessel_path)
    before_voxels = int(np.count_nonzero(vessel))
    report["shape"] = list(vessel_image.shape)
    report["affine"] = np.asarray(vessel_image.affine).tolist()
    report["spacing_mm"] = spacing.tolist()
    report["spatial_units"] = spatial_units
    report["raw_sha256"] = raw_sha256
    report["voxel_count_before"] = before_voxels
    report["geometry"] = {
        "shape": list(vessel_image.shape),
        "affine": np.asarray(vessel_image.affine).tolist(),
        "spacing_mm": spacing.tolist(),
        "spatial_units": spatial_units,
        "raw_dtype": str(np.dtype(vessel_image.get_data_dtype())),
        "dtype": "uint8",
    }

    if not liver_path.is_file():
        report["status"] = "skipped_missing_liver"
        report["audit_state"] = "unavailable"
        report["voxel_count_after"] = before_voxels
        report["added_voxels"] = 0
        warnings.append(f"liver mask is missing: {liver_path}")
        return finish()

    liver = _load_matching_mask(
        liver_path, vessel_image, label="liver"
    )

    tumor = np.zeros(vessel.shape, dtype=bool)
    if not tumor_paths:
        warnings.append("tumor_constraint_unavailable")
    for index, path in enumerate(tumor_paths):
        if not path.is_file():
            raise FileNotFoundError(f"tumor[{index}] mask is missing: {path}")
        tumor |= _load_matching_mask(
            path, vessel_image, label=f"tumor[{index}]"
        )

    other_vessel = np.zeros(vessel.shape, dtype=bool)
    if other_vessel_path is None:
        warnings.append("other_vessel_constraint_unavailable")
    else:
        if not other_vessel_path.is_file():
            raise FileNotFoundError(
                f"other vessel mask is missing: {other_vessel_path}"
            )
        other_vessel = _load_matching_mask(
            other_vessel_path,
            vessel_image,
            label="other vessel",
        )

    if not np.any(vessel):
        report["status"] = "skipped_empty_vessel"
        report["audit_state"] = "unavailable"
        report["voxel_count_after"] = 0
        report["added_voxels"] = 0
        warnings.append("vessel mask is empty")
        return finish()

    optimized = vessel.copy()
    accepted_connections = []
    rejected_candidate_counts = {
        reason: 0 for reason in _REJECTION_REASONS
    }
    components_before = None
    for pass_index in range(1, OPTIMIZATION_PASSES + 1):
        input_voxels = int(np.count_nonzero(optimized))
        result = optimize_vessel_array(
            optimized,
            spacing,
            liver_mask=liver,
            tumor_mask=tumor,
            other_vessel_mask=other_vessel,
            config=config,
        )
        if components_before is None:
            components_before = result.components_before
        proposed = np.asarray(result.optimized_mask, dtype=bool)
        proposed_voxels = int(np.count_nonzero(proposed))
        total_growth_fraction = (
            (proposed_voxels - before_voxels) / before_voxels
        )
        pass_accepted = list(result.accepted_connections)
        pass_warnings = list(result.warnings)
        if total_growth_fraction > config.max_cumulative_growth_fraction:
            rejected_candidate_counts["growth_cumulative"] += max(
                len(pass_accepted), 1
            )
            pass_warnings.append(
                f"pass_{pass_index}_global_cumulative_growth_rejected"
            )
            proposed = optimized
            proposed_voxels = input_voxels
            pass_accepted = []

        for reason, count in result.rejected_candidate_counts.items():
            rejected_candidate_counts[reason] = (
                rejected_candidate_counts.get(reason, 0) + int(count)
            )
        for connection in pass_accepted:
            accepted_connections.append(
                dict(connection) | {"pass_index": pass_index}
            )
        warnings.extend(pass_warnings)
        optimized = proposed
        report["pass_summaries"].append(
            {
                "pass_index": pass_index,
                "input_voxels": input_voxels,
                "output_voxels": proposed_voxels,
                "added_voxels": proposed_voxels - input_voxels,
                "accepted_connections": len(pass_accepted),
                "warnings": pass_warnings,
            }
        )
        report["optimization_passes_completed"] = pass_index

    if sha256_file(vessel_path) != raw_sha256:
        raise RuntimeError(
            "raw vessel changed during optimization; refusing publication"
        )
    _write_optimized_nifti_atomic(optimized, vessel_image, output_path)

    after_voxels = int(np.count_nonzero(optimized))
    warnings[:] = list(dict.fromkeys(warnings))
    audited_connections = []
    for connection in accepted_connections:
        item = dict(connection)
        endpoints_ijk = np.asarray(item["endpoints_ijk"], dtype=float)
        item["endpoints_world_mm"] = nib.affines.apply_affine(
            vessel_image.affine,
            endpoints_ijk,
        ).tolist()
        audited_connections.append(item)
    _, components_after = ndimage.label(optimized, _CONNECTIVITY_26)
    report.update(
        {
            "status": "optimized",
            "audit_state": "validated",
            "optimized_sha256": sha256_file(output_path),
            "voxel_count_before": before_voxels,
            "voxel_count_after": after_voxels,
            "added_voxels": after_voxels - before_voxels,
            "components_before": components_before,
            "components_after": int(components_after),
            "accepted_connections": audited_connections,
            "rejected_candidate_counts": rejected_candidate_counts,
        }
    )
    return finish()

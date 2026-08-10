"""Conservative removal and cross-class relabelling of detached vessel fragments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class VesselFragmentFilterConfig:
    max_noise_length_mm: float = 10.0
    max_noise_volume_mm3: float = 50.0
    max_reclassify_length_mm: float = 15.0
    max_reclassify_volume_mm3: float = 100.0
    max_class_volume_fraction: float = 0.005
    max_other_median_distance_mm: float = 1.5
    max_other_p90_distance_mm: float = 2.5
    other_neighborhood_mm: float = 2.0
    min_other_neighborhood_fraction: float = 0.70
    max_direction_angle_deg: float = 20.0
    min_radius_ratio: float = 0.5
    max_radius_ratio: float = 2.0
    min_distance_margin_mm: float = 2.0
    isolated_noise_distance_mm: float = 2.5
    tumor_protection_distance_mm: float = 2.0


@dataclass
class VesselFragmentFilterResult:
    hepatic_mask: np.ndarray
    portal_mask: np.ndarray
    audit: dict


def _skeleton_length_mm(skeleton: np.ndarray, spacing: np.ndarray) -> float:
    points = np.argwhere(skeleton)
    point_set = {tuple(point) for point in points}
    length = 0.0
    positive_offsets = [
        offset
        for offset in np.ndindex(3, 3, 3)
        if offset != (1, 1, 1)
        and next((value for value in offset if value != 1), 1) > 1
    ]
    for point in points:
        for encoded in positive_offsets:
            offset = np.asarray(encoded, dtype=int) - 1
            neighbor = tuple((point + offset).tolist())
            if neighbor in point_set:
                length += float(np.linalg.norm(offset * spacing))
    return length


def _principal_direction(points: np.ndarray, spacing: np.ndarray):
    if len(points) < 3:
        return None
    physical = points.astype(float) * spacing
    centered = physical - physical.mean(axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 0.0 or values[-1] < 3.0 * max(values[-2], 1e-9):
        return None
    direction = vectors[:, -1]
    return direction / np.linalg.norm(direction)


def _angle_deg(first, second) -> float:
    if first is None or second is None:
        return float("inf")
    cosine = np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _component_records(mask: np.ndarray, spacing: np.ndarray):
    labels, count = ndimage.label(mask, _CONNECTIVITY_26)
    if count == 0:
        return labels, None, []
    sizes = np.bincount(labels.ravel())
    main_label = int(np.argmax(sizes[1:]) + 1)
    records = []
    voxel_volume = float(np.prod(spacing))
    total_volume = float(np.count_nonzero(mask)) * voxel_volume
    for label in range(1, count + 1):
        component = labels == label
        skeleton = skeletonize(component)
        records.append({
            "label": label,
            "mask": component,
            "skeleton": skeleton,
            "points": np.argwhere(skeleton),
            "volume_mm3": float(sizes[label]) * voxel_volume,
            "volume_fraction": (
                float(sizes[label]) * voxel_volume / total_volume
                if total_volume else 0.0
            ),
            "length_mm": _skeleton_length_mm(skeleton, spacing),
            "is_main": label == main_label,
        })
    return labels, main_label, records


def _median_radius(mask: np.ndarray, skeleton: np.ndarray, spacing: np.ndarray):
    values = ndimage.distance_transform_edt(mask, sampling=spacing)[skeleton]
    return float(np.median(values)) if len(values) else 0.0


def _evaluate_source(
    source_name: str,
    source: np.ndarray,
    other_name: str,
    other: np.ndarray,
    spacing: np.ndarray,
    tumor: np.ndarray,
    config: VesselFragmentFilterConfig,
):
    _, main_label, components = _component_records(source, spacing)
    other_labels, other_main_label, other_components = _component_records(
        other, spacing
    )
    if main_label is None:
        return [], []
    source_main = next(item["mask"] for item in components if item["is_main"])
    other_main = (
        other_labels == other_main_label
        if other_main_label is not None
        else np.zeros_like(other)
    )
    source_distance = ndimage.distance_transform_edt(
        ~source_main, sampling=spacing
    )
    other_distance = (
        ndimage.distance_transform_edt(~other_main, sampling=spacing)
        if np.any(other_main)
        else np.full(source.shape, np.inf)
    )
    tumor_distance = (
        ndimage.distance_transform_edt(~tumor, sampling=spacing)
        if np.any(tumor)
        else np.full(source.shape, np.inf)
    )
    other_skeleton = skeletonize(other_main)
    other_points = np.argwhere(other_skeleton)
    other_tree = (
        cKDTree(other_points.astype(float) * spacing)
        if len(other_points)
        else None
    )
    source_radius_map = ndimage.distance_transform_edt(source, sampling=spacing)
    other_radius_map = ndimage.distance_transform_edt(other, sampling=spacing)
    decisions = []
    removal_masks = []

    for component in components:
        if component["is_main"]:
            continue
        points = component["points"]
        if not len(points):
            continue
        indices = tuple(points.T)
        own_distances = source_distance[indices]
        other_distances = other_distance[indices]
        tumor_near = float(np.min(tumor_distance[component["mask"]])) <= (
            config.tumor_protection_distance_mm
        )
        median_other = float(np.median(other_distances))
        p90_other = float(np.percentile(other_distances, 90))
        near_fraction = float(np.mean(
            other_distances <= config.other_neighborhood_mm
        ))
        median_own = float(np.median(own_distances))
        direction = _principal_direction(points, spacing)
        nearby_other_points = np.empty((0, 3), dtype=int)
        if other_tree is not None:
            neighborhoods = other_tree.query_ball_point(
                points.astype(float) * spacing,
                r=config.max_other_p90_distance_mm,
            )
            nearby_ids = sorted({item for group in neighborhoods for item in group})
            if nearby_ids:
                nearby_other_points = other_points[nearby_ids]
        other_direction = _principal_direction(nearby_other_points, spacing)
        angle = _angle_deg(direction, other_direction)
        source_radius = float(np.median(source_radius_map[indices]))
        if len(nearby_other_points):
            other_radius = float(np.median(
                other_radius_map[tuple(nearby_other_points.T)]
            ))
        else:
            other_radius = 0.0
        radius_ratio = (
            source_radius / other_radius if other_radius > 0.0 else float("inf")
        )
        small_for_reclassification = (
            component["length_mm"] <= config.max_reclassify_length_mm
            and component["volume_mm3"] <= config.max_reclassify_volume_mm3
            and component["volume_fraction"] <= config.max_class_volume_fraction
        )
        other_supported = (
            median_other <= config.max_other_median_distance_mm
            and p90_other <= config.max_other_p90_distance_mm
            and near_fraction >= config.min_other_neighborhood_fraction
            and angle <= config.max_direction_angle_deg
            and config.min_radius_ratio <= radius_ratio <= config.max_radius_ratio
            and median_own - median_other >= config.min_distance_margin_mm
        )
        noise = (
            component["length_mm"] <= config.max_noise_length_mm
            and component["volume_mm3"] <= config.max_noise_volume_mm3
            and median_own > config.isolated_noise_distance_mm
            and median_other > config.isolated_noise_distance_mm
        )
        if tumor_near:
            decision = "review_required"
            reason = "tumor_proximity"
        elif small_for_reclassification and other_supported:
            decision = f"reclassified_to_{other_name}"
            reason = "other_tree_geometry_supported"
            removal_masks.append((component["mask"], other_name))
        elif noise:
            decision = "removed_noise"
            reason = "small_isolated_from_both_trees"
            removal_masks.append((component["mask"], None))
        else:
            decision = "review_required"
            reason = "insufficient_confidence"
        decisions.append({
            "source_class": source_name,
            "component_label": int(component["label"]),
            "decision": decision,
            "reason": reason,
            "volume_mm3": component["volume_mm3"],
            "class_volume_fraction": component["volume_fraction"],
            "centerline_length_mm": component["length_mm"],
            "median_distance_to_own_main_mm": median_own,
            "median_distance_to_other_main_mm": median_other,
            "p90_distance_to_other_main_mm": p90_other,
            "other_neighborhood_fraction": near_fraction,
            "direction_angle_deg": angle,
            "radius_ratio": radius_ratio,
            "tumor_protected": tumor_near,
        })
    return decisions, removal_masks


def filter_cross_class_fragments(
    hepatic_mask,
    portal_mask,
    spacing,
    *,
    tumor_mask=None,
    config=VesselFragmentFilterConfig(),
) -> VesselFragmentFilterResult:
    """Remove detached noise and conservatively relabel detached fragments."""
    hepatic = np.asarray(hepatic_mask, dtype=bool)
    portal = np.asarray(portal_mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if hepatic.shape != portal.shape or hepatic.ndim != 3:
        raise ValueError("hepatic and portal masks must be matching 3D arrays")
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive finite values")
    tumor = (
        np.zeros_like(hepatic)
        if tumor_mask is None
        else np.asarray(tumor_mask, dtype=bool)
    )
    if tumor.shape != hepatic.shape:
        raise ValueError("tumor mask shape must match vessel masks")

    hepatic_decisions, hepatic_actions = _evaluate_source(
        "hepatic", hepatic, "portal", portal, spacing, tumor, config
    )
    portal_decisions, portal_actions = _evaluate_source(
        "portal", portal, "hepatic", hepatic, spacing, tumor, config
    )
    cleaned_hepatic = hepatic.copy()
    cleaned_portal = portal.copy()
    for component, target in hepatic_actions:
        cleaned_hepatic[component] = False
        if target == "portal":
            cleaned_portal[component] = True
    for component, target in portal_actions:
        cleaned_portal[component] = False
        if target == "hepatic":
            cleaned_hepatic[component] = True

    decisions = hepatic_decisions + portal_decisions
    return VesselFragmentFilterResult(
        hepatic_mask=cleaned_hepatic,
        portal_mask=cleaned_portal,
        audit={
            "status": "completed",
            "config": asdict(config),
            "decisions": decisions,
            "removed_noise_components": sum(
                item["decision"] == "removed_noise" for item in decisions
            ),
            "reclassified_components": sum(
                item["decision"].startswith("reclassified_to_")
                for item in decisions
            ),
            "review_required_components": sum(
                item["decision"] == "review_required" for item in decisions
            ),
        },
    )

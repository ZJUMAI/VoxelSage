"""Physical measurement regressions matching the frozen Chapter 7 E1 families."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from Tool_Box.liver_analysis import (
    compute_tumor_diameter,
    compute_tumor_vessel_distance,
    compute_vessel_volume,
)
from skills.builtin.liver_analysis.main import run as liver_analysis
from skills.builtin.tumor_diameter.main import run as tumor_diameter
from skills.builtin.tumor_vessel_distance.main import run as tumor_vessel_distance
from skills.builtin.vessel_volume.main import run as vessel_volume
from skills.models import SkillContext


@pytest.fixture(params=["isotropic", "anisotropic", "rotated"])
def affine(request: pytest.FixtureRequest) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = [31.0, -29.0, 13.0]
    if request.param != "isotropic":
        result[:3, :3] = np.diag([0.8, 1.2, 3.0])
    if request.param == "rotated":
        angle = np.deg2rad(37.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        result[:3, :3] = rotation @ result[:3, :3]
    return result


def _save_case(
    root: Path, tumor: np.ndarray, vessel: np.ndarray, affine: np.ndarray
) -> SkillContext:
    masks = root / "masks"
    masks.mkdir()
    for name, mask in {
        "liver": np.ones_like(tumor), "tumor_1": tumor, "hepatic": vessel,
    }.items():
        nib.save(nib.Nifti1Image(mask, affine), str(masks / f"{name}.nii.gz"))
    ct_path = root / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros_like(tumor), affine), str(ct_path))
    return SkillContext("synthetic", str(ct_path), str(masks), str(root))


def _assert_metric(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    else:
        # Compare rounded public outputs to unrounded physical ground truth.
        assert actual == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("condition", [
    "large", "small", "separated_x", "separated_y", "separated_z",
    "overlap", "empty", "below_threshold",
])
def test_frozen_e1_measurements(
    tmp_path: Path, affine: np.ndarray, condition: str
) -> None:
    tumor = np.zeros((32, 32, 32), dtype=np.uint8)
    vessel = np.zeros_like(tumor)
    vessel[10:14, 10:14, 10:14] = 1
    regions = {
        "large": (slice(20, 25), slice(19, 24), slice(18, 22)),
        "small": (slice(20, 23), slice(19, 22), slice(18, 21)),
        "separated_x": (slice(16, 18), slice(10, 12), slice(10, 13)),
        "separated_y": (slice(10, 12), slice(17, 19), slice(10, 13)),
        "separated_z": (slice(10, 12), slice(10, 12), slice(16, 19)),
        "overlap": (slice(12, 15), slice(12, 15), slice(12, 15)),
        "below_threshold": (slice(20, 21), slice(20, 21), slice(20, 25)),
    }
    if condition in regions:
        tumor[regions[condition]] = 1
    ctx = _save_case(tmp_path, tumor, vessel, affine)
    tumor_points = nib.affines.apply_affine(affine, np.argwhere(tumor))
    vessel_points = nib.affines.apply_affine(affine, np.argwhere(vessel))
    expected_volume = float(vessel.sum()) * abs(np.linalg.det(affine[:3, :3]))
    expected_diameter = None
    if len(tumor_points) >= 10:
        expected_diameter = max(
            float(np.linalg.norm(tumor_points - point, axis=1).max())
            for point in tumor_points
        )
    expected_distance = None
    if len(tumor_points):
        expected_distance = min(
            float(np.linalg.norm(vessel_points - point, axis=1).min())
            for point in tumor_points
        )

    direct_volume = compute_vessel_volume(ctx.get_mask_path("hepatic"))
    direct_diameter = compute_tumor_diameter(ctx.get_mask_path("tumor_1"))
    direct_distance = compute_tumor_vessel_distance(
        tumor, vessel, ctx.get_voxel_spacing()
    )
    focused_volume = vessel_volume(ctx)["vessels"]["hepatic"]
    focused_diameter = tumor_diameter(ctx)["tumors"][0]
    focused_distance = tumor_vessel_distance(ctx)["distances"][0]
    integrated = liver_analysis(ctx)
    integrated_tumor = integrated["tumor_results"]["tumor_1"]
    for result in [direct_volume, focused_volume, integrated["vessel_volumes"]["hepatic"]]:
        _assert_metric(result["volume_mm3"], expected_volume)
    for result in [direct_diameter, focused_diameter, integrated_tumor["diameter"]]:
        _assert_metric(result["max_diameter_mm"], expected_diameter)
    for result in [direct_distance, focused_distance]:
        _assert_metric(result["min_distance_mm"], expected_distance)
        assert result["tumor_contacts_vessel"] == (condition == "overlap")
    integrated_distance = integrated_tumor["vessel_distances"]["hepatic"]
    _assert_metric(
        integrated_distance["min_distance_mm"],
        expected_distance if len(tumor_points) >= 10 else None,
    )
    assert focused_distance["mask_variant"] == "raw"


def test_empty_vessel_has_no_fabricated_distance(
    tmp_path: Path, affine: np.ndarray
) -> None:
    tumor = np.zeros((8, 8, 8), dtype=np.uint8)
    tumor[2:5, 2:5, 2:5] = 1
    vessel = np.zeros_like(tumor)
    ctx = _save_case(tmp_path, tumor, vessel, affine)
    direct = compute_tumor_vessel_distance(tumor, vessel, ctx.get_voxel_spacing())
    focused = tumor_vessel_distance(ctx)["distances"][0]
    integrated = liver_analysis(ctx)["tumor_results"]["tumor_1"]["vessel_distances"]["hepatic"]
    for result in [direct, focused, integrated]:
        assert result["min_distance_mm"] is None
        assert result["tumor_contacts_vessel"] is False
        assert result["tumor_voxels"] == 27


def test_anisotropic_spacing_changes_the_nearest_vessel() -> None:
    tumor = np.zeros((8, 8, 8), dtype=np.uint8)
    vessel = np.zeros_like(tumor)
    tumor[2, 2, 2] = 1
    vessel[6, 2, 2] = 1
    vessel[2, 2, 4] = 1
    result = compute_tumor_vessel_distance(tumor, vessel, (0.8, 1.2, 3.0))
    assert result["min_distance_mm"] == pytest.approx(3.2)

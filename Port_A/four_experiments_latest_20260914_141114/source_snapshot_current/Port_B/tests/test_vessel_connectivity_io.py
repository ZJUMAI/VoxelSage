import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from Tool_Box import vessel_connectivity as vessel_connectivity_module
from Tool_Box.vessel_optimization_contract import AUDIT_SCHEMA
from Tool_Box.vessel_connectivity import (
    VesselOptimizationResult,
    optimize_vessel_mask,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_nifti(path, data, affine=None, *, spatial_unit="mm"):
    image = nib.Nifti1Image(
        np.asarray(data),
        np.eye(4) if affine is None else affine,
    )
    if spatial_unit is not None:
        image.header.set_xyzt_units(spatial_unit)
    nib.save(image, path)


def test_nifti_optimizer_preserves_raw_and_geometry(tmp_path):
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    vessel = np.zeros((24, 16, 16), dtype=np.uint8)
    vessel[3:10, 8, 8] = 1
    vessel[11:18, 8, 8] = 1
    liver = np.ones_like(vessel)
    raw = tmp_path / "hepatic.nii.gz"
    liver_path = tmp_path / "liver.nii.gz"
    out = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, vessel, affine)
    _save_nifti(liver_path, liver, affine)
    before = _digest(raw)

    report = optimize_vessel_mask(raw, liver_path, [], out)

    assert _digest(raw) == before
    optimized = nib.load(out)
    assert optimized.shape == vessel.shape
    assert np.allclose(optimized.affine, affine)
    assert optimized.get_data_dtype() == np.dtype(np.uint8)
    assert set(np.unique(optimized.get_fdata())) <= {0.0, 1.0}
    assert report["source_path"].endswith("hepatic.nii.gz")


def test_nifti_optimizer_runs_three_passes_before_writing_final_mask(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    vessel = np.zeros((8, 8, 8), dtype=np.uint8)
    vessel.ravel()[:20] = 1
    _save_nifti(raw, vessel)
    _save_nifti(liver, np.ones_like(vessel))
    calls = []

    def fake_optimize(current, *args, **kwargs):
        current = np.asarray(current, dtype=bool).copy()
        calls.append(current.copy())
        new_index = 20 + len(calls) - 1
        current.ravel()[new_index] = True
        return VesselOptimizationResult(
            optimized_mask=current,
            accepted_connections=[{
                "endpoints_ijk": ((0, 0, 0), (0, 0, 1)),
                "distance_mm": 1.0,
            }],
            rejected_candidate_counts={},
            components_before=4 - len(calls),
            components_after=3 - len(calls),
            warnings=[],
        )

    with patch(
        "Tool_Box.vessel_connectivity.optimize_vessel_array",
        side_effect=fake_optimize,
    ), patch(
        "Tool_Box.vessel_connectivity._write_optimized_nifti_atomic",
        wraps=vessel_connectivity_module._write_optimized_nifti_atomic,
    ) as write_output:
        report = optimize_vessel_mask(raw, liver, [], output)

    assert len(calls) == 3
    assert [int(mask.sum()) for mask in calls] == [20, 21, 22]
    assert write_output.call_count == 1
    assert int(np.count_nonzero(nib.load(output).get_fdata())) == 23
    assert report["optimization_passes_requested"] == 3
    assert report["optimization_passes_completed"] == 3
    assert [item["pass_index"] for item in report["accepted_connections"]] == [1, 2, 3]


def test_missing_liver_skips_without_inventing_output(tmp_path):
    raw = tmp_path / "portal.nii.gz"
    _save_nifti(raw, np.ones((4, 4, 4), dtype=np.uint8))
    report = optimize_vessel_mask(
        raw,
        tmp_path / "missing.nii.gz",
        [],
        tmp_path / "portal_optimized.nii.gz",
    )
    assert report["status"] == "skipped_missing_liver"
    assert not (tmp_path / "portal_optimized.nii.gz").exists()


def test_empty_vessel_validates_auxiliary_mask_geometry_before_skipping(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    tumor = tmp_path / "tumor.nii.gz"
    other = tmp_path / "portal.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, np.zeros((4, 4, 4), dtype=np.uint8))
    _save_nifti(liver, np.ones((4, 4, 4), dtype=np.uint8))
    _save_nifti(tumor, np.zeros((3, 4, 4), dtype=np.uint8))
    _save_nifti(other, np.zeros((4, 4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="tumor\\[0\\] mask shape"):
        optimize_vessel_mask(raw, liver, [tumor], output, other_vessel_path=other)

    assert not output.exists()


def test_empty_vessel_skips_without_generating_output(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, np.zeros((4, 4, 4), dtype=np.uint8))
    _save_nifti(liver, np.ones((4, 4, 4), dtype=np.uint8))

    report = optimize_vessel_mask(raw, liver, [], output)

    assert report["status"] == "skipped_empty_vessel"
    assert not output.exists()


def test_rejects_uncompressed_nifti_output_path(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    _save_nifti(raw, np.ones((4, 4, 4), dtype=np.uint8))
    _save_nifti(liver, np.ones((4, 4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="must end with .nii.gz"):
        optimize_vessel_mask(raw, liver, [], tmp_path / "hepatic_optimized.nii")


def test_report_is_reloaded_as_json_safe_data(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    report_path = tmp_path / "hepatic_optimized.json"
    vessel = np.zeros((24, 16, 16), dtype=np.uint8)
    vessel[3:10, 8, 8] = 1
    vessel[11:18, 8, 8] = 1
    _save_nifti(raw, vessel)
    _save_nifti(liver, np.ones_like(vessel))

    report = optimize_vessel_mask(raw, liver, [], output, report_path=report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_failed_output_write_cleans_up_temporary_file(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, np.ones((4, 4, 4), dtype=np.uint8))
    _save_nifti(liver, np.ones((4, 4, 4), dtype=np.uint8))

    def partial_write_then_fail(_, path):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    with patch("Tool_Box.vessel_connectivity.nib.save", partial_write_then_fail):
        with pytest.raises(OSError, match="simulated write failure"):
            optimize_vessel_mask(raw, liver, [], output)

    assert not output.exists()
    assert not (tmp_path / "hepatic_optimized.nii.gz.tmp.nii.gz").exists()


@pytest.mark.parametrize(
    "max_gap_mm",
    [0.0, -1.0, float("nan"), float("inf"), 4.0001],
)
def test_nifti_entry_point_enforces_public_four_mm_cap(tmp_path, max_gap_mm):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, np.ones((4, 4, 4), dtype=np.uint8))
    _save_nifti(liver, np.ones((4, 4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="max_gap_mm"):
        optimize_vessel_mask(
            raw,
            liver,
            [],
            output,
            max_gap_mm=max_gap_mm,
        )


def test_nifti_entry_point_accepts_exact_public_four_mm_cap(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, np.ones((4, 4, 4), dtype=np.uint8))
    _save_nifti(liver, np.ones((4, 4, 4), dtype=np.uint8))

    report = optimize_vessel_mask(raw, liver, [], output, max_gap_mm=4.0)

    assert report["config"]["max_gap_mm"] == 4.0


def test_rotated_translated_affine_is_accepted_and_world_endpoints_are_audited(
    tmp_path,
):
    theta = np.deg2rad(30.0)
    affine = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0, 20.0],
            [np.sin(theta), np.cos(theta), 0.0, -10.0],
            [0.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    vessel = np.zeros((24, 16, 16), dtype=np.uint8)
    vessel[3:10, 8, 8] = 1
    vessel[12:19, 8, 8] = 1
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, vessel, affine)
    _save_nifti(liver, np.ones_like(vessel), affine)

    report = optimize_vessel_mask(raw, liver, [], output)

    assert report["status"] == "optimized"
    assert report["spatial_units"] == "mm"
    assert len(report["accepted_connections"]) == 1
    connection = report["accepted_connections"][0]
    expected = [
        nib.affines.apply_affine(affine, endpoint).tolist()
        for endpoint in connection["endpoints_ijk"]
    ]
    assert np.allclose(connection["endpoints_world_mm"], expected)


def test_sheared_affine_is_rejected_before_automatic_reconnection(tmp_path):
    affine = np.eye(4)
    affine[0, 1] = 0.25
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(raw, np.ones((4, 4, 4), dtype=np.uint8), affine)
    _save_nifti(liver, np.ones((4, 4, 4), dtype=np.uint8), affine)

    with pytest.raises(ValueError, match="orthogonal"):
        optimize_vessel_mask(raw, liver, [], output)

    assert not output.exists()


def test_unknown_spatial_units_are_rejected_conservatively(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    _save_nifti(
        raw,
        np.ones((4, 4, 4), dtype=np.uint8),
        spatial_unit=None,
    )
    _save_nifti(
        liver,
        np.ones((4, 4, 4), dtype=np.uint8),
        spatial_unit=None,
    )

    with pytest.raises(ValueError, match="millimeters"):
        optimize_vessel_mask(raw, liver, [], output)


def test_audit_schema_contract_and_unavailable_constraints_are_recorded(tmp_path):
    raw = tmp_path / "hepatic.nii.gz"
    liver = tmp_path / "liver.nii.gz"
    output = tmp_path / "hepatic_optimized.nii.gz"
    vessel = np.zeros((24, 16, 16), dtype=np.uint8)
    vessel[3:10, 8, 8] = 1
    vessel[12:19, 8, 8] = 1
    _save_nifti(raw, vessel)
    _save_nifti(liver, np.ones_like(vessel))

    report = optimize_vessel_mask(raw, liver, [], output)

    assert report["schema"] == AUDIT_SCHEMA
    assert report["schema_version"] == 2
    assert "voxel_count_before" in report
    assert "voxel_count_after" in report
    assert "vessel_voxels_before" not in report
    assert "vessel_voxels_after" not in report
    assert "distance" in report["rejected_candidate_counts"]
    assert "max_added_volume_mm3_per_connection" in report["config"]
    assert "max_cumulative_growth_fraction" in report["config"]
    assert "max_tube_radius_mm" in report["config"]
    assert "tumor_constraint_unavailable" in report["warnings"]
    assert "other_vessel_constraint_unavailable" in report["warnings"]

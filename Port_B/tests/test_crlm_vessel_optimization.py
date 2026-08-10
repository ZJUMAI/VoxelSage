import ast
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
from pathlib import Path
import threading
import time
from typing import List
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from Tool_Box.crlm_postprocess import optimize_crlm_vessels
from Tool_Box.vessel_optimization_contract import AUDIT_SCHEMA


def _save(path, data):
    image = nib.Nifti1Image(data.astype(np.uint8), np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


def _load_api_postprocessing(logs):
    source_path = Path(__file__).parents[1] / "API.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_crlm_run_postprocessing"
    )
    namespace = {
        "List": List,
        "Path": Path,
        "glob": glob,
        "os": os,
        "_log": logs.append,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["_crlm_run_postprocessing"]


def test_crlm_optimizer_writes_both_variants_and_case_report(tmp_path):
    liver = np.ones((24, 16, 16), dtype=bool)
    hepatic = np.zeros_like(liver)
    portal = np.zeros_like(liver)
    hepatic[3:10, 7, 7] = True
    hepatic[12:19, 7, 7] = True
    portal[3:10, 10, 10] = True
    portal[12:19, 10, 10] = True
    _save(tmp_path / "liver.nii.gz", liver)
    _save(tmp_path / "hepatic.nii.gz", hepatic)
    _save(tmp_path / "portal.nii.gz", portal)

    report_path = tmp_path.parent / "vessel_optimization_report.json"
    report = optimize_crlm_vessels(tmp_path, report_path)

    assert (tmp_path / "hepatic_optimized.nii.gz").exists()
    assert (tmp_path / "portal_optimized.nii.gz").exists()
    assert set(report["vessels"]) == {"hepatic", "portal"}
    saved = json.loads(report_path.read_text("utf-8"))
    assert saved == report
    assert saved["max_gap_mm"] == 4.0
    assert saved["schema"] == AUDIT_SCHEMA
    assert saved["schema_version"] == 2
    assert isinstance(saved["generation_id"], str)
    assert saved["generation_id"]
    for vessel_name in ("hepatic", "portal"):
        vessel_report = saved["vessels"][vessel_name]
        assert vessel_report["generation_id"] == saved["generation_id"]
        assert vessel_report["status"] == "optimized"
        assert vessel_report["audit_state"] == "validated"
        assert len(vessel_report["raw_sha256"]) == 64
        assert len(vessel_report["optimized_sha256"]) == 64
        assert vessel_report["geometry"]["spatial_units"] == "mm"
    assert not report_path.with_name(report_path.name + ".tmp").exists()


def test_crlm_optimizer_uses_raw_counterpart_for_each_vessel(tmp_path):
    for name in ("liver", "hepatic", "portal", "tumor_2", "tumor_1"):
        (tmp_path / f"{name}.nii.gz").touch()
    calls = []

    def record(vessel_path, liver_path, tumor_paths, output_path, **kwargs):
        calls.append(
            {
                "vessel": Path(vessel_path).name,
                "liver": Path(liver_path).name,
                "tumors": [Path(path).name for path in tumor_paths],
                "output": Path(output_path).name,
                "other": Path(kwargs["other_vessel_path"]).name,
                "max_gap_mm": kwargs["max_gap_mm"],
            }
        )
        return {"status": "recorded", "vessel": Path(vessel_path).name}

    with patch(
        "Tool_Box.crlm_postprocess.optimize_vessel_mask",
        side_effect=record,
    ):
        report = optimize_crlm_vessels(
            tmp_path,
            tmp_path / "report.json",
            max_gap_mm=3.5,
        )

    assert calls == [
        {
            "vessel": "hepatic.nii.gz",
            "liver": "liver.nii.gz",
            "tumors": ["tumor_1.nii.gz", "tumor_2.nii.gz"],
            "output": "hepatic_optimized.nii.gz",
            "other": "portal.nii.gz",
            "max_gap_mm": 3.5,
        },
        {
            "vessel": "portal.nii.gz",
            "liver": "liver.nii.gz",
            "tumors": ["tumor_1.nii.gz", "tumor_2.nii.gz"],
            "output": "portal_optimized.nii.gz",
            "other": "hepatic.nii.gz",
            "max_gap_mm": 3.5,
        },
    ]
    assert report["vessels"]["hepatic"]["vessel"] == "hepatic.nii.gz"
    assert report["vessels"]["portal"]["vessel"] == "portal.nii.gz"
    assert len(report["vessels"]["hepatic"]["raw_sha256"]) == 64
    assert len(report["vessels"]["portal"]["raw_sha256"]) == 64


def test_crlm_optimizer_preserves_success_and_reports_other_vessel_error(tmp_path):
    shape = (4, 4, 4)
    liver = np.ones(shape, dtype=bool)
    hepatic = np.zeros(shape, dtype=bool)
    portal = np.zeros(shape, dtype=bool)
    hepatic[:2, :, :] = True
    portal[2:, :, :] = True
    _save(tmp_path / "liver.nii.gz", liver)
    _save(tmp_path / "hepatic.nii.gz", hepatic)
    _save(tmp_path / "portal.nii.gz", portal)
    report_path = tmp_path / "vessel_optimization_report.json"
    prior_portal = tmp_path / "portal_optimized.nii.gz"
    prior_portal.write_bytes(b"old-portal")
    report_path.write_text('{"status": "stale"}\n', encoding="utf-8")

    def optimize_one(vessel_path, liver_path, tumor_paths, output_path, **kwargs):
        if Path(vessel_path).name.startswith("portal"):
            raise ValueError("portal geometry mismatch")
        raw = nib.load(vessel_path)
        nib.save(
            nib.Nifti1Image(
                np.asanyarray(raw.dataobj).astype(np.uint8),
                raw.affine,
                raw.header,
            ),
            output_path,
        )
        return {"status": "optimized", "output_path": str(output_path)}

    with patch(
        "Tool_Box.crlm_postprocess.optimize_vessel_mask",
        side_effect=optimize_one,
    ):
        report = optimize_crlm_vessels(tmp_path, report_path)

    assert (tmp_path / "hepatic_optimized.nii.gz").is_file()
    assert np.array_equal(
        np.asanyarray(
            nib.load(tmp_path / "hepatic_optimized.nii.gz").dataobj
        ),
        hepatic.astype(np.uint8),
    )
    assert prior_portal.read_bytes() == b"old-portal"
    assert report["vessels"]["hepatic"]["status"] == "optimized"
    assert report["vessels"]["portal"]["status"] == "error"
    assert "portal geometry mismatch" in report["vessels"]["portal"]["error"]
    assert json.loads(report_path.read_text("utf-8")) == report


def test_crlm_optimizer_missing_one_vessel_does_not_block_the_other(tmp_path):
    shape = (4, 4, 4)
    for name in ("liver", "portal"):
        _save(tmp_path / f"{name}.nii.gz", np.ones(shape, dtype=bool))
    calls = []

    def optimize_portal(vessel_path, liver_path, tumor_paths, output_path, **kwargs):
        calls.append(
            (Path(vessel_path).name, kwargs["other_vessel_path"])
        )
        raw = nib.load(vessel_path)
        nib.save(
            nib.Nifti1Image(
                np.asanyarray(raw.dataobj).astype(np.uint8),
                raw.affine,
                raw.header,
            ),
            output_path,
        )
        return {"status": "optimized", "output_path": str(output_path)}

    with patch(
        "Tool_Box.crlm_postprocess.optimize_vessel_mask",
        side_effect=optimize_portal,
    ):
        report = optimize_crlm_vessels(tmp_path, tmp_path / "report.json")

    assert calls == [("portal.nii.gz", None)]
    assert report["vessels"]["hepatic"]["status"] == "skipped_missing_vessel"
    assert "hepatic.nii.gz" in report["vessels"]["hepatic"]["error"]
    assert report["vessels"]["portal"]["status"] == "optimized"
    assert (tmp_path / "portal_optimized.nii.gz").is_file()


def test_report_commit_failure_restores_prior_consistent_snapshot(tmp_path):
    for name in ("liver", "hepatic", "portal"):
        (tmp_path / f"{name}.nii.gz").touch()
    hepatic_output = tmp_path / "hepatic_optimized.nii.gz"
    portal_output = tmp_path / "portal_optimized.nii.gz"
    report_path = tmp_path / "vessel_optimization_report.json"
    hepatic_output.write_bytes(b"old-hepatic")
    report_path.write_text('{"snapshot": "old"}\n', encoding="utf-8")

    def write_new(vessel_path, liver_path, tumor_paths, output_path, **kwargs):
        Path(output_path).write_bytes(
            f"new-{Path(vessel_path).name}".encode("utf-8")
        )
        return {"status": "optimized", "output_path": str(output_path)}

    real_replace = os.replace

    def fail_staged_report_replace(source, destination):
        if (
            Path(destination) == report_path
            and Path(source).name.endswith(".attempt")
        ):
            raise OSError("report commit failed")
        return real_replace(source, destination)

    with (
        patch(
            "Tool_Box.crlm_postprocess.optimize_vessel_mask",
            side_effect=write_new,
        ),
        patch(
            "Tool_Box.crlm_postprocess.os.replace",
            side_effect=fail_staged_report_replace,
        ),
        pytest.raises(OSError, match="report commit failed"),
    ):
        optimize_crlm_vessels(tmp_path, report_path)

    assert hepatic_output.read_bytes() == b"old-hepatic"
    assert not portal_output.exists()
    assert json.loads(report_path.read_text("utf-8")) == {"snapshot": "old"}


def test_api_portable_gate_preserves_prior_outputs_when_orchestrator_fails(
    tmp_path,
):
    shape = (4, 4, 4)
    for name in ("liver", "hepatic", "portal"):
        _save(tmp_path / f"{name}.nii.gz", np.ones(shape, dtype=bool))
    prior_outputs = {}
    for name in ("hepatic", "portal"):
        output_path = tmp_path / f"{name}_optimized.nii.gz"
        _save(output_path, np.ones(shape, dtype=bool))
        prior_outputs[name] = output_path.read_bytes()

    events = []
    logs = []

    def rename(mask_dir):
        events.append("rename")

    def split(mask_dir):
        events.append("split")
        return []

    def fail_report_commit(mask_dir, report_path, max_gap_mm=4.0):
        events.append("optimize")
        raise OSError("simulated report commit failure")

    run_postprocessing = _load_api_postprocessing(logs)
    with (
        patch("Tool_Box.crlm_postprocess.rename_vessel_masks", side_effect=rename),
        patch("Tool_Box.crlm_postprocess.split_hepatic_tumor", side_effect=split),
        patch(
            "Tool_Box.crlm_postprocess.optimize_crlm_vessels",
            side_effect=fail_report_commit,
        ),
    ):
        mask_names = run_postprocessing(str(tmp_path))

    assert events == ["rename", "split", "optimize"]
    assert mask_names == ["hepatic", "liver", "portal"]
    for name, prior_bytes in prior_outputs.items():
        assert (tmp_path / f"{name}_optimized.nii.gz").read_bytes() == prior_bytes
    assert any(
        str(tmp_path) in message
        and "simulated report commit failure" in message
        and "continuing with existing logical masks" in message
        for message in logs
    )
    assert not any("using raw masks" in message for message in logs)
    assert not any(
        "optimization succeeded" in message.lower()
        or "optimization complete" in message.lower()
        for message in logs
    )


def test_api_success_returns_logical_names_not_optimized_variants(tmp_path):
    shape = (4, 4, 4)
    for name in ("liver", "hepatic", "portal"):
        _save(tmp_path / f"{name}.nii.gz", np.ones(shape, dtype=bool))

    def write_optimized_variants(mask_dir, report_path, max_gap_mm=4.0):
        for name in ("hepatic", "portal"):
            _save(
                Path(mask_dir) / f"{name}_optimized.nii.gz",
                np.ones(shape, dtype=bool),
            )
        return {"version": 1, "max_gap_mm": max_gap_mm, "vessels": {}}

    run_postprocessing = _load_api_postprocessing([])
    with (
        patch("Tool_Box.crlm_postprocess.rename_vessel_masks"),
        patch("Tool_Box.crlm_postprocess.split_hepatic_tumor"),
        patch(
            "Tool_Box.crlm_postprocess.optimize_crlm_vessels",
            side_effect=write_optimized_variants,
        ),
    ):
        mask_names = run_postprocessing(str(tmp_path))

    assert mask_names == ["hepatic", "liver", "portal"]
    assert (tmp_path / "hepatic_optimized.nii.gz").exists()
    assert (tmp_path / "portal_optimized.nii.gz").exists()


@pytest.mark.parametrize(
    "max_gap_mm",
    [0.0, -0.1, float("nan"), float("inf"), 4.0001],
)
def test_crlm_public_entry_point_enforces_four_mm_cap(tmp_path, max_gap_mm):
    with pytest.raises(ValueError, match="max_gap_mm"):
        optimize_crlm_vessels(
            tmp_path,
            tmp_path / "vessel_optimization_report.json",
            max_gap_mm=max_gap_mm,
        )


def test_final_added_voxel_intersection_rejects_both_vessel_publications(
    tmp_path,
):
    shape = (16, 16, 16)
    liver = np.ones(shape, dtype=np.uint8)
    hepatic = np.zeros(shape, dtype=np.uint8)
    portal = np.zeros(shape, dtype=np.uint8)
    hepatic[2, 8, 8] = 1
    portal[8, 2, 8] = 1
    _save(tmp_path / "liver.nii.gz", liver)
    _save(tmp_path / "hepatic.nii.gz", hepatic)
    _save(tmp_path / "portal.nii.gz", portal)
    report_path = tmp_path / "vessel_optimization_report.json"

    def write_overlapping(vessel_path, liver_path, tumor_paths, output_path, **kwargs):
        raw_image = nib.load(vessel_path)
        optimized = np.asanyarray(raw_image.dataobj).astype(np.uint8)
        optimized[8, 8, 8] = 1
        image = nib.Nifti1Image(optimized, raw_image.affine, raw_image.header)
        image.header.set_xyzt_units("mm")
        nib.save(image, output_path)
        return {
            "status": "optimized",
            "source_path": str(vessel_path),
            "output_path": str(output_path),
            "voxel_count_before": 1,
            "voxel_count_after": 2,
            "added_voxels": 1,
            "accepted_connections": [],
            "rejected_candidate_counts": {"distance": 0},
            "warnings": [],
            "config": {"max_gap_mm": kwargs["max_gap_mm"]},
        }

    with patch(
        "Tool_Box.crlm_postprocess.optimize_vessel_mask",
        side_effect=write_overlapping,
    ):
        report = optimize_crlm_vessels(tmp_path, report_path)

    assert report["vessels"]["hepatic"]["status"] == (
        "rejected_final_cross_vessel_overlap"
    )
    assert report["vessels"]["portal"]["status"] == (
        "rejected_final_cross_vessel_overlap"
    )
    assert report["cross_vessel_added_overlap_voxels"] is None
    assert report["final_cross_vessel_overlap_voxels"] == 1
    assert not (tmp_path / "hepatic_optimized.nii.gz").exists()
    assert not (tmp_path / "portal_optimized.nii.gz").exists()
    assert json.loads(report_path.read_text("utf-8")) == report


def test_concurrent_case_publications_are_serialized(tmp_path):
    shape = (8, 8, 8)
    for name in ("liver", "hepatic", "portal"):
        _save(tmp_path / f"{name}.nii.gz", np.ones(shape, dtype=np.uint8))
    report_path = tmp_path / "vessel_optimization_report.json"
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def slow_optimizer(vessel_path, liver_path, tumor_paths, output_path, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            raw = nib.load(vessel_path)
            nib.save(
                nib.Nifti1Image(
                    np.asanyarray(raw.dataobj).astype(np.uint8),
                    raw.affine,
                    raw.header,
                ),
                output_path,
            )
            return {
                "status": "optimized",
                "source_path": str(vessel_path),
                "output_path": str(output_path),
                "voxel_count_before": int(np.prod(shape)),
                "voxel_count_after": int(np.prod(shape)),
                "added_voxels": 0,
                "accepted_connections": [],
                "rejected_candidate_counts": {"distance": 0},
                "warnings": [],
                "config": {"max_gap_mm": kwargs["max_gap_mm"]},
            }
        finally:
            with state_lock:
                active -= 1

    def publish_once():
        return optimize_crlm_vessels(tmp_path, report_path)

    with patch(
        "Tool_Box.crlm_postprocess.optimize_vessel_mask",
        side_effect=slow_optimizer,
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            reports = list(executor.map(lambda _: publish_once(), range(2)))

    assert len(reports) == 2
    assert max_active == 1
    saved = json.loads(report_path.read_text("utf-8"))
    assert saved["generation_id"] in {
        report["generation_id"] for report in reports
    }

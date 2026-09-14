from contextlib import contextmanager
import sys
import types

import nibabel as nib
import numpy as np
import pytest

import API
from Tool_Box.runtime_diagnostics import (
    CudaEnvironmentError,
    NiftiGeometryError,
    Vista3DEnvironmentError,
    Vista3DInferenceError,
    exception_chain,
    inspect_nifti_geometry,
    is_retryable_cuda_error,
    validate_nifti_geometry,
)


def _save_nifti(path, shape):
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4)), path)


def test_nifti_geometry_accepts_3d_and_rejects_4d(tmp_path):
    valid_path = tmp_path / "valid.nii.gz"
    invalid_path = tmp_path / "four-dimensional.nii.gz"
    _save_nifti(valid_path, (8, 9, 10))
    _save_nifti(invalid_path, (8, 9, 10, 1))

    valid = validate_nifti_geometry(valid_path)
    invalid = inspect_nifti_geometry(invalid_path)

    assert valid["valid"] is True
    assert valid["voxel_spacing_mm"] == [1.0, 1.0, 1.0]
    assert invalid["valid"] is False
    with pytest.raises(NiftiGeometryError, match="expects one 3D CT volume"):
        validate_nifti_geometry(invalid_path)


def test_exception_chain_exposes_monai_root_cause():
    captured = None
    try:
        try:
            raise ValueError("invalid affine spacing")
        except ValueError as root:
            raise RuntimeError("applying transform <Spacingd>") from root
    except RuntimeError as wrapped:
        captured = wrapped
        rendered = exception_chain(wrapped)

    assert "RuntimeError: applying transform <Spacingd>" in rendered
    assert "ValueError: invalid affine spacing" in rendered
    assert captured is not None
    assert not is_retryable_cuda_error(captured)


def test_retry_classifier_only_accepts_cuda_resource_failures():
    assert is_retryable_cuda_error(RuntimeError("CUDA out of memory"))
    assert not is_retryable_cuda_error(RuntimeError("applying transform <Spacingd>"))


def test_vista_spacing_failure_is_not_retried(monkeypatch, tmp_path):
    attempts = []

    class FakeSegmentator:
        def __init__(self, **_kwargs):
            attempts.append("init")

        def segment(self, **_kwargs):
            try:
                raise ValueError("bad voxel spacing")
            except ValueError as root:
                raise RuntimeError("applying transform <Spacingd>") from root

    fake_module = types.ModuleType("vista3d_Segmentator")
    fake_module.Vista3D_Segmentator = FakeSegmentator
    monkeypatch.setitem(sys.modules, "vista3d_Segmentator", fake_module)
    monkeypatch.setattr(
        API,
        "collect_runtime_diagnostics",
        lambda: {
            "python": "test",
            "packages": {"numpy": "test", "monai": "1.3.2", "torch": "test"},
            "vista3d_compatibility_errors": [],
        },
    )
    monkeypatch.setattr(API, "validate_nifti_geometry", lambda _path: {"shape": [1, 1, 1]})
    monkeypatch.setattr(
        API,
        "require_cuda_device",
        lambda _device: {
            "torch": "test",
            "torch_cuda": "test",
            "devices": [{"index": 0, "name": "Test GPU"}],
        },
    )

    @contextmanager
    def allocate(preferred=None):
        yield preferred or "cuda:0"

    monkeypatch.setattr(API._gpu_manager, "allocate", allocate)

    with pytest.raises(Vista3DInferenceError, match="bad voxel spacing"):
        API._run_vista3d_segmentation(
            "ct.nii.gz", str(tmp_path), organ_list=["liver"], device="cuda:0"
        )
    assert attempts == ["init"]


def test_vista_rejects_incompatible_environment_before_loading_model(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        API,
        "collect_runtime_diagnostics",
        lambda: {
            "python": "test",
            "packages": {"numpy": "2.2.6", "monai": "1.6.0", "torch": "test"},
            "vista3d_compatibility_errors": [
                "The VISTA3D research integration expects MONAI 1.3.2."
            ],
        },
    )
    monkeypatch.setattr(
        API,
        "validate_nifti_geometry",
        lambda _path: pytest.fail("input validation should not run"),
    )

    with pytest.raises(Vista3DEnvironmentError, match="MONAI 1.3.2"):
        API._run_vista3d_segmentation(
            "ct.nii.gz", str(tmp_path), organ_list=["liver"], device="cuda:0"
        )


def test_process_lite_http_error_mapping():
    assert API._segmentation_http_error(NiftiGeometryError("bad")) == (
        422,
        "invalid_nifti_geometry",
    )
    assert API._segmentation_http_error(CudaEnvironmentError("bad")) == (
        503,
        "cuda_unavailable",
    )
    assert API._segmentation_http_error(Vista3DEnvironmentError("bad")) == (
        503,
        "vista3d_environment_incompatible",
    )
    assert API._segmentation_http_error(Vista3DInferenceError("bad")) == (
        500,
        "segmentation_failed",
    )

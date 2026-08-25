import json
from pathlib import Path

import pytest

import API


def test_default_and_alias_backend_names(monkeypatch):
    monkeypatch.setattr(API, "_DEFAULT_SEGMENTATION_BACKEND", "vista3d")

    assert API._normalize_segmentation_backend(None) == "vista3d"
    assert API._normalize_segmentation_backend("VISTA") == "vista3d"
    assert API._normalize_segmentation_backend("TotalSeg") == "totalsegmentator"

    with pytest.raises(ValueError, match="Unsupported segmentation backend"):
        API._normalize_segmentation_backend("unknown-model")


def test_dispatcher_selects_only_requested_backend(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        API,
        "_run_vista3d_segmentation",
        lambda **kwargs: calls.append(("vista3d", kwargs)),
    )
    monkeypatch.setattr(
        API,
        "_run_totalsegmentator_segmentation",
        lambda **kwargs: calls.append(("totalsegmentator", kwargs)),
    )

    selected = API._run_segmentation(
        "totalsegmentator",
        nifti_path="ct.nii.gz",
        output_dir=str(tmp_path),
        device="cuda:1",
    )

    assert selected == "totalsegmentator"
    assert calls == [
        (
            "totalsegmentator",
            {
                "nifti_path": "ct.nii.gz",
                "output_dir": str(tmp_path),
                "device": "cuda:1",
            },
        )
    ]


def test_completion_marker_records_backend_atomically(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "liver.nii.gz").write_bytes(b"x" * 1000)

    API._write_segmentation_metadata(str(mask_dir), "total")

    marker = mask_dir / API._SEGMENTATION_METADATA_FILENAME
    assert json.loads(marker.read_text(encoding="utf-8"))["backend"] == "totalsegmentator"
    assert API._read_segmentation_backend(str(mask_dir)) == "totalsegmentator"
    assert API._is_case_complete(str(mask_dir), "totalsegmentator")
    assert not API._is_case_complete(str(mask_dir), "vista3d")
    assert not Path(str(marker) + ".tmp").exists()

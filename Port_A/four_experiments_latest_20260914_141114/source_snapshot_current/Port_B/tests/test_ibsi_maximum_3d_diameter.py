from pathlib import Path

import pytest

from Tool_Box.liver_analysis import compute_tumor_diameter


def test_ibsi_digital_phantom_maximum_3d_diameter():
    phantom = Path(
        "E:/DeepTumour_agent_eval/diameter_validation/ibsi/"
        "nifti/mask/mask.nii.gz"
    )
    if not phantom.exists():
        pytest.skip("IBSI digital phantom is not installed")

    result = compute_tumor_diameter(str(phantom))

    # IBSI reference: 13.1 mm with tolerance 0.1 mm.
    assert result["method"] == "ibsi_surface_mesh"
    assert abs(result["max_diameter_mm"] - 13.1) <= 0.1

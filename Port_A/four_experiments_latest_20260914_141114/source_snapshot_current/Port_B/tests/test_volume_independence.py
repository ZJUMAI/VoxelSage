import sys
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tool_Box.liver_analysis import analyze_liver_case

def test_volume_survives_diameter_failure_and_sums_before_rounding(tmp_path):
    mask = tmp_path / 'tumor.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((3,3,3), dtype=np.uint8), np.eye(4)), mask)
    with patch('Tool_Box.liver_analysis._compute_diameter_on_mask', side_effect=RuntimeError('geometry failure')):
        result = analyze_liver_case(str(mask), {'tumor_1':str(mask), 'tumor_2':str(mask)})
    assert result['total_tumor_volume_cm3'] == 0.05  # 0.027*2; not 0.03*2
    assert result['volume_lesion_count'] == 2
    assert result['largest_tumor_volume_cm3'] == 0.03
    assert all(t['diameter']['max_diameter_mm'] is None for t in result['tumor_results'].values())
    assert all(t['volume_cm3_unrounded'] == 0.027 for t in result['tumor_results'].values())

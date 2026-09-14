from pathlib import Path
import sys
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tool_Box.liver_analysis import analyze_liver_case, generate_liver_report


def test_empty_reference_mask_is_not_a_lesion(tmp_path):
    affine = np.diag([2., 3., 4., 1.])
    paths = {}
    for name in ['ct', 'liver', 'hepatic tumor']:
        data = np.ones((5, 5, 5), dtype=np.uint8) if name == 'liver' else np.zeros((5, 5, 5), dtype=np.uint8)
        path = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(data, affine), path)
        paths[name] = str(path)
    result = analyze_liver_case(paths['ct'], {k: v for k, v in paths.items() if k != 'ct'})
    assert result['tumor_results'] == {}
    assert abs(result['liver_volume_cm3'] - 3.) < 1e-9
    assert '肿瘤总数: 1' not in generate_liver_report(result)


def test_empty_instance_does_not_hide_nonempty_instance(tmp_path):
    affine = np.eye(4)
    paths = {}
    for name in ['ct', 'tumor_1', 'tumor_2']:
        data = np.zeros((5, 5, 5), dtype=np.uint8)
        if name == 'tumor_1':
            data[1:4, 1:4, 1:4] = 1
        path = tmp_path / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(data, affine), path)
        paths[name] = str(path)
    result = analyze_liver_case(paths['ct'], {k: v for k, v in paths.items() if k != 'ct'})
    assert set(result['tumor_results']) == {'tumor_1'}
    assert result['tumor_results']['tumor_1']['tumor_voxels'] == 27
    assert '肿瘤总数: 1' in generate_liver_report(result)

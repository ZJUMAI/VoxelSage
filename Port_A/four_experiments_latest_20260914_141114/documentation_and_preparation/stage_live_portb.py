"""Stage immutable-source working copies for real Port B computation (WSL)."""
import json
import shutil
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

import os
ROOT = Path(os.environ.get('LIVE_DATA_ROOT','/mnt/e/DeepTumour_agent_eval/live_portb_v1'))
REPO = Path(__file__).resolve().parents[1] / 'VoxelSage-main'
sys.path.insert(0, str(REPO / 'Port_B'))
from Tool_Box.crlm_postprocess import split_hepatic_tumor

audit_path = ROOT / 'bodymaps_small_cohort.json'
if not audit_path.exists():
    audit_path = ROOT / 'geometry_audit.json'
audit = [row for row in json.loads(audit_path.read_text()) if row.get('geometry_passed')]
records = []
for row in audit:
    cid = row['case']
    mask_src = ROOT / 'source_cases' / cid
    bodymaps_ct = ROOT / 'bodymaps_verified' / cid / 'ct.nii.gz'
    ct_src = bodymaps_ct if bodymaps_ct.exists() else mask_src / 'ct.nii.gz'
    ct = nib.load(ct_src)
    masks = [nib.load(mask_src / f'{name}.nii.gz') for name in ['liver', 'liver_lesion']]
    assert all(m.shape == ct.shape and np.allclose(m.affine, ct.affine, atol=1e-5) for m in masks)
    case_id = 'LIVE_' + cid
    dest = ROOT / 'working_cases' / case_id
    dest.mkdir(parents=True, exist_ok=True)
    mask_dir = dest / 'masks'
    mask_dir.mkdir(exist_ok=True)
    ct_link = dest / 'ct.nii.gz'
    if ct_link.is_symlink() or ct_link.exists():
        ct_link.unlink()
    ct_link.symlink_to(ct_src)
    for old in mask_dir.glob('*.nii.gz'):
        old.unlink()
    shutil.copy2(mask_src / 'liver.nii.gz', mask_dir / 'liver.nii.gz')
    shutil.copy2(mask_src / 'liver_lesion.nii.gz', mask_dir / 'hepatic tumor.nii.gz')
    components = split_hepatic_tumor(str(mask_dir))
    (dest / 'staged.json').write_text(json.dumps({'ct_source': str(ct_src), 'mask_source': str(mask_src),
                                                  'components': components, 'min_voxels': 10,
                                                  'scope': 'same-grid reference-mask measurements'}))
    link = REPO / 'Port_B/output' / case_id
    if not link.exists():
        link.symlink_to(dest, target_is_directory=True)
    # A separate absent-input fixture causes a genuine 404 from the real API.
    missing_id = 'MISSING_' + cid
    fixture = ROOT / 'working_cases' / missing_id
    fixture.mkdir(exist_ok=True)
    (fixture / 'masks').mkdir(exist_ok=True)
    missing_link = REPO / 'Port_B/output' / missing_id
    if not missing_link.exists():
        missing_link.symlink_to(fixture, target_is_directory=True)
    records.append({'case': cid, 'port_b_case': case_id, 'missing_case': missing_id,
                    'scope': 'same_grid_reference_masks', 'ct_overlay_validated': True})
(ROOT / 'live_cases.json').write_text(json.dumps(records, indent=2))
print('STAGED', len(records), flush=True)

"""
Shared segmentation utilities for the GeoSurge Port B pipeline.

Used by segmentation.py (batch) and API.py (single-case / HTTP API)
to avoid duplicating organ lists, label-dict loading, mask extraction,
and kidney-merging logic.
"""

import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# Organ list
# ============================================================

ORGAN_LIST = [
    "liver",
    "spleen",
    "pancreas",
    "colon",
    "left kidney",
    "right kidney",
    "pancreatic tumor",
    "hepatic tumor",
    "left kidney cyst",
    "right kidney cyst",
    "hepatic vessel",                # ID 25 — VISTA3D 肝静脉
    "portal vein and splenic vein",  # ID 17 — VISTA3D 门静脉+脾静脉
]


# ============================================================
# Label-dict helpers
# ============================================================

def load_vista3d_label_dict(vista3d_root: str | Path) -> dict:
    """Load VISTA3D label_dict.json → {organ_name: label_id}."""
    path = Path(vista3d_root) / "label_dict.json"
    with open(path) as f:
        return json.load(f)


def build_vista3d_object_ids(
    organ_list: list[str],
    vista3d_root: str | Path,
) -> list[int]:
    """Map organ names to VISTA3D integer label IDs."""
    organ2id = load_vista3d_label_dict(vista3d_root)
    return [organ2id[o] for o in organ_list if o in organ2id]


# ============================================================
# NIfTI I/O
# ============================================================

def save_binary_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    header,
    save_path: str | Path,
) -> None:
    """Save a binary mask as uint8 NIfTI."""
    nii = nib.Nifti1Image(mask.astype(np.uint8, copy=False), affine, header)
    nii.set_data_dtype(np.uint8)
    nib.save(nii, str(save_path))


def robust_load_affine_header(
    input_path: str,
    retries: int = 6,
    sleep: float = 0.35,
    tag: str = "",
):
    """
    Best-effort retry loader for NFS/blob transient errors.
    Returns (affine, header) or raises the last exception.
    """
    last_err = None
    for t in range(retries):
        try:
            ref = nib.load(input_path)
            return ref.affine, ref.header
        except Exception as e:
            last_err = e
            time.sleep(sleep * (t + 1))
    raise last_err


# ============================================================
# Mask extraction & merging
# ============================================================

def extract_organ_masks(
    all_mask: np.ndarray,
    organ2label: dict[str, int],
    organ_list: list[str],
    affine: np.ndarray,
    header,
    out_dir: str | Path,
) -> tuple[dict, int]:
    """
    Extract per-organ binary masks from a multi-label segmentation volume.

    Args:
        all_mask: (H,W,D) integer label map from VISTA3D / BiomedParse.
        organ2label: {organ_name: label_id} mapping.
        organ_list: organs to extract.
        affine, header: NIfTI spatial metadata (from the input CT).
        out_dir: directory to save individual .nii.gz masks.

    Returns:
        (label_masks, saved_count)
        label_masks: {label_id: bool_ndarray} for present organs.
        saved_count: number of successfully saved organ masks.
    """
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    needed_ids = {organ2label[o] for o in organ_list if o in organ2label}
    present = np.unique(all_mask)
    label_masks = {lid: (all_mask == lid) for lid in present if lid in needed_ids}

    saved = 0
    for organ in organ_list:
        lid = organ2label.get(organ)
        if lid is None or lid not in label_masks:
            continue
        save_binary_mask(
            label_masks[lid], affine, header,
            out_dir / f"{organ}.nii.gz",
        )
        saved += 1

    return label_masks, saved


def merge_kidney_masks(
    label_masks: dict[int, np.ndarray],
    organ2label: dict[str, int],
    affine: np.ndarray,
    header,
    out_dir: str | Path,
) -> list[str]:
    """
    Merge left+right kidney and left+right kidney cyst into combined masks.

    Only merges when BOTH sides are present in label_masks.

    Returns:
        list of merged output names (e.g. ["kidney", "kidney cyst"]).
    """
    out_dir = Path(out_dir)
    merged = []

    for left_key, right_key, out_name in [
        ("left kidney", "right kidney", "kidney"),
        ("left kidney cyst", "right kidney cyst", "kidney cyst"),
    ]:
        lk = organ2label.get(left_key)
        rk = organ2label.get(right_key)
        if lk is not None and rk is not None and lk in label_masks and rk in label_masks:
            merged_mask = label_masks[lk] | label_masks[rk]
            save_binary_mask(merged_mask, affine, header, out_dir / f"{out_name}.nii.gz")
            merged.append(out_name)

    return merged

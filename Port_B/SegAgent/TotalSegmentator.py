import os
import sys
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from totalsegmentator.config import setup_nnunet
from totalsegmentator.nnunet import nnUNet_predict_image

# VISTA3D 路径
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VISTA3D_DIR = _PROJECT_ROOT / "SegAgent" / "VISTA3d"
if str(_VISTA3D_DIR) not in sys.path:
    sys.path.insert(0, str(_VISTA3D_DIR))

# ============================================================
# TotalSegmentator CRLM Wrapper
# Uses Python API to run Part 1 organs + liver_vessels tasks
# ============================================================

# Part 1 organs class map (task 291)
PART1_ORGANS_MAP = {
    5: "liver",
}

# liver_vessels class map (task 8)
LIVER_VESSELS_MAP = {
    1: "hepatic_vessels",
    2: "liver_tumor",
}

# Total task class map for portal vein (needs Part 3 cardiac model)
# Task 293, class 14 = portal_vein_and_splenic_vein
TOTAL_PORTAL_VEIN_ID = 14  # in Part 3 cardiac class map


def setup_env():
    """Initialize nnUNet environment for TotalSegmentator"""
    os.environ.setdefault(
        "TOTALSEG_HOME_DIR",
        str(Path.home() / ".cache" / "totalsegmentator"),
    )
    setup_nnunet()


def run_part1_organs(input_path, output_dir, device="cuda"):
    """
    Run Part 1 organs model (task 291) to segment liver.
    Saves liver.nii.gz binary mask.
    """
    print("[TotalSeg] Running Part 1 organs model...")
    out_path = str(Path(output_dir) / "part1_organs.nii.gz")

    seg, _, _ = nnUNet_predict_image(
        file_in=input_path,
        file_out=out_path,
        task_id=291,
        model="3d_fullres",
        folds=[0],
        trainer="nnUNetTrainerNoMirroring",
        multilabel_image=True,
        resample=1.5,
        device=device,
    )

    data = seg.get_fdata().astype(np.int32)
    affine = seg.affine

    # Extract liver (ID 5)
    liver_mask = (data == 5).astype(np.uint8)
    liver_path = Path(output_dir) / "liver.nii.gz"
    nib.save(nib.Nifti1Image(liver_mask, affine), str(liver_path))
    print(f"  [TotalSeg] Liver: {np.count_nonzero(liver_mask)} voxels -> {liver_path}")

    # Also extract spleen (ID 1) - useful for CRLM analysis
    spleen_mask = (data == 1).astype(np.uint8)
    spleen_path = Path(output_dir) / "spleen.nii.gz"
    nib.save(nib.Nifti1Image(spleen_mask, affine), str(spleen_path))
    print(f"  [TotalSeg] Spleen: {np.count_nonzero(spleen_mask)} voxels -> {spleen_path}")

    return seg


def run_liver_vessels(input_path, output_dir, device="cuda"):
    """
    Run liver_vessels model (task 8) to segment hepatic vessels and tumor.
    Saves hepatic.nii.gz and tumor multi-label file.
    """
    print("[TotalSeg] Running liver_vessels model...")
    out_path = str(Path(output_dir) / "liver_vessels.nii.gz")

    seg, _, _ = nnUNet_predict_image(
        file_in=input_path,
        file_out=out_path,
        task_id=8,
        model="3d_fullres",
        folds=[0],
        trainer="nnUNetTrainer",  # HepaticVessel uses nnUNetTrainer (not NoMirroring)
        multilabel_image=True,
        resample=1.5,
        device=device,
    )

    data = seg.get_fdata().astype(np.int32)
    affine = seg.affine

    # Extract hepatic vessels (ID 1)
    vessel_mask = (data == 1).astype(np.uint8)
    vessel_path = Path(output_dir) / "hepatic.nii.gz"
    nib.save(nib.Nifti1Image(vessel_mask, affine), str(vessel_path))
    print(f"  [TotalSeg] Hepatic vessels: {np.count_nonzero(vessel_mask)} voxels -> {vessel_path}")

    # Extract liver tumor (ID 2) and split into connected components
    tumor_mask = (data == 2).astype(np.uint8)
    tumor_voxels = np.count_nonzero(tumor_mask)
    print(f"  [TotalSeg] Liver tumor raw: {tumor_voxels} voxels")

    if tumor_voxels > 0:
        # Connected component labeling
        labeled, n_tumors = ndimage.label(tumor_mask)
        tumor_paths = []
        for i in range(1, n_tumors + 1):
            single_tumor = (labeled == i).astype(np.uint8)
            n_voxels = np.count_nonzero(single_tumor)
            if n_voxels < 10:  # Filter noise
                continue
            t_path = Path(output_dir) / f"tumor_{len(tumor_paths) + 1}.nii.gz"
            nib.save(nib.Nifti1Image(single_tumor, affine), str(t_path))
            tumor_paths.append(t_path)
            print(f"  [TotalSeg]   Tumor {len(tumor_paths)}: {n_voxels} voxels -> {t_path}")

        # Save merged tumor mask too
        all_tumor_path = Path(output_dir) / "tumor_all.nii.gz"
        nib.save(nib.Nifti1Image(tumor_mask, affine), str(all_tumor_path))
    else:
        print("  [TotalSeg]   No tumor found")

    return seg


def run_portal_vein(input_path, output_dir, device="cuda"):
    """
    Run Part 3 cardiac model (task 293) for portal vein segmentation.
    Only works if the model is downloaded.
    Returns True if successful, False if model not found.
    """
    model_dir = (
        Path(os.environ.get("TOTALSEG_HOME_DIR", "~/.totalsegmentator")).expanduser()
        / "nnunet"
        / "results"
        / "Dataset293_TotalSegmentator_part3_cardiac_1559subj"
    )

    if not model_dir.exists():
        print("[TotalSeg] Part 3 cardiac model not found, skipping portal vein")
        return False

    print("[TotalSeg] Running Part 3 cardiac model for portal vein...")
    out_path = str(Path(output_dir) / "part3_cardiac.nii.gz")

    seg, _, _ = nnUNet_predict_image(
        file_in=input_path,
        file_out=out_path,
        task_id=293,
        model="3d_fullres",
        folds=[0],
        trainer="nnUNetTrainerNoMirroring",
        multilabel_image=True,
        resample=1.5,
        device=device,
    )

    data = seg.get_fdata().astype(np.int32)
    affine = seg.affine

    # Portal vein is ID 14 in Part 3 cardiac class map
    portal_mask = (data == TOTAL_PORTAL_VEIN_ID).astype(np.uint8)
    portal_path = Path(output_dir) / "portal.nii.gz"
    nib.save(nib.Nifti1Image(portal_mask, affine), str(portal_path))
    print(f"  [TotalSeg] Portal vein: {np.count_nonzero(portal_mask)} voxels -> {portal_path}")

    return True


def run_crlm_segmentation(input_path, output_dir, device="cuda", skip_portal=False):
    """
    Run full CRLM segmentation pipeline with TotalSegmentator.

    Args:
        input_path: Path to CT NIfTI file
        output_dir: Output directory for masks
        device: "cuda" or "cpu"
        skip_portal: Skip portal vein segmentation

    Returns:
        dict with paths to generated masks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_env()

    # Step 1: Part 1 organs -> liver + spleen
    run_part1_organs(input_path, output_dir, device)

    # Step 2: liver_vessels -> hepatic vessels + tumor
    run_liver_vessels(input_path, output_dir, device)

    # Step 3: Portal vein (if model available)
    if not skip_portal:
        run_portal_vein(input_path, output_dir, device)

    # Summarize
    print("\n[TotalSeg] Segmentation complete! Files generated:")
    for f in sorted(output_dir.glob("*.nii.gz")):
        size = f.stat().st_size
        print(f"  {f.name} ({size / 1024:.1f} KB)")

    return {
        "liver": str(output_dir / "liver.nii.gz"),
        "spleen": str(output_dir / "spleen.nii.gz"),
        "hepatic": str(output_dir / "hepatic.nii.gz"),
        "portal": str(output_dir / "portal.nii.gz") if (output_dir / "portal.nii.gz").exists() else None,
        "tumor_all": str(output_dir / "tumor_all.nii.gz") if (output_dir / "tumor_all.nii.gz").exists() else None,
    }


def run_hybrid_segmentation(input_path, output_dir, device="cuda",
                             vista3d_config=None):
    """
    VISTA3D 全结构分割（CRLM 微调权重）

    用 VISTA3D CRLM 微调权重同时分割肝脏、肝静脉、门静脉、肝肿瘤，
    替代混合方案中 TotalSegmentator 负责的部分。

    Args:
        input_path: CT NIfTI 路径
        output_dir: 输出目录
        device: "cuda" 或 "cuda:X"
        vista3d_config: VISTA3D 推理配置文件路径

    Returns:
        dict of mask paths
    """
    from vista3d_Segmentator import Vista3D_Segmentator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── VISTA3D 全结构分割 ──
    print("[VISTA3D] Running CRLM segmentation (IDs: 1=liver, 25=hepatic, 17=portal, 26=tumor) ...")
    if vista3d_config is None:
        vista3d_config = str(_PROJECT_ROOT / "SegAgent" / "VISTA3d" / "configs" / "infer_finetuned_e2.yaml")

    seg = Vista3D_Segmentator(config_file=vista3d_config, device=device)
    raw_path = str(output_dir / "vista3d_raw.nii.gz")

    all_mask = seg.segment(
        input_path=str(input_path),
        output_path=raw_path,
        object_list=[1, 25, 17, 26],  # liver, hepatic, portal, tumor
        save_mask=True,
    )

    # ── 提取各类别 ──
    affine = nib.load(str(input_path)).affine

    # 肝脏 (ID 1)
    liver = (all_mask == 1).astype(np.uint8)
    liver_path = output_dir / "liver.nii.gz"
    nib.save(nib.Nifti1Image(liver, affine), str(liver_path))
    print(f"  [VISTA3D] Liver: {liver.sum():,} vox -> {liver_path}")

    # 肝静脉 (ID 25)
    hepatic = (all_mask == 25).astype(np.uint8)
    hepatic_path = output_dir / "hepatic.nii.gz"
    nib.save(nib.Nifti1Image(hepatic, affine), str(hepatic_path))
    print(f"  [VISTA3D] Hepatic: {hepatic.sum():,} vox -> {hepatic_path}")

    # 门静脉 (ID 17)
    portal = (all_mask == 17).astype(np.uint8)
    portal_path = output_dir / "portal.nii.gz"
    nib.save(nib.Nifti1Image(portal, affine), str(portal_path))
    print(f"  [VISTA3D] Portal: {portal.sum():,} vox -> {portal_path}")

    # 肝肿瘤 (ID 26) → 连通域拆分
    tumor_raw = (all_mask == 26).astype(np.uint8)
    tumor_voxels = tumor_raw.sum()
    print(f"  [VISTA3D] Tumor raw: {tumor_voxels:,} vox")

    if tumor_voxels > 0:
        labeled, n_tumors = ndimage.label(tumor_raw)
        tumor_count = 0
        for i in range(1, n_tumors + 1):
            single = (labeled == i).astype(np.uint8)
            n_voxels = single.sum()
            if n_voxels < 10:  # Filter noise
                continue
            tumor_count += 1
            t_path = output_dir / f"tumor_{tumor_count}.nii.gz"
            nib.save(nib.Nifti1Image(single, affine), str(t_path))
            print(f"  [VISTA3D]   Tumor {tumor_count}: {n_voxels:,} vox -> {t_path}")

        # Save merged tumor mask
        nib.save(nib.Nifti1Image(tumor_raw, affine), str(output_dir / "tumor_all.nii.gz"))

        if tumor_count == 0:
            print("  [VISTA3D]   All tumors filtered as noise (<10 voxels)")
    else:
        print("  [VISTA3D]   No tumor found")

    # ── 汇总 ──
    print("\n[VISTA3D] Segmentation complete! Files generated:")
    for f in sorted(output_dir.glob("*.nii.gz")):
        if f.name != "vista3d_raw.nii.gz":
            size = f.stat().st_size
            print(f"  {f.name} ({size / 1024:.1f} KB)")

    return {
        "liver": str(liver_path),
        "hepatic": str(hepatic_path),
        "portal": str(portal_path),
        "tumor_all": str(output_dir / "tumor_all.nii.gz") if tumor_voxels > 0 else None,
    }


# Legacy CLI interface for VQA pipeline
def run_totalsegmentator_cli(input_path, out_case_dir, organ_list, device="gpu"):
    """
    Call TotalSegmentator via CLI (legacy path for VQA pipeline).
    """
    organ_map = {
        "liver": "liver",
        "spleen": "spleen",
        "pancreas": "pancreas",
        "colon": "colon",
        "left kidney": "kidney_left",
        "right kidney": "kidney_right",
        "left kidney cyst": "kidney_cyst_left",
        "right kidney cyst": "kidney_cyst_right",
    }

    # Build roi_subset from organ_list
    roi_args = [organ_map.get(o, o) for o in organ_list if o in organ_map]

    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", out_case_dir,
        "-ta", "total",
        "-d", device,
        "--ml",  # single multilabel file for speed
    ]
    if roi_args:
        cmd += ["--roi_subset"] + roi_args

    env = os.environ.copy()
    totalseg_home = env.get(
        "TOTALSEG_HOME_DIR",
        str(Path.home() / ".cache" / "totalsegmentator"),
    )
    os.makedirs(totalseg_home, exist_ok=True)
    env["TOTALSEG_HOME_DIR"] = totalseg_home

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TotalSegmentator failed "
            f"(returncode={result.returncode})\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TotalSegmentator CRLM segmentation")
    parser.add_argument("-i", "--input", required=True, help="Input CT NIfTI path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-d", "--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--skip-portal", action="store_true", help="Skip portal vein")
    args = parser.parse_args()

    run_crlm_segmentation(args.input, args.output, args.device, args.skip_portal)

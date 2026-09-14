"""tumor_vessel_distance: 计算肿瘤到血管的最短距离。

对每个肿瘤和每种血管，加载对应的二值掩码，调用
Tool_Box.liver_analysis.compute_tumor_vessel_distance 计算最小表面距离。
"""

import os
import nibabel as nib
from Tool_Box.crlm_postprocess import split_hepatic_tumor
from Tool_Box.liver_analysis import compute_tumor_vessel_distance
from Tool_Box.mask_resolution import resolve_mask_path
from skills.utils import convert_numpy


def _find_existing_tumor_masks(mask_dir: str) -> list:
    """纯发现函数：扫描 mask_dir 中已有的 tumor_* 掩码文件名列表。

    无副作用，仅读取文件系统目录列表。
    """
    import glob

    names = []
    for fname in sorted(glob.glob(os.path.join(mask_dir, "tumor_*.nii.gz"))):
        stem = os.path.splitext(os.path.basename(fname))[0]
        if stem.endswith(".nii"):
            stem = stem[:-4]
        names.append(stem)
    return names


def _ensure_tumor_masks(mask_dir: str, tumor_label: str) -> list:
    """确保肿瘤掩码已分裂为连通分量。

    有副作用：如果存在 ``tumor_label.nii.gz``（如 ``hepatic tumor.nii.gz``），
    调用 ``split_hepatic_tumor`` 将其分裂为 ``tumor_1.nii.gz, tumor_2.nii.gz, ...``。

    返回肿瘤掩码名称列表。
    """
    import glob

    # 先检查是否已有分裂后的文件
    existing = _find_existing_tumor_masks(mask_dir)
    if existing:
        return existing

    # 尝试连通域分裂（写入文件）
    tumor_path = os.path.join(mask_dir, f"{tumor_label}.nii.gz")
    if os.path.exists(tumor_path):
        names = split_hepatic_tumor(mask_dir)
        if names:
            return names

    # 直接使用原始肿瘤掩码（如果没有分裂文件）
    if os.path.exists(tumor_path):
        return [tumor_label]

    return []


def _discover_vessel_masks(mask_dir: str, vessel_names: list) -> dict:
    """发现有效存在的血管掩码文件。"""
    valid = {}
    for name in vessel_names:
        try:
            valid[name] = resolve_mask_path(mask_dir, name)
        except FileNotFoundError:
            continue
    return valid


def run(ctx):
    tumor_label = ctx.params.get("tumor_label", "hepatic tumor")
    vessel_names = ctx.params.get("vessel_names", ["hepatic", "portal"])

    # 确保肿瘤掩码已分裂，再获取列表
    tumor_names = _ensure_tumor_masks(ctx.mask_dir, tumor_label)
    if not tumor_names:
        return {"distances": [], "message": "No tumor masks found"}

    # 发现血管列表
    available_vessels = _discover_vessel_masks(ctx.mask_dir, vessel_names)
    if not available_vessels:
        return {"distances": [], "message": "No vessel masks found"}

    # 获取体素间距
    spacing = ctx.get_voxel_spacing()

    # 对每个肿瘤 × 每种血管组合计算距离
    results = []
    for tumor_name in tumor_names:
        tumor_mask = ctx.get_mask(tumor_name)
        for vessel_name, resolved in available_vessels.items():
            vessel_mask = nib.load(resolved.path).get_fdata() > 0
            ctx.log(f"Computing distance: {tumor_name} → {vessel_name}...")
            dist = compute_tumor_vessel_distance(tumor_mask, vessel_mask, spacing)
            results.append({
                "tumor_id": tumor_name,
                "vessel_name": vessel_name,
                "mask_variant": resolved.variant,
                "min_distance_mm": dist.get("min_distance_mm"),
                "interpretation": dist.get("interpretation", ""),
                "tumor_contacts_vessel": dist.get("tumor_contacts_vessel", False),
                "tumor_voxels": dist.get("tumor_voxels", 0),
            })

    return convert_numpy({"distances": results})

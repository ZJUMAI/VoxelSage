"""tumor_diameter: 计算肿瘤最大直径。

自动对 hepatic tumor 做连通域分裂（调用 crlm_postprocess.split_hepatic_tumor），
对每个肿瘤调用 Tool_Box.liver_analysis.compute_tumor_diameter。
"""

import os
from Tool_Box.crlm_postprocess import split_hepatic_tumor
from Tool_Box.liver_analysis import compute_tumor_diameter
from skills.utils import convert_numpy


def run(ctx):
    tumor_label = ctx.params.get("tumor_label", "hepatic tumor")
    min_voxels = ctx.params.get("min_voxels", 10)
    convex_hull_threshold = ctx.params.get("convex_hull_threshold", 100)

    # Step 1: 检查是否需要连通域分裂
    tumor_path = os.path.join(ctx.mask_dir, f"{tumor_label}.nii.gz")

    if os.path.exists(tumor_path):
        ctx.log(f"Splitting {tumor_label} into connected components...")
        tumor_names = split_hepatic_tumor(ctx.mask_dir, min_voxels=min_voxels)
        if not tumor_names:
            # 分裂后没有有效肿瘤，尝试直接计算原始掩码
            ctx.log("No valid components after split, trying original mask...")
            tumor_names = [tumor_label]
    else:
        # 尝试直接找 tumor_1, tumor_2, ... 的掩码
        import glob
        tumor_names = []
        for fname in sorted(glob.glob(os.path.join(ctx.mask_dir, "tumor_*.nii.gz"))):
            stem = os.path.splitext(os.path.basename(fname))[0]
            if stem.endswith(".nii"):
                stem = stem[:-4]
            tumor_names.append(stem)
        if not tumor_names:
            return {"tumors": [], "message": "No tumor masks found"}

    # Step 2: 对每个肿瘤计算直径
    results = []
    for name in tumor_names:
        try:
            path = ctx.get_mask_path(name)
            ctx.log(f"Computing diameter for {name}...")
            diam = compute_tumor_diameter(
                path,
                min_voxels=min_voxels,
                convex_hull_threshold=convex_hull_threshold,
            )
            results.append({
                "tumor_id": name,
                "max_diameter_mm": diam.get("max_diameter_mm"),
                "method": diam.get("method", "unknown"),
                "voxel_count": diam.get("voxel_count", 0),
                "note": diam.get("note", ""),
            })
        except FileNotFoundError:
            ctx.log(f"Mask for '{name}' not found, skipping")

    return convert_numpy({"tumors": results})

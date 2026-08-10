"""
肝脏肿瘤量化分析工具
======================

供 CRLM 分析管线使用的四大分析函数 + 报告生成。

所有函数都设计为无副作用的纯计算函数，便于测试和复用。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from Tool_Box.mask_resolution import resolve_mask_path

logger = logging.getLogger(__name__)


# ============================================================
# 函数 1：血管体积
# ============================================================

def compute_vessel_volume(mask_path: str) -> Dict:
    """
    计算血管体积。

    Args:
        mask_path: 二值 mask 的 NIfTI 路径。

    Returns:
        {volume_mm3, volume_cm3, voxel_count, voxel_volume_mm3}
    """
    nii = nib.load(mask_path)
    mask = nii.get_fdata() > 0
    affine = nii.affine

    voxel_count = int(mask.sum())
    vox_vol = abs(np.linalg.det(affine[:3, :3]))
    vol_mm3 = voxel_count * vox_vol

    return {
        "volume_mm3": round(vol_mm3, 2),
        "volume_cm3": round(vol_mm3 / 1000.0, 2),
        "voxel_count": voxel_count,
        "voxel_volume_mm3": round(vox_vol, 6),
    }


# ============================================================
# 函数 2：肿瘤直径（凸包法）
# ============================================================

def compute_tumor_diameter(
    mask_path: str,
    min_voxels: int = 10,
    convex_hull_threshold: int = 100,
) -> Dict:
    """
    计算肿瘤最大三维直径（mm）。

    策略：
    - 先过滤最大连通域
    - ≥ convex_hull_threshold 体素 → 凸包法（scipy.spatial.ConvexHull）
    - < convex_hull_threshold 体素 → 轴对齐法（bounding box 最长轴）

    Args:
        mask_path: 二值 mask 的 NIfTI 路径。
        min_voxels: 最小体素数（< 此值标记为微小病灶）。
        convex_hull_threshold: 凸包法切换阈值。

    Returns:
        {
            max_diameter_mm: 最大直径 (mm) | None,
            method: "convex_hull" | "axis_aligned" | "too_small" | "empty",
            voxel_count: 体素数,
            hull_vertices_count: 凸包顶点数 (仅 convex_hull 模式),
            note: str 附注,
        }
    """
    nii = nib.load(mask_path)
    mask = (nii.get_fdata() > 0).astype(np.uint8)
    affine = nii.affine

    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return {"max_diameter_mm": None, "method": "empty",
                "voxel_count": 0, "hull_vertices_count": 0,
                "note": "肿瘤为空"}

    if voxel_count < min_voxels:
        return {"max_diameter_mm": None, "method": "too_small",
                "voxel_count": voxel_count, "hull_vertices_count": 0,
                "note": f"微小病灶（{voxel_count} 体素），跳过精确分析"}

    # ---- 提取体素世界坐标 ----
    coords = np.argwhere(mask > 0).astype(np.float64)
    # 转换为世界坐标
    world_coords = nib.affines.apply_affine(affine, coords)

    if voxel_count < convex_hull_threshold:
        # ---- 轴对齐法 ----
        min_bounds = world_coords.min(axis=0)
        max_bounds = world_coords.max(axis=0)
        diameter = float(np.linalg.norm(max_bounds - min_bounds))
        return {"max_diameter_mm": round(diameter, 2),
                "method": "axis_aligned",
                "voxel_count": voxel_count,
                "hull_vertices_count": 0,
                "note": ""}

    # ---- 凸包法 ----
    from scipy.spatial import ConvexHull
    hull = ConvexHull(world_coords)
    hull_points = world_coords[hull.vertices]

    # 在凸包顶点间搜索最大距离
    max_dist = 0.0
    n_vertices = len(hull_points)
    for i in range(n_vertices):
        diffs = hull_points - hull_points[i]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        max_dist = max(max_dist, dists.max())

    return {"max_diameter_mm": round(max_dist, 2),
            "method": "convex_hull",
            "voxel_count": voxel_count,
            "hull_vertices_count": n_vertices,
            "note": ""}


# ============================================================
# 函数 3：最大连通域
# ============================================================

def compute_largest_cc(
    mask: np.ndarray,
    min_voxels: int = 10,
) -> Dict:
    """
    计算 mask 的最大连通域。

    Args:
        mask: (H, W, D) 二值数组。
        min_voxels: 小于此体素数的分量视为噪声。

    Returns:
        {
            largest_mask: np.ndarray 最大连通域的 mask (uint8),
            total_components: int 总连通域数（含噪声）,
            valid_components: int 有效分量数（≥ min_voxels）,
            largest_voxels: int 最大分量体素数,
            largest_ratio: float 最大分量占比,
            component_sizes: List[int] 各分量体素数（降序）,
        }
    """
    from scipy import ndimage

    binary = (mask > 0).astype(np.uint8)
    labeled, num = ndimage.label(binary)

    if num == 0:
        return {
            "largest_mask": np.zeros_like(mask, dtype=np.uint8),
            "total_components": 0,
            "valid_components": 0,
            "largest_voxels": 0,
            "largest_ratio": 0.0,
            "component_sizes": [],
        }

    sizes = np.bincount(labeled.ravel())
    # sizes[0] 是背景，排除
    component_sizes = sorted([int(s) for s in sizes[1:] if s > 0], reverse=True)
    valid_sizes = [s for s in component_sizes if s >= min_voxels]

    if not valid_sizes:
        return {
            "largest_mask": np.zeros_like(mask, dtype=np.uint8),
            "total_components": num,
            "valid_components": 0,
            "largest_voxels": max(component_sizes) if component_sizes else 0,
            "largest_ratio": 0.0,
            "component_sizes": component_sizes,
        }

    largest_label = np.argmax(sizes[1:]) + 1
    largest_mask = (labeled == largest_label).astype(np.uint8)
    largest_voxels = int(valid_sizes[0])
    total_valid = sum(valid_sizes)

    return {
        "largest_mask": largest_mask,
        "total_components": num,
        "valid_components": len(valid_sizes),
        "largest_voxels": largest_voxels,
        "largest_ratio": round(largest_voxels / max(total_valid, 1), 4),
        "component_sizes": component_sizes,
    }


# ============================================================
# 函数 4：肿瘤-血管距离
# ============================================================

def _distance_label(min_dist_mm: float) -> str:
    """根据临床文献给出距离释义。"""
    if min_dist_mm < 1.0:
        return "高度怀疑血管侵犯 / 紧贴血管"
    elif min_dist_mm < 5.0:
        return "肿瘤邻近血管（需谨慎规划手术切缘）"
    elif min_dist_mm < 20.0:
        return f"中等距离（DTV < 20mm 与复发率升高相关）"
    else:
        return "距离充足"


def compute_tumor_vessel_distance(
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    spacing: Tuple[float, float, float],
) -> Dict:
    """
    计算肿瘤表面到血管表面的最小距离（mm）。

    Args:
        tumor_mask: (H, W, D) 二值数组 — 肿瘤。
        vessel_mask: (H, W, D) 二值数组 — 血管。
        spacing: (sx, sy, sz) 体素间距（mm）。

    Returns:
        {
            min_distance_mm: float | None,
            interpretation: str 临床释义,
            tumor_voxels: int,
            tumor_contacts_vessel: bool 是否直接接触,
        }
    """
    tumor_binary = (tumor_mask > 0).astype(np.uint8)
    vessel_binary = (vessel_mask > 0).astype(np.uint8)

    tumor_voxels = int(tumor_binary.sum())
    if tumor_voxels == 0:
        return {"min_distance_mm": None, "interpretation": "无肿瘤",
                "tumor_voxels": 0, "tumor_contacts_vessel": False}

    # 检查是否有重叠
    overlap = tumor_binary & vessel_binary
    if overlap.sum() > 0:
        return {"min_distance_mm": 0.0,
                "interpretation": "肿瘤与血管直接接触/重叠",
                "tumor_voxels": tumor_voxels,
                "tumor_contacts_vessel": True}

    from scipy import ndimage

    # 对血管补集做距离变换（得到每个体素到最近血管的距离）
    # ndimage.distance_transform_edt 计算的是欧几里得距离（体素单位）
    vessel_complement = 1 - vessel_binary
    dist_vox = ndimage.distance_transform_edt(vessel_complement)

    # 在肿瘤区域取最小值
    tumor_region_dist = dist_vox[tumor_binary > 0]
    if len(tumor_region_dist) == 0:
        return {"min_distance_mm": None, "interpretation": "计算异常",
                "tumor_voxels": tumor_voxels,
                "tumor_contacts_vessel": False}

    min_voxel_dist = float(tumor_region_dist.min())

    # 体素距离 → 物理距离（mm）
    # 取 spacing 的均值作为近似，或用更精确的向量范数
    spacing_arr = np.array(spacing)
    # 距离变换是各向同性的，使用均值 spacing 近似
    # 更好的方式：min_voxel_dist * spacing.mean()
    min_mm = min_voxel_dist * spacing_arr.mean()

    return {
        "min_distance_mm": round(min_mm, 2),
        "interpretation": _distance_label(min_mm),
        "tumor_voxels": tumor_voxels,
        "tumor_contacts_vessel": False,
    }


# ============================================================
# 综合：加载 mask + 执行分析
# ============================================================

def analyze_tumor(
    ct_nifti_path: str,
    tumor_mask_path: str,
    vessel_mask_paths: Dict[str, str],
    min_voxels: int = 10,
) -> Dict:
    """
    对一个肿瘤执行所有分析。

    Args:
        ct_nifti_path: CT NIfTI 路径（用于获取 spacing）。
        tumor_mask_path: 肿瘤 mask NIfTI 路径。
        vessel_mask_paths: {"血管名": mask_path} 字典。
        min_voxels: 最小体素数过滤阈值。

    Returns:
        {
            diameter: Dict,           # compute_tumor_diameter 结果
            largest_cc: Dict,         # compute_largest_cc 结果
            vessel_distances: {       # 对每种血管的距离
                "hepatic": {"min_distance_mm": ..., ...},
                "portal": {...},
            },
            volume_cm3: float,        # 肿瘤体积
        }
    """
    # 加载 CT 获取 spacing
    ct_nii = nib.load(ct_nifti_path)
    spacing = tuple(float(v) for v in (
        np.linalg.norm(ct_nii.affine[:3, 0]),
        np.linalg.norm(ct_nii.affine[:3, 1]),
        np.linalg.norm(ct_nii.affine[:3, 2]),
    ))

    # 加载肿瘤 mask
    tumor_nii = nib.load(tumor_mask_path)
    tumor_data = tumor_nii.get_fdata()

    # 计算肿瘤体积
    vox_vol = abs(np.linalg.det(tumor_nii.affine[:3, :3]))
    tumor_voxels = int((tumor_data > 0).sum())
    vol_mm3 = tumor_voxels * vox_vol

    # ---- 连通域分析 ----
    cc_result = compute_largest_cc(tumor_data, min_voxels=min_voxels)
    largest_mask = cc_result["largest_mask"]

    # ---- 直径（在最大连通域上计算）----
    # 先把最大连通域保存为临时文件，交给 compute_tumor_diameter
    import tempfile
    import os
    tmp_path = None
    try:
        # 直接传递 mask 数据，避免 I/O
        diameter_result = _compute_diameter_on_mask(
            largest_mask, tumor_nii.affine,
            min_voxels=min_voxels,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ---- 到各血管的距离（在最大连通域上计算）----
    vessel_distances = {}
    for vessel_name, vessel_path in vessel_mask_paths.items():
        vessel_nii = nib.load(vessel_path)
        vessel_data = vessel_nii.get_fdata()
        dist_result = compute_tumor_vessel_distance(
            largest_mask, vessel_data, spacing=spacing,
        )
        vessel_distances[vessel_name] = dist_result

    return {
        "diameter": diameter_result,
        "largest_cc": cc_result,
        "vessel_distances": vessel_distances,
        "volume_mm3": round(vol_mm3, 2),
        "volume_cm3": round(vol_mm3 / 1000.0, 2),
        "tumor_voxels": tumor_voxels,
    }


def _compute_diameter_on_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    min_voxels: int = 10,
) -> Dict:
    """与 compute_tumor_diameter 相同，但直接接收 numpy 数组而不通过文件。"""
    voxel_count = int((mask > 0).sum())
    if voxel_count == 0:
        return {"max_diameter_mm": None, "method": "empty",
                "voxel_count": 0, "hull_vertices_count": 0, "note": "肿瘤为空"}
    if voxel_count < min_voxels:
        return {"max_diameter_mm": None, "method": "too_small",
                "voxel_count": voxel_count, "hull_vertices_count": 0,
                "note": f"微小病灶（{voxel_count} 体素）"}

    coords = np.argwhere(mask > 0).astype(np.float64)
    world_coords = nib.affines.apply_affine(affine, coords)

    convex_hull_threshold = 100
    if voxel_count < convex_hull_threshold:
        min_b = world_coords.min(axis=0)
        max_b = world_coords.max(axis=0)
        d = float(np.linalg.norm(max_b - min_b))
        return {"max_diameter_mm": round(d, 2), "method": "axis_aligned",
                "voxel_count": voxel_count, "hull_vertices_count": 0, "note": ""}

    from scipy.spatial import ConvexHull
    hull = ConvexHull(world_coords)
    hp = world_coords[hull.vertices]
    max_d = 0.0
    for i in range(len(hp)):
        diffs = hp - hp[i]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        max_d = max(max_d, dists.max())

    return {"max_diameter_mm": round(max_d, 2), "method": "convex_hull",
            "voxel_count": voxel_count, "hull_vertices_count": len(hp), "note": ""}


# ============================================================
# 肝脏总分析（多肿瘤入口）
# ============================================================

_LOGICAL_VESSELS = {"hepatic", "portal"}


def _resolve_analysis_mask_paths(mask_paths: Dict[str, str]):
    """Deduplicate only canonical vessel variants and retain every other label."""
    normalized = {}
    variants = {}
    for label, supplied_path in mask_paths.items():
        if (
            label.endswith("_optimized")
            and label[:-10] in _LOGICAL_VESSELS
        ):
            continue
        if label not in _LOGICAL_VESSELS:
            normalized[label] = supplied_path
            continue
        try:
            resolved = resolve_mask_path(Path(supplied_path).parent, label)
        except FileNotFoundError:
            normalized[label] = supplied_path
            variants[label] = "raw"
            continue
        normalized[label] = resolved.path
        variants[label] = resolved.variant
    return normalized, variants


def _normalize_vessel_labels(vessel_labels, mask_paths):
    normalized = []
    for label in vessel_labels:
        logical = (
            label[:-10]
            if label.endswith("_optimized") and label[:-10] in _LOGICAL_VESSELS
            else label
        )
        if logical in mask_paths and logical not in normalized:
            normalized.append(logical)
    return normalized


def analyze_liver_case(
    ct_nifti_path: str,
    mask_paths: Dict[str, str],
    liver_label: str = "liver",
    tumor_labels: Optional[List[str]] = None,
    vessel_labels: Optional[List[str]] = None,
) -> Dict:
    """
    对一个病例执行完整的肝脏肿瘤分析。

    Args:
        ct_nifti_path: CT NIfTI 路径。
        mask_paths: {label: path} 所有 mask 路径。
        liver_label: 肝脏 mask 的 label（默认 "liver"）。
        tumor_labels: 肿瘤 mask 的 label 列表（默认自动检测含 "tumor" 的 label）。
        vessel_labels: 血管 mask 的 label 列表（默认自动检测 "hepatic" 和 "portal"）。

    Returns:
        完整的分析结果字典，包含肝脏体积、逐个肿瘤分析等。
    """
    mask_paths, mask_variants = _resolve_analysis_mask_paths(mask_paths)

    if tumor_labels is None:
        tumor_labels = [k for k in mask_paths if "tumor" in k.lower()]
    if vessel_labels is None:
        # Match both CRLM-renamed ("hepatic", "portal") and VISTA3D-native
        # ("hepatic vessel", "portal vein and splenic vein") vessel labels.
        vessel_labels = [
            k for k in mask_paths
            if ("hepatic" in k.lower() or "portal" in k.lower())
            and "tumor" not in k.lower()
        ]
    else:
        vessel_labels = _normalize_vessel_labels(vessel_labels, mask_paths)

    # 肝脏体积
    liver_volume = None
    if liver_label in mask_paths:
        liver_vol_result = compute_vessel_volume(mask_paths[liver_label])
        liver_volume = liver_vol_result["volume_cm3"]

    # 各血管体积
    vessel_volumes = {}
    for vl in vessel_labels:
        if vl in mask_paths:
            vessel_volumes[vl] = {
                **compute_vessel_volume(mask_paths[vl]),
                "mask_variant": mask_variants.get(vl, "raw"),
            }

    # 各肿瘤分析
    tumor_results = {}
    for tl in tumor_labels:
        if tl not in mask_paths:
            continue
        # 构建该肿瘤可及的血管 mask 字典
        avail_vessels = {vl: mask_paths[vl] for vl in vessel_labels if vl in mask_paths}
        result = analyze_tumor(ct_nifti_path, mask_paths[tl], avail_vessels)
        for vl, distance in result["vessel_distances"].items():
            distance["mask_variant"] = mask_variants.get(vl, "raw")
        tumor_results[tl] = result

    return {
        "liver_volume_cm3": liver_volume,
        "vessel_volumes": vessel_volumes,
        "tumor_results": tumor_results,
    }


# ============================================================
# 结构化报告生成
# ============================================================

def generate_liver_report(analysis_result: Dict) -> str:
    """
    生成肝癌专项分析的格式化文本报告。

    格式：先汇总 → 逐肿瘤展开 → 每项含到各血管的距离和直径。

    Args:
        analysis_result: analyze_liver_case() 的返回值。

    Returns:
        格式化的报告文本。
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  肝脏肿瘤专项分析报告")
    lines.append("=" * 60)
    lines.append("")

    # ---- 肝脏 ----
    lv = analysis_result.get("liver_volume_cm3")
    if lv is not None:
        lines.append(f"  肝脏体积: {lv:.1f} cm³")
    lines.append("")

    # ---- 血管 ----
    vessel_vols = analysis_result.get("vessel_volumes", {})
    if vessel_vols:
        for vn, vv in vessel_vols.items():
            lines.append(f"  血管 [{vn}]: {vv['volume_cm3']:.2f} cm³ ({vv['voxel_count']:,} 体素)")
            if lv and lv > 0:
                ratio = vv['volume_cm3'] / lv * 100
                lines.append(f"            占肝脏体积 {ratio:.1f}%")
        lines.append("")

    # ---- 肿瘤 ----
    tumors = analysis_result.get("tumor_results", {})
    if not tumors:
        lines.append("  📌 未检测到肿瘤")
        lines.append("=" * 60)
        return "\n".join(lines)

    # 汇总
    tumor_count = len(tumors)
    diameters = []
    min_distances = {}  # vessel_name -> [(tumor_name, dist), ...]
    # 统计各血管情况
    all_vessel_names = set()
    for tl, tr in tumors.items():
        if tr["diameter"]["max_diameter_mm"] is not None:
            diameters.append(tr["diameter"]["max_diameter_mm"])
        for vn, vd in tr.get("vessel_distances", {}).items():
            all_vessel_names.add(vn)
            if vn not in min_distances:
                min_distances[vn] = []
            if vd["min_distance_mm"] is not None:
                min_distances[vn].append((tl, vd["min_distance_mm"]))

    lines.append(f"  📊 汇总")
    lines.append(f"      肿瘤总数: {tumor_count}")
    if diameters:
        lines.append(f"      最大直径: {max(diameters):.1f} mm")
        lines.append(f"      最小直径: {min(diameters):.1f} mm")
    lines.append("")

    # 血管距离汇总
    for vn in sorted(all_vessel_names):
        dists = min_distances.get(vn, [])
        if dists:
            closest = min(dists, key=lambda x: x[1])
            lines.append(f"      距 [{vn}] 最近: {closest[1]:.1f} mm ({closest[0]})")
            furthest = max(dists, key=lambda x: x[1])
            lines.append(f"      距 [{vn}] 最远: {furthest[1]:.1f} mm ({furthest[0]})")
    lines.append("")

    # ---- 逐肿瘤展开 ----
    lines.append("-" * 60)
    lines.append("  各肿瘤详情")
    lines.append("-" * 60)
    lines.append("")

    for idx, (tl, tr) in enumerate(sorted(tumors.items())):
        lines.append(f"  ⚕️  {tl}")
        # 体积
        lines.append(f"      体积: {tr['volume_cm3']:.2f} cm³  ({tr['tumor_voxels']:,} 体素)")
        # 直径
        dia = tr["diameter"]
        if dia["max_diameter_mm"] is not None:
            lines.append(f"      最大直径: {dia['max_diameter_mm']:.2f} mm ({dia['method']})")
        if dia["note"]:
            lines.append(f"      ⚠ {dia['note']}")
        # 连通域
        cc = tr["largest_cc"]
        if cc["total_components"] > 1:
            sz_desc = ", ".join(str(s) for s in cc["component_sizes"][:3])
            if len(cc["component_sizes"]) > 3:
                sz_desc += f", ... (共 {cc['total_components']} 个分量)"
            lines.append(f"      连通域: {cc['valid_components']}/{cc['total_components']} 个有效分量, "
                         f"最大占 {cc['largest_ratio']*100:.1f}%")
            lines.append(f"      分量体素数: {sz_desc}")
        elif cc["total_components"] == 1:
            lines.append(f"      连通域: 单一连通域")
        else:
            lines.append(f"      连通域: 无有效分量")
        # 到各血管的距离
        for vn, vd in sorted(tr.get("vessel_distances", {}).items()):
            if vd["min_distance_mm"] is not None:
                contact = " [直接接触]" if vd.get("tumor_contacts_vessel") else ""
                lines.append(f"      距 [{vn}]: {vd['min_distance_mm']:.2f} mm — {vd['interpretation']}{contact}")
            else:
                lines.append(f"      距 [{vn}]: 无法计算")
        lines.append("")

    lines.append("=" * 60)
    lines.append("  报告结束")
    lines.append("=" * 60)

    return "\n".join(lines)

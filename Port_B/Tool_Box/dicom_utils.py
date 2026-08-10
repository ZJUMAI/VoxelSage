"""
DICOM → NIfTI 转换工具
========================

供 run_crlm_analysis.py 按需调用，将 CRLM 数据集的 DICOM CT + SEG
转换为 NIfTI 格式并缓存到 data/CRLM/nifti/{case_id}/。

依赖: pydicom, nibabel, numpy
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# DICOM CT → NIfTI
# ============================================================

def _sort_dicom_slices(dicom_dir: str) -> List[str]:
    """
    读取 DICOM CT 目录，按 ImagePositionPatient[2] (Z) 排序。
    返回排序后的文件路径列表。
    """
    import pydicom

    files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            pos = ds.get("ImagePositionPatient")
            if pos is None:
                continue
            slices.append((float(pos[2]), f))
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not slices:
        raise ValueError(f"No valid DICOM slices found in {dicom_dir}")

    # 按 Z 排序（通常是递减方向，统一按升序排列）
    slices.sort(key=lambda x: x[0])
    return [s[1] for s in slices]


def convert_dicom_ct_to_nifti(
    dicom_dir: str,
    output_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 DICOM CT 系列转换为 NIfTI (.nii.gz)。

    Args:
        dicom_dir: DICOM 文件目录（如 .../101-71548/）。
        output_path: 输出 .nii.gz 路径（如 data/CRLM/nifti/CRLM-CT-1001/ct.nii.gz）。

    Returns:
        (volume, affine)
        - volume: (H, W, D) float32 HU 值
        - affine: 4x4 仿射矩阵
    """
    import pydicom

    sorted_files = _sort_dicom_slices(dicom_dir)

    # 读取第一帧获取元数据
    ref_ds = pydicom.dcmread(sorted_files[0], force=True)
    rows, cols = ref_ds.Rows, ref_ds.Columns
    n_slices = len(sorted_files)

    # 读取第一帧和最后一帧的 ImagePositionPatient 来计算 affine
    first_pos = [float(x) for x in ref_ds.ImagePositionPatient]
    last_ds = pydicom.dcmread(sorted_files[-1], force=True)
    last_pos = [float(x) for x in last_ds.ImagePositionPatient]

    # 像素间距
    pixel_spacing = [float(x) for x in ref_ds.PixelSpacing]  # (dy, dx) in DICOM
    spacing_y = pixel_spacing[0]
    spacing_x = pixel_spacing[1]

    # Z 间距：从第一帧和最后一帧的 Z 差计算
    # DICOM Z 通常是从上到下（递减）或从下到上（递增）
    z_diff = last_pos[2] - first_pos[2]
    spacing_z = abs(z_diff) / max(n_slices - 1, 1)
    z_direction = 1 if z_diff > 0 else -1

    # 构建 affine
    # pixel_array shape 是 (rows, cols)，所以 x=cols 对应 PD=Row, y=rows 对应 PD=Column
    # nibabel 用 (x, y, z) 坐标，所以:
    # affine[:3, 0] = pixel_spacing[0] (y direction in DICOM = rows)
    # affine[:3, 1] = pixel_spacing[1] (x direction in DICOM = cols)
    # Actually let me think about this more carefully.
    #
    # DICOM: ImageOrientationPatient gives row and column direction cosines.
    # For axial CT: row = [+1, 0, 0] (patient L->R), column = [0, +1, 0] (patient A->P)
    # But these are stored in ImageOrientationPatient (0020,0037).
    # For a standard axial CT without rotation:
    #   Row direction cosines = [1, 0, 0], Column direction cosines = [0, 1, 0]
    #
    # In nibabel/NIfTI: (i, j, k) maps to (x, y, z) via affine
    # NIfTI convention normally: i = columns (width), j = rows (height), k = slices (depth)
    # So: i maps to x (L/R), j maps to y (A/P), k maps to z (I/S)

    orient = ref_ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    row_dcos = np.array([float(x) for x in orient[:3]])  # row direction = i axis
    col_dcos = np.array([float(x) for x in orient[3:6]])  # col direction = j axis

    # Z direction (slice orientation)
    slice_dcos = np.cross(row_dcos, col_dcos) * z_direction

    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = row_dcos * spacing_x
    affine[:3, 1] = col_dcos * spacing_y
    affine[:3, 2] = slice_dcos * spacing_z
    affine[:3, 3] = first_pos  # origin

    # 读取所有切片并构建 3D 数组
    # nibabel shape = (cols, rows, slices) = (W, H, D)
    # But numpy array shape = (D, H, W) for plotting
    # Actually nibabel stores as (W, H, D) but loads as (H, W, D) via get_fdata()
    volume = np.zeros((rows, cols, n_slices), dtype=np.float32)

    for idx, fpath in enumerate(sorted_files):
        ds = pydicom.dcmread(fpath, force=True)
        arr = ds.pixel_array.astype(np.float32)

        # 应用 RescaleIntercept/Slope 转 HU
        intercept = float(getattr(ds, "RescaleIntercept", -1024))
        slope = float(getattr(ds, "RescaleSlope", 1))
        arr = arr * slope + intercept

        volume[:, :, idx] = arr

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存 NIfTI
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_path)
    logger.info(f"Saved CT NIfTI: {output_path} ({volume.shape}, HU range [{volume.min():.0f}, {volume.max():.0f}])")

    return volume, affine


# ============================================================
# DICOM SEG → NIfTI Masks
# ============================================================

def _get_segment_map(seg_ds) -> Dict[int, str]:
    """从 DICOM SEG 头部提取 SegmentNumber → SegmentLabel 映射。"""
    seg_map = {}
    if hasattr(seg_ds, "SegmentSequence"):
        for item in seg_ds.SegmentSequence:
            seg_map[int(item.SegmentNumber)] = str(item.SegmentLabel)
    return seg_map


def _sanitize_filename(name: str) -> str:
    """将 SegmentLabel 转为安全的文件名。"""
    return name.lower().replace(" ", "_").replace("-", "_")


def convert_dicom_seg_to_masks(
    seg_path: str,
    ct_nifti_path: str,
    output_dir: str,
    skip_empty: bool = True,
) -> Dict[str, str]:
    """
    将 DICOM SEG 文件转换为各结构的二值 NIfTI mask。

    通过 PerFrameFunctionalGroupsSequence 的
    SegmentIdentificationSequence → ReferencedSegmentNumber
    确定每帧对应的结构。

    Args:
        seg_path: DICOM SEG 文件路径。
        ct_nifti_path: 对应 CT 的 NIfTI 路径（用于获取 affine + shape 对齐）。
        output_dir: 输出目录（如 data/CRLM/nifti/CRLM-CT-1001/）。
        skip_empty: True=跳过空结构（不生成全零文件）。

    Returns:
        {label_name: mask_path} 字典，如 {"liver": "...liver.nii.gz", "hepatic": "...hepatic.nii.gz"}
    """
    import pydicom

    os.makedirs(output_dir, exist_ok=True)

    # 读取 SEG
    ds = pydicom.dcmread(seg_path, force=True)
    seg_map = _get_segment_map(ds)
    logger.info(f"SEG segments: {seg_map}")

    # 读取 CT NIfTI 作为参考
    ct_nii = nib.load(ct_nifti_path)
    ct_affine = ct_nii.affine
    ct_shape = ct_nii.shape  # (H, W, D) or (W, H, D)? nibabel: (W, H, D)
    # 看 get_fdata() 返回什么形状
    ct_data = ct_nii.get_fdata()
    ct_shape_3d = ct_data.shape  # (H, W, D)

    # SEG 像素数据: (num_frames, rows, cols)
    seg_pixels = ds.pixel_array  # (N_frames, Rows, Columns)
    n_frames = seg_pixels.shape[0]

    # 按 Segment 分组帧
    from collections import defaultdict
    seg_frames = defaultdict(list)  # {seg_num: [(frame_idx, z_position, frame_data), ...]}

    for frame_idx in range(n_frames):
        frame = ds.PerFrameFunctionalGroupsSequence[frame_idx]
        seg_num = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        pos = frame.PlanePositionSequence[0].ImagePositionPatient
        z_pos = float(pos[2])
        frame_data = seg_pixels[frame_idx]  # (Rows, Cols)
        seg_frames[seg_num].append((frame_idx, z_pos, frame_data))

    # CT 的 Z 坐标范围
    # 从 CT affine 提取各 slice 的 Z 位置
    ct_z_positions = _compute_ct_z_positions(ct_affine, ct_shape_3d)

    result = {}
    for seg_num in sorted(seg_frames.keys()):
        label = seg_map.get(seg_num, f"segment_{seg_num}")
        filename = _sanitize_filename(label)
        output_path = os.path.join(output_dir, f"{filename}.nii.gz")

        frames = seg_frames[seg_num]
        # 创建与 CT 对齐的 mask
        mask = np.zeros(ct_shape_3d, dtype=np.uint8)

        for _, z_pos, frame_data in frames:
            # 找到 CT 中最近的 Z 层面
            nearest_slice = _find_nearest_slice(z_pos, ct_z_positions)
            if nearest_slice is not None:
                mask[:, :, nearest_slice] = (frame_data > 0).astype(np.uint8)

        # 检查是否为空
        voxel_count = int(mask.sum())
        if skip_empty and voxel_count == 0:
            logger.info(f"  Skipping empty segment: {label}")
            continue

        # 保存
        nii = nib.Nifti1Image(mask, ct_affine)
        nib.save(nii, output_path)
        logger.info(f"  Saved: {filename}.nii.gz ({voxel_count} voxels)")
        result[label] = output_path

    return result


def _compute_ct_z_positions(affine: np.ndarray, shape_3d: Tuple[int, int, int]) -> List[float]:
    """
    从 affine 和 shape 计算 CT 每个 slice 的世界坐标 Z 值。

    NIfTI shape: get_fdata() 返回 (H, W, D)，
    affine 映射 (x, y, z) 体素坐标到世界坐标。
    slice 方向 (k/depth) 对应体素坐标的第三个轴。
    """
    # nibabel: (i, j, k) → affine → (x_world, y_world, z_world)
    # get_fdata() shape = (H, W, D)，体素坐标 z 对应 k (第三轴)
    # CT 是轴向扫描，Z 变化主要在第三维
    depth = shape_3d[2]
    z_positions = []
    for k in range(depth):
        # 体素坐标 (0, 0, k) 映射到世界坐标
        vox = np.array([0, 0, k, 1])
        world = affine @ vox
        z_positions.append(float(world[2]))
    return z_positions


def _find_nearest_slice(target_z: float, z_positions: List[float]) -> Optional[int]:
    """在 Z 位置列表中找最接近 target_z 的索引。"""
    if not z_positions:
        return None
    z_arr = np.array(z_positions)
    idx = int(np.argmin(np.abs(z_arr - target_z)))
    return idx


# ============================================================
# CRLM 目录解析
# ============================================================

def find_crlm_dicom_paths(
    crlm_root: str,
    case_id: str,
) -> Tuple[str, str]:
    """
    根据 case ID 查找 DICOM CT 目录和 SEG 文件路径。

    Args:
        crlm_root: CRLM 数据集根目录
                   (e.g. data/CRLM/TCIA/colorectal_liver_metastases/)
        case_id: 如 "CRLM-CT-1001"

    Returns:
        (ct_series_dir, seg_file_path)
        - ct_series_dir: CT DICOM 目录（.../101-xxx/）
        - seg_file_path: SEG DICOM 文件（.../100-Segmentation-xxx/xxx.dcm）

    Raises:
        FileNotFoundError: 如果找不到匹配的 CT/SEG
    """
    import pydicom

    case_dir = os.path.join(crlm_root, case_id)
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"Case not found: {case_dir}")

    # 遍历子目录寻找 CT 和 SEG
    ct_series_dir = None
    seg_file = None

    for root, dirs, files in os.walk(case_dir):
        # 检查目录名是否包含 Segmentation
        dirname = os.path.basename(root)
        if "Segmentation" in dirname:
            # 这个目录下至少有一个 .dcm 文件
            dcm_files = [f for f in files if f.endswith(".dcm")]
            if dcm_files:
                seg_file = os.path.join(root, dcm_files[0])
                # 验证确实是 SEG modality
                try:
                    ds = pydicom.dcmread(seg_file, force=True, stop_before_pixels=True)
                    if ds.get("Modality") == "SEG":
                        seg_file = seg_file  # keep it
                    else:
                        seg_file = None
                except Exception:
                    seg_file = None
                    continue
        else:
            # 检查目录名是否以数字开头（CT series）
            # CT series 通常叫 101-xxx, 105-xxx, 2-xxx 等
            parts = dirname.split("-")
            if parts and parts[0].isdigit():
                dcm_files = [f for f in files if f.endswith(".dcm")]
                if len(dcm_files) > 10:
                    # 验证确实是 CT modality
                    sample = os.path.join(root, dcm_files[0])
                    try:
                        ds = pydicom.dcmread(sample, force=True, stop_before_pixels=True)
                        if ds.get("Modality") == "CT":
                            ct_series_dir = root
                    except Exception:
                        continue

    if ct_series_dir is None:
        raise FileNotFoundError(f"No CT series found for {case_id}")
    if seg_file is None or not os.path.isfile(seg_file):
        raise FileNotFoundError(f"No SEG file found for {case_id}")

    return ct_series_dir, seg_file


def get_crlm_cache_dir(crlm_root: str, case_id: str, nifti_root: str) -> str:
    """返回某个 case 的 NIfTI 缓存目录。"""
    return os.path.join(nifti_root, case_id)


def get_cached_ct_path(cache_dir: str) -> str:
    """返回缓存的 CT NIfTI 路径。"""
    return os.path.join(cache_dir, "ct.nii.gz")


def case_is_cached(cache_dir: str) -> bool:
    """检查 case 是否已经缓存（ct.nii.gz 存在）。"""
    return os.path.isfile(get_cached_ct_path(cache_dir))


# ============================================================
# DICOM → NIfTI 一键转换（按需）
# ============================================================

def ensure_case_converted(
    case_id: str,
    crlm_root: str,
    nifti_root: str,
    force: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """
    确保 case 已被转换为 NIfTI（如果未缓存则执行转换）。

    Args:
        case_id: 如 "CRLM-CT-1001"
        crlm_root: DICOM 根目录
        nifti_root: NIfTI 缓存根目录
        force: True=强制重新转换

    Returns:
        (ct_nifti_path, mask_paths)
        - ct_nifti_path: CT NIfTI 文件路径
        - mask_paths: {label: mask_path} 字典
    """
    cache_dir = get_crlm_cache_dir(crlm_root, case_id, nifti_root)

    if not force and case_is_cached(cache_dir):
        logger.info(f"Case {case_id}: using cached data in {cache_dir}")
        ct_path = get_cached_ct_path(cache_dir)
        # 查找已有的 mask 文件
        mask_paths = {}
        for fname in sorted(glob.glob(os.path.join(cache_dir, "*.nii.gz"))):
            if os.path.basename(fname) == "ct.nii.gz":
                continue
            # 对 .nii.gz 要去掉两层后缀
            label = Path(fname).stem  # -> "liver.nii"
            if label.endswith(".nii"):
                label = label[:-4]   # -> "liver"
            mask_paths[label] = fname
        return ct_path, mask_paths

    # 需要转换
    logger.info(f"Case {case_id}: converting DICOM → NIfTI ...")
    ct_series_dir, seg_file = find_crlm_dicom_paths(crlm_root, case_id)

    # 转换 CT
    ct_path = get_cached_ct_path(cache_dir)
    convert_dicom_ct_to_nifti(ct_series_dir, ct_path)

    # 转换 SEG
    mask_paths = convert_dicom_seg_to_masks(seg_file, ct_path, cache_dir)

    logger.info(f"Case {case_id}: conversion complete. CT: {ct_path}, Masks: {len(mask_paths)}")
    return ct_path, mask_paths

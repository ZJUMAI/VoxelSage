#!/usr/bin/env python3
"""
3D Medical Visualization & Analysis API — 向后兼容导入桩
=============================================================

所有逻辑已迁移到 Visualization.API。此文件为重新导出桩（re-export shim），
确保 ``from Visualization import vis_api as vis`` 继续可用。

使用方式（推荐）：
    from Visualization.API import generate_visualization, process_nifti_file
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # API.py 已移至项目根目录
for _p in (str(_PROJECT_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 从项目根的 API.py 重新导出所有公开函数
from API import (  # noqa: E402
    # 核心 API 函数
    generate_visualization,
    process_nifti_file,
    list_available_organs,
    estimate_output_size,
    select_top_slices,
    select_best_slice,
    save_slice_images,
    save_best_slice_images,
    generate_structural_report,

    # 共享常量
    ORGAN_COLORS,
    ORGAN_DISPLAY_NAMES,

    # visualize_3d 的重新导出（保持 vis_api 旧接口兼容）
    find_mask_files,
    robust_load_nii,
    binarize_mask,
    downsample_volume,
    extract_mesh,
    voxel_to_world,
    mesh_surface_area,
    resolve_case_dir,
    _read_threejs_sources,

    # 内部错误格式
    _error,
)

if __name__ == "__main__":
    print("=" * 60)
    print("3D Medical Visualization & Analysis API")
    print("=" * 60)
    print()
    print("此文件为向后兼容导入桩，所有逻辑已移至 API.py。")
    print()
    print("推荐导入方式：")
    print("  from Visualization.API import process_nifti_file")
    print("  from Visualization.API import generate_visualization")
    print()
    print("CLI 使用方式：")
    print("  # 处理 .nii.gz 文件")
    print("  python Visualization/API.py /path/to/ct.nii.gz --top-k 3")
    print()
    print("  # 启动 HTTP API 服务")
    print("  python Visualization/API.py --server --port 8765")
    print()
    print("详细文档见 README_API.md")
    print("=" * 60)

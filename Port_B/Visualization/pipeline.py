#!/usr/bin/env python3
"""
VoxelSage Port B 全流程管线 — 向后兼容导入桩
==============================================

所有逻辑已迁移到 Visualization.API。此文件为重新导出桩（re-export shim），
确保 ``from Visualization.pipeline import process_nifti_file`` 继续可用。

使用方式（推荐）：
    from Visualization.API import process_nifti_file
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from API import (  # noqa: E402
    process_nifti_file,
    select_top_slices,
    select_best_slice,
    save_slice_images,
    save_best_slice_images,
    generate_structural_report,
    _score_slice_informativeness,
    _score_slice_cflt,
    ORGAN_COLORS,
    ORGAN_DISPLAY_NAMES,
    pipeline_main as main,
)

if __name__ == "__main__":
    main()

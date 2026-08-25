#!/usr/bin/env python3
"""
3D Medical Visualization & Analysis API
=========================================

单一入口模块，整合管线（pipeline）、Python API、FastAPI HTTP 服务。

API 返回内容：
  1. VISTA3D 分割掩码路径列表
  2. 3D 可视化 HTML 文件路径
  3. 语义化结构性报告
  4. 信息量最大的三张片子的序号和渲染图

快速入门（Python）:
    >>> from Visualization.API import process_nifti_file

    # 全流程：.nii.gz → 分割 → 3D HTML + Top-3 切片 + 报告
    >>> result = process_nifti_file("/path/to/ct.nii.gz", top_k=3)
    >>> result["visualization_html"]   # 3D HTML
    >>> result["best_slices"]          # [{index, score, png_path, overlay_path}, ...]
    >>> result["structural_report"]    # 报告文本
    >>> result["mask_files"]           # {organ_name: mask_path, ...}

快速入门（HTTP）:
    # 启动服务
    python Visualization/API.py --server --port 8765

    # 调用全流程
    curl -X POST http://localhost:8765/api/process \\
        -H "Content-Type: application/json" \\
        -d '{"nifti_path": "/path/to/ct.nii.gz"}'
"""

# ======================================================================
# [Section 1]  Imports
# ======================================================================
import os

# GPUManager selects physical GPU indices reported by nvidia-smi.  CUDA's
# default FASTEST_FIRST enumeration can use a different ordering (especially
# on mixed-GPU hosts), so configure PCI ordering before any module can import
# torch.  This makes ``cuda:N`` refer to the same physical GPU as nvidia-smi N.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import json
import hashlib
import time
import uuid
import glob
import re
import shutil
import tempfile
import logging
import traceback
import datetime as _dt
import fcntl
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from typing import Optional, Dict, Any, List, Tuple, Union

import cv2
import numpy as np
import nibabel as nib

# Ensure we can import sibling modules from Visualization/ (progress, visualize_3d)
_THIS_DIR = str(Path(__file__).resolve().parent)
_VIZ_DIR = str(Path(__file__).resolve().parent / "Visualization")
for _p in (_THIS_DIR, _VIZ_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Progress tracker (optional, for real-time feedback in server mode)
from progress import ProgressTracker

from visualize_3d import (
    ORGAN_COLORS as _VIZ_ORGAN_COLORS,
    ORGAN_DISPLAY_NAMES as _VIZ_ORGAN_DISPLAY_NAMES,
    find_mask_files,
    robust_load_nii,
    binarize_mask,
    downsample_volume,
    extract_mesh,
    voxel_to_world,
    mesh_surface_area,
    resolve_case_dir,
    _read_threejs_sources,
    _read_bezier_surface_source,
    _make_threejs_html,
    resample_for_smoothing,
    quantize_vertices,
    get_display_name,
)

from skimage.morphology import remove_small_objects

# Flush-on-every-write log handler — prevents tail-f delay
class ImmediateFileHandler(logging.FileHandler):
    """File handler that flushes after every write for real-time log tailing.

    Used by server_main() uvicorn log config to write to both console and file.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()

# Skills system (built-in and future user-uploaded)
from skills.engine import SkillEngine, SkillNotFoundError, SkillExecutionError
from skills.models import SkillContext

# Global SkillEngine instance — populated at server startup
_skill_engine: SkillEngine = None

# GPU 管理器 — 负责自动选择空闲 GPU 并防止并发 OOM
from Tool_Box.gpu_manager import GPUManager, print_gpu_info as _print_gpu_info
from Tool_Box.mask_resolution import resolve_mask_path
_gpu_manager = GPUManager()

# 病例级锁 — 同一 case_id 的 process-lite 串行执行，不同 case 并行
_case_locks: Dict[str, Lock] = {}
_case_locks_lock = Lock()

def _get_case_lock(case_id: str) -> Lock:
    """获取或创建指定病例的锁。"""
    with _case_locks_lock:
        if case_id not in _case_locks:
            _case_locks[case_id] = Lock()
        return _case_locks[case_id]


class CaseProcessingBusyError(RuntimeError):
    """Raised when process-lite is already running for a case."""


@contextmanager
def _case_processing_lock(case_id: str, output_dir: str):
    """Prevent duplicate process-lite work in this process and across restarts."""
    case_lock = _get_case_lock(case_id)
    if not case_lock.acquire(blocking=False):
        raise CaseProcessingBusyError(
            f"Case '{case_id}' is already being processed"
        )

    lock_handle = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        lock_path = os.path.join(output_dir, ".process-lite.lock")
        lock_handle = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CaseProcessingBusyError(
                f"Case '{case_id}' is already being processed by another server process"
            ) from exc
        yield
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()
        case_lock.release()

# Segmentation editor routes (registered in server_main)
try:
    from skills.builtin.segmentation_modification import editor_routes as _seg_editor_routes
    _HAS_SEG_EDITOR = True
except ImportError:
    _HAS_SEG_EDITOR = False

# FastAPI / Pydantic (optional — only needed for server mode)
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


# ======================================================================
# [Section 2]  Constants
# ======================================================================

# Merge colors from pipeline.py (Chinese display names) and visualize_3d.py
ORGAN_COLORS = {
    "liver":            {"color": "#e8a58f", "opacity": 0.40},
    "spleen":           {"color": "#c4b0d9", "opacity": 0.45},
    "pancreas":         {"color": "#f9c08a", "opacity": 0.45},
    "colon":            {"color": "#a8d4e6", "opacity": 0.45},
    "left kidney":      {"color": "#a8e6cf", "opacity": 0.45},
    "right kidney":     {"color": "#7dceb0", "opacity": 0.45},
    "kidney":           {"color": "#a8e6cf", "opacity": 0.45},
    "pancreatic tumor": {"color": "#e83030", "opacity": 1.0},
    "hepatic tumor":    {"color": "#e84040", "opacity": 1.0},
    "left kidney cyst": {"color": "#f0c040", "opacity": 1.0},
    "right kidney cyst":{"color": "#e8b830", "opacity": 1.0},
    "kidney cyst":      {"color": "#f0c040", "opacity": 1.0},
}
# Fallback: merge any keys from visualize_3d that pipeline doesn't define
for _k, _v in _VIZ_ORGAN_COLORS.items():
    if _k not in ORGAN_COLORS:
        ORGAN_COLORS[_k] = _v

ORGAN_DISPLAY_NAMES = {
    "liver":             "肝脏",
    "spleen":            "脾脏",
    "pancreas":          "胰腺",
    "colon":             "结肠",
    "left kidney":       "左肾",
    "right kidney":      "右肾",
    "kidney":            "肾脏",
    "pancreatic tumor":  "胰腺肿瘤",
    "hepatic tumor":     "肝肿瘤",
    "left kidney cyst":  "左肾囊肿",
    "right kidney cyst": "右肾囊肿",
    "kidney cyst":       "肾囊肿",
}
# Fallback display names
for _k, _v in _VIZ_ORGAN_DISPLAY_NAMES.items():
    if _k not in ORGAN_DISPLAY_NAMES:
        ORGAN_DISPLAY_NAMES[_k] = _v

# CRLM-specific colors (liver cancer analysis, DICOM SEG naming)
_CRLM_ORGAN_COLORS = {
    "liver":          {"color": "#e8a58f", "opacity": 0.35},
    "liver_remnant":  {"color": "#d4a08a", "opacity": 0.35},
    "hepatic":        {"color": "#40b8a0", "opacity": 0.55},  # 青绿色，与红色肿瘤区分
    "portal":         {"color": "#5060e8", "opacity": 0.55},
}
for _k, _v in _CRLM_ORGAN_COLORS.items():
    # CRLM 颜色优先级最高（无条件覆盖，用于 DICOM SEG 命名管线）
    ORGAN_COLORS[_k] = _v

_CRLM_DISPLAY_NAMES = {
    "liver":          "肝脏",
    "liver_remnant":  "残余肝脏",
    "hepatic":        "肝静脉",
    "portal":         "门静脉",
}
for _k, _v in _CRLM_DISPLAY_NAMES.items():
    if _k not in ORGAN_DISPLAY_NAMES:
        ORGAN_DISPLAY_NAMES[_k] = _v

# Lesion / abnormal tissues (fully opaque)
_LESION_NAMES = {"pancreatic tumor", "hepatic tumor", "left kidney cyst",
                  "right kidney cyst", "kidney cyst", "tumor", "lesion",
                  "cyst", "metastasis", "nodule"}
# CRLM tumor_1..tumor_50 pattern
for _i in range(1, 51):
    _LESION_NAMES.add(f"tumor_{_i}")

# Segmentation backends and VISTA paths
_PROJECT_ROOT = Path(__file__).resolve().parent
_VISTA3D_WRAPPER = str(_PROJECT_ROOT / "SegAgent" / "VISTA3d")
if _VISTA3D_WRAPPER not in sys.path:
    sys.path.insert(0, _VISTA3D_WRAPPER)

_SEGMENTATION_BACKEND_ALIASES = {
    "vista": "vista3d",
    "vista3d": "vista3d",
    "total": "totalsegmentator",
    "totalseg": "totalsegmentator",
    "totalsegmentator": "totalsegmentator",
}
_DEFAULT_SEGMENTATION_BACKEND = os.environ.get(
    "SEGMENTATION_BACKEND", "vista3d"
).strip().lower()
_SEGMENTATION_METADATA_FILENAME = ".voxelsage-segmentation.json"


def _normalize_segmentation_backend(backend: Optional[str]) -> str:
    """Return a canonical, installed-on-demand segmentation backend name."""
    requested = (backend or _DEFAULT_SEGMENTATION_BACKEND).strip().lower()
    canonical = _SEGMENTATION_BACKEND_ALIASES.get(requested)
    if canonical is None:
        supported = ", ".join(sorted(set(_SEGMENTATION_BACKEND_ALIASES.values())))
        raise ValueError(
            f"Unsupported segmentation backend '{requested}'. Supported: {supported}"
        )
    return canonical

_VISTA_LABEL_PATH = _PROJECT_ROOT / "SegAgent" / "VISTA3d" / "label_dict.json"
with open(_VISTA_LABEL_PATH) as _f:
    _VISTA_ORGAN2ID = json.load(_f)
    _VISTA_ID2ORGAN = {v: k for k, v in _VISTA_ORGAN2ID.items()}

# Default output directory — all pipeline outputs go under output/
_DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("VOXELSAGE_OUTPUT_DIR", Path(__file__).resolve().parent / "output")
)
DEFAULT_OUTPUT_DIR = _DEFAULT_OUTPUT_ROOT

# Public-facing base URL — API 返回的可直接访问的链接前缀
# 如果前端/统一服务器在另一端口/地址，修改此项为实际可访问的地址
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8898")

# ---- CRLM-specific constants (environment variables override) ----
_CRLM_DEFAULT_ORGANS = [
    "liver",                           # VISTA3D ID 1
    "hepatic vessel",                  # ID 25
    "portal vein and splenic vein",    # ID 17
    "hepatic tumor",                   # ID 26
    "gallbladder",                      # ID 10
    "inferior vena cava",              # ID 7
    "aorta",                            # ID 6
]
_CRLM_GT_MASK_BASENAMES = {
    *(f"{organ}.nii.gz" for organ in _CRLM_DEFAULT_ORGANS),
    "hepatic.nii.gz",
    "portal.nii.gz",
}
_CRLM_TUMOR_MASK_RE = re.compile(r"^tumor_\d+\.nii\.gz$")
_MAX_SIBLING_GT_MASKS = 64
_CRLM_ROOT_DEFAULT = os.environ.get(
    "CRLM_ROOT",
    str(_PROJECT_ROOT / "data" / "CRLM" / "TCIA" / "colorectal_liver_metastases"),
)
_CRLM_NIFTI_ROOT_DEFAULT = os.environ.get(
    "CRLM_NIFTI_ROOT",
    str(_PROJECT_ROOT / "data" / "CRLM" / "nifti"),
)
_CRLM_OUTPUT_ROOT_DEFAULT = os.environ.get(
    "CRLM_OUTPUT_ROOT",
    str(_PROJECT_ROOT / "output" / "crlm_analysis"),
)
_CRLM_CLINICAL_XLSX = os.environ.get(
    "CRLM_CLINICAL_XLSX",
    str(_PROJECT_ROOT / "data" / "CRLM" / "Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx"),
)

# ---- 演示模式：有预分割 GT 掩码时跳过模型推理 ----
# 通过 DEMO_MODE 环境变量控制（scripts/start.sh 传入），默认开启
_DEMO_MODE = os.environ.get("DEMO_MODE", "1") == "1"

# ---- GPU 互斥锁：防止并发请求同时占用 GPU 导致 CUDA OOM / device busy ----
# 【v2.1】改用 GPUManager — 自动选择空闲 GPU，替代旧版全局串行锁
# 旧版: _GPU_LOCK = Lock(); _GPU_LOCK_TIMEOUT = 120
# GPUManager 实例在 imports 区域创建（见 _gpu_manager）


# ======================================================================
# [Section 3]  Logging — _log() writes to both console and optional log file
# ======================================================================

# Set by server_main() to enable dual-output (console + file)
_log_file_path: str = None


def _log(msg: str):
    """Log a timestamped line to console (flush) and optionally to a file."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file_path:
        try:
            with open(_log_file_path, "a", encoding="utf-8") as _f:
                _f.write(line + "\n")
        except OSError:
            pass


# ======================================================================
# [Section 4]  CT loading helpers
# ======================================================================
def _apply_ct_window(vol: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """Apply abdominal CT windowing, return float [0,1]."""
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    vol = np.clip(vol, low, high)
    vol = (vol - low) / (high - low + 1e-6)
    return vol


def _load_ct(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CT NIfTI, canonical orientation, return (uint8 volume (H,W,D), affine)."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    affine = img.affine
    # CT windowing
    data = _apply_ct_window(data, wl=40.0, ww=400.0)
    return (data * 255.0).astype(np.uint8), affine


def _load_ct_raw(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CT NIfTI WITHOUT windowing, return (raw HU float32 (H,W,D), affine)."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    return data, affine


def _reorient_to_radiological(
    slice_img: np.ndarray,
    label_mask: Optional[np.ndarray],
    affine: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reorient a 2D axial slice to standard radiological display convention.

    Standard radiological convention for axial CT (viewed from below):
      - Top    = Anterior  (front of body)
      - Bottom = Posterior (back of body)
      - Left of image  = Patient's Right
      - Right of image = Patient's Left

    When ``affine`` is provided, uses nibabel ``aff2axcodes`` to determine the
    exact axis directions and applies transposes / flips accordingly.  This
    handles RAS+, LAS, LPS, and all common NIfTI orientations.

    When ``affine`` is **None** (default), assumes the volume is in RAS+
    canonical orientation (as returned by ``nib.as_closest_canonical()``,
    which is used by ``_load_ct()``) and applies the default RAS+ → radiological
    transform (``.T[::-1, ::-1]``).

    Args:
        slice_img:  (H, W) uint8 CT slice.
        label_mask: (H, W) uint8 label mask, or None.
        affine:     4×4 affine matrix from the NIfTI header.  If None,
                    assumes RAS+ canonical orientation.

    Returns:
        (reoriented_slice, reoriented_mask)
    """
    img = np.ascontiguousarray(slice_img)
    mask = np.ascontiguousarray(label_mask) if label_mask is not None else None

    # ---- Determine orientation codes ----
    # If affine is provided, read exact orientation; otherwise assume RAS+
    if affine is not None:
        from nibabel.orientations import aff2axcodes
        codes = aff2axcodes(affine)
        ax0_dir, ax1_dir = codes[0], codes[1]
    else:
        # Default RAS+ (canonical, returned by as_closest_canonical())
        ax0_dir, ax1_dir = 'R', 'A'

    # ----------------------------------------------------------------
    # Step 1 — transpose if rows run Left-Right (place A/P axis on rows)
    if ax0_dir in ('L', 'R'):
        img = np.ascontiguousarray(img.T)
        if mask is not None:
            mask = np.ascontiguousarray(mask.T)
        ax0_dir, ax1_dir = ax1_dir, ax0_dir

    # Step 2 — flip rows so Anterior is at row 0 (top of image)
    if ax0_dir == 'A':
        img = img[::-1, :]
        if mask is not None:
            mask = mask[::-1, :]

    # Step 3 — flip columns so Patient's Right is at col 0 (left of image)
    if ax1_dir == 'R':
        img = img[:, ::-1]
        if mask is not None:
            mask = mask[:, ::-1]

    return img, mask


# ======================================================================
# [Section 5]  VISTA3D segmentation wrapper
# ======================================================================
def _save_binary(mask: np.ndarray, affine: np.ndarray,
                 header: nib.Nifti1Header, path: str):
    nii = nib.Nifti1Image(mask.astype(np.uint8, copy=False), affine, header)
    nii.set_data_dtype(np.uint8)
    nib.save(nii, path)


def _run_vista3d_segmentation(
    nifti_path: str,
    output_dir: str,
    organ_list: Optional[List[str]] = None,
    device: str = "auto",
    vista3d_config: Optional[str] = None,
) -> str:
    """
    Run VISTA3D segmentation on a single NIfTI file.

    Returns path to merged ``all.nii.gz`` label map.
    Individual organ binary masks saved alongside.
    """
    if organ_list is None:
        organ_list = _CRLM_DEFAULT_ORGANS

    from vista3d_Segmentator import Vista3D_Segmentator

    if vista3d_config is None:
        vista3d_config = os.environ.get(
            "VISTA3D_CONFIG",
            str(_PROJECT_ROOT / "SegAgent" / "VISTA3d" / "configs" / "infer.yaml"),
        )

    # ---- GPU 分配 + 重试：自动选择空闲 GPU（若 device 为默认值则自动选卡，否则用指定卡） ----
    _seg = None
    _all_mask = None
    _last_err = None
    for _attempt in range(3):
        # 若 device 是 "auto"，让 GPUManager 自动选择；否则直接使用指定卡
        _preferred = None if device == "auto" else device
        try:
            with _gpu_manager.allocate(preferred=_preferred) as _alloc_dev:
                if device == "auto":
                    _log(f"       [GPU] Auto-allocated {_alloc_dev} (attempt {_attempt+1}/3)")
                _seg = Vista3D_Segmentator(config_file=vista3d_config, device=_alloc_dev)
                _backend_prompt = [_VISTA_ORGAN2ID[o] for o in organ_list]
                _all_mask_path = os.path.join(output_dir, "all.nii.gz")
                _all_mask = _seg.segment(
                    input_path=nifti_path,
                    output_path=_all_mask_path,
                    object_list=_backend_prompt,
                    save_mask=True,
                )
            break  # 加载 + 推理成功，跳出重试
        except RuntimeError as _e:
            _last_err = _e
            _log(f"       [RETRY {_attempt+1}/3] GPU error: {_e}")
            time.sleep(5 * (_attempt + 1))  # 递增等待

    if _all_mask is None:
        raise RuntimeError(f"VISTA3D segmentation failed after 3 attempts: {_last_err}")

    seg = _seg
    all_mask = _all_mask
    all_mask_path = _all_mask_path
    backend_prompt = _backend_prompt

    # Save individual binary masks
    ref_nii = nib.load(nifti_path)
    affine = ref_nii.affine
    header = ref_nii.header

    needed_ids = {_VISTA_ORGAN2ID[o] for o in organ_list}
    present = np.unique(all_mask)
    label_masks = {lid: (all_mask == lid) for lid in present if lid in needed_ids}

    # 并行保存各器官二值掩码（NIfTI 写入是 I/O 密集，线程安全）
    _save_tasks = []
    for organ in organ_list:
        lid = _VISTA_ORGAN2ID[organ]
        if lid not in label_masks:
            continue
        _save_tasks.append((organ, label_masks[lid]))

    with ThreadPoolExecutor(max_workers=min(8, len(_save_tasks) or 1)) as _pool:
        for _organ, _mask in _save_tasks:
            _out_path = os.path.join(output_dir, f"{_organ}.nii.gz")
            _pool.submit(_save_binary, _mask, affine, header, _out_path)

    # Merged kidney （在当前线程执行，依赖前面的独立掩码计算）
    if ("left kidney" in organ_list and "right kidney" in organ_list and
        _VISTA_ORGAN2ID["left kidney"] in label_masks and
        _VISTA_ORGAN2ID["right kidney"] in label_masks):
        merged = (label_masks[_VISTA_ORGAN2ID["left kidney"]] |
                  label_masks[_VISTA_ORGAN2ID["right kidney"]])
        _save_binary(merged, affine, header, os.path.join(output_dir, "kidney.nii.gz"))

    # Merged kidney cyst
    if ("left kidney cyst" in organ_list and "right kidney cyst" in organ_list and
        _VISTA_ORGAN2ID["left kidney cyst"] in label_masks and
        _VISTA_ORGAN2ID["right kidney cyst"] in label_masks):
        merged_cyst = (label_masks[_VISTA_ORGAN2ID["left kidney cyst"]] |
                       label_masks[_VISTA_ORGAN2ID["right kidney cyst"]])
        _save_binary(merged_cyst, affine, header,
                     os.path.join(output_dir, "kidney cyst.nii.gz"))

    return all_mask_path


def _run_totalsegmentator_segmentation(
    nifti_path: str,
    output_dir: str,
    device: str = "auto",
) -> None:
    """Run the optional TotalSegmentator CRLM adapter."""
    try:
        from SegAgent.TotalSegmentator import run_crlm_segmentation
    except ImportError as exc:
        raise RuntimeError(
            "TotalSegmentator is not installed. Run "
            "./scripts/setup.sh --with-totalsegmentator first."
        ) from exc

    preferred = None if device == "auto" else device
    with _gpu_manager.allocate(preferred=preferred) as allocated_device:
        run_crlm_segmentation(
            input_path=nifti_path,
            output_dir=output_dir,
            device=allocated_device,
        )


def _run_segmentation(
    backend: str,
    nifti_path: str,
    output_dir: str,
    organ_list: Optional[List[str]] = None,
    device: str = "auto",
    vista3d_config: Optional[str] = None,
) -> str:
    """Dispatch segmentation without importing optional backends eagerly."""
    backend = _normalize_segmentation_backend(backend)
    if backend == "vista3d":
        _run_vista3d_segmentation(
            nifti_path=nifti_path,
            output_dir=output_dir,
            organ_list=organ_list,
            device=device,
            vista3d_config=vista3d_config,
        )
    elif backend == "totalsegmentator":
        _run_totalsegmentator_segmentation(
            nifti_path=nifti_path,
            output_dir=output_dir,
            device=device,
        )
    return backend


def _segmentation_metadata_path(mask_dir: str) -> Path:
    return Path(mask_dir) / _SEGMENTATION_METADATA_FILENAME


def _read_segmentation_backend(mask_dir: str) -> Optional[str]:
    path = _segmentation_metadata_path(mask_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or not payload.get("backend"):
            return None
        return _normalize_segmentation_backend(payload["backend"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # Older VISTA3D outputs predate the metadata marker.
        legacy_all = Path(mask_dir) / "all.nii.gz"
        legacy_liver = Path(mask_dir) / "liver.nii.gz"
        if legacy_all.is_file() and legacy_liver.is_file():
            return "vista3d"
        return None


def _write_segmentation_metadata(mask_dir: str, backend: str) -> None:
    path = _segmentation_metadata_path(mask_dir)
    payload = {
        "backend": _normalize_segmentation_backend(backend),
        "status": "complete",
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _list_mask_files(mask_dir: str, organ_list: Optional[List[str]] = None) -> Dict[str, str]:
    """Return logical masks mapped to trusted resolved payload paths."""
    from Tool_Box.mask_resolution import scan_logical_masks

    if organ_list is None:
        organ_list = _CRLM_DEFAULT_ORGANS
    resolved = scan_logical_masks(mask_dir)
    return {
        organ: resolved[organ].path
        for organ in organ_list
        if organ in resolved
    }


def _is_case_complete(mask_dir: str, backend: Optional[str] = None) -> bool:
    """Return whether the minimum reusable segmentation result is present.

    New outputs have an atomic completion marker recording their backend.
    Older VISTA3D outputs are recognized by ``all.nii.gz`` plus ``liver.nii.gz``.
    """
    if not os.path.isdir(mask_dir):
        return False

    recorded_backend = _read_segmentation_backend(mask_dir)
    if recorded_backend is None:
        return False
    if backend is not None and recorded_backend != _normalize_segmentation_backend(backend):
        return False

    required_files = ["liver.nii.gz"]
    if not _segmentation_metadata_path(mask_dir).is_file():
        required_files.append("all.nii.gz")
    for filename in required_files:
        path = os.path.join(mask_dir, filename)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) < 1000:
                return False
        except OSError:
            return False
    return True


def _file_sha256(path: str) -> str:
    """Calculate a file digest without loading the complete CT into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ct_files_match(existing_ct_path: str, incoming_ct_path: str) -> bool:
    """Return whether two CT files contain exactly the same bytes.

    Uploads are copied into request-specific directories, so ``samefile``
    alone would reject an identical CT re-upload merely because it has a new
    inode.  Retain it as a cheap fast path, then compare file size and a
    streaming SHA-256 digest for distinct files.
    """
    try:
        if os.path.samefile(existing_ct_path, incoming_ct_path):
            return True

        if os.path.getsize(existing_ct_path) != os.path.getsize(incoming_ct_path):
            return False

        return _file_sha256(existing_ct_path) == _file_sha256(incoming_ct_path)
    except OSError as exc:
        _log(
            f"[process-lite] Unable to compare existing CT with upload: {exc}"
        )
        return False


def _list_existing_mask_files(mask_dir: str) -> Dict[str, str]:
    """Return every existing logical mask using the shared trust resolver."""
    from Tool_Box.mask_resolution import scan_logical_masks

    return {
        name: resolved.path
        for name, resolved in scan_logical_masks(mask_dir).items()
    }


# ======================================================================
# [Section 6]  Slice informativeness scoring
# ======================================================================

def _score_slice_informativeness(
    ct_slice: np.ndarray,           # (H, W) uint8 windowed CT
    organ_present: List[str],       # organ names visible in this slice
    organ_areas: List[int],         # corresponding pixel counts
    has_lesion: bool,
    slice_z: int,
    organ_z_range: Dict[str, Tuple[int, int]],
    volume_depth: int,
) -> float:
    """
    Compute composite 'informativeness' score for one axial slice.

    Factors (normalised → 0‑1, then weighted):
      - Organ diversity (0.25)
      - Organ coverage (0.25)
      - Lesion bonus (0.20)
      - Centrality within organ z‑span (0.15)
      - Image entropy (0.15)
    """
    score = 0.0

    # 1. Organ diversity (up to 0.25)
    n_organs = len(organ_present)
    diversity = min(n_organs / 6.0, 1.0)
    score += diversity * 0.25

    # 2. Organ coverage fraction (up to 0.25)
    h, w = ct_slice.shape
    total_px = h * w
    coverage = sum(organ_areas) / max(total_px, 1)
    coverage = min(coverage * 5.0, 1.0)
    score += coverage * 0.25

    # 3. Lesion bonus (up to 0.20)
    if has_lesion:
        score += 0.20

    # 4. Centrality within organ z‑range (up to 0.15)
    if organ_z_range:
        low = min(v[0] for v in organ_z_range.values())
        high = max(v[1] for v in organ_z_range.values())
        span = max(high - low, 1)
        dist_from_center = abs(slice_z - (low + high) / 2)
        centrality = 1.0 - dist_from_center / (span / 2 + 1)
        centrality = max(0.0, centrality)
        score += centrality * 0.15
    else:
        dist_from_center = abs(slice_z - volume_depth / 2)
        centrality = 1.0 - dist_from_center / (volume_depth / 2 + 1)
        centrality = max(0.0, centrality)
        score += centrality * 0.15

    # 5. Image entropy (up to 0.15)
    hist = cv2.calcHist([ct_slice], [0], None, [64], [0, 256])
    hist = hist.ravel() / max(hist.sum(), 1)
    entropy = -np.sum(hist * np.log2(hist + 1e-12))
    norm_entropy = min(entropy / 6.0, 1.0)
    score += norm_entropy * 0.15

    return score


def _score_slice_cflt(
    ct_slice_hu: np.ndarray,   # (H, W) raw HU values (float32)
    lesion_mask: np.ndarray,   # (H, W) binary — lesion pixels
) -> float:
    """
    CFLT-style lesion-focused score for one slice.

    CFLT-style Eq.3 approximation:
      final = 0.5 × lesion_area_norm + 0.3 × HU_heterogeneity_norm + 0.2 × richness_norm

    NOTE: Normalisation is per-slice (min-max within this slice's metrics),
    not global. For global ranking use select_top_slices().
    """
    from scipy import ndimage

    score = 0.0

    # --- Lesion area (normalised by slice area) ---
    total_px = ct_slice_hu.shape[0] * ct_slice_hu.shape[1]
    lesion_px = int((lesion_mask > 0).sum())
    area_ratio = lesion_px / max(total_px, 1)
    # Normalise: 10% coverage → 1.0
    area_norm = min(area_ratio * 10.0, 1.0)
    score += 0.5 * area_norm

    if lesion_px < 5:
        return score  # no further metrics for tiny lesions

    # --- HU heterogeneity (std of HU within lesion) ---
    lesion_hu = ct_slice_hu[lesion_mask > 0]
    if lesion_hu.size > 1:
        hu_std = float(lesion_hu.std())
        # Typical HU std range ~0-200; cap at 100 for normalisation
        hu_norm = min(hu_std / 100.0, 1.0)
        score += 0.3 * hu_norm

    # --- Richness (number of connected components) ---
    labelled, num_features = ndimage.label((lesion_mask > 0).astype(np.uint8))
    richness_norm = min(num_features / 5.0, 1.0)
    score += 0.2 * richness_norm

    return score


# ======================================================================
# [Section 7]  Slice selection — Top-K
# ======================================================================

def select_top_slices(
    ct_volume: np.ndarray,           # (H, W, D) uint8 windowed CT
    mask_dir: str,                   # directory with per-organ .nii.gz masks
    organ_list: Optional[List[str]] = None,
    top_k: int = 3,
    scoring_mode: str = "crlm",      # "composite" | "cflt" | "crlm" (default)
    ct_volume_raw: Optional[np.ndarray] = None,  # (H,W,D) raw HU, needed for "cflt"
    min_gap: int = 5,                # minimum z-distance between selected slices
    ct_affine: Optional[np.ndarray] = None,  # affine of the canonical CT grid
) -> Tuple[List[int], List[np.ndarray], List[float]]:
    """
    Score every slice and return top-K most informative slices.

    Args:
        ct_volume: Windowed CT (uint8).
        mask_dir: Directory with per-organ .nii.gz masks.
        organ_list: Organs to consider.
        top_k: Number of top slices to return.
        scoring_mode:
            "composite" — use _score_slice_informativeness (organ diversity + coverage + etc.)
            "cflt" — use _score_slice_cflt (lesion-focused: area + HU hetero + richness)
        ct_volume_raw: Raw HU data required when scoring_mode="cflt".
        min_gap: Minimum z-index gap between selected slices (default: 5).
        ct_affine: Affine for ``ct_volume``.  When provided, every mask is
            converted to canonical orientation and resampled onto this exact
            grid, matching the convention used by ``_load_ct``.

    Returns:
        (indices, label_masks, scores)
        - indices: list of z-indices, sorted by score descending
        - label_masks: corresponding (H,W) uint8 label masks
        - scores: float scores in same order
    """
    if organ_list is None:
        organ_list = _CRLM_DEFAULT_ORGANS

    depth = ct_volume.shape[2]

    # ---- Load organ masks ----
    organ_masks: Dict[str, np.ndarray] = {}
    organ_z_range: Dict[str, Tuple[int, int]] = {}
    is_lesion_organ: Dict[str, bool] = {}

    for organ in organ_list:
        try:
            m_path = resolve_mask_path(mask_dir, organ).path
        except FileNotFoundError:
            continue
        try:
            nii = nib.load(m_path)
            # _load_ct() returns a canonical RAS+ array, whereas segmentation
            # masks may retain the source orientation (commonly LPS).
            # Canonicalize first and, when the CT geometry is available, align
            # the mask to its exact grid.
            if ct_affine is not None:
                nii = nib.as_closest_canonical(nii)
                if (
                    nii.shape != ct_volume.shape
                    or not np.allclose(nii.affine, ct_affine, rtol=1e-5, atol=1e-4)
                ):
                    from nibabel.processing import resample_from_to
                    nii = resample_from_to(
                        nii,
                        (ct_volume.shape, ct_affine),
                        order=0,
                        mode="constant",
                        cval=0,
                    )
            mask = np.asanyarray(nii.dataobj) > 0
        except Exception as exc:
            _log(f"       [WARN] Could not align mask '{organ}' to CT grid: {exc}")
            continue
        if mask.ndim != 3:
            continue
        if mask.shape != ct_volume.shape:
            _log(
                f"       [WARN] Skipping mask '{organ}': canonical shape "
                f"{mask.shape} does not match CT shape {ct_volume.shape}"
            )
            continue
        organ_masks[organ] = mask
        is_lesion_organ[organ] = organ.lower() in {o.lower() for o in _LESION_NAMES}

        # Compute z‑range
        z_present = np.where(mask.any(axis=(0, 1)))[0]
        if len(z_present) > 0:
            organ_z_range[organ] = (int(z_present[0]), int(z_present[-1]))
        else:
            organ_z_range[organ] = (0, depth - 1)

    # Some legacy DICOM-SEG masks were copied from pixel frames in
    # (Rows, Columns) order into a NIfTI (X, Y) array.  Their affine and shape
    # match the CT, so affine-based resampling cannot reveal the error; the
    # overlay appears rotated/reflected across the diagonal.  Use the liver as
    # a robust anatomical anchor: a correctly aligned liver should almost
    # entirely cover abdominal soft tissue, not air outside the patient.  Only
    # accept the XY transpose when it wins by a wide margin, so ordinary RAS
    # and correctly converted DICOM cases remain untouched.
    liver_name = next(
        (name for name in organ_masks if name.strip().lower() == "liver"),
        None,
    )
    if liver_name is not None:
        liver_mask = organ_masks[liver_name]
        transposed_liver = np.swapaxes(liver_mask, 0, 1)
        if transposed_liver.shape == ct_volume.shape and liver_mask.sum() >= 100:
            if ct_volume_raw is not None and ct_volume_raw.shape == ct_volume.shape:
                alignment_ct = ct_volume_raw
                soft_low, soft_high = -100.0, 300.0
            else:
                # Invert the fixed WL=40/WW=400 window approximately. Values
                # outside the window are clipped, which is sufficient for the
                # soft-tissue-versus-air test used here.
                alignment_ct = ct_volume.astype(np.float32) / 255.0 * 400.0 - 160.0
                soft_low, soft_high = -100.0, 240.1

            def _anatomical_alignment_score(candidate: np.ndarray) -> float:
                values = alignment_ct[candidate]
                if values.size == 0:
                    return 0.0
                finite = np.isfinite(values)
                if not finite.any():
                    return 0.0
                values = values[finite]
                soft_tissue = np.mean((values > soft_low) & (values < soft_high))
                inside_body = np.mean(values > -500.0)
                return float(0.8 * soft_tissue + 0.2 * inside_body)

            native_score = _anatomical_alignment_score(liver_mask)
            transpose_score = _anatomical_alignment_score(transposed_liver)
            if transpose_score >= 0.85 and transpose_score > native_score + 0.10:
                organ_masks = {
                    name: np.ascontiguousarray(np.swapaxes(mask, 0, 1))
                    for name, mask in organ_masks.items()
                }
                _log(
                    "       [Slice alignment] Corrected legacy DICOM-SEG XY "
                    f"axis swap (native={native_score:.3f}, "
                    f"transpose={transpose_score:.3f})"
                )

    if not organ_masks:
        # Fallback: pick central slice with highest entropy
        scores = np.zeros(depth, dtype=np.float32)
        for z in range(depth):
            hist = cv2.calcHist([ct_volume[:, :, z]], [0], None, [64], [0, 256])
            hist = hist.ravel() / max(hist.sum(), 1)
            ent = -np.sum(hist * np.log2(hist + 1e-12))
            scores[z] = ent
        best_indices = np.argsort(scores)[::-1][:top_k].tolist()
        empty_mask = np.zeros((ct_volume.shape[0], ct_volume.shape[1]), dtype=np.uint8)
        return best_indices, [empty_mask] * len(best_indices), scores[best_indices].tolist()

    # ---- Score each slice (parallelized) ----
    organ_names = list(organ_masks.keys())

    def _gather_slice_organs(z: int):
        """Collect per-organ data for one slice — shared across scoring modes.

        Returns (organ_present, organ_areas, has_lesion, has_vessel).
        """
        present: List[str] = []
        areas: List[int] = []
        has_lesion = False
        has_vessel = False
        _VESSEL_NAMES = {"hepatic", "portal", "hepatic vessel", "portal vein"}
        for organ, mask in organ_masks.items():
            area = int(mask[:, :, z].sum())
            if area > 0:
                present.append(organ)
                areas.append(area)
                if is_lesion_organ.get(organ, False):
                    has_lesion = True
                if organ.lower() in _VESSEL_NAMES:
                    has_vessel = True
        return present, areas, has_lesion, has_vessel

    def _gather_combined_lesion_slice(z: int):
        """Merge all lesion masks into a single binary slice."""
        merged = np.zeros((ct_volume.shape[0], ct_volume.shape[1]), dtype=np.uint8)
        for organ, mask in organ_masks.items():
            if is_lesion_organ.get(organ, False):
                merged[mask[:, :, z] > 0] = 1
        return merged

    def _score_slice(z: int) -> float:
        """Score a single axial slice (thread-safe, GIL-released on numpy ops)."""
        ct_slice = ct_volume[:, :, z]

        if scoring_mode == "cflt":
            # Try lesion-focused CFLT score first
            combined_lesion = _gather_combined_lesion_slice(z)
            if combined_lesion.sum() > 0 and ct_volume_raw is not None:
                return float(_score_slice_cflt(ct_volume_raw[:, :, z], combined_lesion))
            # Fallback: no lesion → use general informativeness
            present, areas, has_l, _ = _gather_slice_organs(z)
            return float(_score_slice_informativeness(
                ct_slice, present, areas, has_l, z, organ_z_range, depth,
            ))

        present, areas, has_l, has_v = _gather_slice_organs(z)
        base = float(_score_slice_informativeness(
            ct_slice, present, areas, has_l, z, organ_z_range, depth,
        ))

        if scoring_mode == "crlm":
            if has_l and has_v:
                base += 0.20
            elif has_v:
                base += 0.10
            return min(base, 1.0)

        return base  # composite (default)

    # 分块并行评分：numpy/scipy 操作释放 GIL，线程安全
    _n_scoring_threads = min(8, depth)
    if depth >= 32 and _n_scoring_threads > 1:
        with ThreadPoolExecutor(max_workers=_n_scoring_threads) as pool:
            per_slice_scores = np.array(list(pool.map(_score_slice, range(depth))), dtype=np.float32)
    else:
        per_slice_scores = np.array([_score_slice(z) for z in range(depth)], dtype=np.float32)

    # ---- Select top-K with min_gap constraint ----
    sorted_indices = np.argsort(per_slice_scores)[::-1]
    selected = []
    selected_set = set()
    min_gap = min_gap

    for idx in sorted_indices:
        if per_slice_scores[idx] < 1e-8:
            break
        if any(abs(idx - s) < min_gap for s in selected_set):
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        if len(selected) >= top_k:
            break

    # Relax constraint if not enough
    if len(selected) < top_k:
        for idx in sorted_indices:
            if int(idx) in selected_set:
                continue
            if per_slice_scores[idx] < 1e-8:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= top_k:
                break

    if not selected:
        selected = [depth // 2]

    # Build label masks for selected slices
    label_masks = []
    for sel_z in selected:
        label_mask = np.zeros((ct_volume.shape[0], ct_volume.shape[1]), dtype=np.uint8)
        for idx, organ in enumerate(organ_names):
            label_mask[organ_masks[organ][:, :, sel_z] > 0] = (idx + 1)
        label_masks.append(label_mask)

    scores = [float(per_slice_scores[z]) for z in selected]
    return selected, label_masks, scores


def select_best_slice(
    ct_volume: np.ndarray,
    mask_dir: str,
    organ_list: Optional[List[str]] = None,
    ct_affine: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray]:
    """
    Legacy wrapper — returns (best_z, label_mask) for the single best slice.

    Calls select_top_slices(top_k=1).
    """
    indices, masks, _ = select_top_slices(
        ct_volume, mask_dir, organ_list=organ_list, top_k=1, scoring_mode="crlm",
        ct_affine=ct_affine,
    )
    if not indices:
        return ct_volume.shape[2] // 2, np.zeros((ct_volume.shape[0], ct_volume.shape[1]), dtype=np.uint8)
    return indices[0], masks[0]


# ======================================================================
# [Section 8]  Slice image saving
# ======================================================================

def save_slice_images(
    ct_volume: np.ndarray,
    slice_indices: List[int],
    label_masks: List[np.ndarray],
    output_dir: str,
    case_name: str = "case",
    organ_names: Optional[List[str]] = None,
    slice_scores: Optional[List[float]] = None,
    affine: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Save PNG images for multiple slices with color-coded overlays.

    For each slice:
      - ``{case_name}_slice_{rank}_{index}.png`` (raw CT slice)
      - ``{case_name}_slice_{rank}_{index}_overlay.png`` (with organ contour overlays)

    Overlay colors (BGR for OpenCV):
      - **Red** ``(0, 0, 255)`` — tumors / lesions
      - **Blue** ``(255, 128, 0)`` — vessels (hepatic / portal)
      - **Cyan** ``(255, 255, 0)`` — other organs

    A legend and case info text are stamped onto the overlay image.

    Args:
        ct_volume: windowed CT (H, W, D) uint8
        slice_indices: z‑indices to save
        label_masks: per-slice uint8 label maps
        output_dir: output directory
        case_name: case identifier (used in filename and overlay text)
        organ_names: list of organ names aligned with label_mask values
                     (label_val = idx + 1).  When ``None``, all contours
                     are drawn in green (legacy behaviour).
        slice_scores: informativeness scores aligned with slice_indices
        affine: 4×4 NIfTI affine matrix. If provided, the orientation is
                detected precisely via nibabel's orientation codes; if omitted
                (default), RAS+ canonical orientation is assumed.  In both cases
                the slice is automatically reoriented to standard radiological
                display convention (Anterior at top, Patient's Right at left).
                (Anterior at top, Patient's Right at left of image).

    Returns:
        list of dicts: [{"index": ..., "png_path": ..., "overlay_path": ...}, ...]

    If a slice fails to save it is skipped with a warning rather than raising
    an exception.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Build label → color / name mapping
    label_to_color = {}  # label_val → BGR tuple
    label_to_name = {}   # label_val → str
    if organ_names:
        for idx, name in enumerate(organ_names):
            lbl = idx + 1
            nl = name.lower()
            if any(t in nl for t in ("tumor", "lesion", "cyst", "metastasis", "nodule")):
                label_to_color[lbl] = (0, 0, 255)       # Red
            elif any(v in nl for v in ("hepatic", "portal", "vein")):
                label_to_color[lbl] = (255, 128, 0)     # Blue
            else:
                label_to_color[lbl] = (255, 255, 0)     # Cyan
            label_to_name[lbl] = name

    legend_colors = {}
    for lbl, col in label_to_color.items():
        nl = label_to_name.get(lbl, "").lower()
        if any(t in nl for t in ("tumor", "lesion", "cyst", "metastasis", "nodule")):
            legend_colors["Tumor"] = (0, 0, 255)
        elif any(v in nl for v in ("hepatic", "portal", "vein")):
            legend_colors["Vessel"] = (255, 128, 0)
        else:
            legend_colors["Organ"] = (255, 255, 0)

    dt_str = _dt.datetime.now().strftime("%Y-%m-%d")

    def _save_one_slice(rank_z_mask):
        """Save a single slice's raw + overlay PNGs. Returns result dict or None."""
        rank, z, label_mask = rank_z_mask
        try:
            # Extract raw slice and reorient to radiological convention
            img = ct_volume[:, :, z]
            img, label_mask = _reorient_to_radiological(img, label_mask, affine)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # --- Info text lines (shared by raw + overlay) ---
            info_lines = [
                f"Case: {case_name}",
                f"Slice: z={z}",
            ]
            if slice_scores and rank < len(slice_scores):
                info_lines.append(f"Score: {slice_scores[rank]:.3f}")
            info_lines.append(dt_str)

            def _draw_info(img_with_text):
                for i, line in enumerate(info_lines):
                    cv2.putText(img_with_text, line, (10, 16 + i * 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                y_off = 16 + len(info_lines) * 16 + 8
                for legend_text, bgr in sorted(legend_colors.items()):
                    cv2.rectangle(img_with_text, (10, y_off - 7),
                                  (25, y_off + 5), bgr, -1)
                    cv2.putText(img_with_text, legend_text, (30, y_off + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_off += 16

            # --- Raw slice ---
            raw_display = img_rgb.copy()
            raw_name = f"{case_name}_slice_{rank}_{z}.png"
            raw_path = os.path.join(output_dir, raw_name)
            cv2.imwrite(raw_path, raw_display)

            # --- Overlay ---
            overlay = img_rgb.copy()
            if label_mask is not None and label_mask.sum() > 0 and organ_names:
                unique_labels = np.unique(label_mask)
                for label_val in unique_labels:
                    if label_val == 0:
                        continue
                    binary = (label_mask == label_val).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
                    color = label_to_color.get(label_val, (0, 255, 0))
                    cv2.drawContours(overlay, contours, -1, color, 2)
            elif label_mask is not None and label_mask.sum() > 0:
                unique_labels = np.unique(label_mask)
                for label_val in unique_labels:
                    if label_val == 0:
                        continue
                    binary = (label_mask == label_val).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

            _draw_info(overlay)

            overlay_name = f"{case_name}_slice_{rank}_{z}_overlay.png"
            overlay_path = os.path.join(output_dir, overlay_name)
            cv2.imwrite(overlay_path, overlay)

            return {
                "index": z,
                "png_path": raw_path,
                "overlay_path": overlay_path,
                "score": float(slice_scores[rank]) if slice_scores and rank < len(slice_scores) else 0.0,
            }
        except Exception as e:
            _log(f"       [WARN] Failed to save slice z={z}: {e}")
            return None

    # 并行保存各切片（IMWRITE 是 I/O 密集 + cv2 释放 GIL）
    items = [(i, z, mask) for i, (z, mask) in enumerate(zip(slice_indices, label_masks))]
    if len(items) >= 2:
        with ThreadPoolExecutor(max_workers=min(len(items), 4)) as pool:
            results = [r for r in pool.map(_save_one_slice, items) if r is not None]
    else:
        results = [r for r in (_save_one_slice(item) for item in items) if r is not None]

    return results


def save_best_slice_images(
    ct_volume: np.ndarray,
    best_z: int,
    label_mask: np.ndarray,
    output_dir: str,
    case_name: str = "case",
    affine: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Legacy wrapper — saves a single slice and returns (raw_path, overlay_path).
    """
    results = save_slice_images(ct_volume, [best_z], [label_mask], output_dir,
                                 case_name, affine=affine)
    if results:
        return results[0]["png_path"], results[0]["overlay_path"]
    return None, None


# ======================================================================
# [Section 9]  Structural report generation
# ======================================================================

# Import medical commonsense validation from Structural_Report module
try:
    _SR_PATH = str(Path(__file__).resolve().parent / "Structural_Report")
    if _SR_PATH not in sys.path:
        sys.path.insert(0, _SR_PATH)
    from utils.medical_commonsense import validate_organ_volume
    _HAS_MEDICAL_CHECK = True
except ImportError:
    _HAS_MEDICAL_CHECK = False


def generate_structural_report(
    ct_volume: np.ndarray,
    affine: np.ndarray,
    mask_dir: str,
    organ_list: Optional[List[str]] = None,
) -> str:
    """
    Generate formatted text structural report summarising organ volumes,
    HU statistics, and lesion info. Includes medical commonsense validation
    to flag unrealistic organ volumes.
    """
    if organ_list is None:
        organ_list = _CRLM_DEFAULT_ORGANS

    # Map display organ names to medical_commonsense validation keys
    _ORGAN_TO_KEY = {
        "liver": "liver",
        "spleen": "spleen",
        "pancreas": "pancreas",
        "colon": "colon",
        "left kidney": "kidney_left",
        "right kidney": "kidney_right",
        "kidney": "kidney_total",
        "pancreatic tumor": "pancreas",
        "hepatic tumor": "liver",
        "left kidney cyst": "kidney_left",
        "right kidney cyst": "kidney_right",
        "kidney cyst": "kidney_total",
    }

    vox_vol = abs(np.linalg.det(affine[:3, :3]))
    lines = []
    lines.append("=" * 60)
    lines.append("  STRUCTURAL REPORT — CT SCAN ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"  Volume shape: {ct_volume.shape}")
    lines.append(f"  Voxel spacing: "
                 f"{np.linalg.norm(affine[:3, 0]):.2f} × "
                 f"{np.linalg.norm(affine[:3, 1]):.2f} × "
                 f"{np.linalg.norm(affine[:3, 2]):.2f} mm")
    lines.append(f"  Number of slices: {ct_volume.shape[2]}")
    lines.append("")

    total_organs = 0

    def _process_organ_report(organ: str) -> Tuple[List[str], bool]:
        """Load mask, compute stats, return (text_lines, detected) for one organ."""
        try:
            m_path = resolve_mask_path(mask_dir, organ).path
        except FileNotFoundError:
            return [], False
        try:
            nii = nib.load(m_path)
            mask = nii.get_fdata()
        except Exception:
            return [], False

        voxels = int((mask > 0).sum())
        if voxels < 1:
            return [], False

        vol_mm3 = voxels * vox_vol
        vol_cm3 = vol_mm3 / 1000.0

        # ---- 医学常识校验 ----
        vol_warning = None
        if _HAS_MEDICAL_CHECK:
            is_lesion_type = organ.lower() in {o.lower() for o in _LESION_NAMES}
            if not is_lesion_type:
                organ_key = _ORGAN_TO_KEY.get(organ, organ)
                _, vol_warning = validate_organ_volume(organ_key, vol_cm3)

        # HU statistics: reverse the windowing applied by _apply_ct_window()
        # The CT volume was windowed with WL=40, WW=400 (abdominal setting).
        # Reverse: hu = (pixel/255.0) * WW - WW/2 + WL
        #        i.e. (pixel/255.0) * 400 - 160
        # NOTE: This assumes WL=40, WW=400. If the window changes (e.g. lung
        #       WL=-600, WW=1500) this formula MUST be updated accordingly.
        masked_hu = ct_volume.astype(np.float32) * (mask > 0)
        nonzero = masked_hu > 0
        if nonzero.sum() > 0:
            hu_values = (masked_hu[nonzero] / 255.0) * 400.0 - 160.0
            # Note: reconstruction assumes abdominal window (WL=40, WW=400).
            # If _apply_ct_window defaults change, update this formula accordingly.
            mean_hu = hu_values.mean()
            std_hu = hu_values.std()
            min_hu = hu_values.min()
            max_hu = hu_values.max()
        else:
            mean_hu = std_hu = min_hu = max_hu = 0.0

        z_present = np.where(mask.any(axis=(0, 1)))[0]
        z_range = f"[{z_present[0]}, {z_present[-1]}]" if len(z_present) > 0 else "N/A"

        display = get_display_name(organ)
        is_lesion = organ.lower() in {o.lower() for o in _LESION_NAMES}
        marker = "⚠️  LESION" if is_lesion else "   ORGAN"
        organ_lines = [
            f"  {marker}: {display}",
            f"       Volume: {vol_cm3:.2f} mL ({vol_mm3:.0f} mm³)",
        ]
        if vol_warning:
            organ_lines.append(f"       ⚠ {vol_warning}")
        organ_lines += [
            f"       Voxels: {voxels:,}",
            f"       Mean HU: {mean_hu:.1f} ± {std_hu:.1f}  (range: {min_hu:.1f} ~ {max_hu:.1f})",
            f"       Z-range: {z_range}",
            "",
        ]
        return organ_lines, True

    # 并行处理各器官（I/O + numpy，线程安全）
    _n_report_threads = min(8, len(organ_list))
    if len(organ_list) >= 4 and _n_report_threads > 1:
        with ThreadPoolExecutor(max_workers=_n_report_threads) as pool:
            organ_results = list(pool.map(_process_organ_report, organ_list))
    else:
        organ_results = [_process_organ_report(o) for o in organ_list]

    for organ_lines, detected in organ_results:
        if detected:
            total_organs += 1
            lines.extend(organ_lines)

    lines.append("-" * 60)
    lines.append(f"  Total organs detected: {total_organs}")
    lesion_count = sum(1 for o in organ_list if o.lower() in {x.lower() for x in _LESION_NAMES}
                       and os.path.exists(os.path.join(mask_dir, f'{o}.nii.gz')))
    lines.append(f"  Lesions/tumours: {lesion_count}")
    lines.append("=" * 60)

    return "\n".join(lines)


# ======================================================================
# [Section 10]  3D rendering helpers
# ======================================================================

def _generate_3d_html(
    mask_dir: str,
    output_path: str,
    step_size: int = 2,
    downsample_factor: float = 1.0,
    title: str = "3D Segmentation",
) -> str:
    """Generate 3D HTML from mask directory. Returns output_path."""
    result = generate_visualization(
        case_dir=mask_dir,
        output_dir=str(Path(output_path).parent),
        output_filename=Path(output_path).stem,
        step_size=step_size,
        downsample_factor=downsample_factor,
        title=title,
    )
    if result["status"] == "error":
        raise RuntimeError(f"3D visualisation failed: {result['message']}")
    return result["file_path"]


def _generate_threejs(
    case_dir: str,
    output_path: str,
    masks: Dict[str, Any],
    step_size: int,
    downsample_factor: float,
    isotropic_resample: bool,
    gaussian_sigma: float,
    smooth: bool,
    default_opacity: float,
    prob_threshold: float,
    skip_empty: bool,
    title: str,
) -> Dict[str, Any]:
    """Three.js rendering with per-organ segmentation statistics."""
    import numpy.linalg as LA
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_one_organ(item):
        """Process a single organ mask and return (mesh_entry, stats_entry) or (None, None)."""
        organ_name, mask_src = item
        try:
            color_cfg = ORGAN_COLORS.get(organ_name, {"color": "#CCCCCC", "opacity": default_opacity})
            display_name = get_display_name(organ_name)
            opacity = color_cfg["opacity"]
            color = color_cfg["color"]

            mask_file_path = None
            if isinstance(mask_src, dict) and "_data" in mask_src:
                mask_data = mask_src["_data"]
                affine = mask_src["_affine"]
            else:
                mask_file_path = mask_src
                nii = robust_load_nii(mask_src)
                mask_data = nii.get_fdata()
                affine = nii.affine

            mask_data = binarize_mask(mask_data, prob_threshold=prob_threshold)

            # ---- 碎片过滤：移除小连通分量，对肝脏只保留最大块 ----
            _org_lower = organ_name.lower()
            _is_lesion = _org_lower in {o.lower() for o in _LESION_NAMES}

            # 选择阈值：实体器官 500 体素，血管 100，肿瘤 50
            _min_size = 500 if organ_name == "liver" else (100 if not _is_lesion else 50)
            mask_data = remove_small_objects(mask_data.astype(bool, copy=False),
                                             min_size=_min_size)

            # 肝脏只保留最大连通分量（解决"很多片肝"）
            if organ_name == "liver":
                from scipy import ndimage as ndi
                labeled, num = ndi.label(mask_data)
                if num > 1:
                    sizes = np.bincount(labeled.ravel())[1:]  # [1:] 跳过背景
                    mask_data = (labeled == (np.argmax(sizes) + 1))

            voxel_count = int(mask_data.sum())
            voxel_volume_mm3 = abs(LA.det(affine[:3, :3]))
            volume_mm3 = float(voxel_count * voxel_volume_mm3)

            if skip_empty and voxel_count < 1:
                return None, None

            if isotropic_resample:
                mask_data, affine = resample_for_smoothing(mask_data, affine)

            if downsample_factor > 1.0:
                orig_shape = mask_data.shape
                mask_data = downsample_volume(mask_data, downsample_factor)
                spacing_mult = np.array(orig_shape) / np.array(mask_data.shape)
            else:
                spacing_mult = np.ones(3)

            if gaussian_sigma > 0:
                from scipy.ndimage import gaussian_filter
                mask_data = gaussian_filter(mask_data, sigma=gaussian_sigma, mode='constant')

            orig_spacing = LA.norm(affine[:3, :3], axis=0)
            effective_spacing = orig_spacing * spacing_mult

            verts, faces = extract_mesh(
                mask_data,
                spacing=tuple(effective_spacing),
                step_size=step_size,
                smooth=smooth,
            )
            if verts is None:
                return None, None

            verts_vox = verts / effective_spacing
            verts_world = voxel_to_world(verts_vox, affine)
            quantize_vertices(verts_world, precision_mm=0.1)

            mesh_entry = {
                "name": organ_name,
                "display_name": display_name,
                "original_name": organ_name,
                "color": color,
                "opacity": opacity,
                "volume_cm3": round(volume_mm3 / 1000.0, 2),
                "vertices": verts_world.astype(np.float32),
                "faces": faces.astype(np.int32),
            }

            area = mesh_surface_area(verts, faces) if len(faces) > 0 else 0.0
            stats_entry = {
                "display_name": display_name,
                "color": color,
                "opacity": float(opacity),
                "mask_file": mask_file_path,
                "voxel_count": voxel_count,
                "volume_mm3": round(volume_mm3, 2),
                "volume_cm3": round(volume_mm3 / 1000.0, 2),
                "vertices": int(len(verts)),
                "faces": int(len(faces)),
                "surface_area_mm2": round(area, 2),
            }
            return mesh_entry, stats_entry
        except Exception:
            return None, None

    mesh_list = []
    organ_stats = {}
    items = list(sorted(masks.items()))
    n_threads = min(8, len(items)) if items else 1
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        fut_map = {pool.submit(_process_one_organ, item): item[0] for item in items}
        for fut in as_completed(fut_map):
            name = fut_map[fut]
            try:
                mesh_entry, stats_entry = fut.result()
                if mesh_entry is not None:
                    mesh_list.append(mesh_entry)
                    organ_stats[name] = stats_entry
            except Exception:
                pass

    if not mesh_list:
        return {
            "status": "error",
            "message": "No meshes were generated. All masks empty?",
            "file_path": output_path,
            "organ_stats": {},
        }

    # ---- Write 3D JSON file (meshes data, separate from HTML) ----
    import json as _json

    # Center vertices for Three.js viewing
    all_verts = np.concatenate([m["vertices"] for m in mesh_list], axis=0)
    center = all_verts.mean(axis=0).tolist()

    meshes_json = []
    for m in mesh_list:
        v = (m["vertices"] - center).ravel().tolist()
        f = m["faces"].ravel().tolist()
        meshes_json.append({
            "name": m["display_name"],
            "original_name": m.get("original_name", ""),
            "color": m["color"],
            "opacity": m["opacity"],
            "volume_cm3": m.get("volume_cm3", 0),
            "verts": v,
            "faces": f,
        })

    json_path = str(Path(output_path).with_suffix(".json"))

    # Preserve existing resection_planes and tumor_cloud if present
    # (these are added by plan_resection skill, which runs after 3D reconstruction)
    existing_resection = []
    existing_tumor_cloud = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_resection = existing.get("resection_planes", [])
            existing_tumor_cloud = existing.get("tumor_cloud", [])
        except Exception:
            pass

    json_data = {
        "center_offset": center,
        "meshes": meshes_json,
        "resection_planes": existing_resection,
    }
    if existing_tumor_cloud:
        json_data["tumor_cloud"] = existing_tumor_cloud

    # Sanitize NaN/Inf to null (browser JSON.parse rejects them)
    json_str = _json.dumps(json_data, separators=(",", ":"), allow_nan=False)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    # ---- Generate HTML that fetches the JSON ----
    json_url = Path(json_path).name  # e.g. "P026_3d.json"
    # Skills API addresses the prepared output directory, so use its actual
    # basename. This also supports manually prepared aliases such as
    # output/CRLM-1012 while the source file is named CRLM-CT-1012_3d.html.
    _page_case_name = Path(output_path).parent.name
    html = _make_threejs_html(
        title=title, three_sources=_read_threejs_sources(),
        json_url=json_url, bezier_source=_read_bezier_surface_source(),
        case_name=_page_case_name,
        mask_dir=str(Path(case_dir).resolve()),
        output_dir=str(Path(output_path).parent.resolve()),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return {"status": "ok", "file_path": output_path, "organ_stats": organ_stats}


# ======================================================================
# [Section 11]  Python API — generate_visualization
# ======================================================================

def generate_visualization(
    case_dir: str,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    step_size: int = 2,
    downsample_factor: float = 1.0,
    isotropic_resample: bool = True,
    gaussian_sigma: float = 0.3,
    smooth: bool = True,
    default_opacity: float = 0.85,
    prob_threshold: float = 0.5,
    skip_empty: bool = True,
    title: Optional[str] = None,
    timeout_minutes: Optional[float] = None,
) -> Dict[str, Any]:
    """
    从分割掩码生成 3D 可视化 HTML。

    Args:
        case_dir: 分割结果目录，包含各器官的 .nii.gz 掩码文件
        output_dir: 输出 HTML 的目录（默认: output/）
        output_filename: 输出文件名（不含 .html）
        step_size: marching cubes 步长（1=最精细慢，2=推荐）
        downsample_factor: 体素降采样倍率（1.0=不降采样）
        isotropic_resample: 是否对各向异性体素做平滑插值（消除层面间锯齿）
        smooth: 是否对网格做拉普拉斯平滑
        default_opacity: 器官默认不透明度（0.0~1.0）
        prob_threshold: 概率图二值化阈值（0~1）
        skip_empty: 是否跳过空掩码
        title: HTML 页面标题
        timeout_minutes: 超时时间（分钟），None=不超时

    Returns:
        dict，包含 status / file_path / organs / segmentation_results / stats 等字段。
        segmentation_results: 每类器官的详细分割统计（体积 mm³/mL、表面积 mm²、体素数、网格顶点/面数等）
    """
    start_time = time.time()
    case_dir = str(Path(case_dir).resolve())

    if not os.path.isdir(case_dir):
        return _error(f"Case directory not found: {case_dir}")

    output_dir = str(Path(output_dir or DEFAULT_OUTPUT_DIR).resolve())
    os.makedirs(output_dir, exist_ok=True)

    case_name = Path(case_dir).name
    if output_filename:
        if output_filename.endswith(".html"):
            output_filename = output_filename[:-5]
        output_path = str(Path(output_dir) / f"{output_filename}.html")
    else:
        output_path = str(Path(output_dir) / f"{case_name}_3d.html")

    title = title or f"3D Visualization - {case_name}"

    try:
        masks = find_mask_files(case_dir)
    except FileNotFoundError as e:
        return _error(str(e))

    if not masks:
        return _error(f"No mask files found in {case_dir}. "
                      f"Ensure the directory contains *.nii.gz mask files.")

    try:
        vis_kwargs = dict(
            step_size=step_size,
            downsample_factor=downsample_factor,
            isotropic_resample=isotropic_resample,
            gaussian_sigma=gaussian_sigma,
            smooth=smooth,
            default_opacity=default_opacity,
            prob_threshold=prob_threshold,
            skip_empty=skip_empty,
            title=title,
        )
        result = _generate_threejs(
            case_dir, output_path, masks, **vis_kwargs,
        )
    except Exception as e:
        return _error(f"Visualization generation failed: {e}")

    elapsed = time.time() - start_time
    file_path = result.get("file_path", output_path)
    filename = Path(file_path).name

    organs = list(masks.keys())
    organ_display = [ORGAN_DISPLAY_NAMES.get(o, o) for o in organs]
    organ_stats = result.get("organ_stats", {})

    # 计算相对于 OUTPUT 根目录的路径，确保 URL 可访问
    try:
        _rel = str(Path(file_path).resolve().relative_to(DEFAULT_OUTPUT_DIR.resolve()))
        _url = f"{PUBLIC_BASE_URL}/output/{_rel}"
    except ValueError:
        # fallback: 文件在 OUTPUT_DIR 之外，直接用文件名
        _url = f"{PUBLIC_BASE_URL}/output/{filename}"

    return {
        "status": result.get("status", "ok"),
        "message": f"Visualization generated in {elapsed:.1f}s",
        "file_path": file_path,
        "filename": filename,
        "url": _url,
        "organs": organ_display,
        "segmentation_results": organ_stats,
        "stats": _collect_stats(case_dir, masks),
        "elapsed_seconds": round(elapsed, 1),
    }


def _collect_stats(case_dir: str, masks: Dict[str, Any]) -> Dict[str, Any]:
    """Collect basic organ display info."""
    stats = {}
    for organ_name in sorted(masks.keys()):
        stats[organ_name] = {
            "display_name": ORGAN_DISPLAY_NAMES.get(organ_name, organ_name),
            "color": ORGAN_COLORS.get(organ_name, {}).get("color", "#CCCCCC"),
            "opacity": ORGAN_COLORS.get(organ_name, {}).get("opacity", 0.85),
        }
    return stats


def _error(message: str) -> Dict[str, Any]:
    """Standard error response dict."""
    return {
        "status": "error",
        "message": message,
        "file_path": None,
        "filename": None,
        "url": None,
        "organs": [],
        "segmentation_results": {},
        "stats": {},
        "elapsed_seconds": 0.0,
    }


def list_available_organs() -> List[Dict[str, Any]]:
    """List all renderable organs with colours."""
    result = []
    for name, cfg in ORGAN_COLORS.items():
        result.append({
            "name": name,
            "display_name": ORGAN_DISPLAY_NAMES.get(name, name),
            "color": cfg["color"],
            "opacity": cfg["opacity"],
            "is_lesion": cfg["opacity"] >= 1.0,
        })
    return result


def estimate_output_size(case_dir: str, downsample_factor: float = 2.0) -> Dict[str, Any]:
    """
    Estimate 3D HTML output size without generating meshes.
    """
    masks = find_mask_files(case_dir)
    total_raw_mb = 0.0
    organ_sizes = {}

    for organ_name, mask_src in sorted(masks.items()):
        if isinstance(mask_src, dict) and "_data" in mask_src:
            size_mb = mask_src["_data"].nbytes / (1024 * 1024)
        else:
            size_mb = os.path.getsize(mask_src) / (1024 * 1024)
        total_raw_mb += size_mb
        organ_sizes[organ_name] = round(size_mb, 2)

    scale = 1.0 / (downsample_factor ** 3)
    estimated_kb = total_raw_mb * 1024 * 0.5 * scale * 100 / 1024

    return {
        "num_organs": len(masks),
        "organ_sizes_mb": organ_sizes,
        "total_raw_mb": round(total_raw_mb, 2),
        "estimated_file_size_kb": round(estimated_kb, 0),
        "downsample_factor": downsample_factor,
    }


# ======================================================================
# [Section 11b]  CRLM Pipeline Helpers
# ======================================================================

def _find_sibling_gt_masks(input_path: str) -> List[str]:
    """Return only explicitly named CRLM masks beside a NIfTI CT.

    A dataset directory can contain thousands of unrelated CT volumes, so
    sibling ``*.nii.gz`` files must never be treated as masks based only on
    their extension.
    """
    input_path = os.path.abspath(input_path)
    input_dir = os.path.dirname(input_path)
    candidates = []
    for path in glob.glob(os.path.join(input_dir, "*.nii.gz")):
        if os.path.abspath(path) == input_path:
            continue
        basename = os.path.basename(path)
        if (
            basename in _CRLM_GT_MASK_BASENAMES
            or _CRLM_TUMOR_MASK_RE.fullmatch(basename)
        ):
            candidates.append(path)

    candidates.sort()
    if len(candidates) > _MAX_SIBLING_GT_MASKS:
        raise RuntimeError(
            f"Refusing to copy {len(candidates)} sibling GT masks "
            f"(limit: {_MAX_SIBLING_GT_MASKS}) from {input_dir}"
        )
    return candidates


def _ensure_ct_symlink(ct_nifti_path: str, output_dir: str) -> str:
    """Atomically point ``output_dir/ct.nii.gz`` at the current CT input."""
    target = os.path.abspath(ct_nifti_path)
    ct_link = os.path.join(output_dir, "ct.nii.gz")

    if os.path.lexists(ct_link):
        if os.path.islink(ct_link):
            current = os.readlink(ct_link)
            if not os.path.isabs(current):
                current = os.path.join(os.path.dirname(ct_link), current)
            if os.path.abspath(current) == target:
                return ct_link
        elif os.path.samefile(ct_link, target):
            return ct_link
        else:
            raise RuntimeError(
                f"Cannot replace non-symlink CT file with a different input: {ct_link}"
            )

    temporary_link = f"{ct_link}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        os.symlink(target, temporary_link)
        os.replace(temporary_link, ct_link)
    finally:
        if os.path.lexists(temporary_link):
            os.unlink(temporary_link)
    return ct_link


def _crlm_prepare_input(
    input_path: str,
    output_dir: str,
    force: bool = False,
) -> Tuple[str, bool]:
    """
    Prepare CT NIfTI input for CRLM pipeline.

    If input is a DICOM directory: convert CT to NIfTI, extract SEG GT masks.
    If input is a .nii.gz file: check for existing GT masks alongside it.

    Returns:
        (ct_nifti_path, has_seg_gt)
    """
    ct_nifti_path = None
    has_seg_gt = False

    if os.path.isdir(input_path):
        # ---------- DICOM 输入 ----------
        from Tool_Box.dicom_utils import (
            find_crlm_dicom_paths,
            convert_dicom_ct_to_nifti,
            convert_dicom_seg_to_masks,
        )
        input_path = input_path.rstrip("/")
        # 从路径中提取 CRLM-CT-XXXX，支持裸数字回退 (如 "1033" → "CRLM-CT-1033")
        _path_str = input_path.replace("\\", "/")
        case_id = _resolve_crlm_case_id(_path_str)

        nifti_cache = os.path.join(_CRLM_NIFTI_ROOT_DEFAULT, case_id)
        os.makedirs(nifti_cache, exist_ok=True)
        ct_nifti_path = os.path.join(nifti_cache, "ct.nii.gz")

        # Step 1: Convert DICOM CT → NIfTI (or use cached)
        if not force and os.path.isfile(ct_nifti_path):
            _log(f"[CRLM Prep] Using cached CT: {ct_nifti_path}")
        else:
            try:
                ct_series_dir, _ = find_crlm_dicom_paths(_CRLM_ROOT_DEFAULT, case_id)
                convert_dicom_ct_to_nifti(ct_series_dir, ct_nifti_path)
            except FileNotFoundError as e:
                _log(f"[CRLM Prep] DICOM CT not found: {e}, trying cache")
                if not os.path.isfile(ct_nifti_path):
                    raise

        # Step 2: Always attempt SEG GT extraction (independent of CT caching).
        #          This ensures demo mode can find GT masks on subsequent runs.
        try:
            _, seg_file = find_crlm_dicom_paths(_CRLM_ROOT_DEFAULT, case_id)
            gt_dir = os.path.join(output_dir, "gt_masks")
            os.makedirs(gt_dir, exist_ok=True)
            _log(f"[CRLM Prep] Found DICOM SEG, extracting GT to {gt_dir}")
            convert_dicom_seg_to_masks(seg_file, ct_nifti_path, gt_dir)
            # Verify extraction actually produced mask files
            gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz")))
            if gt_files:
                has_seg_gt = True
                _log(f"[CRLM Prep] SEG extraction produced {len(gt_files)} mask(s)")
            else:
                _log(f"[CRLM Prep] ⚠ SEG extraction yielded no masks for {case_id}")
        except (FileNotFoundError, Exception):
            _log(f"[CRLM Prep] No DICOM SEG GT found for {case_id}")
    else:
        # ---------- NIfTI 输入 ----------
        ct_nifti_path = input_path
        input_dir = os.path.dirname(input_path)
        gt_dir = os.path.join(output_dir, "gt_masks")

        # Only accept explicit CRLM mask names.  Treating every sibling NIfTI
        # as GT can copy an entire multi-case CT dataset into one case output.
        gt_candidates = _find_sibling_gt_masks(input_path)
        if gt_candidates:
            os.makedirs(gt_dir, exist_ok=True)
            _log(
                f"[CRLM Prep] Copying {len(gt_candidates)} explicitly named "
                f"GT mask(s) from {input_dir}"
            )
            for f in gt_candidates:
                shutil.copy2(f, os.path.join(gt_dir, os.path.basename(f)))
            has_seg_gt = True
            _log(f"[CRLM Prep] Copied {len(gt_candidates)} GT masks from {input_dir}")
        else:
            # NIfTI 文件名含 CRLM 或裸数字 → 尝试从 DICOM SEG 提取 GT 掩码
            _path_str = input_path.replace("\\", "/")
            _case_id = _resolve_crlm_case_id(_path_str)
            if _case_id.startswith("CRLM-CT-"):
                case_id = _case_id
                _log(f"[CRLM Prep] NIfTI input contains CRLM case: {case_id}, trying DICOM SEG ...")
                try:
                    from Tool_Box.dicom_utils import (
                        find_crlm_dicom_paths,
                        convert_dicom_seg_to_masks,
                    )
                    _, seg_file = find_crlm_dicom_paths(_CRLM_ROOT_DEFAULT, case_id)
                    os.makedirs(gt_dir, exist_ok=True)
                    _log(f"[CRLM Prep] Found DICOM SEG, extracting GT to {gt_dir}")
                    convert_dicom_seg_to_masks(seg_file, ct_nifti_path, gt_dir)
                    has_seg_gt = True
                except FileNotFoundError as e:
                    _log(f"[CRLM Prep] DICOM SEG not found for {case_id}: {e}")
                except Exception as e:
                    _log(f"[CRLM Prep] Failed to extract DICOM SEG for {case_id}: {e}")

    if has_seg_gt:
        gt_count = len(glob.glob(os.path.join(output_dir, "gt_masks", "*.nii.gz")))
        _log(f"[CRLM Prep] GT masks: {gt_dir} ({gt_count} files)")
    else:
        _log("[CRLM Prep] No DICOM SEG GT found")

    return ct_nifti_path, has_seg_gt


def _use_gt_masks(output_dir: str, mask_dir: str) -> None:
    """
    Copy ground truth masks from ``output_dir/gt_masks/`` (or nifti cache)
    to ``mask_dir`` so the rest of the pipeline can use them directly.

    Priority:
      1. ``output_dir/gt_masks/`` — extracted by ``_crlm_prepare_input`` from DICOM SEG.
      2. nifti cache (``data/CRLM/nifti/<case_id>/``) — pre-computed from prior runs.

    Normalizes DICOM SEG naming (underscores) to VISTA3D naming (spaces)
    so downstream postprocessing (``_crlm_run_postprocessing``) works correctly.

    Only copies files matching the normal VISTA3D pipeline's expected output
    names, so demo mode and normal mode return identical mask file structures.
    """
    # ---- Build allowed name set matching normal VISTA3D pipeline output ----
    # VISTA3D native names (before postprocessing rename + split)
    _VISTA3D_NAMES = {f"{o}.nii.gz" for o in _CRLM_DEFAULT_ORGANS}
    # After rename: hepatic vessel → hepatic, portal vein → portal
    _RENAMED_NAMES = {"hepatic.nii.gz", "portal.nii.gz"}
    # After tumor split: hepatic tumor → tumor_1, tumor_2, ...
    # (matched by pattern below)
    _ALLOWED_NAMES = _VISTA3D_NAMES | _RENAMED_NAMES
    _TUMOR_PAT = re.compile(r"^tumor_\d+\.nii\.gz$")

    # Normalize DICOM SEG → VISTA3D naming
    _GT_NAME_FIX = {
        "hepatic_tumor.nii.gz": "hepatic tumor.nii.gz",
    }

    gt_dirs = []

    # Priority 1: freshly extracted GT masks
    gt_dir = os.path.join(output_dir, "gt_masks")
    if os.path.isdir(gt_dir):
        gt_dirs.append(gt_dir)

    # Priority 2: nifti cache (if we can deduce case_id)
    _path_str = output_dir.replace("\\", "/")
    _cache_case = _resolve_crlm_case_id(_path_str)
    if _cache_case.startswith("CRLM-CT-"):
        cache_dir = os.path.join(_CRLM_NIFTI_ROOT_DEFAULT, _cache_case)
        if os.path.isdir(cache_dir):
            gt_dirs.append(cache_dir)

    seen = set()
    for src_dir in gt_dirs:
        for fname in sorted(glob.glob(os.path.join(src_dir, "*.nii.gz"))):
            fbase = os.path.basename(fname)
            if fbase == "ct.nii.gz":
                continue
            # Normalize filename: DICOM SEG naming → VISTA3D naming
            dst_base = _GT_NAME_FIX.get(fbase, fbase)
            # Filter: only keep files matching normal VISTA3D pipeline output
            if dst_base not in _ALLOWED_NAMES and not _TUMOR_PAT.match(dst_base):
                if dst_base not in seen:
                    _log(f"[GT Copy]  ⏭ Skipping (not in VISTA3D output): {dst_base}")
                continue
            if dst_base in seen:
                continue
            dst = os.path.join(mask_dir, dst_base)
            if os.path.isfile(dst):
                continue
            shutil.copy2(fname, dst)
            seen.add(dst_base)

    _log(f"[GT Copy] Copied {len(seen)} GT mask(s) to {mask_dir}")
    if not seen:
        # Fallback: try extracting from DICOM SEG directly (for cases where
        # convert_dicom_seg_to_masks created the gt_masks dir but no files).
        _path_str = output_dir.replace("\\", "/")
        _m = re.search(r"(CRLM-CT-\d+)", _path_str)
        if _m:
            try:
                from Tool_Box.dicom_utils import (
                    find_crlm_dicom_paths,
                    convert_dicom_seg_to_masks,
                )
                case_id = _m.group(1)
                ct_nifti_path = os.path.join(
                    _CRLM_NIFTI_ROOT_DEFAULT, case_id, "ct.nii.gz",
                )
                fallback_dir = os.path.join(output_dir, "gt_masks")
                os.makedirs(fallback_dir, exist_ok=True)
                _, seg_file = find_crlm_dicom_paths(_CRLM_ROOT_DEFAULT, case_id)
                _log(f"[GT Copy]  ⚠ Re-extracting SEG from {seg_file}")
                convert_dicom_seg_to_masks(seg_file, ct_nifti_path, fallback_dir)
                # Copy the freshly extracted masks
                for fname in sorted(glob.glob(os.path.join(fallback_dir, "*.nii.gz"))):
                    fbase = os.path.basename(fname)
                    if fbase == "ct.nii.gz":
                        continue
                    dst_base = _GT_NAME_FIX.get(fbase, fbase)
                    if dst_base not in _ALLOWED_NAMES and not _TUMOR_PAT.match(dst_base):
                        continue
                    if dst_base in seen:
                        continue
                    shutil.copy2(fname, os.path.join(mask_dir, dst_base))
                    seen.add(dst_base)
            except Exception:
                pass
        if seen:
            _log(f"[GT Copy] Fallback SEG extraction produced {len(seen)} mask(s)")
        else:
            _log(f"[GT Copy]  ⚠ No GT masks found in {gt_dirs}")


def _crlm_run_postprocessing(mask_dir: str) -> List[str]:
    """
    CRLM postprocessing:
    1. Rename vessels: hepatic vessel → hepatic, portal vein and splenic vein → portal
    2. Split hepatic tumor into tumor_1, tumor_2, ... via connected components.
    3. Optimize hepatic and portal vessel continuity from the same raw inputs.

    Returns list of final mask names (stems) in mask_dir.
    """
    from Tool_Box.crlm_postprocess import (
        optimize_crlm_vessels,
        rename_vessel_masks,
        split_hepatic_tumor,
    )
    from Tool_Box.mask_resolution import scan_logical_masks

    _log("[CRLM Postprocessing] Renaming vessel masks ...")
    rename_vessel_masks(mask_dir)

    _log("[CRLM Postprocessing] Splitting hepatic tumor ...")
    split_hepatic_tumor(mask_dir)

    _log("[CRLM Postprocessing] Optimizing vessel masks ...")
    report_path = Path(mask_dir).parent / "vessel_optimization_report.json"
    try:
        optimize_crlm_vessels(mask_dir, report_path, max_gap_mm=4.0)
    except Exception as exc:
        _log(
            f"[CRLM Postprocessing] Vessel optimization failed for "
            f"{mask_dir}: {exc}; continuing with existing logical masks"
        )

    mask_names = sorted(scan_logical_masks(mask_dir))
    _log(f"[CRLM Postprocessing] Final masks: {mask_names}")
    return mask_names


def _crlm_run_analysis_and_report(
    ct_nifti_path: str,
    mask_dir: str,
    output_dir: str,
    case_id: str,
    base_report: str,
) -> str:
    """
    Run CRLM quantitative analysis and append liver report to base structural report.

    Returns combined report string (base_report + liver_report).
    """
    from Tool_Box.liver_analysis import analyze_liver_case, generate_liver_report

    # Build mask_paths dict from mask_dir
    mask_paths = {}
    for fname in sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz"))):
        stem = Path(fname).stem
        if stem.endswith(".nii"):
            stem = stem[:-4]
        mask_paths[stem] = fname

    tumor_labels = sorted(k for k in mask_paths if "tumor" in k.lower() and "all" not in k.lower())
    # Match both CRLM-renamed ("hepatic", "portal") and VISTA3D-native
    # ("hepatic vessel", "portal vein and splenic vein") vessel labels.
    vessel_labels = [
        k for k in mask_paths
        if ("hepatic" in k.lower() or "portal" in k.lower())
        and "tumor" not in k.lower()
    ]
    liver_label = "liver" if "liver" in mask_paths else None

    _log(f"[CRLM Analysis] Liver: {liver_label}, Tumors: {tumor_labels}, Vessels: {vessel_labels}")

    result = analyze_liver_case(
        ct_nifti_path=ct_nifti_path,
        mask_paths=mask_paths,
        liver_label=liver_label,
        tumor_labels=tumor_labels,
        vessel_labels=vessel_labels,
    )

    liver_report = generate_liver_report(result)
    combined = base_report + "\n\n" + liver_report

    # Clinical sanity check (non-blocking info logging)
    _crlm_clinical_sanity_check(case_id, result)

    return combined


def _crlm_clinical_sanity_check(case_id: str, result: Dict) -> None:
    """
    Cross-check measured max tumor diameter against CRLM clinical Excel data.
    Non-blocking: only logs info/warning, never raises.
    """
    xlsx_path = _CRLM_CLINICAL_XLSX
    if not os.path.isfile(xlsx_path):
        _log("[CRLM Clinical] Excel data not found, skipping cross-check")
        return

    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        _log(f"[CRLM Clinical] Could not read Excel: {e}")
        return

    row = df[df["Patient-ID"] == case_id]
    if row.empty:
        _log(f"[CRLM Clinical] No clinical record for {case_id}")
        return

    clinical_diameter = row.iloc[0].get("max_tumor_size")
    if clinical_diameter is None or clinical_diameter == -999:
        _log(f"[CRLM Clinical] {case_id} max_tumor_size missing")
        return

    diameters = [
        tr.get("diameter", {}).get("max_diameter_mm")
        for tr in result.get("tumor_results", {}).values()
        if tr.get("diameter", {}).get("max_diameter_mm") is not None
    ]

    if not diameters:
        _log("[CRLM Clinical] No measurable tumors detected")
        return

    max_measured_mm = max(diameters)
    max_clinical_mm = float(clinical_diameter) * 10  # cm → mm
    ratio = max_measured_mm / max_clinical_mm if max_clinical_mm > 0 else 0

    if ratio < 0.5:
        _log(f"[CRLM Clinical] WARNING: measured {max_measured_mm:.1f}mm vs "
             f"clinical {max_clinical_mm:.1f}mm (ratio {ratio:.2f})")
    else:
        _log(f"[CRLM Clinical] OK: measured {max_measured_mm:.1f}mm vs "
             f"clinical {max_clinical_mm:.1f}mm (ratio {ratio:.2f})")


def _resolve_crlm_case_id(path_str: str) -> str:
    """
    Extract CRLM case ID from a path, handling both full and shorthand forms.

    Priority:
      1. ``CRLM-CT-NNNN`` in path (e.g. ``.../CRLM-CT-1033/...``) → ``CRLM-CT-1033``
      2. Bare number in path and ``CRLM-CT-{num}`` exists in dataset → ``CRLM-CT-{num}``
      3. Fallback to path basename.

    Args:
        path_str: Input path (directory or file) possibly containing a CRLM case ID.

    Returns:
        The resolved case ID string.
    """
    _m = re.search(r"(CRLM-CT-\d+)", path_str)
    if _m:
        return _m.group(1)

    # Try bare number → CRLM-CT-{num} lookup
    _num_m = re.search(r"(\d{4})", path_str)
    if _num_m:
        _candidate = f"CRLM-CT-{_num_m.group(1)}"
        if os.path.isdir(os.path.join(_CRLM_ROOT_DEFAULT, _candidate)):
            _log(f"[CRLM Case] Resolved bare number {_num_m.group(1)} → {_candidate}")
            return _candidate

    # Fallback
    return os.path.basename(path_str.rstrip("/"))


def _crlm_list_all_cases() -> List[str]:
    """List all CRLM-CT-XXXX case IDs in the CRLM dataset root."""
    cases = []
    if not os.path.isdir(_CRLM_ROOT_DEFAULT):
        _log(f"[CRLM] Dataset root not found: {_CRLM_ROOT_DEFAULT}")
        return cases
    for entry in sorted(os.listdir(_CRLM_ROOT_DEFAULT)):
        path = os.path.join(_CRLM_ROOT_DEFAULT, entry)
        if os.path.isdir(path) and entry.startswith("CRLM-CT-"):
            cases.append(entry)
    return cases


def _detect_pipeline_type(
    input_path: str,
    explicit: str = "auto",
) -> str:
    """
    Pipeline type detection — always returns "crlm".
    CRLM is the only pipeline.
    """
    return "crlm"


# ======================================================================
# [Section 12]  Pipeline — process_nifti_file
# ======================================================================

def process_nifti_file(
    input_path: str,
    output_dir: Optional[str] = None,
    case_name: Optional[str] = None,
    device: str = "auto",
    organ_list: Optional[List[str]] = None,
    step_size: int = 2,
    downsample_factor: float = 1.0,
    keep_masks: bool = True,
    verbose: bool = True,
    progress_tracker: Optional[ProgressTracker] = None,
    top_k: int = 3,
    scoring_mode: str = "crlm",
    nifti_path: Optional[str] = None,
    force: bool = False,
    skip_viz: bool = False,
    vista3d_config: Optional[str] = None,
    demo_mode: Optional[bool] = None,
    seg_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行 CRLM 肝癌专项管线。

    Args:
        input_path: DICOM 目录路径 或 .nii.gz 文件路径。
        output_dir: 输出目录（默认: output/<case_name>/）。
        case_name: 病例名（默认从文件名或 DICOM 目录名自动推断）。
        device: 自动选择（"auto"）或指定 CUDA 设备（如 "cuda:0"）。
        organ_list: 要分割的器官列表（默认: CRLM 专用列表）。
        step_size: Marching cubes 步长（1=最精细，2=推荐）。
        downsample_factor: 体素降采样倍率（1.0=不降采样）。
        keep_masks: 是否保留中间掩码文件（默认 True，设为 False 则删除）。
        verbose: 打印进度。
        progress_tracker: 进度追踪器（HTTP 异步模式用）。
        top_k: 返回的 Top 信息量切片数（默认 3）。
        scoring_mode: 切片评分模式: "composite" / "cflt" / "crlm"（默认）。
        nifti_path: [已弃用] 向后兼容别名，请使用 input_path。
        force: 强制重新转换 DICOM。
        skip_viz: 跳过 3D 可视化。
        demo_mode: 演示模式（默认: True）。开启时若存在预分割 GT 掩码，跳过模型推理。
        seg_backend: 分割后端；默认读取 SEGMENTATION_BACKEND（默认 vista3d）。

    Returns:
        包含 visualization_html / best_slices / structural_report / mask_files 等字段的 dict。
        CRLM 分析结果追加于 structural_report 末尾。
    """
    start = time.time()
    _t = lambda s, p, m="": progress_tracker.update(s, p, m) if progress_tracker else None
    try:
        effective_backend = _normalize_segmentation_backend(seg_backend)
    except ValueError as exc:
        return _pipeline_error(str(exc))

    # ---- Backward compat: accept either input_path or nifti_path ----
    resolved_input = str(Path(input_path or nifti_path).resolve())

    # ---- Always CRLM pipeline ----
    if verbose:
        _log("=" * 60)
        _log(f"  CRLM Pipeline — {resolved_input}")
        _log("=" * 60)

    # ---- Input preparation ----
    ct_nifti_path = resolved_input
    has_seg_gt = False

    if os.path.isdir(resolved_input):
        # DICOM input
        path_str = resolved_input.replace("\\", "/")
        case_id_from_path = _resolve_crlm_case_id(path_str)
        if case_name is None:
            case_name = case_id_from_path
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent /
                             "output" / case_name)
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            _log(f"[0/5] Preparing DICOM input ...")
        _t("DICOM preparation", 2)
        ct_nifti_path, has_seg_gt = _crlm_prepare_input(resolved_input, output_dir, force=force)
        if not os.path.isfile(ct_nifti_path):
            return _pipeline_error(f"CRLM prep failed: no CT at {ct_nifti_path}")
    else:
        # .nii.gz input
        if not os.path.isfile(ct_nifti_path):
            return _pipeline_error(f"File not found: {ct_nifti_path}")
        if case_name is None:
            stem = Path(ct_nifti_path).stem
            if stem.endswith(".nii"):
                stem = stem[:-4]
            case_name = stem
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent /
                             "output" / case_name)
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)

        # 尝试从 CRLM DICOM SEG 提取 GT 掩码（文件名含 CRLM-CT-NNNN 时）
        _, has_seg_gt = _crlm_prepare_input(resolved_input, output_dir, force=force)

    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    # ---- Use CRLM default organ list if none specified ----
    if organ_list is None:
        effective_organ_list = _CRLM_DEFAULT_ORGANS
    else:
        effective_organ_list = organ_list

    # ---- Step 1: Load CT ----
    if verbose:
        _log(f"[1/5] Loading CT volume …")
    _t("Loading CT", 5)
    try:
        ct_volume, affine = _load_ct(ct_nifti_path)
    except Exception as e:
        return _pipeline_error(f"Failed to load CT: {e}")
    if verbose:
        _log(f"       Shape: {ct_volume.shape}, dtype: {ct_volume.dtype}")

    # Also load raw HU for CFLT / crlm scoring mode
    ct_raw = None
    if scoring_mode in ("cflt", "crlm"):
        try:
            ct_raw, _ = _load_ct_raw(ct_nifti_path)
        except Exception:
            if verbose:
                _log(f"       [WARN] Could not load raw HU, falling back to composite")
            scoring_mode = "composite"

    # ---- 演示模式：有 GT 掩码时跳过模型推理 ----
    _demo_mode = _DEMO_MODE if demo_mode is None else demo_mode

    # ---- Step 2: Segmentation backend or demo-mode GT ----
    if _demo_mode and has_seg_gt:
        if verbose:
            _log(f"[2/5] 🎯 演示模式 — 使用预分割 GT 掩码（跳过模型推理）")
        _t("GT mask copy", 10, "Copying GT masks ...")
        _use_gt_masks(output_dir, mask_dir)
        # 检查是否有实际可用的掩码，没有则回退到选定后端
        gt_count = len(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        if gt_count == 0:
            if verbose:
                _log(f"       ⚠ GT 掩码为空，自动回退到 {effective_backend} 分割")
            _demo_mode = False  # 标记已回退，避免后续误判
            if verbose:
                _log(f"[2/5] Running {effective_backend} segmentation (device={device}) …")
            _t(f"{effective_backend} segmentation", 10, "Initializing model...")
            try:
                _run_segmentation(
                    effective_backend, ct_nifti_path, mask_dir,
                    organ_list=effective_organ_list, device=device,
                    vista3d_config=vista3d_config,
                )
            except Exception as e:
                tb = traceback.format_exc()
                return _pipeline_error(f"Segmentation failed: {e}\n{tb}")
            if verbose:
                _log(f"       Masks saved to: {mask_dir}")
            _t(f"{effective_backend} segmentation", 65, "Masks saved")
        else:
            if verbose:
                _log(f"       GT masks copied to: {mask_dir}")
    else:
        if verbose:
            _log(f"[2/5] Running {effective_backend} segmentation (device={device}) …")
        _t(f"{effective_backend} segmentation", 10, "Initializing model...")
        try:
            _run_segmentation(
                effective_backend, ct_nifti_path, mask_dir,
                organ_list=effective_organ_list, device=device,
                vista3d_config=vista3d_config,
            )
        except Exception as e:
            tb = traceback.format_exc()
            return _pipeline_error(f"Segmentation failed: {e}\n{tb}")
        if verbose:
            _log(f"       Masks saved to: {mask_dir}")
        _t(f"{effective_backend} segmentation", 65, "Masks saved")

    # ---- Step 2b: CRLM postprocessing (vessel rename + tumor split) ----
    if verbose:
        _log(f"[2b/5] CRLM postprocessing (vessel rename + tumor split) …")
    effective_organ_list = _crlm_run_postprocessing(mask_dir)
    _write_segmentation_metadata(mask_dir, effective_backend)

    # ---- Collect mask files ----
    mask_files = _list_mask_files(mask_dir, organ_list=effective_organ_list)

    # ---- Steps 3-5: 结构报告 / 3D渲染 / 切片选择 / CRLM分析（全部并行） ----
    if verbose:
        _log(f"[3-5/5] Running report + 3D viz + slice selection + CRLM analysis (parallel) …")
    _t("Parallel tasks", 70)

    html_path = os.path.join(output_dir, f"{case_name}_3d.html")
    report_path = os.path.join(output_dir, f"{case_name}_report.txt")
    best_slice_result = {}

    def _run_report_and_crlm():
        """Generate structural report → save → run CRLM analysis. 串行在此线程内，与外部任务并行。"""
        rep = generate_structural_report(ct_volume, affine, mask_dir,
                                          organ_list=effective_organ_list)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(rep)
        combined = _crlm_run_analysis_and_report(
            ct_nifti_path, mask_dir, output_dir, case_name, rep,
        )
        if combined != rep:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(combined)
        return combined

    def _do_slice_selection(eol=effective_organ_list):
        indices, label_masks, scores = select_top_slices(
            ct_volume, mask_dir, organ_list=eol,
            top_k=top_k, scoring_mode="crlm",
            ct_volume_raw=ct_raw,
            ct_affine=affine,
        )
        slice_results = save_slice_images(
            ct_volume, indices, label_masks, output_dir, case_name,
            organ_names=eol,
            slice_scores=list(scores),
            affine=affine,
        )
        return {
            "slices": slice_results,
            "best_z": slice_results[0]["index"] if slice_results else ct_volume.shape[2] // 2,
            "raw_path": slice_results[0]["png_path"] if slice_results else None,
            "overlay_path": slice_results[0]["overlay_path"] if slice_results else None,
            "scores": scores,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_viz = None
        if not skip_viz:
            fut_viz = pool.submit(
                _generate_3d_html, mask_dir, html_path,
                step_size=step_size,
                downsample_factor=downsample_factor,
                title=f"3D Visualization — {case_name}",
            )
        fut_slices = pool.submit(_do_slice_selection)
        fut_report = pool.submit(_run_report_and_crlm)

        # Collect slice results
        try:
            best_slice_result = fut_slices.result()
        except Exception as e:
            _log(f"       [WARN] Slice selection failed: {e}")
            best_slice_result = {
                "slices": [],
                "best_z": ct_volume.shape[2] // 2,
                "raw_path": None,
                "overlay_path": None,
                "scores": [],
            }

        # Collect 3D HTML
        if fut_viz:
            try:
                html_path = fut_viz.result()
            except Exception as e:
                if verbose:
                    _log(f"       [WARN] 3D rendering failed: {e}")
                html_path = None

        # Collect combined report (report + CRLM analysis)
        combined_report = report_path
        try:
            combined_report = fut_report.result()
        except Exception as e:
            if verbose:
                _log(f"       [WARN] Report/CRLM analysis failed: {e}")
            # Fallback: try reading report from disk
            if os.path.isfile(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    combined_report = f.read()
            else:
                combined_report = None

    if verbose:
        _log(f"       Top slices: {[s['index'] for s in best_slice_result.get('slices', [])]}")
        if html_path:
            _log(f"       3D HTML: {html_path}")

    # ---- Cleanup optional masks ----
    if not keep_masks and os.path.isdir(mask_dir):
        shutil.rmtree(mask_dir, ignore_errors=True)

    _t("Finalizing", 100, "Pipeline complete")
    elapsed = time.time() - start

    # Build URL from filesystem path
    _proc_out = Path(__file__).resolve().parent / "output"
    _vis_url = None
    if html_path:
        try:
            _rel = str(Path(html_path).resolve().relative_to(_proc_out.resolve()))
            _vis_url = f"{PUBLIC_BASE_URL}/process-output/{_rel}"
        except ValueError:
            pass

    result = {
        "status": "ok",
        "message": f"Pipeline completed in {elapsed:.1f}s",
        "visualization_html": html_path,
        "visualization_url": _vis_url,
        "best_slice_png": best_slice_result.get("raw_path"),
        "best_slice_overlay_png": best_slice_result.get("overlay_path"),
        "best_slice_index": best_slice_result.get("best_z"),
        "best_slices": best_slice_result.get("slices", []),
        "top_k": top_k,
        "scoring_mode": scoring_mode,
        "mask_files": mask_files,
        "structural_report": combined_report,
        "structural_report_path": report_path,
        "elapsed_seconds": round(elapsed, 1),
        "case_name": case_name,
    }

    if verbose:
        _log("=" * 60)
        _log(f"  Pipeline completed in {elapsed:.1f}s")
        _log("=" * 60)

    return result


def _pipeline_error(msg: str) -> Dict[str, Any]:
    """Error dict for pipeline return."""
    return {
        "status": "error",
        "message": msg,
        "visualization_html": None,
        "visualization_url": None,
        "best_slice_png": None,
        "best_slice_overlay_png": None,
        "best_slice_index": None,
        "best_slices": [],
        "top_k": 0,
        "scoring_mode": None,
        "mask_files": {},
        "structural_report": None,
        "structural_report_path": None,
        "elapsed_seconds": 0.0,
        "case_name": None,
    }

# ======================================================================
# [Section 14]  CLI entry — pipeline mode
# ======================================================================

def pipeline_main(argv: Optional[List[str]] = None):
    """
    CLI entry point for the CRLM pipeline, with batch support.

    Usage:
      python API.py pipeline /path/to/DICOM/dir/ [options]
      python API.py pipeline /path/to/ct.nii.gz [options]
      python API.py pipeline --all [options]
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="CRLM Pipeline: DICOM/.nii.gz → segmentation → 3D HTML + top-K slices + report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", type=str, default=None,
                        help="Path to DICOM directory or .nii.gz CT file")
    parser.add_argument("--all", action="store_true",
                        help="Batch process all CRLM-CT-XXXX cases")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: output/<case_name>/)")
    parser.add_argument("--device", type=str, default="auto",
                        help="CUDA device ('auto'=auto-select, 'cuda:0'='cuda:1'=specific, default: auto)")
    parser.add_argument(
        "--seg-backend",
        default=None,
        choices=sorted(set(_SEGMENTATION_BACKEND_ALIASES.values())),
        help="Segmentation backend (default: SEGMENTATION_BACKEND or vista3d)",
    )
    parser.add_argument("--step-size", type=int, default=2,
                        help="Marching cubes step (default: 2)")
    parser.add_argument("--downsample", type=float, default=1.0,
                        help="Volume downsampling (default: 1.0)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove intermediate mask files after pipeline (default: keep)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top slices to return (default: 3)")
    parser.add_argument("--scoring", type=str, default="crlm",
                        choices=["composite", "cflt", "crlm"],
                        help="Slice scoring mode (default: crlm)")
    parser.add_argument("--force", action="store_true",
                        help="Force DICOM re-conversion")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip 3D visualization")
    parser.add_argument("--demo", action="store_true", dest="demo", default=None,
                        help="Enable demo mode (use GT masks if available, skip model inference)")
    parser.add_argument("--no-demo", action="store_false", dest="demo", default=None,
                        help="Disable demo mode (always run the selected backend)")

    args = parser.parse_args(argv)

    if not args.input and not args.all:
        parser.print_help()
        sys.exit(1)

    # ---- Batch mode ----
    if args.all:
        case_list = _crlm_list_all_cases()
        if not case_list:
            _log(f"No CRLM cases found in {_CRLM_ROOT_DEFAULT}")
            sys.exit(1)
        _log(f"Found {len(case_list)} CRLM cases to process")
        results = []
        for case_id in case_list:
            try:
                case_full_path = os.path.join(_CRLM_ROOT_DEFAULT, case_id)
                res = process_nifti_file(
                    input_path=case_full_path,
                    force=args.force,
                    skip_viz=args.skip_viz,
                    device=args.device,
                    step_size=args.step_size,
                    downsample_factor=args.downsample,
                    keep_masks=not args.cleanup,
                    verbose=not args.quiet,
                    top_k=args.top_k,
                    scoring_mode=args.scoring,
                    demo_mode=args.demo,
                    seg_backend=args.seg_backend,
                )
                results.append(res)
                status = "OK" if res["status"] == "ok" else "FAILED"
                _log(f"  [{case_id}] {status} ({res.get('elapsed_seconds', 0):.1f}s)")
            except Exception as e:
                _log(f"  [{case_id}] ERROR: {e}")
                import traceback
                _log(traceback.format_exc())

        success_count = sum(1 for r in results if r["status"] == "ok")
        total_time = sum(r.get("elapsed_seconds", 0) for r in results)
        _log(f"\nBatch complete: {success_count}/{len(case_list)} succeeded, "
             f"total time {total_time:.1f}s")
        return

    # ---- Single case ----
    result = process_nifti_file(
        input_path=args.input,
        output_dir=args.output_dir,
        device=args.device,
        step_size=args.step_size,
        downsample_factor=args.downsample,
        keep_masks=not args.cleanup,
        verbose=not args.quiet,
        top_k=args.top_k,
        scoring_mode=args.scoring,
        force=args.force,
        skip_viz=args.skip_viz,
        demo_mode=args.demo,
        seg_backend=args.seg_backend,
    )

    if result["status"] == "ok":
        print(f"\n✅ Pipeline OK ({result['elapsed_seconds']:.1f}s)")
        if result.get("visualization_html"):
            _log(f"   3D HTML:       {result['visualization_html']}")
        for i, slc in enumerate(result.get("best_slices", [])):
            _log(f"   Slice #{i+1}:    index={slc['index']}  score={slc.get('score', 'N/A'):.3f}  PNG={slc['png_path']}")
        if result.get("mask_files"):
            _log(f"   Masks:         {len(result['mask_files'])} organs")
        if result.get("structural_report_path"):
            _log(f"   Report:        {result['structural_report_path']}")
    else:
        print(f"\n❌ Pipeline failed: {result['message']}")
        sys.exit(1)


# ======================================================================
# [Section 15]  Pydantic models (FastAPI)
# ======================================================================

if _HAS_FASTAPI:

    class VisualizeRequest(BaseModel):
        """POST /api/visualize request body."""
        case_dir: Optional[str] = Field(
            None, description="Segmentation mask directory (or use image_id)"
        )
        image_id: Optional[str] = Field(
            None, description="Case ID (requires data_root + seg_backend)"
        )
        data_root: Optional[str] = Field(
            None, description="Optional root directory containing prepared cases"
        )
        seg_backend: Optional[str] = Field(
            None, description="Segmentation backend (vista3d / totalsegmentator)"
        )
        step_size: int = Field(2, description="Marching cubes step: 1~4")
        downsample: float = Field(1.0, description="Voxel downsampling: 1.0=no downsampling")
        isotropic_resample: bool = Field(True, description="Isotropic resampling for smoother surfaces")
        gaussian_sigma: float = Field(0.3, description="Gaussian blur sigma (0=off)")
        smooth: bool = Field(True, description="Smooth mesh")
        prob_threshold: float = Field(0.5, description="Probability threshold (0~1)")
        output_filename: Optional[str] = Field(None, description="Custom output filename (without .html)")
        title: Optional[str] = Field(None, description="HTML page title")
        async_mode: bool = Field(False, description="Async mode (returns job_id)")

    class ProcessNiftiRequest(BaseModel):
        """POST /api/process request body."""
        input: str = Field(..., description="Path to DICOM directory or .nii.gz CT file")
        nifti_path: Optional[str] = Field(None, description="[Deprecated] Use `input` instead")
        case_name: Optional[str] = Field(None, description="Case name (from filename if omitted)")
        device: str = Field("auto", description="CUDA device ('auto'=auto-select, 'cuda:0'='cuda:1'=specific)")
        step_size: int = Field(2, description="Marching cubes step: 1~4")
        downsample: float = Field(1.0, description="Voxel downsampling")
        async_mode: bool = Field(False, description="Async mode (returns job_id)")
        top_k: int = Field(3, description="Number of top informative slices (default: 3)")
        scoring_mode: str = Field("crlm", description="Slice scoring mode: 'composite', 'cflt', or 'crlm' (default)")
        force: bool = Field(False, description="Force DICOM re-conversion")
        skip_viz: bool = Field(False, description="Skip 3D visualization")
        demo_mode: Optional[bool] = Field(None, description="Demo mode: use GT masks if available and skip model inference")
        seg_backend: Optional[str] = Field(
            None,
            description="Segmentation backend: vista3d (default) or totalsegmentator",
        )

    class VisualizeResponse(BaseModel):
        """POST /api/visualize sync response."""
        status: str
        message: str
        file_path: Optional[str] = None
        filename: Optional[str] = None
        url: Optional[str] = None
        organs: List[str] = []
        segmentation_results: Dict[str, Any] = Field(default_factory=dict)
        stats: Dict[str, Any] = Field(default_factory=dict)
        elapsed_seconds: float = 0.0

    class SliceInfo(BaseModel):
        """Information about a single informative slice."""
        index: int = Field(..., description="Z-index of the slice")
        score: float = Field(0.0, description="Informativeness score")
        png_path: Optional[str] = Field(None, description="Path to the raw slice PNG")
        overlay_path: Optional[str] = Field(None, description="Path to the overlay PNG")
        png_url: Optional[str] = Field(None, description="HTTP URL to the raw slice PNG")
        overlay_url: Optional[str] = Field(None, description="HTTP URL to the overlay PNG")

    class ProcessNiftiResponse(BaseModel):
        """POST /api/process response — full pipeline result."""
        status: str
        message: str
        visualization_html: Optional[str] = None
        visualization_url: Optional[str] = Field(
            None, description="HTTP URL to view the 3D HTML in a browser",
        )
        # Backward compat fields
        best_slice_png: Optional[str] = None
        best_slice_overlay_png: Optional[str] = None
        best_slice_index: Optional[int] = None
        # New fields
        best_slices: List[SliceInfo] = Field(
            default_factory=list,
            description="Top-K most informative slices with scores and images",
        )
        top_k: int = 3
        scoring_mode: Optional[str] = None
        mask_files: Dict[str, str] = Field(
            default_factory=dict,
            description="Organ mask files {organ_name: file_path}",
        )
        structural_report: Optional[str] = None
        structural_report_path: Optional[str] = None
        elapsed_seconds: float = 0.0
        case_name: Optional[str] = None

    class JobStatusResponse(BaseModel):
        """GET /api/status/{job_id} response."""
        job_id: str
        status: str  # "running" | "completed" | "error"
        created_at: float
        completed_at: Optional[float] = None
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        progress: Optional[Dict[str, Any]] = Field(
            None, description="Progress info: {step, percentage, message}"
        )

    # ── Skills API models ────────────────────────────────────────

    class ProcessLiteRequest(BaseModel):
        """POST /api/process-lite request body — segmentation only."""
        input: str = Field(..., description="Path to DICOM directory or .nii.gz CT file")
        case_name: Optional[str] = Field(None, description="Case name")
        device: str = Field("auto", description="CUDA device ('auto'=auto-select, 'cuda:0'='cuda:1'=specific)")
        force: bool = Field(False, description="Force DICOM re-conversion")
        demo_mode: Optional[bool] = Field(None, description="Demo mode: use GT masks if available and skip model inference")
        seg_backend: Optional[str] = Field(
            None,
            description="Segmentation backend: vista3d (default) or totalsegmentator",
        )

    class ProcessLiteResponse(BaseModel):
        """POST /api/process-lite response — segmentation results."""
        status: str
        message: str
        case_id: Optional[str] = None
        ct_nifti_path: Optional[str] = None
        mask_dir: Optional[str] = None
        output_dir: Optional[str] = None
        mask_files: Dict[str, str] = Field(
            default_factory=dict,
            description="Organ mask files {organ_name: file_path}",
        )
        elapsed_seconds: float = 0.0
        case_reused: bool = False
        # Backward-compatible spelling used by the original reuse proposal.
        reused: bool = False

    class SkillRunRequest(BaseModel):
        """POST /api/skills/run request body."""
        skill_name: str = Field(..., description="Skill name (from GET /api/skills/list)")
        case_id: str = Field(..., description="Case ID (returned by /api/process-lite)")
        params: Dict[str, Any] = Field(
            default_factory=dict,
            description="Parameters for the skill, as defined in its schema",
        )

    class SkillRunResponse(BaseModel):
        """POST /api/skills/run response."""
        status: str
        result: Dict[str, Any] = Field(default_factory=dict)
        execution_time_ms: Optional[float] = None
        message: Optional[str] = None
        error_code: Optional[str] = None
        retryable: Optional[bool] = None

    class SkillRegisterRequest(BaseModel):
        """POST /api/skills/register request body — 从本地目录注册。"""
        skill_dir: str = Field(..., description="Skill 目录路径（必须含 skill.yaml + main.py）")

    class SkillRegisterResponse(BaseModel):
        """POST /api/skills/register response."""
        status: str
        name: Optional[str] = None
        version: Optional[str] = None
        message: Optional[str] = None

    class PlanResectionRequest(BaseModel):
        """POST /api/plan-resection request body."""
        case_dir: str = Field(..., description="掩码目录路径")
        output_dir: str = Field(..., description="输出目录路径")
        case_name: str = Field("", description="病例名称")
        tumor_margin_mm: float = Field(5.0, description="肿瘤切缘（mm）")

    class SaveResectionPlaneRequest(BaseModel):
        """POST /api/resection-plane/save request body."""
        output_dir: str = Field(..., description="3D JSON 所在的病例输出目录")
        json_file: str = Field(..., description="3D JSON 文件名")
        plane_index: int = Field(..., ge=0, description="当前候选剖面的数组下标")
        control_points_3d: List[List[List[float]]] = Field(
            ..., description="编辑后的 4x4 三维控制点（网页坐标系）"
        )
        candidate_name: str = Field("", description="候选剖面名称")
        source: str = Field("three_d_editor", description="保存来源")

    class InvalidateResectionPlaneRequest(BaseModel):
        """POST /api/resection-plane/invalidate request body."""
        output_dir: str = Field(..., description="3D JSON 所在的病例输出目录")
        json_file: str = Field(..., description="3D JSON 文件名")
        plane_index: int = Field(..., ge=0, description="当前候选剖面的数组下标")

    class RestoreResectionPlaneRequest(InvalidateResectionPlaneRequest):
        """POST /api/resection-plane/restore request body."""
        original_control_points_3d: List[List[List[float]]] = Field(
            ..., description="页面加载时保留的原始 4x4 三维控制点（网页坐标系）"
        )


# ======================================================================
# [Section 16]  FastAPI app + endpoints
# ======================================================================

if _HAS_FASTAPI:
    OUTPUT_DIR = _DEFAULT_OUTPUT_ROOT
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Pipeline outputs go here — also served as static files for browser access
    PROCESS_OUTPUT_DIR = _DEFAULT_OUTPUT_ROOT
    PROCESS_OUTPUT_DIR.mkdir(exist_ok=True)

    _jobs: Dict[str, Dict[str, Any]] = {}
    _MAX_JOBS = 500          # max completed/failed jobs to retain
    _JOB_TTL_SECONDS = 3600  # expire jobs older than 1 hour

    def _cleanup_old_jobs():
        """Remove expired & excess jobs from _jobs to prevent memory leak."""
        now = time.time()
        # Remove by TTL
        expired = [jid for jid, j in list(_jobs.items())
                   if j.get("status") in ("completed", "error")
                   and (now - j.get("completed_at", now)) > _JOB_TTL_SECONDS]
        for jid in expired:
            del _jobs[jid]
        # Enforce max count
        running = {jid: j for jid, j in _jobs.items() if j.get("status") == "running"}
        done = {jid: j for jid, j in _jobs.items() if jid not in running}
        if len(done) > _MAX_JOBS:
            # Keep the newest completed/failed jobs, oldest first
            to_del = sorted(done, key=lambda jid: done[jid].get("completed_at", 0))[:len(done) - _MAX_JOBS]
            for jid in to_del:
                del _jobs[jid]

    app = FastAPI(
        title="VoxelSage API",
        description="将 CT 分割结果（NIfTI 掩码）渲染为交互式 3D HTML 的可视化服务。",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
    app.mount("/process-output", StaticFiles(directory=str(PROCESS_OUTPUT_DIR)), name="process-output")

    # ======================================================================
    # Middleware: log every request + response (compact one-liner)
    # ======================================================================
    from starlette.requests import Request as _Request

    @app.middleware("http")
    async def _log_request_response(request: _Request, call_next):
        """Log incoming request params and outgoing response for every endpoint."""
        method = request.method
        path = request.url.path
        qs = str(request.url.query)

        # Build request line
        req_line = f"→ {method} {path}" + (f"?{qs}" if qs else "")

        # Capture body for POST/PUT
        body_str = ""
        if method in ("POST", "PUT", "PATCH"):
            try:
                raw = await request.body()
                txt = raw.decode("utf-8", errors="replace")
                if len(txt) > 2000:
                    txt = txt[:2000] + f" … [truncated {len(txt)} bytes]"
                body_str = txt
            except Exception:
                body_str = "<unreadable>"

        if body_str:
            # Compact: inline the body after the arrow line
            _log(f"{req_line} — {body_str}")
        else:
            _log(req_line)

        t0 = time.time()

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = time.time() - t0
            _log(f"← {method} {path} — EXCEPTION: {exc} ({elapsed:.3f}s)")
            raise

        elapsed = time.time() - t0

        # Capture response body
        resp_preview = ""
        if hasattr(response, "body"):
            raw = response.body
            txt = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            if len(txt) > 3000:
                txt = txt[:3000] + f" … [truncated {len(txt)} bytes]"
            resp_preview = txt

        _log(f"← {method} {path} → {response.status_code} ({elapsed:.3f}s)"
             + (f" — {resp_preview[:200]}" if resp_preview else ""))

        return response

    def _make_visualization_url(fs_path: Optional[str]) -> Optional[str]:
        """Convert a process_output filesystem path to an HTTP URL."""
        if not fs_path:
            return None
        try:
            rel = str(Path(fs_path).resolve().relative_to(PROCESS_OUTPUT_DIR.resolve()))
            return f"{PUBLIC_BASE_URL}/process-output/{rel}"
        except ValueError:
            return None

    @app.get("/")
    def root():
        """API 根路径，返回接口概览。"""
        return {
            "name": "3D Medical Visualization API",
            "version": "1.0.0",
            "endpoints": {
                "POST /api/visualize": "从分割掩码生成 3D 可视化",
                "POST /api/process": "从 .nii.gz 全流程（分割 + 3D + Top-K 切片 + 报告）",
                "GET  /api/organs": "列出支持的器官",
                "GET  /api/status/{job_id}": "查询后台任务状态",
                "GET  /output/{filename}": "获取生成的 HTML 文件（静态）",
                "GET  /process-output/{path}": "获取全流程管线产物（3D HTML / 切片 PNG / 报告）",
                "GET  /health": "健康检查",
            },
            "output_directory": str(OUTPUT_DIR),
        }

    @app.get("/debug/threads")
    async def debug_threads():
        """返回所有 Python 线程的堆栈（诊断用，无权限限制）。"""
        import sys, traceback
        frames = {}
        for thread_id, frame in sys._current_frames().items():
            frames[str(thread_id)] = traceback.format_stack(frame)
        return {
            "status": "ok",
            "thread_count": len(frames),
            "threads": frames,
        }

    @app.get("/health")
    async def health():
        """健康检查。"""
        gpu_summary = _gpu_manager.summary()
        return {
            "status": "ok",
            "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": str(OUTPUT_DIR),
            "output_dir_exists": OUTPUT_DIR.exists(),
            "output_dir_writable": os.access(str(OUTPUT_DIR), os.W_OK) if OUTPUT_DIR.exists() else False,
            "gpu": {
                "detected": gpu_summary["detected"],
                "in_use": gpu_summary["in_use"],
                "devices": gpu_summary["devices"],
                "count": len(gpu_summary["gpus"]),
            },
        }

    @app.get("/api/organs")
    def list_organs():
        """列出所有可渲染的器官及其颜色配置。"""
        organs = list_available_organs()
        return {
            "status": "ok",
            "organs": organs,
            "count": len(organs),
        }

    @app.post("/api/visualize", response_model=VisualizeResponse)
    async def visualize(req: VisualizeRequest):
        """生成 3D 可视化。"""
        if req.case_dir:
            case_dir = req.case_dir
        elif req.image_id:
            try:
                data_root = req.data_root or os.environ.get(
                    "DATA_ROOT", str(_PROJECT_ROOT / "data")
                )
                seg_backend = req.seg_backend or "VISTA3D"
                case_dir = resolve_case_dir(req.image_id, data_root, seg_backend)
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'case_dir' or 'image_id'",
            )

        if not os.path.isdir(case_dir):
            raise HTTPException(status_code=404, detail=f"Case directory not found: {case_dir}")

        if req.async_mode:
            job_id = str(uuid.uuid4())[:8]
            _cleanup_old_jobs()
            _jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "created_at": time.time(),
                "completed_at": None,
                "result": None,
                "error": None,
                "progress": None,
            }
            thread = Thread(
                target=_run_visualization_bg,
                args=(job_id, case_dir, req),
                daemon=True,
            )
            thread.start()
            return VisualizeResponse(
                status="accepted",
                message=f"Task accepted. Check status at GET /api/status/{job_id}",
            )

        try:
            result = generate_visualization(
                case_dir=case_dir,
                output_dir=str(OUTPUT_DIR),
                output_filename=req.output_filename,
                step_size=req.step_size,
                downsample_factor=req.downsample,
                isotropic_resample=req.isotropic_resample,
                gaussian_sigma=req.gaussian_sigma,
                smooth=req.smooth,
                prob_threshold=req.prob_threshold,
                skip_empty=True,
                title=req.title,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return VisualizeResponse(
            status="ok",
            message=result["message"],
            file_path=result["file_path"],
            filename=result["filename"],
            url=result["url"],
            organs=result["organs"],
            segmentation_results=result.get("segmentation_results", {}),
            stats=result.get("stats", {}),
            elapsed_seconds=result["elapsed_seconds"],
        )

    @app.post("/api/process", response_model=ProcessNiftiResponse)
    async def process_nifti(req: ProcessNiftiRequest):
        """全流程管线：输入 → 可选分割后端 → 3D/切片/结构报告。"""
        # Backward compat: accept either `input` or `nifti_path`
        input_path = req.input or req.nifti_path
        if not input_path:
            raise HTTPException(
                status_code=400,
                detail="Either 'input' or 'nifti_path' is required",
            )

        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"Input not found: {input_path}")

        if req.async_mode:
            job_id = str(uuid.uuid4())[:8]
            _cleanup_old_jobs()
            _jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "created_at": time.time(),
                "completed_at": None,
                "result": None,
                "error": None,
                "progress": None,
            }
            thread = Thread(
                target=_run_process_bg,
                args=(job_id, req),
                daemon=True,
            )
            thread.start()
            return ProcessNiftiResponse(
                status="accepted",
                message=f"Task accepted. Check status at GET /api/status/{job_id}",
            )

        try:
            result = process_nifti_file(
                input_path=input_path,
                case_name=req.case_name,
                device=req.device,
                step_size=req.step_size,
                downsample_factor=req.downsample,
                verbose=False,
                top_k=req.top_k,
                scoring_mode=req.scoring_mode,
                force=req.force,
                skip_viz=req.skip_viz,
                demo_mode=req.demo_mode,
                seg_backend=req.seg_backend,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        # Construct HTTP URLs from filesystem paths for browser access
        visualization_url = _make_visualization_url(result.get("visualization_html"))
        best_slices = []
        for s in result.get("best_slices", []):
            slc = SliceInfo(**s)
            if s.get("png_path"):
                if u := _make_visualization_url(s["png_path"]):
                    slc.png_url = u
            if s.get("overlay_path"):
                if u := _make_visualization_url(s["overlay_path"]):
                    slc.overlay_url = u
            best_slices.append(slc)

        # ---- 构造响应体 ----
        resp = ProcessNiftiResponse(
            status="ok",
            message=result["message"],
            visualization_html=result.get("visualization_html"),
            visualization_url=visualization_url,
            best_slice_png=result.get("best_slice_png"),
            best_slice_overlay_png=result.get("best_slice_overlay_png"),
            best_slice_index=result.get("best_slice_index"),
            best_slices=best_slices,
            top_k=result.get("top_k", 3),
            scoring_mode=result.get("scoring_mode"),
            mask_files=result.get("mask_files", {}),
            structural_report=result.get("structural_report"),
            structural_report_path=result.get("structural_report_path"),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
            case_name=result.get("case_name"),
        )

        # ---- 把完整返回体写日志（JSON 序列化），方便排查 ----
        try:
            _log(f"[api/process response] {resp.model_dump_json(indent=None, exclude={'structural_report'})}")
        except Exception:
            _log(f"[api/process response] (serialization failed) status=ok case={result.get('case_name')}")

        return resp

    @app.get("/api/status/{job_id}", response_model=JobStatusResponse)
    def get_job_status(job_id: str):
        """查询后台异步任务的状态。"""
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return JobStatusResponse(**job)

    @app.get("/api/jobs")
    def list_jobs():
        """列出所有后台任务。"""
        return {
            "status": "ok",
            "jobs": [
                {
                    "job_id": j["job_id"],
                    "status": j["status"],
                    "created_at": j["created_at"],
                    "completed_at": j["completed_at"],
                }
                for j in _jobs.values()
            ],
        }

    # ── Skills API endpoints ─────────────────────────────────

    @app.get("/api/skills/list")
    def list_skills():
        """返回所有已注册 Skills 的元数据（供 Port A LLM 构建 function calling）。"""
        if _skill_engine is None:
            raise HTTPException(status_code=503, detail="Skills engine not initialized")
        return {
            "status": "ok",
            "skills": _skill_engine.list_skills(),
            "tools": _skill_engine.list_skills_openai_tools(),
        }

    @app.post("/api/skills/run", response_model=SkillRunResponse)
    def run_skill(req: SkillRunRequest):
        """执行指定 Skill（Port A LLM 按需调用）。"""
        if _skill_engine is None:
            raise HTTPException(status_code=503, detail="Skills engine not initialized")

        # 查找病例上下文（从 _jobs 或从磁盘重建）
        # 这里从请求参数重建上下文，由 process-lite 提前准备好数据
        from pathlib import Path
        output_root = Path(__file__).resolve().parent / "output"
        case_output = output_root / req.case_id
        if not case_output.is_dir() and req.case_id.startswith("CRLM-CT-"):
            short_case = output_root / req.case_id.replace("CRLM-CT-", "CRLM-", 1)
            if short_case.is_dir():
                case_output = short_case
        ct_nifti = case_output / "ct.nii.gz"
        mask_dir = case_output / "masks"
        if not ct_nifti.exists() and mask_dir.is_dir():
            # Some manually generated visualization cases contain masks but
            # no copied CT file. Skills only need a valid NIfTI context path.
            mask_candidates = sorted(mask_dir.glob("*.nii.gz"))
            if mask_candidates:
                ct_nifti = mask_candidates[0]

        if not ct_nifti.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Case '{req.case_id}' not found. Run /api/process-lite first.",
            )

        ctx = SkillContext(
            case_id=req.case_id,
            ct_nifti_path=str(ct_nifti),
            mask_dir=str(mask_dir),
            output_dir=str(case_output),
            params=req.params,
        )

        start = time.time()
        try:
            result = _skill_engine.run(
                skill_name=req.skill_name,
                ctx=ctx,
            )
            elapsed = (time.time() - start) * 1000  # ms
            resp = SkillRunResponse(
                status="ok",
                result=result,
                execution_time_ms=round(elapsed, 1),
            )
            _log(f"[skill:{req.skill_name}] OK ({elapsed:.0f}ms) → {json.dumps(result, ensure_ascii=False, default=str)[:2000]}")
            return resp
        except SkillNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except SkillExecutionError as e:
            elapsed = (time.time() - start) * 1000
            resp = SkillRunResponse(
                status="error",
                message=f"Skill '{req.skill_name}' execution failed: {e}",
                error_code="SKILL_EXECUTION_ERROR",
                retryable=False,
                execution_time_ms=round(elapsed, 1),
            )
            _log(f"[skill:{req.skill_name}] ERROR ({elapsed:.0f}ms) → {e}")
            return resp
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            resp = SkillRunResponse(
                status="error",
                message=str(e),
                error_code="UNEXPECTED_ERROR",
                retryable=True,
                execution_time_ms=round(elapsed, 1),
            )
            _log(f"[skill:{req.skill_name}] UNEXPECTED ({elapsed:.0f}ms) → {e}")
            return resp

    @app.post("/api/skills/register", response_model=SkillRegisterResponse)
    def register_skill(req: SkillRegisterRequest):
        """从本地目录注册一个用户 Skill。

        目录必须包含 skill.yaml + main.py。
        """
        if _skill_engine is None:
            raise HTTPException(status_code=503, detail="Skills engine not initialized")

        skill_dir = str(Path(req.skill_dir).resolve())
        if not os.path.isdir(skill_dir):
            raise HTTPException(status_code=400, detail=f"Directory not found: {skill_dir}")

        try:
            meta = _skill_engine.register_user_skill(skill_dir)
            return SkillRegisterResponse(
                status="ok",
                name=meta.name,
                version=meta.version,
                message=f"Skill '{meta.name}' v{meta.version} registered",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

    @app.delete("/api/skills/{name}")
    def unregister_skill(name: str):
        """注销指定 Skill。"""
        if _skill_engine is None:
            raise HTTPException(status_code=503, detail="Skills engine not initialized")
        try:
            _skill_engine.unregister_skill(name)
            return {"status": "ok", "message": f"Skill '{name}' unregistered"}
        except SkillNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/plan-resection")
    def api_plan_resection(req: PlanResectionRequest):
        """3D 网页按需计算肝脏手术最优切除剖面。

        从 3D HTML 页面的 toggle 开关调用，执行 VoxelSage 引擎
        计算 Bézier 切除剖面并追加到已有 3D JSON 文件中。
        """
        import glob as _glob
        from skills.models import SkillContext
        from skills.builtin.plan_resection.main import run as _run_plan_resection

        case_dir = req.case_dir
        output_dir = req.output_dir
        case_name = req.case_name
        margin_mm = req.tumor_margin_mm

        if not case_dir or not output_dir:
            return {"status": "error", "message": "缺少 case_dir 或 output_dir 参数"}

        # 使用任意一个掩码文件作为 CT NIfTI（所有掩码共享同一 affine）
        mask_files = sorted(_glob.glob(os.path.join(case_dir, "*.nii.gz")))
        if not mask_files:
            return {"status": "error", "message": f"未找到掩码文件: {case_dir}"}
        ct_nifti_path = mask_files[0]

        ctx = SkillContext(
            case_id=case_name or Path(case_dir).name,
            ct_nifti_path=ct_nifti_path,
            mask_dir=case_dir,
            output_dir=output_dir,
            params={"tumor_margin_mm": margin_mm, "case_name": case_name},
        )

        try:
            _log(f"[plan-resection] Computing resection plane for {case_name}...")
            result = _run_plan_resection(ctx)
            _log(f"[plan-resection] Done: margin_min={result.get('margin_min_mm', 'N/A')}mm, "
                 f"success={result.get('margin_success', False)}")
            return {"status": "ok", **result}
        except Exception as e:
            _log(f"[plan-resection] Error: {e}")
            return {"status": "error", "message": str(e)}

    @app.post("/api/resection-plane/save")
    def api_save_resection_plane(req: SaveResectionPlaneRequest):
        """持久化三维网页中用户最终确认的剖面。

        路径规划器只接受带有 user_saved=true 的剖面，避免误用尚未确认的
        候选或仅在网页中临时拖动过的控制点。
        """
        import datetime as _datetime
        import json as _json

        output_dir = Path(req.output_dir).resolve()
        json_name = Path(req.json_file).name
        json_path = output_dir / json_name
        if not output_dir.is_dir() or not json_path.is_file() or json_path.suffix != ".json":
            raise HTTPException(status_code=400, detail="无效的 3D JSON 路径")
        if len(req.control_points_3d) != 4 or any(
            len(row) != 4 or any(len(point) != 3 for point in row)
            for row in req.control_points_3d
        ):
            raise HTTPException(status_code=400, detail="控制点必须是 4x4x3")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            planes = data.get("resection_planes", [])
            if req.plane_index >= len(planes):
                raise HTTPException(status_code=400, detail="剖面下标超出范围")
            now = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
            plane = planes[req.plane_index]
            if "original_control_points_3d" not in plane and plane.get("control_points_3d"):
                # Preserve the pre-edit baseline for legacy JSON files before
                # the first manual save overwrites their active control points.
                plane["original_control_points_3d"] = plane["control_points_3d"]
            plane["control_points_3d"] = req.control_points_3d
            plane["user_saved"] = True
            plane["saved_at"] = now
            plane["save_source"] = req.source
            plane["saved_candidate_name"] = req.candidate_name or plane.get("candidate_name", "")
            data["resection_planes"] = planes
            data["resection_sequence_available"] = False
            data["selected_resection_plane_index"] = req.plane_index
            data["selected_resection_plane_source"] = req.source
            data["selected_resection_plane_saved_at"] = now
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            return {
                "status": "ok",
                "json_path": str(json_path),
                "plane_index": req.plane_index,
                "saved_at": now,
                "user_saved": True,
            }
        except HTTPException:
            raise
        except Exception as e:
            _log(f"[resection-plane/save] Error: {e}")
            raise HTTPException(status_code=500, detail=f"保存剖面失败: {e}")

    @app.post("/api/resection-plane/invalidate")
    def api_invalidate_resection_plane(req: InvalidateResectionPlaneRequest):
        """用户开始修改控制点后，使此前保存的剖面失效。"""
        import json as _json

        output_dir = Path(req.output_dir).resolve()
        json_path = output_dir / Path(req.json_file).name
        if not output_dir.is_dir() or not json_path.is_file() or json_path.suffix != ".json":
            raise HTTPException(status_code=400, detail="无效的 3D JSON 路径")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            planes = data.get("resection_planes", [])
            if req.plane_index >= len(planes):
                raise HTTPException(status_code=400, detail="剖面下标超出范围")
            plane = planes[req.plane_index]
            plane["user_saved"] = False
            plane["unsaved_changes"] = True
            plane.pop("saved_at", None)
            data["resection_planes"] = planes
            data["resection_sequence_available"] = False
            if data.get("selected_resection_plane_index") == req.plane_index:
                data["selected_resection_plane_source"] = "unsaved_editor_changes"
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            return {"status": "ok", "plane_index": req.plane_index, "user_saved": False}
        except HTTPException:
            raise
        except Exception as e:
            _log(f"[resection-plane/invalidate] Error: {e}")
            raise HTTPException(status_code=500, detail=f"剖面失效处理失败: {e}")

    @app.post("/api/resection-plane/restore")
    def api_restore_resection_plane(req: RestoreResectionPlaneRequest):
        """放弃人工控制点修改并持久化恢复最初的 Bézier 剖面。"""
        import json as _json

        output_dir = Path(req.output_dir).resolve()
        json_path = output_dir / Path(req.json_file).name
        if not output_dir.is_dir() or not json_path.is_file() or json_path.suffix != ".json":
            raise HTTPException(status_code=400, detail="无效的 3D JSON 路径")
        if len(req.original_control_points_3d) != 4 or any(
            len(row) != 4 or any(len(point) != 3 for point in row)
            for row in req.original_control_points_3d
        ):
            raise HTTPException(status_code=400, detail="原始控制点必须是 4x4x3")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            planes = data.get("resection_planes", [])
            if req.plane_index >= len(planes):
                raise HTTPException(status_code=400, detail="剖面下标超出范围")
            plane = planes[req.plane_index]
            # New outputs carry the optimizer baseline in JSON. For older
            # outputs, the page-load snapshot is the best available baseline.
            original = plane.get("original_control_points_3d")
            if original is None:
                original = req.original_control_points_3d
                plane["original_control_points_3d"] = original
            plane["control_points_3d"] = original
            plane["user_saved"] = False
            plane["unsaved_changes"] = False
            for key in ("saved_at", "save_source", "saved_candidate_name"):
                plane.pop(key, None)
            data["resection_planes"] = planes
            data["resection_sequence_available"] = False
            if data.get("selected_resection_plane_index") == req.plane_index:
                data["selected_resection_plane_source"] = "restored_original"
                data.pop("selected_resection_plane_saved_at", None)
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            return {
                "status": "ok",
                "plane_index": req.plane_index,
                "user_saved": False,
                "control_points_3d": original,
            }
        except HTTPException:
            raise
        except Exception as e:
            _log(f"[resection-plane/restore] Error: {e}")
            raise HTTPException(status_code=500, detail=f"复原剖面失败: {e}")

    @app.post("/api/process-lite", response_model=ProcessLiteResponse)
    def process_lite(req: ProcessLiteRequest):
        """轻量管线：仅分割 + 后处理，不执行 3D/切片/分析。

        返回 case_id + 文件路径，供 Port A 后续逐个调用 Skills。
        """
        input_path = str(Path(req.input).resolve())
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"Input not found: {input_path}")
        try:
            effective_backend = _normalize_segmentation_backend(req.seg_backend)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        start = time.time()
        case_name = req.case_name

        try:
            # Resolve the case before taking the lock.  All writes for the case,
            # including CT-link and GT preparation, must happen under the lock.
            if os.path.isdir(input_path):
                path_str = input_path.replace("\\", "/")
                case_name = case_name or _resolve_crlm_case_id(path_str)
            else:
                case_name = case_name or _resolve_crlm_case_id(input_path.replace("\\", "/"))
                if case_name.endswith(".nii"):
                    case_name = case_name[:-4]

            if not case_name:
                raise HTTPException(status_code=400, detail="Unable to resolve case_name")
            if Path(case_name).name != case_name or case_name in (".", ".."):
                raise HTTPException(status_code=400, detail=f"Invalid case_name: {case_name}")

            output_dir = str(_PROJECT_ROOT / "output" / case_name)
            try:
                with _case_processing_lock(case_name, output_dir):
                    _log(f"[process-lite] Acquired case lock for {case_name}")

                    mask_dir = os.path.join(output_dir, "masks")
                    if _is_case_complete(mask_dir):
                        recorded_backend = _read_segmentation_backend(mask_dir)
                        if recorded_backend != effective_backend:
                            timestamp = _dt.datetime.now().strftime("%y%m%d%H%M")
                            new_case_id = f"{case_name}_{effective_backend}_{timestamp}"
                            collision_index = 1
                            while (_PROJECT_ROOT / "output" / new_case_id).exists():
                                new_case_id = (
                                    f"{case_name}_{effective_backend}_{timestamp}_"
                                    f"{collision_index:02d}"
                                )
                                collision_index += 1
                            _log(
                                f"[process-lite] Case {case_name} uses "
                                f"{recorded_backend}; using new case_id {new_case_id} "
                                f"for {effective_backend}"
                            )
                            req.case_name = new_case_id
                            return process_lite(req)

                        # Keep the case CT link synchronized for NIfTI uploads.
                        # DICOM inputs need conversion when the Skills CT link
                        # is absent, or explicitly when ``force`` is requested.
                        ct_nifti_path = os.path.join(output_dir, "ct.nii.gz")
                        if os.path.isdir(input_path):
                            if req.force or not os.path.isfile(ct_nifti_path):
                                prepared_ct_path, _ = _crlm_prepare_input(
                                    input_path, output_dir, force=req.force
                                )
                                ct_nifti_path = _ensure_ct_symlink(
                                    prepared_ct_path, output_dir
                                )
                        elif not os.path.isfile(ct_nifti_path):
                            ct_nifti_path = _ensure_ct_symlink(input_path, output_dir)
                        elif not os.path.isdir(input_path):
                            # A case ID identifies one CT + segmentation pair.
                            # Reusing masks with a different upload would make
                            # all downstream measurements internally invalid.
                            if not _ct_files_match(ct_nifti_path, input_path):
                                # Preserve the existing case and allocate a new,
                                # human-readable ID for this distinct CT. Holding the
                                # original case lock while recursing prevents concurrent
                                # requests from assigning competing replacement IDs.
                                timestamp = _dt.datetime.now().strftime("%y%m%d%H%M")
                                new_case_id = f"{case_name}_{timestamp}"
                                collision_index = 1
                                while (_PROJECT_ROOT / "output" / new_case_id).exists():
                                    new_case_id = (
                                        f"{case_name}_{timestamp}_{collision_index:02d}"
                                    )
                                    collision_index += 1
                                _log(
                                    f"[process-lite] CT differs from case {case_name}; "
                                    f"using new case_id {new_case_id}"
                                )
                                req.case_name = new_case_id
                                return process_lite(req)

                        mask_files = _list_existing_mask_files(mask_dir)
                        elapsed = time.time() - start
                        _log(
                            f"[process-lite] Case {case_name} already processed, "
                            f"reusing {len(mask_files)} masks..."
                        )
                        return ProcessLiteResponse(
                            status="ok",
                            message=f"Reused existing segmentation in {elapsed:.1f}s",
                            case_id=case_name,
                            ct_nifti_path=ct_nifti_path,
                            mask_dir=mask_dir,
                            output_dir=output_dir,
                            mask_files=mask_files,
                            elapsed_seconds=round(elapsed, 1),
                            case_reused=True,
                            reused=True,
                        )

                    if os.path.isdir(input_path):
                        # DICOM → NIfTI conversion
                        _log(f"[process-lite] Preparing DICOM input for {case_name}...")
                        ct_nifti_path, has_seg_gt = _crlm_prepare_input(
                            input_path, output_dir, force=req.force
                        )
                        if not os.path.isfile(ct_nifti_path):
                            raise HTTPException(
                                status_code=500,
                                detail=f"CRLM prep failed: no CT at {ct_nifti_path}",
                            )
                        # Skills resolve CT from ``output/<case>/ct.nii.gz``.
                        # Keep that link available for DICOM as well as NIfTI.
                        _ensure_ct_symlink(ct_nifti_path, output_dir)
                    else:
                        # NIfTI input.  Keep the Skills CT link synchronized
                        # even when a case ID is reused with a new upload path.
                        ct_nifti_path = input_path
                        _ensure_ct_symlink(ct_nifti_path, output_dir)
                        _, has_seg_gt = _crlm_prepare_input(
                            input_path, output_dir, force=req.force
                        )

                    os.makedirs(mask_dir, exist_ok=True)

                    # ---- 演示模式：有 GT 掩码时跳过模型推理 ----
                    _demo_mode = _DEMO_MODE if req.demo_mode is None else req.demo_mode
                    if _demo_mode and has_seg_gt:
                        _log("[process-lite] 🎯 演示模式 — 使用预分割 GT 掩码（跳过模型推理）")
                        _use_gt_masks(output_dir, mask_dir)
                        # 检查是否有实际可用的掩码，没有则回退到选定后端
                        gt_list = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
                        gt_list = [
                            f for f in gt_list
                            if (
                                os.path.basename(f) in _CRLM_GT_MASK_BASENAMES
                                or _CRLM_TUMOR_MASK_RE.fullmatch(os.path.basename(f))
                            )
                        ]
                        if not gt_list:
                            _log(
                                f"       ⚠ GT 掩码为空，自动回退到 "
                                f"{effective_backend} 分割"
                            )
                            _demo_mode = False
                        else:
                            effective_organs = sorted(
                                Path(f).stem.replace(".nii", "") for f in gt_list
                            )
                            _log(f"[process-lite] GT masks: {effective_organs}")

                    if not (_demo_mode and has_seg_gt):
                        _log(
                            f"[process-lite] Running {effective_backend} "
                            f"segmentation for {case_name}..."
                        )
                        _run_segmentation(
                            backend=effective_backend,
                            nifti_path=ct_nifti_path,
                            output_dir=mask_dir,
                            organ_list=_CRLM_DEFAULT_ORGANS,
                            device=req.device,
                        )

                    # CRLM postprocessing（血管重命名 + 肿瘤分裂）
                    _log("[process-lite] CRLM postprocessing...")
                    effective_organs = _crlm_run_postprocessing(mask_dir)
                    _write_segmentation_metadata(mask_dir, effective_backend)

                    mask_files = _list_mask_files(
                        mask_dir, organ_list=effective_organs
                    )

                    elapsed = time.time() - start
                    _log(
                        f"[process-lite] Complete in {elapsed:.1f}s — "
                        f"{len(mask_files)} masks"
                    )

                    return ProcessLiteResponse(
                        status="ok",
                        message=f"Segmentation complete in {elapsed:.1f}s",
                        case_id=case_name,
                        ct_nifti_path=ct_nifti_path,
                        mask_dir=mask_dir,
                        output_dir=output_dir,
                        mask_files=mask_files,
                        elapsed_seconds=round(elapsed, 1),
                        case_reused=False,
                        reused=False,
                    )
            except CaseProcessingBusyError as exc:
                _log(f"[process-lite] BUSY: {exc}")
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        except HTTPException:
            raise
        except Exception as e:
            elapsed = time.time() - start
            _log(f"[process-lite] FAILED after {elapsed:.1f}s: {e}")
            _log(traceback.format_exc())
            return ProcessLiteResponse(
                status="error",
                message=f"Process-lite failed: {e}",
                case_id=case_name,
                elapsed_seconds=round(elapsed, 1),
                case_reused=False,
                reused=False,
            )

    def _run_visualization_bg(job_id: str, case_dir: str, req: VisualizeRequest):
        """Background thread for visualization (timeout: 30 min)."""
        _BG_TIMEOUT_SECONDS = 1800
        try:
            tracker = ProgressTracker(target=_jobs[job_id])
            tracker.update("Starting", 0, "Loading masks")

            def _do_viz():
                return generate_visualization(
                    case_dir=case_dir,
                    output_dir=str(OUTPUT_DIR),
                    output_filename=req.output_filename,
                    step_size=req.step_size,
                    downsample_factor=req.downsample,
                    isotropic_resample=req.isotropic_resample,
                    gaussian_sigma=req.gaussian_sigma,
                    smooth=req.smooth,
                    prob_threshold=req.prob_threshold,
                    skip_empty=True,
                    title=req.title,
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_do_viz)
                result = fut.result(timeout=_BG_TIMEOUT_SECONDS)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = time.time()
            _jobs[job_id]["result"] = result
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["completed_at"] = time.time()
            _jobs[job_id]["error"] = str(e)

    def _run_process_bg(job_id: str, req: ProcessNiftiRequest):
        """Background thread for full pipeline (timeout: 60 min)."""
        _BG_PIPELINE_TIMEOUT = 3600
        try:
            tracker = ProgressTracker(target=_jobs[job_id])
            tracker.update("Starting", 0, "Pipeline initialized")
            input_path = req.input or req.nifti_path

            def _do_pipeline():
                return process_nifti_file(
                    input_path=input_path,
                    case_name=req.case_name,
                    device=req.device,
                    step_size=req.step_size,
                    downsample_factor=req.downsample,
                    verbose=False,
                    progress_tracker=tracker,
                    top_k=req.top_k,
                    scoring_mode=req.scoring_mode,
                    force=req.force,
                    skip_viz=req.skip_viz,
                    demo_mode=req.demo_mode,
                    seg_backend=req.seg_backend,
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_do_pipeline)
                result = fut.result(timeout=_BG_PIPELINE_TIMEOUT)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = time.time()
            _jobs[job_id]["result"] = result
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["completed_at"] = time.time()
            _jobs[job_id]["error"] = str(e)


# ======================================================================
# [Section 17]  CLI entry — server mode
# ======================================================================

def server_main(argv: Optional[List[str]] = None):
    """CLI entry point for starting the FastAPI server."""
    # 启用 faulthandler：进程卡死时发 SIGUSR2 即可 dump 线程堆栈到 stderr
    import faulthandler, signal
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR2, chain=True)

    if not _HAS_FASTAPI:
        print("[ERROR] fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(
        description="VoxelSage API — CT 影像分割、三维可视化与肝癌分析服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to log file (default: server.log in project root)")
    args = parser.parse_args(argv)

    # ---- Logger setup: _log() writes to both console and this file ----
    global _log_file_path
    log_file = args.log_file or str(_PROJECT_ROOT / "server.log")
    _log_file_path = str(Path(log_file).resolve())
    Path(_log_file_path).parent.mkdir(parents=True, exist_ok=True)

    # ---- Uvicorn logging: inject file handler into uvicorn's log config ----
    # uvicorn.run() calls logging.config.dictConfig() internally, which replaces
    # all logger configurations.  To add file output without losing console,
    # modify uvicorn's default LOGGING_CONFIG before passing it to run().
    import uvicorn.config as _uv_cfg
    _log_cfg = _uv_cfg.LOGGING_CONFIG.copy()
    _log_cfg["disable_existing_loggers"] = False

    # File handler — wraps ImmediateFileHandler so every log line flushes to disk
    _log_cfg["handlers"]["logfile"] = {
        "class": f"{__name__}.ImmediateFileHandler",
        "filename": _log_file_path,
        "mode": "a",
        "encoding": "utf-8",
        "formatter": "default",
    }

    # Attach file handler to uvicorn loggers
    _log_cfg["loggers"]["uvicorn"]["handlers"] = ["default", "logfile"]
    _log_cfg["loggers"]["uvicorn.access"]["handlers"] = ["access", "logfile"]

    # ---- Initialize Skills engine ----
    global _skill_engine
    _skill_engine = SkillEngine()
    count = 0  # will be updated below if skills dir exists
    skills_dir = str(_PROJECT_ROOT / "skills")
    if os.path.isdir(skills_dir):
        builtin_dir = os.path.join(skills_dir, "builtin")
        _log("[Skills] Initializing SkillEngine...")
        _skill_engine.register_builtin(builtin_dir)
        count = len(_skill_engine.list_skills())
        _log(f"[Skills] {count} built-in skill(s) registered")
    else:
        _log(f"[Skills] Skills directory not found: {skills_dir}")

    # ---- Initialize segmentation editor routes ----
    if _HAS_SEG_EDITOR:
        try:
            _seg_editor_routes.register_routes(app)
            _log("[SegEditor] Segmentation editor routes registered")
        except Exception as e:
            _log(f"[SegEditor] Failed to register editor routes: {e}")
    else:
        _log("[SegEditor] Segmentation editor not available (import skipped)")

    # ---- Periodic cleanup of expired editor sessions ----
    @app.on_event("startup")
    async def _start_seg_editor_cleanup():
        import asyncio

        def _do_cleanup(editor_sessions):
            """在线程池执行清理，避免阻塞事件循环。"""
            before = editor_sessions.count_active()
            editor_sessions.cleanup_expired()
            after = editor_sessions.count_active()
            return before, after

        async def _cleanup_loop():
            while True:
                await asyncio.sleep(300)  # every 5 minutes
                try:
                    from skills.builtin.segmentation_modification.session_manager import editor_sessions
                    before, after = await asyncio.to_thread(_do_cleanup, editor_sessions)
                    if before != after:
                        _log(f"[SegEditor] Cleaned {before - after} expired session(s), {after} active")
                except Exception:
                    pass

        asyncio.create_task(_cleanup_loop())

    # ---- Banner ----
    sep = "=" * 60
    _log(sep)
    _log("  VoxelSage API Server v2.0.0")
    _log(sep)
    _log(f"  Start:     http://{args.host}:{args.port}")
    _log(f"  API Docs:  http://{args.host}:{args.port}/docs")
    _log(f"  Output:    {OUTPUT_DIR}")
    _log(f"  Segment:   {_normalize_segmentation_backend(None)}")
    _log(f"  Skills:    {count if _skill_engine else 0} registered")
    _log(f"  Log file:  {_log_file_path}")
    # GPU info
    _log("  GPU Status:")
    _print_gpu_info()
    _log(f"  GPUManager: {_gpu_manager.in_use_count()} in use")
    _log(sep)

    uvicorn.run(app, host=args.host, port=args.port, log_config=_log_cfg)


# ======================================================================
# [Section 18]  CLI dispatcher
# ======================================================================

if __name__ == "__main__":
    # Unified CLI dispatcher
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("=" * 60)
        print("3D Medical Visualization & Analysis API")
        print("=" * 60)
        print()
        print("Usage:")
        print("  # CRLM pipeline (DICOM directory or .nii.gz)")
        print("  python API.py pipeline /path/to/CRLM-CT-1001/ [options]")
        print("  python API.py pipeline /path/to/ct.nii.gz [options]")
        print()
        print("  # Batch process all CRLM cases")
        print("  python API.py pipeline --all [options]")
        print()
        print("  # Start HTTP API server")
        print("  python API.py server --port 8765")
        print()
        print("API Server Endpoints:")
        print("  POST /api/process      — Full pipeline (unchanged)")
        print("  POST /api/process-lite — Segmentation only (for Skills workflow)")
        print("  GET  /api/skills/list  — List registered Skills")
        print("  POST /api/skills/run   — Execute a Skill")
        print("  POST /api/skills/register — Register a user Skill")
        print("  DELETE /api/skills/{name} — Unregister a Skill")
        print()
        print("Available functions:")
        print("  process_nifti_file()     — CRLM pipeline")
        print("  generate_visualization() — 3D visualization from masks (Three.js)")
        print("  select_top_slices()      — Top-K informative slices")
        print("  list_available_organs()  — Supported organs")
        print("  estimate_output_size()   — Estimate output file size")
        print()
        print("See README_API.md for details")
        print("=" * 60)
    elif sys.argv[1] == "pipeline":
        pipeline_main(sys.argv[2:])
    elif sys.argv[1] in ("server", "--server"):
        server_main(sys.argv[2:])
    else:
        # Backward compat: treat first positional arg as input path
        pipeline_main(sys.argv[1:])

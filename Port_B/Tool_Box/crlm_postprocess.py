"""
CRLM 后处理工具
================

VISTA3D 输出 → CRLM 分析期望格式的转换：

1. 血管重命名  — hepatic vessel → hepatic, portal vein → portal
2. 肿瘤拆分    — hepatic tumor → tumor_1, tumor_2, ...
"""

from contextlib import contextmanager
import json
import logging
import os
import shutil
import threading
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np

from Tool_Box.vessel_connectivity import optimize_vessel_mask
from Tool_Box.vessel_fragment_filter import filter_cross_class_fragments
from Tool_Box.vessel_optimization_contract import (
    AUDIT_SCHEMA,
    AUDIT_SCHEMA_VERSION,
    MAX_ALLOWED_GAP_MM,
    sha256_file,
    validate_max_gap_mm,
)

logger = logging.getLogger(__name__)

_CASE_LOCKS: dict[str, threading.Lock] = {}
_CASE_LOCKS_GUARD = threading.Lock()


# ============================================================
# 1. 血管重命名
# ============================================================

_VESSEL_RENAME_MAP = {
    "hepatic vessel.nii.gz": "hepatic.nii.gz",
    "portal vein and splenic vein.nii.gz": "portal.nii.gz",
}


def rename_vessel_masks(mask_dir: str) -> None:
    """
    将 VISTA3D 输出的血管 mask 重命名为 CRLM 分析期望的名称。

    - hepatic vessel.nii.gz → hepatic.nii.gz
    - portal vein and splenic vein.nii.gz → portal.nii.gz

    如果源文件不存在则跳过；如果目标已存在则不重复复制。
    """
    mask_dir = Path(mask_dir)
    for src_name, dst_name in _VESSEL_RENAME_MAP.items():
        src = mask_dir / src_name
        dst = mask_dir / dst_name
        if not src.is_file():
            continue
        if dst.exists():
            logger.info(f"  [后处理] {dst_name} 已存在，跳过重命名")
            os.remove(str(src))
            continue
        shutil.copy2(str(src), str(dst))
        os.remove(str(src))
        logger.info(f"  [后处理] {src_name} → {dst_name}")


# ============================================================
# 2. 肿瘤拆分（连通域）
# ============================================================

def split_hepatic_tumor(mask_dir: str, min_voxels: int = 10) -> list[str]:
    """
    将 VISTA3D 输出的单张 hepatic tumor mask 按连通域拆分为独立 tumor_N mask。

    步骤：
    1. 加载 ``hepatic tumor.nii.gz``
    2. ``scipy.ndimage.label`` 标记连通域
    3. 滤除 < ``min_voxels`` 的噪声分量
    4. 按体素数降序保存为 ``tumor_1.nii.gz``, ``tumor_2.nii.gz``, ...
    5. 删除原始的 ``hepatic tumor.nii.gz``

    Args:
        mask_dir: VISTA3D 输出 mask 目录。
        min_voxels: 最小体素数（小于此值的连通域视为噪声忽略）。

    Returns:
        list[str]: 生成的 tumor_N 列表（如 [tumor_1, tumor_2]）。
    """
    from scipy import ndimage

    mask_dir = Path(mask_dir)
    src_path = mask_dir / "hepatic tumor.nii.gz"

    if not src_path.is_file():
        logger.info("  [后处理] hepatic tumor.nii.gz 不存在，跳过拆分")
        return []

    # 加载 mask
    nii = nib.load(str(src_path))
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    # 连通域标记
    binary = (data > 0).astype(np.uint8)
    labeled, num = ndimage.label(binary)

    if num == 0:
        logger.info("  [后处理] hepatic tumor 为空，跳过拆分")
        return []

    # 统计各分量大小
    sizes = np.bincount(labeled.ravel())
    # sizes[0] = 背景，排除
    component_sizes = [(int(sizes[i]), i) for i in range(1, num + 1) if sizes[i] > 0]
    component_sizes.sort(key=lambda x: x[0], reverse=True)

    # 滤除噪声
    valid = [(voxels, label_id) for voxels, label_id in component_sizes
             if voxels >= min_voxels]

    if not valid:
        logger.info(f"  [后处理] 无有效肿瘤分量（均 < {min_voxels} 体素）")
        return []

    saved = []
    for rank, (voxels, label_id) in enumerate(valid, start=1):
        comp_mask = (labeled == label_id).astype(np.uint8)
        out_name = f"tumor_{rank}.nii.gz"
        out_path = mask_dir / out_name
        out_nii = nib.Nifti1Image(comp_mask, affine, header)
        nib.save(out_nii, str(out_path))
        saved.append(f"tumor_{rank}")
        logger.info(f"  [后处理] tumor_{rank}: {voxels} 体素 → {out_name}")

    # 删除原始合并 mask
    os.remove(str(src_path))
    logger.info(f"  [后处理] 删除了原始 hepatic tumor.nii.gz（拆分为 {len(saved)} 个）")

    return saved


def _write_json_atomic(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_name(report_path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        json.loads(temp_path.read_text(encoding="utf-8"))
        os.replace(temp_path, report_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _copy_snapshot(source: Path, backup: Path) -> None:
    shutil.copy2(source, backup)
    if backup.stat().st_size != source.stat().st_size:
        raise OSError(f"incomplete snapshot copy: {source}")


def _vessel_error_report(
    vessel_path: Path,
    output_path: Path,
    other_vessel_path: Path | None,
    exc: Exception,
    *,
    status: str = "error",
) -> dict:
    return {
        "status": status,
        "source_path": str(vessel_path),
        "other_vessel_path": (
            None if other_vessel_path is None else str(other_vessel_path)
        ),
        "output_path": str(output_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


@contextmanager
def _case_publication_lock(mask_dir: Path):
    """Serialize one case's optimization and publication across threads/processes."""
    key = os.path.normcase(str(mask_dir.resolve(strict=False)))
    with _CASE_LOCKS_GUARD:
        thread_lock = _CASE_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        lock_path = mask_dir / ".vessel-optimization.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _geometry_record(image: nib.spatialimages.SpatialImage) -> dict:
    return {
        "shape": list(image.shape),
        "affine": np.asarray(image.affine, dtype=float).tolist(),
        "spacing_mm": np.asarray(
            nib.affines.voxel_sizes(image.affine),
            dtype=float,
        ).tolist(),
        "spatial_units": image.header.get_xyzt_units()[0],
        "raw_dtype": str(np.dtype(image.get_data_dtype())),
        "dtype": "uint8",
    }


def _finalize_vessel_record(
    vessel_name: str,
    vessel_path: Path,
    output_path: Path,
    staged_output_path: Path,
    record: dict,
    generation_id: str,
    working_vessel_path: Path | None = None,
) -> dict:
    """Bind a per-vessel audit record to fully loaded, validated payloads."""
    item = dict(record)
    item.update(
        {
            "generation_id": generation_id,
            "raw_filename": vessel_path.name,
            "optimized_filename": output_path.name,
            "source_path": str(vessel_path),
            "output_path": str(output_path),
        }
    )
    if vessel_path.is_file():
        item["raw_sha256"] = sha256_file(vessel_path)
        try:
            raw_image = nib.load(str(vessel_path))
            raw_data = np.asanyarray(raw_image.dataobj)
            item["geometry"] = _geometry_record(raw_image)
            item["voxel_count_before"] = int(np.count_nonzero(raw_data))
        except Exception as exc:
            item["geometry"] = None
            item.setdefault("warnings", []).append(
                f"raw_audit_unavailable:{type(exc).__name__}"
            )
    else:
        raw_image = None
        raw_data = None
        item["raw_sha256"] = None
        item["geometry"] = None
        item["voxel_count_before"] = None

    if item.get("status") != "optimized":
        item["audit_state"] = "unavailable"
        item["optimized_sha256"] = None
        item.setdefault("voxel_count_after", None)
        item.setdefault("added_voxels", None)
        return item
    if raw_image is None or raw_data is None or not staged_output_path.is_file():
        raise RuntimeError(
            f"{vessel_name} optimizer reported success without auditable payloads"
        )

    working_vessel_path = working_vessel_path or vessel_path
    working_image = nib.load(str(working_vessel_path))
    working_data = np.asanyarray(working_image.dataobj)
    optimized_image = nib.load(str(staged_output_path))
    optimized_data = np.asanyarray(optimized_image.dataobj)
    if (
        optimized_image.shape != raw_image.shape
        or not np.allclose(
            optimized_image.affine,
            raw_image.affine,
            rtol=0.0,
            atol=1e-4,
        )
    ):
        raise ValueError(f"{vessel_name} optimized geometry mismatch")
    if np.dtype(optimized_image.get_data_dtype()) != np.dtype(np.uint8):
        raise ValueError(f"{vessel_name} optimized payload is not uint8")
    if not set(np.unique(optimized_data).tolist()).issubset({0, 1}):
        raise ValueError(f"{vessel_name} optimized payload is not binary")
    raw_binary = raw_data != 0
    optimized_binary = optimized_data != 0
    working_binary = working_data != 0
    if np.any(working_binary & ~optimized_binary):
        raise ValueError(
            f"{vessel_name} optimized payload is not a filtered-mask superset"
        )
    if (
        raw_image.header.get_xyzt_units()[0] != "mm"
        or optimized_image.header.get_xyzt_units()[0] != "mm"
    ):
        raise ValueError(f"{vessel_name} payload spatial units are not mm")
    before = int(np.count_nonzero(raw_binary))
    after = int(np.count_nonzero(optimized_binary))
    added = int(np.count_nonzero(optimized_binary & ~raw_binary))
    removed = int(np.count_nonzero(raw_binary & ~optimized_binary))
    item.update(
        {
            "audit_state": "validated",
            "optimized_sha256": sha256_file(staged_output_path),
            "voxel_count_before": before,
            "voxel_count_after": after,
            "added_voxels": added,
            "removed_voxels": removed,
            "net_voxel_change": after - before,
            "filtered_voxel_count": int(np.count_nonzero(working_binary)),
            "removed_before_reconnection": int(np.count_nonzero(
                raw_binary & ~working_binary
            )),
            "geometry": _geometry_record(raw_image),
        }
    )
    return item


def _prepare_fragment_filtered_masks(
    hepatic_path: Path,
    portal_path: Path,
    tumor_paths: list[Path],
    attempt_dir: Path,
) -> tuple[dict[str, Path], dict]:
    """Create temporary cleaned class masks while preserving both raw files."""
    if not hepatic_path.is_file() or not portal_path.is_file():
        return {"hepatic": hepatic_path, "portal": portal_path}, {
            "status": "skipped_missing_vessel",
            "decisions": [],
        }
    hepatic_image = nib.load(str(hepatic_path))
    portal_image = nib.load(str(portal_path))
    if (
        hepatic_image.shape != portal_image.shape
        or not np.allclose(
            hepatic_image.affine, portal_image.affine, rtol=0.0, atol=1e-4
        )
    ):
        raise ValueError("hepatic and portal geometry mismatch")
    hepatic = np.asanyarray(hepatic_image.dataobj) != 0
    portal = np.asanyarray(portal_image.dataobj) != 0
    tumor = np.zeros_like(hepatic)
    for path in tumor_paths:
        image = nib.load(str(path))
        if (
            image.shape != hepatic_image.shape
            or not np.allclose(
                image.affine, hepatic_image.affine, rtol=0.0, atol=1e-4
            )
        ):
            raise ValueError(f"tumor mask geometry mismatch: {path}")
        tumor |= np.asanyarray(image.dataobj) != 0
    spacing = np.asarray(
        nib.affines.voxel_sizes(hepatic_image.affine), dtype=float
    )
    result = filter_cross_class_fragments(
        hepatic,
        portal,
        spacing,
        tumor_mask=tumor,
    )
    working_paths = {}
    for name, data in (
        ("hepatic", result.hepatic_mask),
        ("portal", result.portal_mask),
    ):
        path = attempt_dir / f"{name}_filtered.nii.gz"
        image = nib.Nifti1Image(
            np.asarray(data, dtype=np.uint8),
            hepatic_image.affine,
            hepatic_image.header.copy(),
        )
        image.set_data_dtype(np.uint8)
        nib.save(image, str(path))
        working_paths[name] = path
    return working_paths, result.audit


def _added_voxel_overlap(
    raw_first: Path,
    optimized_first: Path,
    raw_second: Path,
    optimized_second: Path,
) -> int:
    first_raw = np.asanyarray(nib.load(str(raw_first)).dataobj) != 0
    first_optimized = (
        np.asanyarray(nib.load(str(optimized_first)).dataobj) != 0
    )


def _mask_overlap(first: Path, second: Path) -> int:
    first_mask = np.asanyarray(nib.load(str(first)).dataobj) != 0
    second_mask = np.asanyarray(nib.load(str(second)).dataobj) != 0
    return int(np.count_nonzero(first_mask & second_mask))
    second_raw = np.asanyarray(nib.load(str(raw_second)).dataobj) != 0
    second_optimized = (
        np.asanyarray(nib.load(str(optimized_second)).dataobj) != 0
    )
    return int(
        np.count_nonzero(
            (first_optimized & ~first_raw)
            & (second_optimized & ~second_raw)
        )
    )


def optimize_crlm_vessels(
    mask_dir,
    report_path,
    *,
    max_gap_mm=MAX_ALLOWED_GAP_MM,
) -> dict:
    """Optimize and publish one trusted, serialized CRLM vessel generation."""
    mask_dir = Path(mask_dir)
    report_path = Path(report_path)
    max_gap_mm = validate_max_gap_mm(max_gap_mm)
    with _case_publication_lock(mask_dir):
        return _optimize_crlm_vessels_locked(
            mask_dir,
            report_path,
            max_gap_mm=max_gap_mm,
        )


def _optimize_crlm_vessels_locked(
    mask_dir: Path,
    report_path: Path,
    *,
    max_gap_mm: float,
) -> dict:
    liver_path = mask_dir / "liver.nii.gz"
    hepatic_path = mask_dir / "hepatic.nii.gz"
    portal_path = mask_dir / "portal.nii.gz"
    tumor_paths = sorted(mask_dir.glob("tumor_*.nii.gz"))
    attempt_id = uuid.uuid4().hex
    attempt_dir = mask_dir / f".vessel-optimization-{attempt_id}"
    staged_report_path = report_path.with_name(
        f".{report_path.name}.{attempt_id}.attempt"
    )
    report_backup_path = report_path.with_name(
        f".{report_path.name}.{attempt_id}.backup"
    )
    attempt_dir.mkdir(parents=True)
    preserve_recovery_artifacts = False

    try:
        vessel_reports = {}
        pending_outputs = {}
        raw_hash_snapshot = {
            path: sha256_file(path)
            for path in (hepatic_path, portal_path)
            if path.is_file()
        }
        try:
            working_paths, fragment_filter_report = (
                _prepare_fragment_filtered_masks(
                    hepatic_path,
                    portal_path,
                    tumor_paths,
                    attempt_dir,
                )
            )
        except Exception as exc:
            logger.exception("Cross-class vessel fragment filtering failed")
            working_paths = {
                "hepatic": hepatic_path,
                "portal": portal_path,
            }
            fragment_filter_report = {
                "status": "error_raw_masks_used",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "decisions": [],
            }
        vessel_specs = (
            (
                "hepatic",
                hepatic_path,
                working_paths["hepatic"],
                working_paths["portal"],
            ),
            (
                "portal",
                portal_path,
                working_paths["portal"],
                working_paths["hepatic"],
            ),
        )
        for (
            vessel_name,
            vessel_path,
            working_vessel_path,
            working_other_path,
        ) in vessel_specs:
            output_path = mask_dir / f"{vessel_name}_optimized.nii.gz"
            staged_output_path = attempt_dir / output_path.name
            other_vessel_path = (
                working_other_path if working_other_path.is_file() else None
            )
            if not vessel_path.is_file():
                missing = FileNotFoundError(
                    f"vessel mask is missing: {vessel_path}"
                )
                vessel_reports[vessel_name] = _vessel_error_report(
                    vessel_path,
                    output_path,
                    other_vessel_path,
                    missing,
                    status="skipped_missing_vessel",
                )
                vessel_reports[vessel_name] = _finalize_vessel_record(
                    vessel_name,
                    vessel_path,
                    output_path,
                    staged_output_path,
                    vessel_reports[vessel_name],
                    attempt_id,
                )
                continue

            try:
                vessel_report = dict(
                    optimize_vessel_mask(
                        working_vessel_path,
                        liver_path,
                        tumor_paths,
                        staged_output_path,
                        other_vessel_path=other_vessel_path,
                        max_gap_mm=max_gap_mm,
                        generation_id=attempt_id,
                    )
                )
                vessel_report = _finalize_vessel_record(
                    vessel_name,
                    vessel_path,
                    output_path,
                    staged_output_path,
                    vessel_report,
                    attempt_id,
                    working_vessel_path=working_vessel_path,
                )
                if vessel_report.get("status") == "optimized":
                    pending_outputs[output_path] = staged_output_path
                elif staged_output_path.exists():
                    staged_output_path.unlink()
                vessel_reports[vessel_name] = vessel_report
            except Exception as exc:
                staged_output_path.unlink(missing_ok=True)
                vessel_reports[vessel_name] = _vessel_error_report(
                    vessel_path,
                    output_path,
                    other_vessel_path,
                    exc,
                )
                vessel_reports[vessel_name] = _finalize_vessel_record(
                    vessel_name,
                    vessel_path,
                    output_path,
                    staged_output_path,
                    vessel_reports[vessel_name],
                    attempt_id,
                )

        raw_changed = any(
            not path.is_file() or sha256_file(path) != digest
            for path, digest in raw_hash_snapshot.items()
        )
        if raw_changed:
            for vessel_name, item in vessel_reports.items():
                if item.get("status") == "optimized":
                    item["status"] = "error_raw_changed_during_generation"
                    item["audit_state"] = "unavailable"
                    item["optimized_sha256"] = None
                    item.setdefault("warnings", []).append(
                        "raw_changed_during_generation"
                    )
            for staged_output in pending_outputs.values():
                staged_output.unlink(missing_ok=True)
            pending_outputs.clear()

        cross_vessel_added_overlap_voxels = 0
        final_cross_vessel_overlap_voxels = 0
        hepatic_output = mask_dir / "hepatic_optimized.nii.gz"
        portal_output = mask_dir / "portal_optimized.nii.gz"
        if (
            hepatic_output in pending_outputs
            and portal_output in pending_outputs
        ):
            cross_vessel_added_overlap_voxels = _added_voxel_overlap(
                hepatic_path,
                pending_outputs[hepatic_output],
                portal_path,
                pending_outputs[portal_output],
            )
            final_cross_vessel_overlap_voxels = _mask_overlap(
                pending_outputs[hepatic_output],
                pending_outputs[portal_output],
            )
            if (
                cross_vessel_added_overlap_voxels
                or final_cross_vessel_overlap_voxels
            ):
                for vessel_name, output_path in (
                    ("hepatic", hepatic_output),
                    ("portal", portal_output),
                ):
                    item = vessel_reports[vessel_name]
                    item["status"] = (
                        "rejected_cross_vessel_added_overlap"
                        if cross_vessel_added_overlap_voxels
                        else "rejected_final_cross_vessel_overlap"
                    )
                    item["audit_state"] = "rejected"
                    item["optimized_sha256"] = None
                    item.setdefault("warnings", []).append(
                        "cross_vessel_added_overlap"
                        if cross_vessel_added_overlap_voxels
                        else "final_cross_vessel_overlap"
                    )
                    pending_outputs.pop(output_path).unlink(
                        missing_ok=True
                    )

        has_errors = any(
            item.get("status") != "optimized"
            for item in vessel_reports.values()
        )
        report = {
            "schema": AUDIT_SCHEMA,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "generation_id": attempt_id,
            "status": (
                "completed_with_errors" if has_errors else "completed"
            ),
            "max_gap_mm": max_gap_mm,
            "cross_vessel_added_overlap_voxels": (
                cross_vessel_added_overlap_voxels
            ),
            "final_cross_vessel_overlap_voxels": (
                final_cross_vessel_overlap_voxels
            ),
            "fragment_filter": fragment_filter_report,
            "vessels": vessel_reports,
        }
        _write_json_atomic(report, staged_report_path)

        output_backups = {}
        for output_path in pending_outputs:
            if output_path.is_file():
                backup_path = attempt_dir / f"{output_path.name}.previous"
                _copy_snapshot(output_path, backup_path)
                output_backups[output_path] = backup_path
            else:
                output_backups[output_path] = None

        report_existed = report_path.is_file()
        if report_existed:
            _copy_snapshot(report_path, report_backup_path)

        committed_outputs = []
        try:
            for output_path, staged_output_path in pending_outputs.items():
                os.replace(staged_output_path, output_path)
                committed_outputs.append(output_path)
            os.replace(staged_report_path, report_path)
        except Exception as commit_exc:
            rollback_errors = []
            for output_path in reversed(committed_outputs):
                backup_path = output_backups[output_path]
                try:
                    if backup_path is None:
                        output_path.unlink(missing_ok=True)
                    else:
                        os.replace(backup_path, output_path)
                except Exception as rollback_exc:
                    rollback_errors.append(
                        f"{output_path}: {rollback_exc}"
                    )
            try:
                if report_existed:
                    os.replace(report_backup_path, report_path)
                else:
                    report_path.unlink(missing_ok=True)
            except Exception as rollback_exc:
                rollback_errors.append(f"{report_path}: {rollback_exc}")

            if rollback_errors:
                preserve_recovery_artifacts = True
                raise RuntimeError(
                    "vessel optimization commit failed and automatic "
                    "rollback was incomplete; recovery artifacts retained "
                    f"in {attempt_dir}: {'; '.join(rollback_errors)}"
                ) from commit_exc
            raise

        return report
    finally:
        if not preserve_recovery_artifacts:
            shutil.rmtree(attempt_dir, ignore_errors=True)
            staged_report_path.unlink(missing_ok=True)
            report_backup_path.unlink(missing_ok=True)

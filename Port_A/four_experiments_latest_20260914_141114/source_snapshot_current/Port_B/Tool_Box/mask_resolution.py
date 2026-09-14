from dataclasses import dataclass
import json
from pathlib import Path
import logging

import nibabel as nib
import numpy as np

from Tool_Box.vessel_optimization_contract import (
    AUDIT_SCHEMA,
    AUDIT_SCHEMA_VERSION,
    CASE_MANIFEST_FILENAME,
    OPTIMIZABLE_VESSELS,
    is_supported_audit_schema,
    sha256_file,
    validate_max_gap_mm,
)


logger = logging.getLogger(__name__)
_OPTIMIZABLE = OPTIMIZABLE_VESSELS
_SKIP = {"all", "ct", "mask"}


@dataclass(frozen=True)
class ResolvedMask:
    logical_name: str
    path: str
    variant: str
    warning: str | None = None


class _ManifestValidationError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def _manifest_path(root: Path) -> Path | None:
    candidates = (
        root / CASE_MANIFEST_FILENAME,
        root.parent / CASE_MANIFEST_FILENAME,
    )
    return next((path for path in candidates if path.is_file()), None)


def _validate_manifest_header(manifest) -> tuple[str, dict]:
    if not isinstance(manifest, dict):
        raise _ManifestValidationError("optimized_manifest_invalid")
    if (
        not is_supported_audit_schema(manifest.get("schema"))
        or manifest.get("schema_version") != AUDIT_SCHEMA_VERSION
        or manifest.get("status") not in {"completed", "completed_with_errors"}
    ):
        raise _ManifestValidationError("optimized_manifest_invalid")
    generation_id = manifest.get("generation_id")
    vessels = manifest.get("vessels")
    if not isinstance(generation_id, str) or not generation_id:
        raise _ManifestValidationError("optimized_manifest_invalid")
    if not isinstance(vessels, dict):
        raise _ManifestValidationError("optimized_manifest_invalid")
    try:
        validate_max_gap_mm(manifest.get("max_gap_mm"))
    except ValueError as exc:
        raise _ManifestValidationError("optimized_manifest_invalid") from exc
    return generation_id, vessels


def _validate_optimized_entry(
    root: Path,
    logical_name: str,
    item: dict,
    generation_id: str,
) -> None:
    if (
        not isinstance(item, dict)
        or item.get("generation_id") != generation_id
        or item.get("status") != "optimized"
        or item.get("audit_state") != "validated"
        or item.get("raw_filename") != f"{logical_name}.nii.gz"
        or item.get("optimized_filename")
        != f"{logical_name}_optimized.nii.gz"
    ):
        raise _ManifestValidationError("optimized_manifest_invalid")
    raw = root / item["raw_filename"]
    optimized = root / item["optimized_filename"]
    if not raw.is_file() or not optimized.is_file():
        raise _ManifestValidationError("optimized_payload_missing")
    if item.get("raw_sha256") != sha256_file(raw):
        raise _ManifestValidationError("optimized_raw_hash_mismatch")
    if item.get("optimized_sha256") != sha256_file(optimized):
        raise _ManifestValidationError("optimized_hash_mismatch")
    try:
        raw_image = nib.load(str(raw))
        optimized_image = nib.load(str(optimized))
        raw_data = np.asanyarray(raw_image.dataobj)
        optimized_data = np.asanyarray(optimized_image.dataobj)
    except Exception as exc:
        raise _ManifestValidationError("optimized_unreadable") from exc
    if (
        len(raw_image.shape) != 3
        or optimized_image.shape != raw_image.shape
        or not np.allclose(
            optimized_image.affine,
            raw_image.affine,
            rtol=0.0,
            atol=1e-4,
        )
    ):
        raise _ManifestValidationError("optimized_geometry_mismatch")
    if np.dtype(optimized_image.get_data_dtype()) != np.dtype(np.uint8):
        raise _ManifestValidationError("optimized_non_uint8")
    if not set(np.unique(optimized_data).tolist()).issubset({0, 1}):
        raise _ManifestValidationError("optimized_non_binary")
    if np.any((raw_data != 0) & (optimized_data == 0)):
        raise _ManifestValidationError("optimized_not_raw_superset")
    geometry = item.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("shape") != list(raw_image.shape)
        or geometry.get("spatial_units") != "mm"
        or geometry.get("dtype") != "uint8"
        or raw_image.header.get_xyzt_units()[0] != "mm"
        or optimized_image.header.get_xyzt_units()[0] != "mm"
    ):
        raise _ManifestValidationError("optimized_audit_mismatch")
    try:
        recorded_affine = np.asarray(geometry.get("affine"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise _ManifestValidationError("optimized_audit_mismatch") from exc
    if recorded_affine.shape != (4, 4) or not np.allclose(
        recorded_affine,
        raw_image.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise _ManifestValidationError("optimized_audit_mismatch")
    before = int(np.count_nonzero(raw_data))
    after = int(np.count_nonzero(optimized_data))
    if (
        item.get("voxel_count_before") != before
        or item.get("voxel_count_after") != after
        or item.get("added_voxels") != after - before
    ):
        raise _ManifestValidationError("optimized_audit_mismatch")


def _trusted_manifest(root: Path, requested_name: str) -> tuple[dict, str | None]:
    path = _manifest_path(root)
    if path is None:
        return {}, "optimized_manifest_missing"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        generation_id, vessels = _validate_manifest_header(manifest)
        requested_item = vessels.get(requested_name)
        if not isinstance(requested_item, dict):
            return {}, "optimized_manifest_entry_missing"
        for name, item in sorted(vessels.items()):
            if (
                name in _OPTIMIZABLE
                and isinstance(item, dict)
                and item.get("status") == "optimized"
            ):
                try:
                    _validate_optimized_entry(
                        root,
                        name,
                        item,
                        generation_id,
                    )
                except _ManifestValidationError as exc:
                    if name == requested_name:
                        return {}, exc.reason
                    return {}, "optimized_generation_invalid"
        status = requested_item.get("status")
        if status != "optimized":
            return {}, f"optimized_manifest_status_{status}"
        return requested_item, None
    except _ManifestValidationError as exc:
        return {}, exc.reason
    except Exception:
        return {}, "optimized_manifest_invalid"


def resolve_mask_path(mask_dir, logical_name: str) -> ResolvedMask:
    root = Path(mask_dir)
    raw = root / f"{logical_name}.nii.gz"
    if not raw.is_file():
        raise FileNotFoundError(f"Mask '{logical_name}' not found in {root}")
    optimized = root / f"{logical_name}_optimized.nii.gz"
    if logical_name in _OPTIMIZABLE and optimized.is_file():
        _, warning = _trusted_manifest(root, logical_name)
        if warning is None:
            return ResolvedMask(logical_name, str(optimized), "optimized")
        logger.warning(
            "%s for %s; falling back to raw mask",
            warning,
            logical_name,
        )
        return ResolvedMask(logical_name, str(raw), "raw", warning)
    return ResolvedMask(logical_name, str(raw), "raw")


def resolve_visualization_mask_path(mask_dir, logical_name: str) -> ResolvedMask:
    """Prefer a readable optimized vessel mask for non-quantitative 3D display.

    Quantitative consumers continue to use :func:`resolve_mask_path`, which
    requires the full audit manifest. Visualization deliberately accepts an
    optimized payload without a trusted manifest so an absent or stale audit
    record does not silently force the reconstructed mesh back to the raw mask.
    """
    root = Path(mask_dir)
    raw = root / f"{logical_name}.nii.gz"
    if not raw.is_file():
        raise FileNotFoundError(f"Mask '{logical_name}' not found in {root}")

    optimized = root / f"{logical_name}_optimized.nii.gz"
    if logical_name in _OPTIMIZABLE and optimized.is_file():
        try:
            image = nib.load(str(optimized))
            if len(image.shape) != 3:
                raise ValueError("optimized mask is not three-dimensional")
            return ResolvedMask(
                logical_name,
                str(optimized),
                "optimized",
                "optimized_visualization_unverified",
            )
        except Exception:
            logger.warning(
                "optimized_unreadable for %s; falling back to raw mask",
                logical_name,
            )
            return ResolvedMask(
                logical_name,
                str(raw),
                "raw",
                "optimized_unreadable",
            )
    return ResolvedMask(logical_name, str(raw), "raw")


def _logical_mask_names(root: Path) -> set[str]:
    return {
        _stem(path)
        for path in root.glob("*.nii.gz")
        if _stem(path) not in _SKIP
        and not (
            _stem(path).endswith("_optimized")
            and _stem(path)[:-10] in _OPTIMIZABLE
        )
    }


def scan_logical_masks(mask_dir) -> dict[str, ResolvedMask]:
    root = Path(mask_dir)
    return {
        name: resolve_mask_path(root, name)
        for name in sorted(_logical_mask_names(root))
    }


def scan_visualization_masks(mask_dir) -> dict[str, ResolvedMask]:
    """Scan masks using the relaxed optimized-first visualization policy."""
    root = Path(mask_dir)
    return {
        name: resolve_visualization_mask_path(root, name)
        for name in sorted(_logical_mask_names(root))
    }

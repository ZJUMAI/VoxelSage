"""Runtime checks shared by VISTA3D inference and the deployment doctor."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import re
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np


class NiftiGeometryError(ValueError):
    """Raised when a volume cannot be safely resampled by MONAI."""


class CudaEnvironmentError(RuntimeError):
    """Raised when the selected CUDA runtime or device is unavailable."""


class Vista3DInferenceError(RuntimeError):
    """Raised for a non-retryable VISTA3D preprocessing/inference failure."""


class Vista3DEnvironmentError(RuntimeError):
    """Raised for a known-incompatible VISTA3D Python environment."""


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_nifti_geometry(path: str | Path) -> Dict[str, Any]:
    """Inspect geometry without materializing the voxel array."""
    resolved = Path(path).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    result: Dict[str, Any] = {
        "path": str(resolved),
        "valid": False,
        "errors": errors,
        "warnings": warnings,
    }

    try:
        image = nib.load(str(resolved))
    except Exception as exc:
        errors.append(f"NIfTI cannot be opened: {type(exc).__name__}: {exc}")
        return result

    shape = tuple(int(value) for value in image.shape)
    affine = np.asarray(image.affine, dtype=np.float64)
    result.update(
        {
            "shape": list(shape),
            "dtype": str(image.get_data_dtype()),
            "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"]),
        }
    )

    if len(shape) != 3:
        errors.append(
            f"VISTA3D expects one 3D CT volume, but the input shape is {shape}."
        )
    elif any(size <= 0 for size in shape):
        errors.append(f"NIfTI contains an empty spatial dimension: {shape}.")

    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        errors.append("NIfTI affine must be a finite 4x4 matrix.")
    else:
        spatial = affine[:3, :3]
        determinant = float(np.linalg.det(spatial))
        result["affine_determinant"] = determinant
        try:
            voxel_sizes = np.asarray(nib.affines.voxel_sizes(affine), dtype=np.float64)
        except Exception as exc:
            errors.append(f"Unable to derive voxel spacing: {type(exc).__name__}: {exc}")
        else:
            result["voxel_spacing_mm"] = [float(value) for value in voxel_sizes]
            if (
                voxel_sizes.shape != (3,)
                or not np.all(np.isfinite(voxel_sizes))
                or np.any(voxel_sizes <= 0)
            ):
                errors.append(f"Invalid voxel spacing derived from affine: {voxel_sizes}.")
        if not math.isfinite(determinant) or abs(determinant) < 1e-8:
            errors.append("NIfTI affine is singular and cannot be resampled.")

    if result["qform_code"] == 0 and result["sform_code"] == 0:
        warnings.append(
            "Both qform_code and sform_code are zero; nibabel fallback geometry will be used."
        )

    result["valid"] = not errors
    return result


def validate_nifti_geometry(path: str | Path) -> Dict[str, Any]:
    result = inspect_nifti_geometry(path)
    if not result["valid"]:
        raise NiftiGeometryError(" ".join(result["errors"]))
    return result


def cuda_runtime_info(requested_device: Optional[str] = None) -> Dict[str, Any]:
    """Return authoritative PyTorch CUDA information and validate a device."""
    result: Dict[str, Any] = {
        "available": False,
        "requested_device": requested_device,
        "errors": [],
    }
    try:
        import torch
    except Exception as exc:
        result["errors"].append(f"PyTorch import failed: {type(exc).__name__}: {exc}")
        return result

    result.update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    )
    if not result["available"]:
        result["errors"].append(
            "torch.cuda.is_available() is false; check the NVIDIA driver and PyTorch CUDA build."
        )
        return result

    devices = []
    for index in range(result["device_count"]):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "capability": f"{properties.major}.{properties.minor}",
                "memory_total_mb": round(properties.total_memory / 1024**2),
            }
        )
    result["devices"] = devices

    if requested_device and requested_device != "auto":
        match = re.fullmatch(r"cuda:(\d+)", requested_device.strip().lower())
        if not match:
            result["errors"].append(
                f"VISTA3D requires a CUDA device such as cuda:0, not {requested_device!r}."
            )
        elif int(match.group(1)) >= result["device_count"]:
            result["errors"].append(
                f"Requested {requested_device}, but PyTorch detects only "
                f"{result['device_count']} CUDA device(s)."
            )
    return result


def require_cuda_device(device: str) -> Dict[str, Any]:
    result = cuda_runtime_info(device)
    if result["errors"]:
        raise CudaEnvironmentError(" ".join(result["errors"]))
    return result


def exception_chain(exc: BaseException) -> str:
    """Render nested MONAI/PyTorch exceptions without losing the root cause."""
    messages = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen and len(messages) < 12:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(messages)


def is_retryable_cuda_error(exc: BaseException) -> bool:
    message = exception_chain(exc).lower()
    retryable_markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "device is busy or unavailable",
        "all gpus are currently in use",
    )
    return any(marker in message for marker in retryable_markers)


def collect_runtime_diagnostics(nifti_path: Optional[str] = None) -> Dict[str, Any]:
    packages = {
        name: _package_version(distribution)
        for name, distribution in {
            "numpy": "numpy",
            "scipy": "scipy",
            "nibabel": "nibabel",
            "scikit_image": "scikit-image",
            "monai": "monai",
            "torch": "torch",
        }.items()
    }
    compatibility_errors = []
    warnings = []
    if packages["monai"] != "1.3.2":
        compatibility_errors.append(
            f"The VISTA3D research integration expects MONAI 1.3.2; found {packages['monai']}."
        )
    numpy_version = packages["numpy"] or ""
    if numpy_version.startswith("2."):
        compatibility_errors.append(
            "NumPy 2.x is outside the upstream VISTA3D research environment; rerun scripts/setup.sh."
        )

    result: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda_runtime_info(),
        "vista3d_compatibility_errors": compatibility_errors,
        "warnings": warnings,
    }
    if nifti_path:
        result["nifti"] = inspect_nifti_geometry(nifti_path)
    return result

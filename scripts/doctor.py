#!/usr/bin/env python3
"""Print deployment diagnostics without starting VoxelSage services."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Port_B"))

from Tool_Box.runtime_diagnostics import collect_runtime_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Python packages, NVIDIA/PyTorch CUDA, and optional NIfTI geometry."
    )
    parser.add_argument("nifti", nargs="?", help="Optional .nii/.nii.gz file to inspect")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero when PyTorch cannot use CUDA",
    )
    parser.add_argument(
        "--require-vista-compatible",
        action="store_true",
        help="Exit non-zero for known VISTA3D package incompatibilities",
    )
    args = parser.parse_args()

    diagnostics = collect_runtime_diagnostics(args.nifti)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    failed = False
    if args.require_cuda and not diagnostics["cuda"]["available"]:
        failed = True
    if args.require_vista_compatible and diagnostics["vista3d_compatibility_errors"]:
        failed = True
    if args.nifti and not diagnostics["nifti"]["valid"]:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

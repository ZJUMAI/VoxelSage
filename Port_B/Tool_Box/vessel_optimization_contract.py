"""Shared safety and audit contract for experimental vessel optimization."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


MAX_ALLOWED_GAP_MM = 4.0
AUDIT_SCHEMA = "geosurge.vessel-optimization-audit"
AUDIT_SCHEMA_VERSION = 2
LEGACY_AUDIT_SCHEMAS = frozenset({"3dmedagent.vessel-optimization-audit"})
SUPPORTED_AUDIT_SCHEMAS = frozenset({AUDIT_SCHEMA, *LEGACY_AUDIT_SCHEMAS})
CASE_MANIFEST_FILENAME = "vessel_optimization_report.json"
OPTIMIZABLE_VESSELS = frozenset({"hepatic", "portal"})


def is_supported_audit_schema(value) -> bool:
    """Return whether a manifest schema is accepted by this release.

    New manifests use ``AUDIT_SCHEMA``.  The legacy identifier remains
    readable so previously generated, otherwise valid audit manifests retain
    their safety-validation path after the GeoSurge naming migration.
    """
    return value in SUPPORTED_AUDIT_SCHEMAS


def validate_max_gap_mm(value) -> float:
    """Return a safe public gap threshold or raise before any processing."""
    try:
        validated = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_gap_mm must be within (0, {MAX_ALLOWED_GAP_MM}]"
        ) from exc
    if (
        not math.isfinite(validated)
        or validated <= 0.0
        or validated > MAX_ALLOWED_GAP_MM
    ):
        raise ValueError(
            f"max_gap_mm must be within (0, {MAX_ALLOWED_GAP_MM}]"
        )
    return validated


def sha256_file(path: str | Path) -> str:
    """Calculate a streaming SHA-256 digest for an immutable audit binding."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Compatibility tests for the GeoSurge vessel-optimization audit schema."""

import unittest

from Tool_Box.vessel_optimization_contract import (
    AUDIT_SCHEMA,
    LEGACY_AUDIT_SCHEMAS,
    is_supported_audit_schema,
)


class AuditSchemaCompatibilityTests(unittest.TestCase):
    def test_new_schema_is_the_writer_identifier(self):
        self.assertEqual(AUDIT_SCHEMA, "geosurge.vessel-optimization-audit")
        self.assertTrue(is_supported_audit_schema(AUDIT_SCHEMA))

    def test_legacy_schema_remains_readable(self):
        legacy_schema = next(iter(LEGACY_AUDIT_SCHEMAS))
        self.assertTrue(is_supported_audit_schema(legacy_schema))

    def test_unknown_schema_is_rejected(self):
        self.assertFalse(is_supported_audit_schema("unknown.audit-schema"))


if __name__ == "__main__":
    unittest.main()

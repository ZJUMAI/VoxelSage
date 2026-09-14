import json
import sys
import tempfile
import unittest
from pathlib import Path

# Some lightweight HTML tests replace scientific packages in sys.modules with
# no-op modules during collection. Restore the real packages before API import
# so this endpoint test remains order-independent.
for _module_name in (
    "skimage.measure", "skimage", "scipy.ndimage", "scipy", "numpy", "nibabel",
):
    _module = sys.modules.get(_module_name)
    if _module is not None and getattr(_module, "__file__", None) is None:
        del sys.modules[_module_name]

from API import (
    RestoreResectionPlaneRequest,
    SaveResectionPlaneRequest,
    api_restore_resection_plane,
    api_save_resection_plane,
)


def _control_points(offset):
    return [
        [[offset + i, offset + j, offset + i + j] for j in range(4)]
        for i in range(4)
    ]


class RestoreResectionPlaneTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.json_path = self.output_dir / "case_3d.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_case(self, plane):
        self.json_path.write_text(
            json.dumps({
                "resection_planes": [plane],
                "selected_resection_plane_index": 0,
                "selected_resection_plane_saved_at": "old",
                "resection_sequence_available": True,
            }),
            encoding="utf-8",
        )

    def _restore(self, points):
        return api_restore_resection_plane(RestoreResectionPlaneRequest(**{
            "output_dir": str(self.output_dir),
            "json_file": self.json_path.name,
            "plane_index": 0,
            "original_control_points_3d": points,
        }))

    def test_restore_prefers_persisted_optimizer_baseline(self):
        original = _control_points(0)
        self._write_case({
            "control_points_3d": _control_points(100),
            "original_control_points_3d": original,
            "user_saved": True,
            "unsaved_changes": True,
            "saved_at": "old",
        })

        response = self._restore(_control_points(200))

        self.assertEqual(response["status"], "ok")
        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        plane = data["resection_planes"][0]
        self.assertEqual(plane["control_points_3d"], original)
        self.assertFalse(plane["user_saved"])
        self.assertFalse(plane["unsaved_changes"])
        self.assertNotIn("saved_at", plane)
        self.assertFalse(data["resection_sequence_available"])
        self.assertEqual(data["selected_resection_plane_source"], "restored_original")
        self.assertNotIn("selected_resection_plane_saved_at", data)

    def test_restore_backfills_legacy_baseline(self):
        original = _control_points(0)
        self._write_case({"control_points_3d": _control_points(100)})

        response = self._restore(original)

        self.assertEqual(response["status"], "ok")
        plane = json.loads(self.json_path.read_text(encoding="utf-8"))["resection_planes"][0]
        self.assertEqual(plane["control_points_3d"], original)
        self.assertEqual(plane["original_control_points_3d"], original)

    def test_first_save_preserves_legacy_baseline(self):
        original = _control_points(0)
        edited = _control_points(100)
        self._write_case({"control_points_3d": original})

        response = api_save_resection_plane(SaveResectionPlaneRequest(**{
            "output_dir": str(self.output_dir),
            "json_file": self.json_path.name,
            "plane_index": 0,
            "control_points_3d": edited,
            "candidate_name": "candidate",
        }))

        self.assertEqual(response["status"], "ok")
        plane = json.loads(self.json_path.read_text(encoding="utf-8"))["resection_planes"][0]
        self.assertEqual(plane["control_points_3d"], edited)
        self.assertEqual(plane["original_control_points_3d"], original)


if __name__ == "__main__":
    unittest.main()

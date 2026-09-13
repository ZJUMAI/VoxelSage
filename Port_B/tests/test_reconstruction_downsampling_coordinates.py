"""Verify exported mesh coordinates without loading the Linux HTTP server."""

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import numpy as np
import pytest
from skimage.morphology import remove_small_objects

from Visualization import visualize_3d as visualization


def _mesh_generator():
    # API imports Linux-only fcntl. Execute the actual mesh-generation function
    # with real geometry dependencies, while omitting unrelated server startup.
    source = Path(__file__).resolve().parents[1] / "API.py"
    function = next(
        node for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "_generate_threejs"
    )
    namespace = {
        "Any": Any, "Dict": Dict, "np": np, "os": os, "Path": Path,
        "json": json, "ORGAN_COLORS": visualization.ORGAN_COLORS,
        "_LESION_NAMES": {"tumor"}, "remove_small_objects": remove_small_objects,
        "_read_threejs_sources": lambda: None,
        "_read_bezier_surface_source": lambda: "",
        "_make_threejs_html": lambda **kwargs: "<html></html>",
    }
    for name in (
        "get_display_name", "robust_load_nii", "binarize_mask",
        "resample_for_smoothing", "downsample_volume", "extract_mesh",
        "voxel_to_world", "quantize_vertices", "mesh_surface_area",
    ):
        namespace[name] = getattr(visualization, name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_generate_threejs"]


@pytest.mark.parametrize("family", ["isotropic", "anisotropic", "rotated"])
@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_exported_cuboid_preserves_physical_coordinates(
    tmp_path: Path, family: str, factor: float
) -> None:
    affine = np.eye(4)
    affine[:3, 3] = [31.0, -29.0, 13.0]
    if family != "isotropic":
        affine[:3, :3] = np.diag([0.8, 1.2, 3.0])
    if family == "rotated":
        angle = np.deg2rad(37.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        affine[:3, :3] = rotation @ affine[:3, :3]
    mask = np.zeros((33, 33, 33), dtype=np.uint8)
    mask[8:25, 8:25, 8:25] = 1
    result = _mesh_generator()(
        case_dir=str(tmp_path), output_path=str(tmp_path / "scene.html"),
        masks={"liver": {"_data": mask, "_affine": affine}},
        step_size=1, downsample_factor=factor, isotropic_resample=False,
        gaussian_sigma=0.0, smooth=False, default_opacity=0.5,
        prob_threshold=0.5, skip_empty=True, title="synthetic",
    )
    assert result["status"] == "ok"
    scene = json.loads((tmp_path / "scene.json").read_text(encoding="utf-8"))
    world = np.asarray(scene["meshes"][0]["verts"]).reshape(-1, 3)
    world += scene["center_offset"]
    source_indices = nib.affines.apply_affine(np.linalg.inv(affine), world)

    # Factor 2 gives 16 samples spanning original voxel centres 0..32.
    # Nearest-neighbour sampling retains indices 4..11, so the isosurface
    # lies at 3.5 and 11.5 of these 15 intervals, independently on each axis.
    lower, upper = (7.5, 24.5) if factor == 1.0 else (112.0 / 15.0, 368.0 / 15.0)
    np.testing.assert_allclose(source_indices.min(axis=0), lower, atol=2e-5)
    np.testing.assert_allclose(source_indices.max(axis=0), upper, atol=2e-5)
    assert result["organ_stats"]["liver"]["voxel_count"] == 17 ** 3

    if factor == 1.0:
        original, _ = visualization.extract_mesh(mask, (1.0, 1.0, 1.0), 1, smooth=False)
        expected = nib.affines.apply_affine(affine, original)
        np.testing.assert_allclose(world, expected, atol=1e-5)

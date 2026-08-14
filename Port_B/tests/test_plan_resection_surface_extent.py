import numpy as np

from skills.builtin.plan_resection.main import (
    _bernstein3_values,
    _fit_surface_to_liver_intersection,
    _fit_surface_to_liver_projection,
    _invalidate_previous_resection_state,
    _surface_resolution_for_cell_size,
    _select_refinement_candidates,
)


def _height(grid, u_norm, v_norm):
    return float(
        _bernstein3_values(u_norm)
        @ np.asarray(grid, dtype=np.float64)
        @ _bernstein3_values(v_norm)
    )


def _surface():
    return {
        "reference_plane": {
            "origin_mm": [10.0, -20.0, 3.0],
            "normal_world": [0.0, 0.0, 1.0],
            "u_axis_world": [1.0, 0.0, 0.0],
            "v_axis_world": [0.0, 1.0, 0.0],
            "u_range_mm": [-100.0, 100.0],
            "v_range_mm": [-80.0, 80.0],
        },
        "height_control_4x4_mm": [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0, 4.0],
            [3.0, 6.0, 7.0, 5.0],
            [1.0, 3.0, 4.0, 2.0],
        ],
    }


def test_surface_domain_encloses_complete_liver_projection_with_padding():
    surface = _surface()
    liver_xyz = np.array([
        [-65.0, -70.0, 0.0],
        [95.0, 55.0, 8.0],
        [20.0, 30.0, -2.0],
    ])

    _fit_surface_to_liver_projection(surface, liver_xyz, padding_mm=5.0)

    # Coordinates are relative to origin [10, -20, 3].
    assert surface["reference_plane"]["u_range_mm"] == [-80.0, 90.0]
    assert surface["reference_plane"]["v_range_mm"] == [-55.0, 80.0]


def test_surface_reparameterisation_preserves_original_polynomial_geometry():
    surface = _surface()
    old_grid = np.asarray(surface["height_control_4x4_mm"], dtype=np.float64)
    old_u = np.asarray(surface["reference_plane"]["u_range_mm"])
    old_v = np.asarray(surface["reference_plane"]["v_range_mm"])
    liver_xyz = np.array([
        [-65.0, -70.0, 0.0],
        [95.0, 55.0, 8.0],
    ])

    control_points = np.asarray(
        _fit_surface_to_liver_projection(surface, liver_xyz, padding_mm=5.0)
    )
    new_grid = np.asarray(surface["height_control_4x4_mm"])
    new_u = np.asarray(surface["reference_plane"]["u_range_mm"])
    new_v = np.asarray(surface["reference_plane"]["v_range_mm"])

    for u_norm, v_norm in ((0.0, 0.0), (0.23, 0.71), (0.5, 0.5), (1.0, 1.0)):
        u_mm = new_u[0] + u_norm * np.diff(new_u)[0]
        v_mm = new_v[0] + v_norm * np.diff(new_v)[0]
        old_u_norm = (u_mm - old_u[0]) / np.diff(old_u)[0]
        old_v_norm = (v_mm - old_v[0]) / np.diff(old_v)[0]
        assert np.isclose(
            _height(new_grid, u_norm, v_norm),
            _height(old_grid, old_u_norm, old_v_norm),
            atol=1e-10,
        )

    # Browser-authoritative 3D control points and the reference representation
    # must describe the same bicubic surface.
    assert np.allclose(control_points[:, :, 2], 3.0 + new_grid)


def test_refinement_shortlist_keeps_selected_candidate_and_family_diversity():
    candidates = [{"name": str(index)} for index in range(5)]
    scored = [
        {"score": 0.90, "candidate_family": "plane"},
        {"score": 0.80, "candidate_family": "plane"},
        {"score": 0.70, "candidate_family": "local"},
        {"score": 0.60, "candidate_family": "portal_vessel"},
        {"score": 0.10, "candidate_family": "other"},
    ]

    shortlist = _select_refinement_candidates(candidates, scored, selected_index=4, max_candidates=4)

    assert [index for index, _ in shortlist] == [4, 0, 2, 3]

def test_surface_expansion_does_not_extrapolate_local_cubic():
    surface = _surface()
    original_grid = np.asarray(surface["height_control_4x4_mm"], dtype=np.float64)
    liver_xyz = np.array([
        [-240.0, -180.0, 0.0],
        [260.0, 210.0, 8.0],
    ])

    control_points = np.asarray(
        _fit_surface_to_liver_projection(surface, liver_xyz, padding_mm=5.0)
    )
    expanded_grid = np.asarray(surface["height_control_4x4_mm"])

    # Expanding a local cubic must not evaluate it outside [0, 1], where its
    # control heights can explode.  The bounded grid is stretched instead.
    assert np.array_equal(expanded_grid, original_grid)
    assert np.max(np.abs(control_points[:, :, 2] - 3.0)) <= np.max(np.abs(original_grid))


def test_final_surface_retains_three_grid_cells_around_liver_cut_perimeter():
    coordinates = np.arange(-10.0, 10.1, 1.0)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    liver_xyz = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    liver_xyz = liver_xyz[np.sum(liver_xyz ** 2, axis=1) <= 10.0 ** 2]
    surface = {
        "reference_plane": {
            "origin_mm": [0.0, 0.0, 8.0],
            "normal_world": [0.0, 0.0, 1.0],
            "u_axis_world": [1.0, 0.0, 0.0],
            "v_axis_world": [0.0, 1.0, 0.0],
            "u_range_mm": [-15.0, 15.0],
            "v_range_mm": [-15.0, 15.0],
        },
        "height_control_4x4_mm": np.zeros((4, 4)).tolist(),
    }

    _fit_surface_to_liver_intersection(
        surface, liver_xyz, band_mm=0.6, cell_size_mm=4.0,
    )

    # The z=8 cut of a radius-10 sphere has an in-plane radius of 6 mm.  With
    # three 4 mm safety cells on each edge, outward alignment yields ±20 mm.
    u_range = surface["reference_plane"]["u_range_mm"]
    v_range = surface["reference_plane"]["v_range_mm"]
    assert u_range == [-20.0, 20.0]
    assert v_range == [-20.0, 20.0]
    assert _surface_resolution_for_cell_size(surface, 4.0) == [11, 11]


def test_intersection_bounds_are_aligned_outward_without_extra_padding():
    surface = _surface()
    intersection_xyz = np.array([
        [-6.1, -4.2, 3.0],
        [7.9, 5.1, 3.0],
    ])

    _fit_surface_to_liver_projection(
        surface,
        intersection_xyz,
        padding_mm=0.0,
        grid_alignment_mm=4.0,
    )

    # Coordinates are relative to origin [10, -20, 3]:
    # u=[-16.1,-2.1] -> [-20,0], v=[15.8,25.1] -> [12,28].
    assert surface["reference_plane"]["u_range_mm"] == [-20.0, 0.0]
    assert surface["reference_plane"]["v_range_mm"] == [12.0, 28.0]
    assert _surface_resolution_for_cell_size(surface, 4.0) == [6, 5]


def test_replanning_invalidates_stale_plane_selection():
    json_data = {
        "selected_resection_plane_index": 2,
        "selected_resection_plane_source": "editor_save",
        "selected_resection_plane_saved_at": "2026-01-01T00:00:00Z",
        "resection_sequence_available": True,
        "unrelated": "preserved",
    }

    _invalidate_previous_resection_state(json_data)

    assert "selected_resection_plane_index" not in json_data
    assert "selected_resection_plane_source" not in json_data
    assert "selected_resection_plane_saved_at" not in json_data
    assert json_data["resection_sequence_available"] is False
    assert json_data["unrelated"] == "preserved"

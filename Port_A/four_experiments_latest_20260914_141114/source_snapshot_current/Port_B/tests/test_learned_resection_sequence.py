import os
from pathlib import Path

import numpy as np
import pytest

from skills.builtin.plan_resection_sequence.learned_shielded import (
    EXPECTED_CHECKPOINT_SHA256,
    build_scenario,
    configured_checkpoint_path,
    plan_learned_shielded,
    validate_checkpoint,
)
from skills.builtin.plan_resection_sequence.main import (
    _enclose_boundary_vessel_proxies,
    _learned_surface_resolution,
    _learned_target_crop,
    _remap_adapter_steps,
)


def _flat_control_points(u_mm=40.0, v_mm=80.0):
    cp = np.zeros((4, 4, 3), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            cp[i, j] = [u_mm * i / 3.0, v_mm * j / 3.0, 0.0]
    return cp


def test_learned_resolution_uses_four_mm_surface_cells():
    assert _learned_surface_resolution(_flat_control_points()) == [11, 21]


def test_learned_target_crop_removes_only_outside_liver_display_border():
    target = np.zeros((32, 35), dtype=bool)
    target[3:28, 3:31] = True

    crop = _learned_target_crop(target, 32, 35)

    assert (crop["rows"], crop["cols"]) == (25, 28)
    assert (crop["row0"], crop["col0"]) == (3, 3)
    assert crop["local_to_source"][0] == 3 * 35 + 3
    assert crop["local_to_source"][-1] == 27 * 35 + 30
    assert np.all(target.reshape(-1)[crop["local_to_source"]])
    assert np.count_nonzero(crop["source_to_local"] >= 0) == 25 * 28


def test_learned_target_crop_rejects_true_target_outside_frozen_envelope():
    target = np.ones((29, 42), dtype=bool)

    with pytest.raises(ValueError, match=r"目标 29x42"):
        _learned_target_crop(target, 29, 42)


def test_adapter_steps_map_back_to_complete_saved_surface_grid():
    target = np.zeros((6, 8), dtype=bool)
    target[2:5, 3:7] = True
    crop = _learned_target_crop(target, 6, 8)
    steps = [
        {"step": 0, "cell": 0, "grid_ij": [0, 0]},
        {"step": 1, "cell": 5, "grid_ij": [1, 1]},
    ]

    remapped = _remap_adapter_steps(
        steps,
        crop["local_to_source"],
        source_cols=8,
        adapter_cols=4,
    )

    assert remapped[0]["cell"] == 2 * 8 + 3
    assert remapped[0]["grid_ij"] == [2, 3]
    assert remapped[0]["adapter_cell"] == 0
    assert remapped[1]["cell"] == 3 * 8 + 4
    assert remapped[1]["grid_ij"] == [3, 4]
    assert remapped[1]["adapter_grid_ij"] == [1, 1]


def test_boundary_vessel_proxy_is_enclosed_with_sampled_liver_support():
    core = np.zeros((5, 5), dtype=bool)
    core[1:4, 1:4] = True
    support = core.copy()
    support[0, 2] = True
    vascular_safe = np.ones((5, 5), dtype=bool)
    vascular_safe[1, 2] = False

    target, audit = _enclose_boundary_vessel_proxies(
        core, support, vascular_safe, rows=5, cols=5
    )

    assert target.reshape(5, 5)[0, 2]
    assert audit == {
        "core_cell_count": 9,
        "enclosure_added_cell_count": 1,
        "initial_boundary_vessel_cell_count": 1,
    }


def test_boundary_vessel_proxy_fails_when_liver_support_cannot_enclose_it():
    core = np.zeros((5, 5), dtype=bool)
    core[1:4, 1:4] = True
    vascular_safe = np.ones((5, 5), dtype=bool)
    vascular_safe[1, 2] = False

    with pytest.raises(ValueError, match="支撑区域不足"):
        _enclose_boundary_vessel_proxies(
            core, core, vascular_safe, rows=5, cols=5
        )


def test_checkpoint_validation_fails_closed_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="冻结切除排序模型不存在"):
        validate_checkpoint(tmp_path / "missing.pt")


def test_checkpoint_validation_fails_closed_on_hash_mismatch(tmp_path):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not-the-frozen-model")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint(wrong)


def test_surface_adapter_keeps_start_component_and_vessel_proxies():
    liver = np.array([
        True, True, False,
        True, True, False,
        False, False, True,
    ])
    vascular_safe = np.ones(9, dtype=bool)
    vascular_safe[4] = False
    scenario, component = build_scenario(
        liver, vascular_safe, start=0, rows=3, cols=3, cell_side_mm=4.0
    )
    assert component.tolist() == [True, True, False, True, True, False, False, False, False]
    assert scenario["domain_cells"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert scenario["obstacle_cells"] == [[1, 1]]
    assert scenario["start_cell"] == [0, 0]


@pytest.mark.skipif(
    not configured_checkpoint_path().is_file(),
    reason="frozen v10.6 checkpoint is intentionally not distributed",
)
def test_frozen_c4_runs_on_surface_parameter_grid():
    liver = np.ones(16, dtype=bool)
    vascular_safe = np.ones(16, dtype=bool)
    vascular_safe[5] = False
    result = plan_learned_shielded(
        liver, vascular_safe, start=0, rows=4, cols=4, cell_side_mm=4.0
    )
    assert result["policy_id"] == "clinical_v106_c4_learned_shielded"
    assert result["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert result["simulator"]["safety_invariant_violations"] == 0
    assert set(result["covered_cells"]) == set(range(16))
    assert result["path"][0]["cell"] == 0

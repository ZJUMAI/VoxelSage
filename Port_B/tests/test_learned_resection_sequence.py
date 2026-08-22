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
from skills.builtin.plan_resection_sequence.main import _learned_surface_resolution


def _flat_control_points(u_mm=40.0, v_mm=80.0):
    cp = np.zeros((4, 4, 3), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            cp[i, j] = [u_mm * i / 3.0, v_mm * j / 3.0, 0.0]
    return cp


def test_learned_resolution_uses_four_mm_surface_cells():
    assert _learned_surface_resolution(_flat_control_points()) == [11, 21]


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

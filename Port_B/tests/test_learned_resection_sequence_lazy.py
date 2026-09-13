"""Public Skill coverage and eager/lazy equivalence on identical vessel grids."""

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from skills.builtin.plan_resection_sequence.learned_shielded import (
    EXPECTED_CHECKPOINT_SHA256,
    _replay_actions,
    _research_modules,
    build_scenario,
    configured_checkpoint_path,
    plan_learned_shielded,
)
from skills.builtin.plan_resection_sequence.main import _vascular_safe_mask
from skills.engine import SkillEngine


FROZEN_CONFIG = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
    "cell_side_mm": 4.0,
}
requires_checkpoint = pytest.mark.skipif(
    not configured_checkpoint_path().is_file(),
    reason="frozen v10.6 checkpoint is intentionally not distributed",
)


@requires_checkpoint
@pytest.mark.parametrize("geometry", ["plain", "interior_vessel", "boundary_barrier", "disconnected"])
def test_lazy_adapter_matches_eager_on_identical_targets_and_vessels(geometry):
    rows = cols = 5
    liver = np.ones((rows, cols), dtype=bool)
    safe = np.ones_like(liver)
    if geometry == "interior_vessel":
        safe[2, 2] = False
    elif geometry == "boundary_barrier":
        safe[:, 2] = False
    elif geometry == "disconnected":
        liver[:, 3] = False
        safe[2, 1] = False

    result = plan_learned_shielded(liver, safe, start=0, rows=rows, cols=cols)
    scenario, component = build_scenario(
        liver, safe, start=0, rows=rows, cols=cols, cell_side_mm=4.0
    )
    _research_modules()
    from confirmation_controllers_v107 import rollout_controller

    eager = rollout_controller(
        "C4", scenario,
        baseline_blood=result["simulator"]["baseline_simulated_blood_ml"],
        margin_ml=result["simulator"]["margin_ml"], cfg=FROZEN_CONFIG,
        checkpoint_path=configured_checkpoint_path(),
    )
    eager_path, eager_covered = _replay_actions(
        scenario, eager["actions"], clinical_config=FROZEN_CONFIG, cols=cols
    )

    assert eager["completion"]
    assert result["scenario"] == scenario
    assert result["simulator"]["verify_mode"] == "lazy"
    assert result["simulator"]["action_sequence_hash"] == eager["action_sequence_hash"]
    assert result["path"] == eager_path
    assert result["covered_cells"] == eager_covered == np.flatnonzero(component).tolist()
    for actual, expected in (
        ("elapsed_minutes", "elapsed_minutes"),
        ("simulated_blood_ml", "realized_episode_B_ml"),
        ("budget_ml", "budget_ml"),
        ("shield_intervention_count", "shield_intervention_count"),
        ("safety_invariant_violations", "safety_invariant_violations"),
    ):
        assert result["simulator"][actual] == eager[expected]
    assert result["simulator"]["simulated_blood_ml"] <= result["simulator"]["budget_ml"]


def _saved_surface_context(tmp_path):
    cp = np.zeros((4, 4, 3), dtype=float)
    for i in range(4):
        for j in range(4):
            cp[i, j] = [16.0 * i / 3.0, 16.0 * j / 3.0, 1.0]
    source = tmp_path / "constructed_3d.json"
    source.write_text(json.dumps({
        "center_offset": [0, 0, 0],
        "resection_planes": [{"user_saved": True, "control_points_3d": cp.tolist()}],
    }), encoding="utf-8")
    nib.save(nib.Nifti1Image(np.ones((17, 17, 3), dtype=np.uint8), np.eye(4)),
             tmp_path / "liver.nii.gz")
    vessel = np.zeros((17, 17, 3), dtype=np.uint8)
    vessel[6, 6, 1] = 1
    nib.save(nib.Nifti1Image(vessel, np.eye(4)), tmp_path / "portal.nii.gz")
    return SimpleNamespace(
        case_id="constructed", output_dir=str(tmp_path), mask_dir=str(tmp_path),
        params={"algorithm": "learned_shielded", "vascular_safe_distance_mm": 0.1},
    )


def _public_skill_module():
    path = Path(__file__).resolve().parents[1] / "skills/builtin/plan_resection_sequence/main.py"
    return SkillEngine()._load_module("plan_resection_sequence", str(path))


@requires_checkpoint
def test_public_saved_surface_skill_runs_lazy_without_dropping_vessels(tmp_path):
    ctx = _saved_surface_context(tmp_path)
    response = _public_skill_module().run(ctx)
    saved = json.loads(Path(response["result_path"]).read_text(encoding="utf-8"))

    assert response["status"] == "ok"
    assert response["algorithm"] == "learned_shielded"
    assert response["policy_id"] == "clinical_v106_c4_learned_shielded"
    assert response["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert response["simulator"]["controller"] == "C4"
    assert response["simulator"]["verify_mode"] == "lazy"
    assert response["coverage"] == 1.0
    assert saved["covered_cells"] == list(range(16))
    assert saved["cell_states"][5]["state"] == "vascular_risk"
    assert saved["parameters"]["liver_intersection_min_samples"] == 4
    assert response["simulator"]["simulated_blood_ml"] <= response["simulator"]["budget_ml"]


@requires_checkpoint
def test_lazy_adapter_refuses_when_no_candidate_satisfies_budget():
    with pytest.raises(RuntimeError, match="infeasible_no_safe_candidate"):
        plan_learned_shielded(
            np.ones(16, dtype=bool), np.ones(16, dtype=bool),
            start=0, rows=4, cols=4, margin_ml=-1.0,
        )


@pytest.mark.parametrize("centers", [np.array([[6., 6., 1.]]), np.array([[100., 100., 1.]])])
def test_vessel_mask_retains_all_unsafe_or_out_of_volume_samples(tmp_path, centers):
    ctx = _saved_surface_context(tmp_path)
    assert not _vascular_safe_mask(ctx, centers, [0, 0, 0], 0.1).any()


def test_public_skill_rejects_fully_vascular_surface(tmp_path):
    ctx = _saved_surface_context(tmp_path)
    nib.save(nib.Nifti1Image(np.ones((17, 17, 3), dtype=np.uint8), np.eye(4)),
             tmp_path / "portal.nii.gz")
    with pytest.raises(ValueError, match="没有可规划单元"):
        _public_skill_module().run(ctx)
    assert not (tmp_path / "constructed_resection_sequence.json").exists()


@requires_checkpoint
def test_public_skill_accepts_empty_vessel_masks_inside_the_volume(tmp_path):
    ctx = _saved_surface_context(tmp_path)
    for name in ("portal", "hepatic"):
        nib.save(nib.Nifti1Image(np.zeros((17, 17, 3), dtype=np.uint8), np.eye(4)),
                 tmp_path / f"{name}.nii.gz")
    # No vessel is within any finite threshold; out-of-volume points remain invalid.
    ctx.params["vascular_safe_distance_mm"] = 1000.0
    centers = np.array([[2., 2., 1.], [100., 100., 1.]])
    assert _vascular_safe_mask(ctx, centers, [0, 0, 0], 1000.0).tolist() == [True, False]

    response = _public_skill_module().run(ctx)
    saved = json.loads(Path(response["result_path"]).read_text(encoding="utf-8"))
    assert response["status"] == "ok"
    assert response["coverage"] == 1.0
    assert response["simulator"]["verify_mode"] == "lazy"
    assert saved["vascular_excluded_cell_count"] == 0
    assert saved["covered_cells"] == list(range(16))

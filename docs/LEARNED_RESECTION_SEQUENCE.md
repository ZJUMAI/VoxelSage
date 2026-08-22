# Learned, shielded resection-sequence Skill

`plan_resection_sequence` can apply the frozen v10.6 C4 target ranker and its exact simulator shield to the two-dimensional parameter grid of a user-confirmed 3D Bezier resection surface. The returned cell sequence is mapped back into the existing Three.js viewer.

> [!CAUTION]
> This is an opt-in research mode, not a clinical surgical trajectory. Vessel-near cells and simulated blood are uncalibrated two-dimensional proxies. Passing the simulator shield does not establish patient safety, clinical validity, or a patient-specific blood-loss prediction.

## What changed

- `nearest`, `dfs`, and `spanning_tree` remain deterministic defaults.
- `learned_shielded` must be selected explicitly in the Skill call or 3D viewer.
- The adapter derives an approximately 4 mm grid from the confirmed surface and rejects incompatible grid overrides.
- Missing or altered weights fail closed before planning.
- Results include the policy ID, checkpoint SHA-256, shield interventions, proxy budget, runtime, action-sequence hash, and a scope warning.

## Prerequisites

1. Run `three_d_reconstruction` with `generate_resection_plane=true`.
2. Open the generated 3D page, edit the Bezier surface, and save it.
3. Provide the frozen checkpoint locally.

Model weights are intentionally excluded from this repository. The accepted checkpoint is `epoch_05.pt` from v10.6 config 05, seed `2026081603`, epoch 5, with SHA-256:

```text
c07904502d6b71a74484adb1c27971c77cdf6a61bb20b04f1f39d786d61a70be
```

Place the authorized checkpoint at `Port_B/models/resection_sequence/epoch_05.pt`, or configure an absolute path:

```bash
export VOXELSAGE_RESECTION_MODEL_CHECKPOINT=/absolute/path/to/epoch_05.pt
sha256sum "$VOXELSAGE_RESECTION_MODEL_CHECKPOINT"
```

The research code, split manifests, frozen hashes, controller definitions, and result summaries are under [`Research/planar-resection-planning`](../Research/planar-resection-planning/README.md). Checkpoints and medical data are not redistributed.

## Use from the Skill API

After the surface is saved, call the Skill with the experimental algorithm:

```json
{
  "skill_name": "plan_resection_sequence",
  "params": {
    "case_name": "your-case",
    "algorithm": "learned_shielded",
    "preview_only": false,
    "learned_cell_side_mm": 4.0
  }
}
```

The 3D viewer exposes the same option as **Frozen learned ranker + simulator shield (experimental)**. Changing the algorithm invalidates the previous start selection, so preview the valid cells and select a start again before generating the path.

## Failure behavior

The Skill rejects the request when:

- the checkpoint is absent or its SHA-256 differs;
- the saved surface cannot support a 4 mm grid within the frozen `30 × 40` cap;
- the caller overrides `grid_resolution` inconsistently;
- the start cell lies outside the connected liver/surface target;
- the controller does not complete, violates a shield invariant, or exceeds its simulator budget.

These are visible failures. The implementation does not silently fall back to a deterministic planner under the `learned_shielded` label.

## Verification

From the repository root:

```bash
python -m pytest -q \
  Port_B/tests/test_learned_resection_sequence.py \
  Port_B/tests/test_visualize_3d_html.py
```

The focused tests cover deterministic checkpoint validation, adapter geometry, frozen-controller invocation, fail-closed behavior, and the 3D viewer integration. They are software tests, not clinical validation.

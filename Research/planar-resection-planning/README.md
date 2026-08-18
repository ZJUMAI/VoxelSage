# Planar Resection Planning Research

This directory contains the standalone two-dimensional research simulator used
to study sequential resection planning, behavior cloning (BC), masked PPO,
deterministic baselines, and an exact policy-external safety shield.

The current reported method is **not a final PPO policy**. It combines a
behavior-cloned target-ranking model with a deterministic low-level planner and
an exact safety shield. The shield evaluates candidate continuations against a
simulated blood-loss budget before execution. Earlier PPO experiments remain in
the repository for reproducibility and negative-result analysis.

> [!CAUTION]
> This is an unvalidated 2D methodology simulator. Simulated blood loss,
> tension, strain, timing, and safety budgets are not clinically calibrated.
> Do not use this software for diagnosis, treatment, or surgical decisions.

## What is included

- Grid-based resection environments and scenario generators
- Rule, serpentine, learned-ranking, and teacher planners
- Variable-size masked-PPO and behavior-cloning training pipelines
- Clinical-window environment variants and frozen evaluation gates
- Exact policy-external safety shield and confirmatory-study scripts
- A small FastAPI/HTML simulator for interactive inspection
- Unit and regression tests
- Curated v10.7/v10.7.1 reports, statistics, manifests, and figures under
  `artifacts/`

Large generated scenario pools, per-scenario shards, logs, caches, and model
checkpoints are intentionally omitted. The published manifests retain the
frozen checkpoint hashes needed to audit provenance.

## Install

From the VoxelSage repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Research/planar-resection-planning/requirements.txt
```

PyTorch installation can be platform-specific. If the default wheel is not
suitable for your CUDA setup, install the appropriate PyTorch build first.

## Run the simulator

```bash
cd Research/planar-resection-planning
python app.py --port 8910
```

Then open <http://127.0.0.1:8910/>. The deterministic simulator and mechanics
endpoints work without a checkpoint. Learned-policy endpoints require a
compatible local checkpoint and deliberately fail closed when it is absent.

## Test

```bash
cd Research/planar-resection-planning
python -m pytest -q
```

Tests that require omitted checkpoints may report the model as unavailable;
the API and policy services are designed to expose that state explicitly. The
default `pytest.ini` excludes historical artifact-audit suites whose frozen
inputs are not distributed. Restore those inputs and override `addopts` to run
the corresponding versioned audit tests.

## Research status

The frozen v10.7 replication study reported that the learned-ranking policy
plus exact shield reduced mean simulated time and simulated blood loss relative
to the serpentine baseline on 256 new generated scenarios, without budget
overruns in that replication set. The v10.7.1 correction recomputed sensitivity
comparisons against condition-specific baselines and reported 4/4 perturbation
conditions passing its predefined simulator gates.

These are simulator findings only. See
`artifacts/v10.7-confirmation/report_clinical_v107_confirmation.md` and
`artifacts/v10.7.1-sensitivity/report_sensitivity_v1071.md` for the exact
claims, limitations, and numerical results.

## Entry points

| Purpose | File |
| --- | --- |
| Interactive API and web simulator | `app.py` |
| Base environment | `environment.py` |
| Clinical-window environment | `clinical_window_environment.py` |
| Variable-size PPO/BC training | `train_variable_masked_ppo.py` |
| Frozen v10.6 target-ranking policy | `clinical_target_order_policy_v106.py` |
| Exact safety shield | `clinical_safety_shield_v106.py` |
| v10.7 confirmation evaluation | `evaluate_confirmation_v107.py` |
| v10.7.1 sensitivity evaluation | `evaluate_sensitivity_v1071.py` |
| Detailed Chinese documentation index | `文档索引.md` |

## Reproducibility notes

- Generated outputs belong under `results/` and are ignored by Git.
- Checkpoints and patient or medical-imaging data must never be committed.
- Some historical scripts preserve versioned experiment contracts and should
  not be silently reused as current recommendations.
- The confirmatory manifests pin seeds, code hashes, checkpoint hashes, split
  access order, and gate definitions.

Code in this directory is distributed under the repository's Apache-2.0
license. Third-party Python packages retain their own licenses.

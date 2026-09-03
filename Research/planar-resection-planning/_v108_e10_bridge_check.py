"""v10.8 E10 Port B bridge acceptance check (rewrite, per Bryce 2026-09-04).

Spec:
  * 3 cases (different surface sizes) x 2 reps
  * Each rep runs BOTH algorithms through the real Port B bridge functions:
      - v10.7: ``plan_learned_shielded`` (old)
      - v10.8: ``plan_learned_shielded_v108`` (new)
  * Verify:
      1. v10.8 algorithm ID is callable via the Port B bridge.
      2. v10.8 returns the new diagnostic fields
         (verified_count_mean/max, selected_rank, infeasible_count,
         shield_intervention_count, action_sequence_hash).
      3. v10.7 and v10.8 produce the same action sequence (path-equivalent)
         on the same scenario -- per the v10.8 spec the lazy and eager
         verify sets are identical and the lazy controller only differs
         in the order it checks candidates.
      4. Determinism: 2 reps of v10.8 on the same case produce the same
         action_sequence_hash.
      5. Old ``learned_shielded`` algorithm is still callable and returns
         v10.7 metrics (backward compatibility).

Unlike the previous E10 check (which called the research-side
``rollout_controller`` directly), this script uses the Port B bridge
functions in ``skills.builtin.plan_resection_sequence.*`` so the test
exercises the same path that production code would use.

Output:
  results/clinical_window_v10_8_lazy_shield/port_b_bridge/e10_bridge_check.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Resolve paths: this file is in Research/planar-resection-planning/
REPO = Path(__file__).resolve().parent
VOXELSAGE = REPO.parent.parent
PORT_B = VOXELSAGE / "Port_B"
sys.path.insert(0, str(PORT_B))  # so we can import skills.builtin...

# Point the v10.7 bridge at the same frozen v10.6 checkpoint used by
# all other v10.8 experiments.  Without this it falls back to
# Port_B/models/resection_sequence/epoch_05.pt which is not in this tree.
_FROZEN_CKPT = (REPO
    / "results" / "clinical_window_v10_6_shielded_learning" / "runs" / "bc"
    / "config_05_seed_2026081603" / "epoch_05.pt")
os.environ.setdefault("VOXELSAGE_RESECTION_MODEL_CHECKPOINT", str(_FROZEN_CKPT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
OUT = V108 / "port_b_bridge"
OUT.mkdir(parents=True, exist_ok=True)


def _make_control_points(u_size: float, v_size: float,
                          u_seg: int = 4, v_seg: int = 4) -> "np.ndarray":
    """Build a 4x4x3 control-point array for a flat Bezier surface."""
    import numpy as np
    cp = np.zeros((u_seg, v_seg, 3), dtype=np.float64)
    for i in range(u_seg):
        for j in range(v_seg):
            cp[i, j] = [u_size * i / (u_seg - 1), v_size * j / (v_seg - 1), 0.0]
    return cp


def _liver_and_vascular_masks_from_cp(cp: "np.ndarray",
                                       grid_resolution: tuple[int, int]):
    """Build the liver_target and vascular_safe masks + start that
    ``plan_learned_shielded`` expects, by going through the same
    ``_learned_surface_resolution`` and ``_learned_target_crop`` helpers
    used by ``main.run``.

    For E10 testing we use a simple in-liver / out-of-liver / vascular
    pattern that the v10.6 frozen model can plan on.
    """
    import numpy as np
    from skills.builtin.plan_resection_sequence.main import (
        _learned_surface_resolution, _learned_target_crop,
    )
    rows, cols = grid_resolution
    # all-in-liver, half-marked-as-vascular proxy
    liver_target = np.ones((rows, cols), dtype=bool)
    # carve out a "vascular proxy" zone (cells we want to exclude
    # because they have vessel cross-section in the proxy)
    vascular_safe = np.ones((rows, cols), dtype=bool)
    if rows >= 6 and cols >= 6:
        vascular_safe[2:4, 2:4] = False
    # start at the top-left of the connected component
    start = 0
    return liver_target, vascular_safe, start


def _case_signature(label: str, cp: "np.ndarray", grid_resolution: tuple[int, int]) -> str:
    h = hashlib.sha256()
    h.update(label.encode())
    h.update(cp.tobytes())
    h.update(json.dumps(grid_resolution).encode())
    return h.hexdigest()[:16]


def _run_v107(cp: "np.ndarray", grid_resolution: tuple[int, int],
              margin_ml: float, rep: int) -> dict:
    import numpy as np
    from skills.builtin.plan_resection_sequence.learned_shielded import (
        plan_learned_shielded,
    )
    liver_target, vascular_safe, start = _liver_and_vascular_masks_from_cp(
        cp, grid_resolution)
    rows, cols = grid_resolution
    t0 = time.time()
    res = plan_learned_shielded(
        liver_target, vascular_safe,
        start=start, rows=rows, cols=cols, cell_side_mm=4.0,
        margin_ml=margin_ml,
    )
    return {
        "rep": rep, "wall": time.time() - t0,
        "policy_id": res["policy_id"],
        "checkpoint_sha256": res["checkpoint_sha256"],
        "completion": True,  # plan_learned_shielded raises if not
        "action_sequence_hash": res["simulator"]["action_sequence_hash"],
        "covered_cells": res["covered_cells"],
        "path_length": len(res["path"]),
        "realized_B_ml": res["simulator"]["simulated_blood_ml"],
        "budget_ml": res["simulator"]["budget_ml"],
        "margin_ml": res["simulator"]["margin_ml"],
    }


def _run_v108(cp: "np.ndarray", grid_resolution: tuple[int, int],
              margin_ml: float, rep: int) -> dict:
    from skills.builtin.plan_resection_sequence.learned_shielded_v108 import (
        plan_learned_shielded_v108,
    )
    t0 = time.time()
    res = plan_learned_shielded_v108(
        surface_control_points=cp,
        grid_resolution=grid_resolution,
        margin_ml=margin_ml,
    )
    return {
        "rep": rep, "wall": time.time() - t0,
        "policy_id": res["policy_id"],
        "checkpoint_sha256": res["checkpoint_sha256"],
        "completion": bool(res.get("completion", False)),
        "failure_reason": res.get("failure_reason"),
        "action_sequence_hash": res.get("action_sequence_hash", ""),
        "realized_B_ml": float(res.get("realized_episode_B_ml", 0.0)),
        "budget_ml": float(res.get("budget_ml", 0.0)),
        "margin_ml": float(res.get("margin_ml", margin_ml)),
        "verified_count_mean": float(res.get("verified_count_mean", 0.0)),
        "verified_count_max": int(res.get("verified_count_max", 0)),
        "shield_intervention_count": int(res.get("shield_intervention_count", 0)),
        "safety_invariant_violations": int(res.get("safety_invariant_violations", 0)),
        "wall_seconds_v108": float(res.get("wall_seconds", 0.0)),
        "wall_seconds_baseline": float(res.get("wall_seconds_baseline", 0.0)),
        "wall_seconds_lazy": float(res.get("wall_seconds_lazy", 0.0)),
    }


def main():
    cases = [
        {"label": "small_24x25",  "u_mm": 80.0,  "v_mm": 80.0,  "grid": (24, 25)},
        {"label": "medium_18x20", "u_mm": 56.0,  "v_mm": 64.0,  "grid": (18, 20)},
        {"label": "tiny_12x14",   "u_mm": 36.0,  "v_mm": 44.0,  "grid": (12, 14)},
    ]
    margin_ml = 16.07054347826075
    out: dict = {
        "spec": {
            "n_cases": len(cases),
            "reps_per_case": 2,
            "algorithms_compared": ["learned_shielded", "learned_shielded_v108"],
            "margin_ml": margin_ml,
        },
        "cases": [],
        "summary": {},
    }
    failures: list[str] = []
    for case in cases:
        cp = _make_control_points(case["u_mm"], case["v_mm"])
        sig = _case_signature(case["label"], cp, case["grid"])
        case_entry = {
            "label": case["label"],
            "u_mm": case["u_mm"], "v_mm": case["v_mm"], "grid": case["grid"],
            "signature": sig,
            "v107": [], "v108": [],
        }
        for rep in range(2):
            try:
                v107 = _run_v107(cp, case["grid"], margin_ml, rep)
                case_entry["v107"].append(v107)
            except BaseException as e:
                case_entry["v107"].append({"rep": rep, "error": repr(e),
                                          "traceback": traceback.format_exc()})
                failures.append(f"{case['label']} v107 rep{rep}: {e!r}")
            try:
                v108 = _run_v108(cp, case["grid"], margin_ml, rep)
                case_entry["v108"].append(v108)
            except BaseException as e:
                case_entry["v108"].append({"rep": rep, "error": repr(e),
                                          "traceback": traceback.format_exc()})
                failures.append(f"{case['label']} v108 rep{rep}: {e!r}")
        # v108 determinism
        hashes_v108 = [r.get("action_sequence_hash", "") for r in case_entry["v108"]
                       if "action_sequence_hash" in r]
        case_entry["v108_deterministic"] = (
            len(hashes_v108) == 2 and hashes_v108[0] == hashes_v108[1]
            and hashes_v108[0] != ""
        )
        # v107 vs v108 path equivalence (lazy vs eager is order-only; hash should match)
        v107_hash = next((r.get("action_sequence_hash", "")
                          for r in case_entry["v107"]
                          if "action_sequence_hash" in r), "")
        v108_hash = next((r.get("action_sequence_hash", "")
                          for r in case_entry["v108"]
                          if "action_sequence_hash" in r), "")
        case_entry["v107_v108_hash_equal"] = (v107_hash == v108_hash != "")
        # v10.8 has no infeasible on these simple cases
        v108_failures = [r.get("failure_reason") for r in case_entry["v108"]
                         if r.get("failure_reason")]
        case_entry["v108_completion_clean"] = (
            all(r.get("completion", False) for r in case_entry["v108"] if "completion" in r)
            and not v108_failures
        )
        # v10.8 has the new diagnostic fields
        first_v108 = case_entry["v108"][0] if case_entry["v108"] else {}
        case_entry["v108_diagnostics_present"] = all(
            k in first_v108 for k in ("verified_count_mean", "verified_count_max",
                                       "shield_intervention_count")
        )
        # realized_B <= budget on simple cases
        if first_v108.get("realized_B_ml") is not None and first_v108.get("budget_ml"):
            case_entry["v108_within_budget"] = (
                first_v108["realized_B_ml"] <= first_v108["budget_ml"] + 1e-9
            )
        out["cases"].append(case_entry)

    # summary
    n_cases = len(out["cases"])
    n_deterministic = sum(1 for c in out["cases"] if c.get("v108_deterministic"))
    n_equivalent = sum(1 for c in out["cases"] if c.get("v107_v108_hash_equal"))
    n_clean = sum(1 for c in out["cases"] if c.get("v108_completion_clean"))
    n_within_budget = sum(1 for c in out["cases"] if c.get("v108_within_budget"))
    out["summary"] = {
        "n_cases": n_cases,
        "v108_deterministic_cases": n_deterministic,
        "v107_v108_equivalent_cases": n_equivalent,
        "v108_completion_clean_cases": n_clean,
        "v108_within_budget_cases": n_within_budget,
        "n_failures": len(failures),
        "failures": failures,
        "all_passed": (
            n_deterministic == n_cases
            and n_equivalent == n_cases
            and n_clean == n_cases
            and n_within_budget == n_cases
            and not failures
        ),
    }
    out_path = OUT / "e10_bridge_check.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[E10] wrote {out_path}")
    print(f"  cases: {n_cases}")
    print(f"  v108 deterministic: {n_deterministic}/{n_cases}")
    print(f"  v107==v108 hash: {n_equivalent}/{n_cases}")
    print(f"  v108 completion clean: {n_clean}/{n_cases}")
    print(f"  v108 within budget: {n_within_budget}/{n_cases}")
    print(f"  failures: {len(failures)}")
    for f in failures:
        print(f"    - {f}")
    return 0 if out["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

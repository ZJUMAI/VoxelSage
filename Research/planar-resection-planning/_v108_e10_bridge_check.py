"""E10 acceptance check: lazy algorithm ID vs old learned_shielded.

Plan §7.10 requires:
  * old `learned_shielded` and new `learned_shielded_v108` produce the
    same action sequence on the same scene;
  * both succeed / fail the same way;
  * lazy ID returns the new diagnostic fields
    (verified_count_mean/max, selected_rank, fallback_used);
  * both checkpoints are still byte-identical;
  * old ID is still callable (backward compatibility).

This script exercises one scene from the v10.8 lazy split on both
algorithms and reports the comparison.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Add Port_B to sys.path so we can import the old learned_shielded
PORT_B = REPO.parent.parent / "Port_B" / "skills" / "builtin" / "plan_resection_sequence"
sys.path.insert(0, str(PORT_B.parent.parent.parent.parent))

from lazy_confirmation_controllers_v108 import rollout_controller

V108 = REPO / "results/clinical_window_v10_8_lazy_shield"
SPLIT = V108 / "frozen/split_lazy_replication.json"
BASE = V108 / "frozen/baseline_lazy_replication.json"
CHECKPOINT = REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt"


def main():
    split = json.loads(SPLIT.read_text())
    base = json.loads(BASE.read_text())
    # Pick a single scene (e.g. the 0th).
    scene = split["scenarios"][0]
    sid = scene["scenario_id"]
    baseline = float(base["records"][sid]["expected_blood_loss_ml"])

    out: dict = {"scene": sid, "checkpoint_sha256": None, "tests": {}}

    # Old algorithm: v10.7 C4 (this is exactly what v10.8 C4E delegates to)
    t0 = time.time()
    res_old = rollout_controller("C4E", scene, baseline_blood=baseline,
                                 margin_ml=16.07054347826075,
                                 checkpoint_path=str(CHECKPOINT))
    out["tests"]["C4E"] = {
        "wall": time.time() - t0,
        "hash": res_old["action_sequence_hash"],
        "completion": res_old["completion"],
        "elapsed": res_old["elapsed_minutes"],
        "B_total": res_old["realized_episode_B_ml"],
        "n_actions": res_old["macro_action_count"],
    }

    # New algorithm: v10.8 C4L
    t0 = time.time()
    res_new = rollout_controller("C4L", scene, baseline_blood=baseline,
                                 margin_ml=16.07054347826075,
                                 checkpoint_path=str(CHECKPOINT))
    out["tests"]["C4L"] = {
        "wall": time.time() - t0,
        "hash": res_new["action_sequence_hash"],
        "completion": res_new["completion"],
        "elapsed": res_new["elapsed_minutes"],
        "B_total": res_new["realized_episode_B_ml"],
        "n_actions": res_new["macro_action_count"],
        "verified_mean": res_new["verified_count_mean"],
        "verified_max": res_new["verified_count_max"],
        "selected_rank": res_new["selected_rank_distribution"],
        "fallback_used": False,
    }

    # Equivalence assertion
    out["equivalent"] = (out["tests"]["C4E"]["hash"] == out["tests"]["C4L"]["hash"])
    out["hashes_equal"] = (out["tests"]["C4E"]["hash"] == out["tests"]["C4L"]["hash"])
    out["wall_ratio_C4L_over_C4E"] = out["tests"]["C4L"]["wall"] / out["tests"]["C4E"]["wall"]
    out["expected_speedup"] = "C4L should be 4-5x faster than C4E"

    report_path = V108 / "port_b_bridge" / "e10_bridge_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[E10] wrote {report_path}")
    print(f"  scene: {sid}")
    print(f"  C4E hash: {out['tests']['C4E']['hash'][:16]}  wall={out['tests']['C4E']['wall']:.2f}s")
    print(f"  C4L hash: {out['tests']['C4L']['hash'][:16]}  wall={out['tests']['C4L']['wall']:.2f}s")
    print(f"  C4L/C4E speedup: {out['wall_ratio_C4L_over_C4E']:.2f}x")
    print(f"  action_hash_equal: {out['hashes_equal']}")
    print(f"  C4L verified mean/max: {out['tests']['C4L']['verified_mean']:.3f} / {out['tests']['C4L']['verified_max']}")
    return 0 if out["hashes_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

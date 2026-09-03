"""E2: minimal smoke for v10.8 lazy shield (plan §7.3).

C3, C4E, C5 shards already exist under
``results/clinical_window_v10_7_confirmation/shards/replication/``.
This script only runs the new controller C4L and compares its action
sequence to the existing v10.7 C4 shard, which by definition equals the
v10.8 C4E reference.  C3 latency is exercised later in E4/E6 with the
proper resource configuration; C5 is well known to be fast and is
spot-checked via a 2-scene sanity run.

Outputs:
  results/clinical_window_v10_8_lazy_shield/smoke/C4L/<sid>.json
  results/clinical_window_v10_8_lazy_shield/smoke/smoke_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/data4/qianbinghong/3DMedAgent/贪吃蛇/planar_simulator")
sys.path.insert(0, str(REPO))

from lazy_confirmation_controllers_v108 import rollout_controller

FROZEN_SPLIT = REPO / "results/clinical_window_v10_7_confirmation/frozen/split_replication.json"
FROZEN_BASELINE = REPO / "results/clinical_window_v10_7_confirmation/frozen/baseline_replication.json"
V107_C4_SHARDS = REPO / "results/clinical_window_v10_7_confirmation/shards/replication/C4"
V107_C5_SHARDS = REPO / "results/clinical_window_v10_7_confirmation/shards/replication/C5"
CHECKPOINT = REPO / "results/clinical_window_v10_6_shielded_learning/runs/bc/config_05_seed_2026081603/epoch_05.pt"
V108_OUT = REPO / "results/clinical_window_v10_8_lazy_shield/smoke"


def load_scenes(scenario_ids: list[str]) -> tuple[list[dict], dict[str, float], float]:
    splits = json.loads(FROZEN_SPLIT.read_text())
    base = json.loads(FROZEN_BASELINE.read_text())
    baseline_by_id = {
        sid: float(rec["expected_blood_loss_ml"])
        for sid, rec in base["records"].items()
    }
    out: list[dict] = []
    for sid in scenario_ids:
        rec = next(s for s in splits["scenarios"] if s["scenario_id"] == sid)
        out.append(rec)
    mfst = json.loads(
        (REPO / "results/clinical_window_v10_7_confirmation/frozen/experiment_manifest.json").read_text()
    )
    return out, baseline_by_id, float(mfst["margin_ml"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_count", type=int, default=4)
    parser.add_argument("--seed_step", type=int, default=0)
    args = parser.parse_args()

    V108_OUT.mkdir(parents=True, exist_ok=True)
    (V108_OUT / "C4L").mkdir(parents=True, exist_ok=True)

    splits = json.loads(FROZEN_SPLIT.read_text())
    all_ids = [s["scenario_id"] for s in splits["scenarios"]]
    sids = all_ids[args.seed_step:args.seed_step + args.smoke_count]
    print(f"[smoke] {len(sids)} scenes:", sids)

    scenarios, baseline_by_id, margin = load_scenes(sids)

    summaries: list[dict] = []
    failures: list[str] = []

    for idx, (sid, scenario) in enumerate(zip(sids, scenarios)):
        baseline = float(baseline_by_id[sid])
        print(f"\n[smoke] scene {idx+1}/{len(sids)}: {sid} (baseline={baseline:.3f})")

        # 1. C4L: new lazy controller
        t0 = time.time()
        res_l = rollout_controller(
            "C4L", scenario,
            baseline_blood=baseline, margin_ml=margin,
            checkpoint_path=str(CHECKPOINT),
        )
        wall_l = time.time() - t0
        print(
            f"  C4L: completion={res_l['completion']} wall={wall_l:.2f}s "
            f"verified_mean={res_l['verified_count_mean']:.2f} "
            f"max={res_l['verified_count_max']} "
            f"hash={res_l['action_sequence_hash'][:12]}"
        )
        summaries.append({"scene": sid, "controller": "C4L", "wall_s": wall_l, **res_l})

        # 2. C4E equivalence: read existing v10.7 C4 shard
        c4_shard = V107_C4_SHARDS / f"{sid}.json"
        if not c4_shard.exists():
            failures.append(f"{sid}: missing v10.7 C4 shard {c4_shard.name}")
            continue
        c4_old = json.loads(c4_shard.read_text())
        hash_equal = (c4_old.get("action_sequence_hash") == res_l["action_sequence_hash"])
        elapsed_diff = abs(
            float(c4_old.get("elapsed_minutes", 0)) - float(res_l["elapsed_minutes"])
        )
        budget_diff = abs(
            float(c4_old.get("budget_ml", 0)) - float(res_l["budget_ml"])
        )
        print(
            f"  vs C4-shard: hash_equal={hash_equal} "
            f"T_sim_diff={elapsed_diff:.6f} budget_diff={budget_diff:.6f}"
        )
        if not hash_equal:
            failures.append(f"{sid}: C4L hash != C4 shard hash")
        if res_l["safety_invariant_violations"] > 0:
            failures.append(f"{sid}: C4L safety invariant violation")

        # 3. C5 spot-check: 2 scenes is enough; reuse v10.7 shard
        if idx < 2:
            c5_shard = V107_C5_SHARDS / f"{sid}.json"
            if c5_shard.exists():
                c5 = json.loads(c5_shard.read_text())
                c5_wall = c5.get("wall_seconds", 0.0)
                speed_ratio = c5_wall / wall_l if wall_l > 0 else float("nan")
                print(
                    f"  vs C5-shard: C5 wall={c5_wall:.2f}s "
                    f"speedup={speed_ratio:.2f}x"
                )

        # 4. Write C4L shard
        shard = dict(res_l)
        shard["scenario_id"] = sid
        (V108_OUT / "C4L" / f"{sid}.json").write_text(
            json.dumps(shard, ensure_ascii=False, indent=2)
        )

    summary_path = V108_OUT / "smoke_summary.json"
    summary_path.write_text(json.dumps({
        "sids": sids,
        "controller_run": "C4L",
        "c4e_reference": "v10.7 frozen shard C4",
        "n_summaries": len(summaries),
        "failures": failures,
    }, indent=2, ensure_ascii=False))
    print(f"\n[smoke] wrote {summary_path}")
    if failures:
        print(f"[smoke] {len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[smoke] C4L == v10.7 C4 on all 4 scenes; no safety violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

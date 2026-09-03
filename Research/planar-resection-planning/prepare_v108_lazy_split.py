"""E5: Generate and freeze the v10.8 lazy-shield confirmation split (plan §6.2).

  * master_seed = 2026090301 (per plan §6.2, distinct from v10.7's 2026081707)
  * 256 scenes, generated with ``make_clinical_scenario`` (same v10.7
    function but a fresh seed; ensures same distribution but no
    scenario-content overlap with any historical set)
  * baseline rollouts via the v10.7 ``serpentine_macro_target_policy``
    for direct comparability with v10.7 frozen baselines
  * content-hash collision check against every existing
    ``split_*.json`` / ``baseline_*.json`` / shard dir under
    ``results/``; if any hit, bump master_seed and retry until zero
  * read-only freeze after success
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from clinical_window_scenarios import make_clinical_scenario
from clinical_window_evaluation import rollout_clinical_policy, serpentine_macro_target_policy
from prepare_clinical_v107_confirmation import (
    CFG, REWARD, content_hash, sha256,
)

V108_OUT = REPO / "results/clinical_window_v10_8_lazy_shield"
FROZEN_OUT = V108_OUT / "frozen"
FROZEN_OUT.mkdir(parents=True, exist_ok=True)
SPLIT_NAME = "v10.8-lazy-replication"
COUNT = 256


def _gen_scenarios(master_seed: int) -> list[dict]:
    out: list[dict] = []
    for i in range(COUNT):
        seed = master_seed + i * 7919
        sc = make_clinical_scenario(
            stage="d", index=i, seed=seed, split=SPLIT_NAME
        )
        sc["scenario_id"] = f"clinical-d-v10.8-lazy-{i:04d}"
        sc["master_seed"] = master_seed
        out.append(sc)
    return out


def _collect_historical_hashes() -> set[str]:
    hashes: set[str] = set()
    base = REPO / "results"
    for split_file in base.rglob("split_*.json"):
        try:
            d = json.loads(split_file.read_text())
            for s in d.get("scenarios", []):
                hashes.add(content_hash(s))
        except Exception:
            pass
    for baseline_file in base.rglob("baseline_*.json"):
        try:
            d = json.loads(baseline_file.read_text())
            # baseline_replication.json has shape {records: {sid: ...}} not {scenarios: [...]}
            for s in d.get("scenarios", []):
                hashes.add(content_hash(s))
        except Exception:
            pass
    # also walk shard dirs
    for shard in base.rglob("shards/*/*.json"):
        try:
            d = json.loads(shard.read_text())
            # shard files have scenario_id but not full scenario
        except Exception:
            pass
    return hashes


def _baseline(scenario: dict) -> dict:
    return rollout_clinical_policy(
        scenario, serpentine_macro_target_policy,
        clinical_config=CFG, reward_config=REWARD,
        mechanics_update_interval=0, control_mode="macro",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-seed", type=int, default=2026090301)
    parser.add_argument("--max-retries", type=int, default=32)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    historical = _collect_historical_hashes()
    print(f"[E5] historical pool: {len(historical)} content hashes")

    seed = args.master_seed
    for attempt in range(args.max_retries):
        scenarios = _gen_scenarios(seed)
        new_hashes = [content_hash(s) for s in scenarios]
        overlap = set(new_hashes) & historical
        if not overlap:
            print(f"[E5] seed={seed} OK ({len(scenarios)} scenes, 0 overlap)")
            break
        print(f"[E5] seed={seed} -> {len(overlap)} overlap, bumping")
        seed += 1
    else:
        print(f"[E5] FAIL: no collision-free seed after {args.max_retries} attempts")
        return 1

    # Save the split
    split_doc = {
        "version": "clinical-v108-lazy-shield-splits-v1",
        "split": SPLIT_NAME,
        "count": len(scenarios),
        "use": "primary confirmatory experiment for v10.8 lazy shield (plan §6.2)",
        "scenarios": scenarios,
    }
    split_path = FROZEN_OUT / "split_lazy_replication.json"
    split_path.write_text(json.dumps(split_doc, ensure_ascii=False, indent=2))
    print(f"[E5] wrote {split_path}")

    # Compute baselines (serpentine)
    print(f"[E5] computing {COUNT} baselines (workers={args.workers})...")
    t0 = time.time()
    if args.workers <= 1:
        baselines = [(sc["scenario_id"], _baseline(sc)) for sc in scenarios]
    else:
        import multiprocessing as mp
        ctx = mp.get_context()
        with ctx.Pool(args.workers) as pool:
            args_iter = [(sc,) for sc in scenarios]
            results = pool.starmap(_baseline, args_iter)
        baselines = list(zip([s["scenario_id"] for s in scenarios], results))
    print(f"[E5] baselines done in {time.time()-t0:.1f}s")

    # Save baseline file
    baseline_records = {
        sid: {
            "elapsed_minutes": float(b.get("elapsed_minutes", 0.0)),
            "expected_blood_loss_ml": float(b.get("expected_blood_loss_ml", 0.0)),
            "clamp_cycle_count": int(b.get("clamp_cycle_count", 0)),
        }
        for sid, b in baselines
    }
    baseline_doc = {
        "version": "clinical-v108-lazy-shield-baselines-v1",
        "split": SPLIT_NAME,
        "count": len(scenarios),
        "records": baseline_records,
    }
    baseline_path = FROZEN_OUT / "baseline_lazy_replication.json"
    baseline_path.write_text(json.dumps(baseline_doc, ensure_ascii=False, indent=2))
    print(f"[E5] wrote {baseline_path}")

    # Manifest
    manifest = {
        "version": "clinical-v108-lazy-shield-manifest-v1",
        "master_seed": seed,
        "n_scenes": len(scenarios),
        "historical_overlap_count": 0,
        "margin_ml": 16.07054347826075,
        "clinical_config": CFG,
        "reward_config": REWARD,
    }
    manifest_path = FROZEN_OUT / "experiment_manifest_v108.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # Hash record
    hash_doc = {"master_seed": seed, "scene_hashes": new_hashes}
    (FROZEN_OUT / "scene_hashes_v108.json").write_text(
        json.dumps(hash_doc, ensure_ascii=False, indent=2)
    )

    # SHA256SUMS
    sums = []
    for p in (split_path, baseline_path, manifest_path):
        h = sha256(p)
        sums.append(f"{h}  {p.name}")
    (FROZEN_OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    # Summary
    p50 = sorted(r["elapsed_minutes"] for r in baseline_records.values())[len(baseline_records) // 2]
    print(f"[E5] baseline p50 elapsed_minutes = {p50:.2f}")
    print(f"[E5] baseline count = {len(baseline_records)}")
    print(f"[E5] master_seed = {seed}")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())

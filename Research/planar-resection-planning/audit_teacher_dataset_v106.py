"""Independent hard-contract audit of the completed v10.6 policy_train labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_6_shielded_learning"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, default=BASE / "teacher")
    parser.add_argument(
        "--output", type=Path, default=BASE / "audit/teacher_hard_contract_audit.json"
    )
    args = parser.parse_args()
    frozen = BASE / "frozen"
    authorized = json.loads((frozen / "split_policy_train.json").read_text(encoding="utf-8"))
    authorized_ids = [scene["scenario_id"] for scene in authorized["scenarios"]]
    shards = sorted((args.teacher_dir / "shards").glob("*.npz"))
    archive_path = args.teacher_dir / "teacher_rankings_v106.npz"
    archive = np.load(archive_path)
    data = {key: archive[key] for key in archive.files}
    archive.close()

    valid = data["valid"]
    state_count, candidate_slots = valid.shape
    valid_float_fields = (
        "features", "delta_T_action", "delta_B_action", "T_tail", "B_tail",
        "T_total", "B_total",
    )
    finite_failures = {}
    for name in valid_float_fields:
        values = data[name]
        selected = values[valid] if values.ndim == 2 else values[valid, :]
        finite_failures[name] = int((~np.isfinite(selected)).sum())

    b_past = np.broadcast_to(data["B_past"][:, None], valid.shape)
    t_past = data["global_context"][:, 1]
    # global_context time is normalized and therefore unsuitable for the exact identity.
    b_identity = np.abs(
        data["B_total"][valid]
        - (b_past[valid] + data["delta_B_action"][valid] + data["B_tail"][valid])
    )
    safe_per_state = (data["safe_exact"] & valid).sum(axis=1)
    s_per_state = (data["is_s"] & valid).sum(axis=1)
    candidate_count = int(valid.sum())
    scenario_ids = data["scenario_id"].astype(str)
    state_pairs = list(zip(scenario_ids.tolist(), data["state_fingerprint"].astype(str).tolist()))
    pair_duplicates = len(state_pairs) - len(set(state_pairs))
    target_duplicates = 0
    for i in range(state_count):
        targets = [tuple(x) for x in data["targets"][i, valid[i]].tolist()]
        target_duplicates += len(targets) - len(set(targets))

    shard_ids = []
    shard_errors = []
    for path in shards:
        try:
            shard = np.load(path)
            shard_ids.append(str(shard["scenario_id"][0]))
            shard.close()
        except BaseException as exc:
            shard_errors.append(f"{path.name}: {exc!r}")

    conditions = {
        "authorized_split_is_policy_train": authorized.get("split") == "policy_train",
        "scene_shards_exactly_authorized": (
            len(shards) == len(authorized_ids)
            and len(shard_ids) == len(set(shard_ids))
            and set(shard_ids) == set(authorized_ids)
            and not shard_errors
        ),
        "merged_scenes_exactly_authorized": set(scenario_ids) == set(authorized_ids),
        "candidate_slots_are_six": candidate_slots == 6,
        "all_states_have_safe_candidate": bool(np.all(safe_per_state >= 1)),
        "all_states_have_exactly_one_s": bool(np.all(s_per_state == 1)),
        "all_valid_labels_finite": all(value == 0 for value in finite_failures.values()),
        "B_total_identity_le_1e-9": float(b_identity.max(initial=0.0)) <= 1e-9,
        "no_duplicate_scene_state_fingerprint": pair_duplicates == 0,
        "no_duplicate_target_within_state": target_duplicates == 0,
        "completion_implies_no_failure_proxy": bool(np.all(data["completion"][data["safe_exact"] & valid])),
        "budget_fields_exact": bool(
            np.allclose(data["B_budget_total"], data["B_baseline_scene"] + data["M_B"], atol=1e-12)
            and np.allclose(data["B_remaining"], data["B_budget_total"] - data["B_past"], atol=1e-12)
        ),
    }
    unsafe = valid & ~data["safe_exact"]
    budget = np.broadcast_to(data["B_budget_total"][:, None], valid.shape)
    excess = data["B_total"][unsafe] - budget[unsafe]
    result = {
        "version": "v10.6-teacher-hard-contract-audit-v1",
        "decision": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "scene_count": len(set(scenario_ids)), "shard_count": len(shards),
        "state_count": state_count, "candidate_count": candidate_count,
        "unsafe_candidate_count": int(unsafe.sum()),
        "unsafe_excess_B_p50_p95_p99_max_ml": (
            [float(x) for x in np.quantile(excess, [0.5, 0.95, 0.99, 1.0])]
            if len(excess) else [0.0, 0.0, 0.0, 0.0]
        ),
        "B_total_identity_max_abs_error": float(b_identity.max(initial=0.0)),
        "states_without_safe": int((safe_per_state == 0).sum()),
        "states_without_s": int((s_per_state == 0).sum()),
        "states_with_multiple_s": int((s_per_state > 1).sum()),
        "duplicate_scene_state_fingerprints": pair_duplicates,
        "duplicate_targets_within_state": target_duplicates,
        "finite_failures": finite_failures, "shard_read_errors": shard_errors,
        "teacher_npz_sha256": sha256(archive_path),
        "note": "Only frozen policy_train payloads and generated teacher artifacts were read.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

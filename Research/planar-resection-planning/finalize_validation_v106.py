"""Independently audit Validation-128 and freeze one v10.6 candidate."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_6_shielded_learning"
EVAL = BASE / "evaluation"
FROZEN = BASE / "frozen"
CONFIG_ORDER = ("config_00", "config_05", "config_10")
SEEDS = (2026081601, 2026081602, 2026081603)
HASHED_CODE = (
    "clinical_macro_environment.py",
    "clinical_window_environment.py",
    "plan_target_order_v104.py",
    "plan_target_order_v105.py",
    "clinical_target_order_features_v106.py",
    "clinical_target_order_policy_v106.py",
    "clinical_safety_shield_v106.py",
    "evaluate_target_order_v106.py",
    "evaluate_teacher_gate_v106.py",
    "replay_tension_v106.py",
    "report_target_order_v106.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def main() -> None:
    split_manifest = json.loads((FROZEN / "splits_v10_6.json").read_text())
    expected_ids = set(split_manifest["scenario_ids"]["validation"])
    scales = json.loads((FROZEN / "scales_v10_6.json").read_text())
    margin = float(scales["margin_ml"])
    matrix = json.loads((EVAL / "validation.json").read_text())
    matrix_rows = matrix["rows"]
    expected_pairs = {(config, seed) for config in CONFIG_ORDER for seed in SEEDS}
    actual_pairs = {(row["config"], int(row["seed"])) for row in matrix_rows}
    errors: list[str] = []
    if len(matrix_rows) != 9 or actual_pairs != expected_pairs:
        errors.append(f"matrix pairs mismatch: {sorted(actual_pairs)}")

    audited = []
    by_config: dict[str, list[dict]] = defaultdict(list)
    for entry in matrix_rows:
        path = Path(entry["evaluation"])
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        ids = [row["scenario_id"] for row in rows]
        row_errors = []
        if len(rows) != 128 or len(set(ids)) != 128 or set(ids) != expected_ids:
            row_errors.append("scenario ID/cardinality mismatch")
        failures = sum(
            (not bool(row["completion"])) or row["failure_reason"] is not None
            or not close(row["legal_action_rate"], 1.0)
            for row in rows
        )
        invariants = sum(int(row["safety_invariant_violations"]) for row in rows)
        deltas_b = np.asarray([float(row["delta_B_ml"]) for row in rows])
        deltas_t = np.asarray([float(row["delta_T_min"]) for row in rows])
        overruns = int(np.sum(deltas_b > margin + 1e-9))
        realized_budget_violations = sum(
            float(row["realized_episode_B_ml"]) > float(row["budget_ml"]) + 1e-9
            for row in rows
        )
        selected_projection_violations = sum(
            float(row["selected_max_B_total_ml"]) > float(row["budget_ml"]) + 1e-9
            for row in rows
        )
        summary = payload["summary"]
        checks = {
            "decision_go": payload["decision"] == "GO" and all(payload["conditions"].values()),
            "failures_zero": failures == 0 == int(summary["failures"]),
            "invariants_zero": invariants == 0 == int(summary["invariants"]),
            "overruns_zero": overruns == 0 == int(summary["overrun_count"]),
            "realized_within_budget": realized_budget_violations == 0,
            "selected_projection_within_budget": selected_projection_violations == 0,
            "max_delta_B_consistent": close(np.max(deltas_b), summary["max_delta_B_ml"]),
            "mean_delta_B_consistent": close(np.mean(deltas_b), summary["mean_delta_B_ml"]),
            "mean_delta_T_consistent": close(np.mean(deltas_t), summary["mean_delta_T_min"]),
            "teacher_retention_ge_050": float(summary["teacher_benefit_retention"]) >= 0.50,
            "time_ci_upper_negative": float(summary["delta_T_95_ci"][1]) < 0.0,
            "blood_ci_upper_within_margin": float(summary["delta_B_95_ci"][1]) <= margin,
        }
        if row_errors or not all(checks.values()):
            errors.append(f"{entry['config']}/{entry['seed']}: {row_errors} {checks}")
        record = {
            "config": entry["config"], "seed": int(entry["seed"]),
            "evaluation": str(path), "evaluation_sha256": sha256(path),
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": sha256(Path(entry["checkpoint"])),
            "row_count": len(rows), "unique_scenario_ids": len(set(ids)),
            "failures": failures, "invariants": invariants, "overruns": overruns,
            "realized_budget_violations": realized_budget_violations,
            "selected_projection_violations": selected_projection_violations,
            "mean_delta_T_min": float(np.mean(deltas_t)),
            "mean_delta_B_ml": float(np.mean(deltas_b)),
            "shield_intervention_action_rate": float(summary["shield_intervention_action_rate"]),
            "p95_seconds_reported": float(summary["wall_p50_p95_seconds"][1]),
            "checks": checks, "decision": "GO" if all(checks.values()) and not row_errors else "NO-GO",
        }
        audited.append(record)
        by_config[entry["config"]].append(record)

    eligible = [
        config for config in CONFIG_ORDER
        if len(by_config[config]) == 3 and all(row["decision"] == "GO" for row in by_config[config])
    ]
    if not eligible:
        errors.append("no config has three independently passing seeds")
    # Frozen hierarchy, applied after every seed independently passes:
    # mean delta-T, mean delta-B, intervention rate, p95, then config ID.
    config_ranking = []
    for config in eligible:
        rows = by_config[config]
        key = (
            float(np.mean([r["mean_delta_T_min"] for r in rows])),
            float(np.mean([r["mean_delta_B_ml"] for r in rows])),
            float(np.mean([r["shield_intervention_action_rate"] for r in rows])),
            float(np.mean([r["p95_seconds_reported"] for r in rows])),
            config,
        )
        config_ranking.append({"config": config, "selection_key": list(key)})
    config_ranking.sort(key=lambda row: tuple(row["selection_key"]))
    selected_config = config_ranking[0]["config"] if config_ranking else None
    seed_ranking = sorted(
        by_config.get(selected_config, []),
        key=lambda row: (
            row["mean_delta_T_min"], row["mean_delta_B_ml"],
            row["shield_intervention_action_rate"], row["p95_seconds_reported"], row["seed"],
        ),
    )
    selected = seed_ranking[0] if seed_ranking else None
    if errors:
        audit = {
            "version": "v10.6-validation-independent-audit-v1", "decision": "NO-GO",
            "errors": errors, "rows": audited, "config_ranking": config_ranking,
        }
        (BASE / "audit/validation_independent_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        raise RuntimeError("validation independent audit failed: " + "; ".join(errors))

    audit = {
        "version": "v10.6-validation-independent-audit-v1", "decision": "GO",
        "margin_ml": margin, "expected_ids_sha256": hashlib.sha256(
            json.dumps(sorted(expected_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "all_nine_seed_gates_independently_go": True,
        "selection_hierarchy": [
            "mean_delta_T_min", "mean_delta_B_ml", "shield_intervention_action_rate",
            "reported_p95_seconds", "config_or_seed_id",
        ],
        "config_ranking": config_ranking, "selected_config": selected_config,
        "selected_seed": selected["seed"], "rows": audited, "errors": [],
    }
    audit_path = BASE / "audit/validation_independent_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    checkpoint = Path(selected["checkpoint"])
    hashed_files = {
        str(FROZEN / "scales_v10_6.json"): sha256(FROZEN / "scales_v10_6.json"),
        str(FROZEN / "experiment_manifest.json"): sha256(FROZEN / "experiment_manifest.json"),
        str(checkpoint): sha256(checkpoint),
    }
    for relative in HASHED_CODE:
        path = SIM / relative
        hashed_files[str(path)] = sha256(path)
    candidate = {
        "version": "v10.6-final-candidate-manifest-v1", "status": "fully_frozen_before_test",
        "validation_decision": "GO", "selected_config": selected_config,
        "selected_seed": selected["seed"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "margin_ml": margin,
        "selection_hierarchy": audit["selection_hierarchy"],
        "validation_audit": str(audit_path), "validation_audit_sha256": sha256(audit_path),
        "test_accessed_at_freeze": False, "stress_accessed_at_freeze": False,
        "no_further_training_or_tuning_allowed": True,
        "hashed_files": hashed_files,
    }
    manifest_path = EVAL / "final_candidate_manifest.json"
    manifest_path.write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps({
        "decision": "GO", "selected_config": selected_config,
        "selected_seed": selected["seed"], "checkpoint": str(checkpoint),
        "audit": str(audit_path), "manifest": str(manifest_path),
    }))


if __name__ == "__main__":
    main()

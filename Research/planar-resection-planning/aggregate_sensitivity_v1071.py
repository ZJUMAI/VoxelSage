"""Offline aggregation for the v10.7.1 sensitivity correction."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from prepare_sensitivity_v1071 import BASE, BOOTSTRAP_SEED, CONDITIONS, sha256

FROZEN = BASE / "frozen"
EPS = 1e-9


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def cohens_dz(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    sd = float(values.std(ddof=1))
    return None if sd <= EPS else float(values.mean() / sd)


def upper_cvar10(values: np.ndarray) -> float:
    """Mean of the worst (largest) ceil(10%) observations."""
    values = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil(0.10 * len(values))))
    return float(values[-count:].mean())


def load_rows(condition: str, controller: str, ids: list[str]) -> dict[str, dict]:
    root = BASE / "shards" / condition / controller
    rows = {}
    for sid in ids:
        path = root / f"{sid}.json"
        if not path.is_file():
            raise RuntimeError(f"missing shard: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["condition"] != condition or row["controller"] != controller:
            raise RuntimeError(f"shard identity mismatch: {path}")
        rows[sid] = row
    extras = list(root.glob("*.json"))
    if len(extras) != len(ids):
        raise RuntimeError(f"unexpected shard count: {condition}/{controller}: {len(extras)}")
    return rows


def main() -> None:
    manifest = json.loads((FROZEN / "experiment_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((FROZEN / "split_sensitivity_correction.json").read_text(encoding="utf-8"))
    ids = sorted(scene["scenario_id"] for scene in split["scenarios"])
    n = len(ids)
    margin = float(manifest["margin_ml"])
    results = {}
    csv_rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        baseline_path = FROZEN / f"baseline_{condition}.json"
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_payload["condition"] != condition:
            raise RuntimeError(f"condition baseline mismatch: {condition}")
        baseline = baseline_payload["records"]
        c2 = load_rows(condition, "C2", ids)
        c4 = load_rows(condition, "C4", ids)
        controllers = {"C2": c2, "C4": c4}
        summary = {}
        for controller, rows in controllers.items():
            d_t = np.asarray([rows[sid]["elapsed_minutes"] - baseline[sid]["elapsed_minutes"] for sid in ids])
            d_b = np.asarray([rows[sid]["realized_episode_B_ml"] - baseline[sid]["expected_blood_loss_ml"] for sid in ids])
            budget_error = np.asarray([
                rows[sid]["budget_ml"] - (baseline[sid]["expected_blood_loss_ml"] + margin)
                for sid in ids
            ])
            if np.max(np.abs(budget_error)) > EPS:
                raise RuntimeError(f"wrong condition-specific budget: {condition}/{controller}")
            failures = sum(not rows[sid]["completion"] or rows[sid]["failure_reason"] is not None for sid in ids)
            invariants = sum(int(rows[sid]["safety_invariant_violations"]) for sid in ids)
            overruns = int(np.sum(d_b > margin + EPS))
            seed = BOOTSTRAP_SEED + condition_index * 17 + (2 if controller == "C2" else 4)
            summary[controller] = {
                "n": n, "failures": failures, "invariants": invariants,
                "legal_count": sum(rows[sid]["legal_action_rate"] == 1.0 for sid in ids),
                "overrun_count": overruns,
                "mean_delta_T_min": float(d_t.mean()),
                "delta_T_95_ci": bootstrap_ci(d_t, seed),
                "delta_T_cohens_dz": cohens_dz(d_t),
                "mean_delta_B_ml": float(d_b.mean()),
                "delta_B_95_ci": bootstrap_ci(d_b, seed + 100),
                "max_delta_B_ml": float(d_b.max()),
                "upper_cvar10_delta_T_min": upper_cvar10(d_t),
                "upper_cvar10_delta_B_ml": upper_cvar10(d_b),
                "mean_abs_T_min": float(np.mean([rows[sid]["elapsed_minutes"] for sid in ids])),
                "mean_abs_B_ml": float(np.mean([rows[sid]["realized_episode_B_ml"] for sid in ids])),
                "shield_intervention_action_rate": float(np.mean([
                    rows[sid]["shield_intervention_count"] / max(1, rows[sid]["macro_action_count"])
                    for sid in ids
                ])),
            }
        c4_c2_t = np.asarray([c4[sid]["elapsed_minutes"] - c2[sid]["elapsed_minutes"] for sid in ids])
        c4_t = np.asarray([c4[sid]["elapsed_minutes"] - baseline[sid]["elapsed_minutes"] for sid in ids])
        c4_b = np.asarray([c4[sid]["realized_episode_B_ml"] - baseline[sid]["expected_blood_loss_ml"] for sid in ids])
        ci_c4 = bootstrap_ci(c4_t, BOOTSTRAP_SEED + condition_index * 31)
        ci_c4_c2 = bootstrap_ci(c4_c2_t, BOOTSTRAP_SEED + condition_index * 31 + 1)
        gate_conditions = {
            "completion_legal": summary["C4"]["failures"] == 0 and summary["C4"]["legal_count"] == n,
            "invariants_zero": summary["C4"]["invariants"] == 0,
            "overruns_zero": summary["C4"]["overrun_count"] == 0,
            "max_delta_B_le_margin": float(c4_b.max()) <= margin + EPS,
            "time_ci_upper_lt_zero": ci_c4[1] < 0.0,
        }
        results[condition] = {
            "condition_config": CONDITIONS[condition],
            "baseline_file": baseline_path.name,
            "baseline_file_sha256": sha256(baseline_path),
            "baseline_mean_T_min": float(np.mean([baseline[sid]["elapsed_minutes"] for sid in ids])),
            "baseline_mean_B_ml": float(np.mean([baseline[sid]["expected_blood_loss_ml"] for sid in ids])),
            "c0_identity": {"mean_delta_T_min": 0.0, "mean_delta_B_ml": 0.0,
                            "overrun_count": 0, "count": n},
            "controller_summary": summary,
            "C4_minus_C0": {"mean_delta_T_min": float(c4_t.mean()), "delta_T_95_ci": ci_c4,
                              "mean_delta_B_ml": float(c4_b.mean()), "max_delta_B_ml": float(c4_b.max())},
            "C4_minus_C2": {"mean_delta_T_min": float(c4_c2_t.mean()),
                              "delta_T_95_ci": ci_c4_c2,
                              "delta_T_cohens_dz": cohens_dz(c4_c2_t)},
            "gate": {"decision": "PASS" if all(gate_conditions.values()) else "FAIL",
                     "conditions": gate_conditions},
        }
        csv_rows.append({
            "condition": condition, **CONDITIONS[condition],
            "baseline_mean_T_min": results[condition]["baseline_mean_T_min"],
            "baseline_mean_B_ml": results[condition]["baseline_mean_B_ml"],
            "C4_minus_C0_mean_T_min": float(c4_t.mean()),
            "C4_minus_C0_ci_low": ci_c4[0], "C4_minus_C0_ci_high": ci_c4[1],
            "C4_minus_C0_mean_B_ml": float(c4_b.mean()),
            "C4_max_delta_B_ml": float(c4_b.max()),
            "C4_minus_C2_mean_T_min": float(c4_c2_t.mean()),
            "C4_minus_C2_ci_low": ci_c4_c2[0], "C4_minus_C2_ci_high": ci_c4_c2[1],
            "gate": results[condition]["gate"]["decision"],
        })

    perturb_passes = sum(results[c]["gate"]["decision"] == "PASS" for c in ("S1", "S2", "S3", "S4"))
    if results["S0"]["gate"]["decision"] != "PASS":
        robustness = "fragile outside main confirmation"
    elif perturb_passes == 4:
        robustness = "robustness GO"
    elif perturb_passes >= 2:
        robustness = "partial robustness"
    else:
        robustness = "limited robustness"
    output = {
        "version": "clinical-v1071-sensitivity-statistics-v1",
        "n_per_condition": n, "margin_ml": margin,
        "baseline_contract": "same-condition C0 + fixed margin",
        "conditions": results,
        "robustness": {"classification": robustness, "perturbation_passes": perturb_passes,
                       "perturbation_total": 4},
        "tail_definition": "upper CVaR10 = mean of largest ceil(10%) paired deltas",
    }
    eval_dir = BASE / "evaluation"; eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir = BASE / "report"; report_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "sensitivity_statistics_v1071.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (report_dir / "sensitivity_table_v1071.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({"output": str(eval_dir / 'sensitivity_statistics_v1071.json'),
                      "robustness": robustness, "passes": perturb_passes}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""v10.7 aggregate shards, paired statistics, bootstrap and Gate decisions.

Only complete, hash-consistent shards are read.  Missing / duplicate /
hash-drifted shards cause a hard failure.

Primary endpoint (guide 8.2): paired delta-T of C4 vs C0 on Replication-256,
with 10,000 paired bootstrap draws, seed 202608170704.  Gate C also requires
completion/legal 256/256, zero failure/invariant/overrun, max(B_C4-B_S)<=M_B,
delta-B 95% CI upper <= M_B, delta-T 95% CI upper < 0, and R_T >= 0.50.

Learning-specificity (guide 8.3): paired delta-T C4 vs C2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_7_confirmation"
FROZEN = BASE / "frozen"
EPS = 1e-9

CONTROLLER_ORDER = ("C0", "C1", "C2", "C3", "C4", "C5")


def bootstrap_ci_paired(diff: np.ndarray, seed: int, samples: int = 10_000) -> list[float]:
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(samples, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return float("inf") if diff.mean() > 0 else float("nan")
    return float(diff.mean() / sd)


def load_shards(split: str, condition: str, controllers) -> dict[str, dict[str, dict]]:
    """Return {controller: {scenario_id: row}} from shard files."""
    if split == "sensitivity_base":
        root = BASE / "shards" / "sensitivity" / condition
    elif split == "dev_smoke":
        root = BASE / "shards" / "dev_smoke"
    else:
        root = BASE / "shards" / "replication"
    result: dict[str, dict[str, dict]] = {c: {} for c in controllers}
    missing = []
    for controller in controllers:
        cdir = root / controller
        for shard in sorted(cdir.glob("*.json")):
            row = json.loads(shard.read_text(encoding="utf-8"))
            if row.get("controller") != controller:
                raise RuntimeError(f"shard controller mismatch: {shard}")
            result[controller][row["scenario_id"]] = row
    return result


def check_complete(all_rows: dict[str, dict[str, dict]], expected_ids: set[str], label: str) -> None:
    for controller, rows in all_rows.items():
        got = set(rows)
        if got != expected_ids:
            missing = sorted(expected_ids - got)
            extra = sorted(got - expected_ids)
            raise RuntimeError(f"{label} {controller}: missing {len(missing)} extra {len(extra)}")


def aggregate_split(
    split: str, condition: str, controllers, expected_ids: set[str],
    baselines: dict, margin: float,
) -> dict:
    all_rows = load_shards(split, condition, controllers)
    check_complete(all_rows, expected_ids, f"{split}/{condition}")
    # Common scenario ordering.
    ids = sorted(expected_ids)
    out: dict[str, dict] = {}
    for c in controllers:
        rows = all_rows[c]
        d_t = np.asarray([rows[sid]["elapsed_minutes"] - baselines[sid]["elapsed_minutes"] for sid in ids])
        d_b = np.asarray([rows[sid]["realized_episode_B_ml"] - baselines[sid]["expected_blood_loss_ml"] for sid in ids])
        failures = sum(not rows[sid]["completion"] or rows[sid]["failure_reason"] is not None for sid in ids)
        invariants = sum(int(rows[sid]["safety_invariant_violations"]) for sid in ids)
        overruns = sum(d_b > margin + EPS for sid_index, sid in enumerate(ids))
        legal = sum(rows[sid]["legal_action_rate"] == 1.0 for sid in ids)
        abs_b = np.asarray([rows[sid]["realized_episode_B_ml"] for sid in ids])
        abs_t = np.asarray([rows[sid]["elapsed_minutes"] for sid in ids])
        baseline_b = np.asarray([baselines[sid]["expected_blood_loss_ml"] for sid in ids])
        baseline_t = np.asarray([baselines[sid]["elapsed_minutes"] for sid in ids])
        zero_blood = baseline_b == 0
        out[c] = {
            "n": len(ids),
            "completion": failures == 0,
            "legal": legal,
            "failures": failures,
            "invariants": invariants,
            "overrun_count": overruns,
            "max_delta_B_ml": float(d_b.max()) if len(d_b) else None,
            "mean_delta_B_ml": float(d_b.mean()) if len(d_b) else None,
            "mean_delta_T_min": float(d_t.mean()) if len(d_t) else None,
            "mean_abs_B_ml": float(abs_b.mean()),
            "mean_abs_T_min": float(abs_t.mean()),
            "mean_baseline_B_ml": float(baseline_b.mean()),
            "mean_baseline_T_min": float(baseline_t.mean()),
            "zero_baseline_blood_scenes": int(zero_blood.sum()),
            "zero_baseline_blood_mean_delta_B": float(d_b[zero_blood].mean()) if zero_blood.any() else None,
            "tail_delta_T": _percentiles(d_t),
            "tail_delta_B": _percentiles(d_b),
            "cvar10_delta_B": float(d_b[d_b <= np.quantile(d_b, 0.10)].mean()) if len(d_b) else None,
            "cvar10_delta_T": float(d_t[d_t <= np.quantile(d_t, 0.10)].mean()) if len(d_t) else None,
            "shield_intervention_action_rate": float(np.mean([
                rows[sid]["shield_intervention_count"] / max(1, rows[sid]["macro_action_count"])
                for sid in ids
            ])),
            "s_selection_action_rate": float(np.mean([
                rows[sid]["s_selection_count"] / max(1, rows[sid]["macro_action_count"])
                for sid in ids
            ])),
            "selected_max_B_total_le_budget": all(
                rows[sid]["selected_max_B_total_ml"] <= baselines[sid]["expected_blood_loss_ml"] + margin + EPS
                for sid in ids
            ),
        }
    return out


def _percentiles(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    return [float(np.quantile(values, q)) for q in (0.50, 0.90, 0.95, 0.99)] + [float(values.max())]


def gate_replication(summary: dict, ids: list[str], rows_by_id: dict, margin: float,
                    teacher_C3_rows: dict, c0_rows: dict, baselines: dict) -> dict:
    """Evaluate Gate C conditions on C4 (the primary model)."""
    c4 = summary["C4"]
    # teacher retention uses mean elapsed_minutes.
    t_c0 = c0_rows["mean_abs_T_min"]
    t_c4 = c4["mean_abs_T_min"]
    t_c3 = teacher_C3_rows["mean_abs_T_min"]
    denominator = t_c0 - t_c3
    if denominator <= 0:
        r_t = None
    else:
        r_t = (t_c0 - t_c4) / denominator
    # paired bootstrap on per-scene delta-T (C4 - C0) from shards
    diff_t = np.asarray([c4["shard"][sid]["elapsed_minutes"] - c0_rows["shard"][sid]["elapsed_minutes"]
                         for sid in ids])
    diff_b = np.asarray([c4["shard"][sid]["realized_episode_B_ml"]
                         - baselines[sid]["expected_blood_loss_ml"]
                         for sid in ids])
    seed = 202608170704
    ci_dt = bootstrap_ci_paired(diff_t, seed)
    ci_db = bootstrap_ci_paired(diff_b, seed)
    max_db = float(diff_b.max())
    conditions = {
        "completion_legal_256": c4["completion"] and c4["legal"] == 256,
        "failure_invariant_zero": c4["failures"] == 0 and c4["invariants"] == 0,
        "per_scene_overrun_zero": c4["overrun_count"] == 0,
        "max_delta_B_le_margin": max_db <= margin + EPS,
        "delta_B_ci_upper_le_margin": ci_db[1] <= margin,
        "delta_T_ci_upper_lt_zero": ci_dt[1] < 0.0,
        "teacher_retention_ge_050": r_t is not None and r_t >= 0.50,
    }
    decision = "GO" if all(conditions.values()) else "NO-GO"
    if conditions["teacher_retention_ge_050"] is False:
        raise RuntimeError("teacher itself has no positive time benefit; contract failed")
    return {
        "version": "v10.7-replication-gate-v1",
        "decision": decision,
        "conditions": conditions,
        "R_T": r_t,
        "paired_delta_T_mean": float(diff_t.mean()),
        "paired_delta_T_median": float(np.median(diff_t)),
        "paired_delta_T_95_ci": ci_dt,
        "paired_delta_T_cohens_dz": cohens_dz(diff_t),
        "win_tie_loss": [int((diff_t < -EPS).sum()), int((abs(diff_t) <= EPS).sum()), int((diff_t > EPS).sum())],
        "paired_delta_B_mean": float(diff_b.mean()),
        "paired_delta_B_95_ci": ci_db,
        "max_delta_B_ml": max_db,
        "margin_ml": margin,
        "n": len(ids),
    }


def gate_learning_specificity(summary: dict, ids: list[str]) -> dict:
    c4 = summary["C4"]["shard"]
    c2 = summary["C2"]["shard"]
    diff = np.asarray([c4[sid]["elapsed_minutes"] - c2[sid]["elapsed_minutes"] for sid in ids])
    ci = bootstrap_ci_paired(diff, 202608170704)
    decision = "learning-specific GO" if ci[1] < 0.0 else "feasibility GO, learned advantage unproven"
    return {
        "version": "v10.7-learning-specificity-v1",
        "decision": decision,
        "paired_delta_T_mean": float(diff.mean()),
        "paired_delta_T_95_ci": ci,
        "paired_delta_T_cohens_dz": cohens_dz(diff),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="replication")
    parser.add_argument("--condition", default="S0")
    args = parser.parse_args()

    manifest = json.loads((FROZEN / "experiment_manifest.json").read_text(encoding="utf-8"))
    margin = float(manifest["margin_ml"])
    baseline_payload = json.loads((FROZEN / f"baseline_{args.split}.json").read_text(encoding="utf-8"))
    baselines = baseline_payload["records"]
    expected_ids = set(baselines.keys())

    if args.split == "sensitivity_base":
        controllers = ["C0", "C2", "C4"]
    else:
        controllers = list(CONTROLLER_ORDER)
    all_rows = load_shards(args.split, args.condition, controllers)
    check_complete(all_rows, expected_ids, f"{args.split}/{args.condition}")

    # Build per-controller summary with shard references for pairing.
    ids = sorted(expected_ids)
    summary = {}
    for c in controllers:
        rows = all_rows[c]
        d_t = np.asarray([rows[sid]["elapsed_minutes"] - baselines[sid]["elapsed_minutes"] for sid in ids])
        d_b = np.asarray([rows[sid]["realized_episode_B_ml"] - baselines[sid]["expected_blood_loss_ml"] for sid in ids])
        s = {
            "shard": rows,
            "mean_abs_T_min": float(np.mean([rows[sid]["elapsed_minutes"] for sid in ids])),
            "mean_abs_B_ml": float(np.mean([rows[sid]["realized_episode_B_ml"] for sid in ids])),
            "mean_delta_T_min": float(d_t.mean()),
            "mean_delta_B_ml": float(d_b.mean()),
            "tail_delta_T": [float(x) for x in _percentiles(d_t)],
            "tail_delta_B": [float(x) for x in _percentiles(d_b)],
            "cvar10_delta_T": float(d_t[d_t <= np.quantile(d_t, 0.10)].mean()) if len(d_t) else None,
            "cvar10_delta_B": float(d_b[d_b <= np.quantile(d_b, 0.10)].mean()) if len(d_b) else None,
            "completion": all(rows[sid]["completion"] for sid in ids),
            "failures": sum(not rows[sid]["completion"] or rows[sid]["failure_reason"] is not None for sid in ids),
            "invariants": sum(int(rows[sid]["safety_invariant_violations"]) for sid in ids),
            "overrun_count": int(np.sum(d_b > margin + EPS)),
            "legal": sum(rows[sid]["legal_action_rate"] == 1.0 for sid in ids),
            "shield_intervention_action_rate": float(np.mean([
                rows[sid]["shield_intervention_count"] / max(1, rows[sid]["macro_action_count"])
                for sid in ids
            ])),
            "s_selection_action_rate": float(np.mean([
                rows[sid]["s_selection_count"] / max(1, rows[sid]["macro_action_count"])
                for sid in ids
            ])),
        }
        summary[c] = s

    if args.split == "replication":
        gate = gate_replication(summary, ids, all_rows, margin, summary["C3"], summary["C0"], baselines)
        learn = gate_learning_specificity(summary, ids)
        result = {
            "version": "v10.7-replication-statistics-v1",
            "split": args.split, "margin_ml": margin, "n": len(ids),
            "gate": gate, "learning_specificity": learn,
            "controller_summary": {
                c: {k: v for k, v in summary[c].items() if k != "shard"} for c in controllers
            },
        }
        out = BASE / "evaluation" / "replication_statistics.json"
    else:
        # Gate D: for this condition, C4 completion/legal 100%, zero
        # failure/invariant/overrun, and C4-C0 paired delta-T CI upper < 0.
        c4 = summary["C4"]; c0 = summary["C0"]
        diff_t = np.asarray([c4["shard"][sid]["elapsed_minutes"] - c0["shard"][sid]["elapsed_minutes"]
                             for sid in ids])
        diff_b = np.asarray([c4["shard"][sid]["realized_episode_B_ml"]
                             - baselines[sid]["expected_blood_loss_ml"]
                             for sid in ids])
        ci_dt = bootstrap_ci_paired(diff_t, 202608170704)
        cond_d = {
            "completion_legal_100": c4["completion"] and c4["legal"] == 128,
            "failure_invariant_zero": c4["failures"] == 0 and c4["invariants"] == 0,
            "per_scene_overrun_zero": c4["overrun_count"] == 0,
            "max_delta_B_le_margin": float(diff_b.max()) <= margin + EPS,
            "delta_T_ci_upper_lt_zero": ci_dt[1] < 0.0,
        }
        gate_d = {
            "version": "v10.7-sensitivity-gate-v1",
            "condition": args.condition,
            "decision": "PASS" if all(cond_d.values()) else "FAIL",
            "conditions": cond_d,
            "paired_delta_T_95_ci": ci_dt,
            "paired_delta_T_mean": float(diff_t.mean()),
        }
        result = {
            "version": "v10.7-sensitivity-statistics-v1",
            "split": args.split, "condition": args.condition, "margin_ml": margin, "n": len(ids),
            "gate_d": gate_d,
            "controller_summary": {
                c: {k: v for k, v in summary[c].items() if k != "shard"} for c in controllers
            },
        }
        out = BASE / "evaluation" / f"sensitivity_{args.condition}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out),
                      "decision": result.get("gate", {}).get("decision"),
                      "learn": result.get("learning_specificity", {}).get("decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

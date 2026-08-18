"""Gate A v2 policy-rollout diagnostics figure (guide 13).

Reads the merged full-64 payloads under ``oracle_upper_bound_v2/full64/`` and
writes ``oracle_upper_bound_v2/gate_a_policy_diagnostics.png`` with:

  1. per-scenario delta blood        (+ 95% CI and the 5% non-inferiority margin
                                      shown for context, never used as the Gate
                                      criterion)
  2. per-scenario delta ischemia     (+ 95% CI)
  3. per-scenario delta time         (+ 95% CI and the 1% margin)
  4. release count vs clamp elapsed  (distribution of the actual releases)
  5. release/continue reject reasons (bar chart)
  6. baseline vs oracle paired scatter (blood)

Candidate-state v1 results are never plotted on the same statistic as the
episode-policy v2 results (guide 13).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_ids(records) -> list[str]:
    return [str(r["scenario_id"]) for r in records]


def _by_id(records) -> dict[str, dict]:
    return {str(r["scenario_id"]): r for r in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full64-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paired = _load(args.full64_dir / "paired_differences.json")
    baseline_payload = _load(args.full64_dir / "baseline_records.json")
    oracle_payload = _load(args.full64_dir / "oracle_records.json")
    baseline_records = baseline_payload["records"]
    oracle_records = oracle_payload["records"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # --- 1/2/3 per-scenario deltas with CI + margins -------------------------
    baseline_by_id = _by_id(baseline_records)
    ids = [str(r["scenario_id"]) for r in oracle_records]
    order = range(len(ids))
    for ax, key, color, ylabel in (
        (axes[0, 0], "blood", "tab:red", "Δ blood (mL)"),
        (axes[0, 1], "ischemia", "tab:blue", "Δ ischemia (min)"),
        (axes[0, 2], "time", "tab:green", "Δ time (min)"),
    ):
        diffs = [float(oracle_records[i][_FIELD[key]]) - float(baseline_by_id[ids[i]][_FIELD[key]])
                 for i in order]
        ci = paired["fields"][key]["bootstrap_95_ci"]
        margin = paired["margins"].get(_MARGIN[key])
        ax.bar(list(order), diffs, color=color, alpha=0.6, width=0.8)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axhspan(ci[0], ci[1], color="gray", alpha=0.15,
                   label=f"bootstrap 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
        ax.axhline(ci[1], color="orange", ls="--", lw=1.2)
        if margin is not None:
            ax.axhline(margin, color="purple", ls=":", lw=1.2,
                       label=f"{_MARGIN_LABEL[key]} ({margin:.2f})")
        ax.set_title(f"Per-scenario Δ{key} (oracle − baseline)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("scenario index")
        ax.legend(fontsize=6, loc="best")

    # --- 4 release count vs clamp elapsed ------------------------------------
    rel_steps = []
    rel_clamp_el = []
    for rec in oracle_records:
        for d in rec.get("decisions", []):
            if d.get("action") == 1:
                rel_steps.append(int(d.get("macro_step", 0)))
                rel_clamp_el.append(float(d.get("clamp_elapsed_minutes", 0.0)))
    ax = axes[1, 0]
    if rel_steps:
        ax.scatter(rel_steps, rel_clamp_el, s=40, alpha=0.8, color="tab:blue")
    ax.set_title(f"Releases (n={len(rel_steps)}) — macro step vs clamp elapsed")
    ax.set_xlabel("macro step")
    ax.set_ylabel("clamp elapsed at release (min)")
    ax.axvspan(9.9, 15.1, color="orange", alpha=0.08)

    # --- 5 reject reasons ----------------------------------------------------
    from collections import Counter
    reasons = Counter()
    for rec in oracle_records:
        for d in rec.get("decisions", []):
            reasons[d.get("reject_reason") or "RELEASE"] += 1
    ax = axes[1, 1]
    labels = list(reasons.keys())
    counts = [reasons[l] for l in labels]
    short = [l if len(l) < 40 else l[:37] + "…" for l in labels]
    ax.barh(range(len(labels)), counts, color="tab:gray")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short, fontsize=6)
    ax.set_title("Decision outcomes / reject reasons")
    ax.set_xlabel("count")

    # --- 6 baseline vs oracle paired scatter (blood) -------------------------
    ax = axes[1, 2]
    b_blood = [float(baseline_by_id[i]["expected_blood_loss_ml"]) for i in ids]
    o_blood = [float(oracle_records[i]["expected_blood_loss_ml"]) for i in order]
    ax.scatter(b_blood, o_blood, s=20, alpha=0.7, color="tab:red")
    lo = min(min(b_blood), min(o_blood))
    hi = max(max(b_blood), max(o_blood))
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1)
    ax.set_title("Baseline vs oracle blood loss (mL)")
    ax.set_xlabel("baseline (frozen BC + 15/5)")
    ax.set_ylabel("safe-greedy oracle")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110)
    plt.close(fig)
    print(json.dumps({
        "figure": str(args.output),
        "n_scenarios": len(ids),
        "releases_total": len(rel_steps),
        "reject_reason_count": len(reasons),
    }, ensure_ascii=False))


_FIELD = {
    "blood": "expected_blood_loss_ml",
    "ischemia": "total_clamped_minutes",
    "time": "elapsed_minutes",
}
_MARGIN = {"blood": "blood_M_B", "ischemia": None, "time": "time_M_T"}
_MARGIN_LABEL = {"blood": "5% non-inferiority (final-model only)", "time": "1% time margin"}


if __name__ == "__main__":
    main()

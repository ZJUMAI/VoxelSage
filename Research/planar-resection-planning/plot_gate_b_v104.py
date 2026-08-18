"""Plot the frozen v10.4 Gate B training and safety diagnostics.

The completed Gate B run used behavior cloning rather than PPO, and the
historical trainer did not persist per-epoch ranking loss.  Accordingly the
"reward" panel below is an explicitly labelled frozen-evaluation clinical
reward proxy, not an invented PPO training-return series.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SIM = Path(__file__).resolve().parent
ROOT = SIM / "results/clinical_window_v10_4_target_order"
RUNS = ROOT / "runs"
REPORT = ROOT / "report"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _reward(mean_t: float, mean_b: float, time_scale: float, blood_scale: float) -> float:
    return -(mean_t / time_scale + mean_b / blood_scale)


def plot_training_diagnostics(evaluation, bc, scales) -> Path:
    names = ["Serpentine", "Depth-1 MPC", "BC scorer"]
    branches = [evaluation["baseline"], evaluation["teacher"], evaluation["model"]]
    colors = ["#7A8793", "#2A9D70", "#E07A2D"]
    time_scale = float(scales["time_scale_minutes"])
    blood_scale = float(scales["blood_scale_ml"])
    rewards = [
        _reward(float(item["mean_T"]), float(item["mean_B"]), time_scale, blood_scale)
        for item in branches
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    fig.suptitle("v10.4 Gate B: frozen evaluation and BC training diagnostics", fontsize=15)

    ax = axes[0, 0]
    x = np.arange(len(names))
    ax.plot(x, rewards, marker="o", linewidth=2.2, color="#326A9A")
    for i, value in enumerate(rewards):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel("Reward proxy (higher is better)")
    ax.set_title("Clinical reward audit (not a PPO training curve)")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01, 0.02,
        "R = -(T/Tscale + B/Bscale); BC training stored no episode-return history",
        transform=ax.transAxes, fontsize=8, color="#4D5966",
    )

    ax = axes[0, 1]
    means = np.asarray([float(item["mean_T"]) for item in branches])
    cis = [item["dT_95_ci"] for item in branches]
    base = means[0]
    lower = np.asarray([base + float(ci[0]) for ci in cis])
    upper = np.asarray([base + float(ci[1]) for ci in cis])
    yerr = np.vstack((means - lower, upper - means))
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#3C4855", capsize=5, linewidth=1.4)
    ax.bar(x, means, color=colors, width=0.62)
    for i, value in enumerate(means):
        ax.text(i, value + 0.18, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel("Total simulated time (min)")
    ax.set_title("Time improved, including BC vs teacher")
    ax.set_ylim(min(lower) - 1.0, max(upper) + 1.2)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    means_b = np.asarray([float(item["mean_B"]) for item in branches])
    cis_b = [item["dB_95_ci"] for item in branches]
    base_b = means_b[0]
    lower_b = np.asarray([base_b + float(ci[0]) for ci in cis_b])
    upper_b = np.asarray([base_b + float(ci[1]) for ci in cis_b])
    yerr_b = np.vstack((means_b - lower_b, upper_b - means_b))
    ax.errorbar(x, means_b, yerr=yerr_b, fmt="none", ecolor="#3C4855", capsize=5, linewidth=1.4)
    ax.bar(x, means_b, color=colors, width=0.62)
    for i, value in enumerate(means_b):
        ax.text(i, value + 14, f"{value:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel("Expected simulated blood loss (mL)")
    ax.set_title("BC mean hides an unsafe heavy tail")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    metrics = ["Top-1", "Top-3", "NDCG@3"]
    values = [float(bc["top1_acc"]), float(bc["top3_acc"]), float(bc["ndcg_at_3"])]
    bars = ax.bar(metrics, values, color=["#326A9A", "#4D91C6", "#77A8CE"], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Training-set ranking score")
    ax.set_title(f"BC fitted teacher ranking ({int(bc['n_states']):,} states)")
    ax.grid(axis="y", alpha=0.25)

    output = REPORT / "gate_b_training_diagnostics.png"
    _save(fig, output)
    return output


def plot_failure_diagnostics(evaluation, diagnostic, teacher_npz: Path) -> Path:
    d = np.load(teacher_npz)
    valid = d["valid"].astype(bool)
    safe = (d["cost_B"] <= d["safe_threshold"][:, None]) & valid
    n_valid = valid.sum(axis=1)
    n_safe = safe.sum(axis=1)
    state_shares = np.asarray([
        np.mean(n_safe == n_valid),
        np.mean((n_safe > 0) & (n_safe < n_valid)),
        np.mean(n_safe == 0),
    ])

    worst = list(diagnostic["worst_10"])
    worst_ids = [item["scenario_id"].replace("train-", "") for item in worst]
    db = np.asarray([float(item["dB"]) for item in worst])
    dt = np.asarray([float(item["dT"]) for item in worst])
    margin = float(evaluation["margin_ml"])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    fig.suptitle("v10.4 Gate B: why the BC policy is unsafe", fontsize=15)

    ax = axes[0, 0]
    labels = ["Median", "90th pct", "Maximum"]
    values = [float(diagnostic["dB_median"]), float(diagnostic["dB_p90"]), float(diagnostic["dB_max"])]
    ax.bar(labels, values, color=["#2A9D70", "#E5A33B", "#C94747"], width=0.62)
    ax.axhline(margin, color="#4D5966", linestyle="--", linewidth=1.4,
               label=f"Non-inferiority margin +{margin:.1f} mL")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylabel("Model - baseline blood loss (mL, symlog)")
    ax.set_title("Heavy tail: median safe-looking, maximum catastrophic")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    labels = ["All candidates safe", "Mixed safety", "All unsafe"]
    bars = ax.bar(labels, state_shares * 100, color=["#7A8793", "#E5A33B", "#C94747"], width=0.62)
    for bar, value in zip(bars, state_shares * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Teacher states (%)")
    ax.set_title("Only a small minority carries a comparative safety signal")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    order = np.arange(len(worst))
    ax.barh(order, db, color="#C94747")
    ax.set_yticks(order, worst_ids)
    ax.invert_yaxis()
    ax.axvline(margin, color="#4D5966", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Model - baseline blood loss (mL)")
    ax.set_ylabel("Scenario ID suffix")
    ax.set_title("Worst ten policy failures")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 1]
    sizes = np.asarray([max(35.0, float(item["vessel_cells"]) * 5.0) for item in worst])
    ax.scatter(dt, db, s=sizes, c="#C94747", alpha=0.82, edgecolors="#7C2929")
    for item in worst[:5]:
        ax.annotate(item["scenario_id"].replace("train-", ""),
                    (float(item["dT"]), float(item["dB"])),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(margin, color="#4D5966", linestyle="--", linewidth=1.3)
    ax.axvline(0, color="#4D5966", linewidth=1.0)
    ax.set_xlabel("Model - baseline time (min)")
    ax.set_ylabel("Model - baseline blood loss (mL)")
    ax.set_title("Worst scenes are faster but unsafe")
    ax.grid(alpha=0.25)

    output = REPORT / "gate_b_failure_diagnostics.png"
    _save(fig, output)
    return output


def main() -> None:
    evaluation = _read_json(RUNS / "gate_b_evaluation.json")
    diagnostic = _read_json(RUNS / "gate_b_failure_diag.json")
    bc = _read_json(RUNS / "target_order_bc.json")
    scales = _read_json(ROOT / "frozen/scales_v10_4.json")
    outputs = [
        plot_training_diagnostics(evaluation, bc, scales),
        plot_failure_diagnostics(evaluation, diagnostic, ROOT / "teacher/teacher_rankings.npz"),
    ]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()

"""Generate v10 training figures and a Markdown evidence report from artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _health_series(run: Path) -> list[dict[str, Any]]:
    def key(path: Path) -> int:
        match = re.search(r"step_(\d+)", path.stem)
        return int(match.group(1)) if match else 0

    return [_load(path) for path in sorted((run / "training_health").glob("step_*.json"), key=key)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bc_files = sorted(args.results_dir.glob("runs/**/behavior_cloning.json"))
    oracle_files = sorted(args.results_dir.glob("**/clamp_oracle_report.json"))
    health_runs = sorted({path.parent.parent for path in args.results_dir.glob("runs/**/training_health/step_*.json")})
    optuna_files = sorted(args.results_dir.glob("**/optuna_summary.json"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    plotted = False
    for path in bc_files:
        data = _load(path)
        epochs = data.get("epoch_summaries", [])
        if not epochs:
            continue
        ax.plot(
            [item["epoch"] for item in epochs],
            [item["mean_policy_loss"] for item in epochs],
            marker="o",
            markersize=3,
            label=path.parent.name,
        )
        plotted = True
    ax.set(title="Target-head behavior cloning", xlabel="BC epoch", ylabel="Mean policy loss")
    if plotted:
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No BC history found", ha="center", va="center", transform=ax.transAxes)

    ax = axes[0, 1]
    plotted = False
    for path in oracle_files:
        history = _load(path).get("history", [])
        if not history:
            continue
        x = [item["epoch"] for item in history]
        ax.plot(x, [item["mean_loss"] for item in history], marker="o", label="oracle loss")
        ax2 = ax.twinx()
        ax2.plot(x, [item["accuracy"] for item in history], color="tab:green", label="accuracy")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.02)
        plotted = True
        break
    ax.set(title="Clamp timing-oracle pretraining", xlabel="Epoch", ylabel="Cross-entropy")
    if not plotted:
        ax.text(0.5, 0.5, "No timing-oracle history found", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 0]
    plotted = False
    for run in health_runs:
        values = _health_series(run)
        if not values:
            continue
        ax.plot(
            [item["timesteps"] for item in values],
            [item["mean_expected_blood_loss_ml"] for item in values],
            marker="o",
            markersize=3,
            label=run.name,
        )
        plotted = True
    ax.set(title="Training-pool rolling blood loss", xlabel="PPO steps", ylabel="Expected blood loss (mL)")
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No training-health history found", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 1]
    plotted = False
    for run in health_runs:
        values = _health_series(run)
        if not values:
            continue
        ax.plot(
            [item["timesteps"] for item in values],
            [item["mean_transfer_overhead"] for item in values],
            marker="o",
            markersize=3,
            label=run.name,
        )
        plotted = True
    ax.set(title="Training-pool transfer overhead", xlabel="PPO steps", ylabel="Transfer overhead")
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No training-health history found", ha="center", va="center", transform=ax.transAxes)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    diagnostics_path = args.output_dir / "v10_training_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)

    pareto_path = args.output_dir / "v10_optuna_pareto.png"
    optuna_rows: list[dict[str, Any]] = []
    if optuna_files:
        data = _load(optuna_files[-1])
        pareto_ids = set(data.get("pareto_trial_numbers", []))
        for trial in data.get("trials", []):
            values = trial.get("values")
            if trial.get("state") != "COMPLETE" or not values:
                continue
            constraints = trial.get("constraints") or []
            feasible = all(float(value) <= 0 for value in constraints)
            optuna_rows.append({
                "number": int(trial["number"]),
                "time": float(values[0]),
                "blood": float(values[1]),
                "transfer": float(values[2]),
                "feasible": feasible,
                "pareto": int(trial["number"]) in pareto_ids,
            })
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if optuna_rows:
        for feasible, marker, label in ((False, "x", "infeasible"), (True, "o", "feasible")):
            rows = [row for row in optuna_rows if row["feasible"] == feasible]
            if not rows:
                continue
            scatter = ax.scatter(
                [row["time"] for row in rows],
                [row["blood"] for row in rows],
                c=[row["transfer"] for row in rows],
                marker=marker,
                s=55,
                cmap="viridis",
                label=label,
            )
        for row in optuna_rows:
            if row["pareto"] and row["feasible"]:
                ax.annotate(str(row["number"]), (row["time"], row["blood"]), xytext=(4, 4), textcoords="offset points")
        fig.colorbar(scatter, ax=ax, label="Transfer overhead")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No Optuna summary found", ha="center", va="center", transform=ax.transAxes)
    ax.set(title="Optuna tuning Pareto candidates", xlabel="Frozen tuning time (min)", ylabel="Frozen tuning blood loss (mL)")
    ax.grid(alpha=0.25)
    fig.savefig(pareto_path, dpi=180)
    plt.close(fig)

    lines = [
        "# Clinical-window v10 training report",
        "",
        "> Automatically generated from frozen artifacts. Training-pool rolling metrics are diagnostic only; model selection must use frozen validation.",
        "",
        "## Training curves",
        "",
        "![v10 training diagnostics](v10_training_diagnostics.png)",
        "",
        "## Optuna Pareto search",
        "",
        "![v10 Optuna Pareto candidates](v10_optuna_pareto.png)",
        "",
        "## Artifact counts",
        "",
        f"- BC histories: {len(bc_files)}",
        f"- Timing-oracle histories: {len(oracle_files)}",
        f"- PPO training-health runs: {len(health_runs)}",
        f"- Complete Optuna trials: {len(optuna_rows)}",
        "",
        "## Required final interpretation",
        "",
        "The training agent must append completion, paired time/blood differences with bootstrap 95% confidence intervals, END safety violations, and multi-seed frozen-validation results. A Pareto point is not a final model until it passes all safety gates.",
    ]
    (args.output_dir / "report_clinical_v10.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(args.output_dir / "report_clinical_v10.md"),
        "training_figure": str(diagnostics_path),
        "optuna_figure": str(pareto_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

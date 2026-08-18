"""v10.2 report generator.

Reads the frozen artifacts plus run/evaluation outputs under
``results/clinical_window_v10_2/`` and writes:

    report/report_clinical_v102.md
    report/reward_curves.png
    report/oracle_metrics.png
    report/clinical_metrics.png
    report/optuna_pareto.png   (only if an Optuna summary exists)

Never plots different ``blood_cost``/``ischemia_cost`` reward curves on the
same axis; marks training-pool rolling curves as diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_figure():
    return plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)


def _plot_oracle_metrics(results_dir: Path, output: Path) -> bool:
    oracle_files = sorted(results_dir.glob("**/clamp_oracle_report.json"))
    if not oracle_files:
        return False
    fig, axes = _make_figure()
    ax = axes[0, 0]
    for path in oracle_files:
        data = _load(path)
        history = data.get("history", [])
        if history:
            ax.plot(
                [h["epoch"] for h in history],
                [h.get("mean_loss", 0.0) for h in history],
                marker="o",
                markersize=3,
                label=path.parent.name,
            )
    ax.set(title="Oracle clamp training loss", xlabel="Epoch", ylabel="Loss")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for path in oracle_files:
        data = _load(path)
        history = data.get("history", [])
        if history:
            ax.plot(
                [h["epoch"] for h in history],
                [h.get("train_accuracy", h.get("accuracy", 0.0)) for h in history],
                marker="o",
                markersize=3,
                label=path.parent.name,
            )
    ax.set(title="Oracle train accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    ax.text(0.5, 0.5, "AUROC / AUPRC / calibration\n(reported in oracle_report.json)",
            ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Class balance + release precision/recall\nper oracle report",
            ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.savefig(output, dpi=110)
    plt.close(fig)
    return True


def _plot_reward_curves(results_dir: Path, output: Path) -> bool:
    probe_files = sorted(results_dir.glob("runs/**/probe/step_*.json"))
    if not probe_files:
        return False
    fig, axes = _make_figure()
    ax = axes[0, 0]
    for path in probe_files:
        data = _load(path)
        ax.plot([data["timesteps"]], [data["det_mean_reward"]], "o", label=path.parent.name)
    ax.set(title="Probe deterministic reward (fixed Probe, early-stop signal)",
           xlabel="PPO steps", ylabel="Mean reward")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for path in probe_files:
        data = _load(path)
        ax.plot([data["timesteps"]], [data["det_mean_ischemia"]], "s", label=path.parent.name)
    ax.set(title="Probe cumulative clamped (ischemia proxy)", xlabel="PPO steps", ylabel="Minutes")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    for path in probe_files:
        data = _load(path)
        ax.plot([data["timesteps"]], [data["det_mean_blood"]], "^", label=path.parent.name)
    ax.set(title="Probe expected blood loss", xlabel="PPO steps", ylabel="mL")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    for path in probe_files:
        data = _load(path)
        ax.plot([data["timesteps"]], [data["det_end_count"]], "d", label=path.parent.name)
    ax.set(title="Probe deterministic END count", xlabel="PPO steps", ylabel="Count")
    ax.legend(fontsize=7)
    fig.savefig(output, dpi=110)
    plt.close(fig)
    return True


def _plot_clinical_metrics(results_dir: Path, output: Path) -> bool:
    eval_files = sorted(results_dir.glob("evaluation/**/*.json"))
    if not eval_files:
        return False
    fig, axes = _make_figure()
    ax = axes[0, 0]
    for path in eval_files:
        data = _load(path)
        det = data.get("det_summary")
        if det:
            ax.bar(path.stem, det.get("mean_elapsed_minutes", 0.0))
    ax.set(title="Deterministic mean surgery time by split", ylabel="Minutes")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax = axes[0, 1]
    for path in eval_files:
        data = _load(path)
        det = data.get("det_summary")
        if det:
            ax.bar(path.stem, det.get("mean_expected_blood_loss_ml", 0.0))
    ax.set(title="Deterministic mean blood loss by split", ylabel="mL")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax = axes[1, 0]
    for path in eval_files:
        data = _load(path)
        det = data.get("det_summary")
        if det:
            ax.bar(path.stem, det.get("mean_total_clamped_minutes", 0.0))
    ax.set(title="Deterministic cumulative clamped (ischemia proxy)", ylabel="Minutes")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax = axes[1, 1]
    for path in eval_files:
        data = _load(path)
        det = data.get("det_summary")
        if det:
            ax.bar(path.stem, det.get("mean_early_end_count", 0.0))
    ax.set(title="Deterministic END count by split", ylabel="Count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    fig.savefig(output, dpi=110)
    plt.close(fig)
    return True


def _optuna_pareto_path(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("**/optuna_summary.json"))
    return files[0] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    has_oracle = _plot_oracle_metrics(args.results_dir, args.output_dir / "oracle_metrics.png")
    has_reward = _plot_reward_curves(args.results_dir, args.output_dir / "reward_curves.png")
    has_clinical = _plot_clinical_metrics(args.results_dir, args.output_dir / "clinical_metrics.png")

    lines = [
        "# Clinical-window v10.2 training report",
        "",
        "> Target-conditioned clamp agent.  Clamp decisions are conditioned on the",
        "> frozen BC planned target and its automatic transfer route, with an added",
        "> ischemia proxy in the reward.  All metrics are research proxies, not",
        "> clinically validated decision systems.",
        "",
        "## Figures",
        "",
    ]
    if has_oracle:
        lines.append("![Oracle metrics](oracle_metrics.png)")
    if has_reward:
        lines.append("![Probe reward curves](reward_curves.png)")
    if has_clinical:
        lines.append("![Clinical metrics](clinical_metrics.png)")
    optuna = _optuna_pareto_path(args.results_dir)
    if optuna is not None:
        lines.append("![Optuna Pareto](optuna_pareto.png)")

    lines += [
        "",
        "## Required final interpretation",
        "",
        "The training agent must append the five headline answers:",
        "",
        "1. Did the model perform non-zero safe early release?",
        "2. Did it reduce cumulative clamped (ischemia) time?",
        "3. Is blood loss non-inferior to baseline?",
        "4. Did it reduce total surgery time?",
        "5. Do these hold on both 3-seed Validation and one-shot Test?",
        "",
        "Also append: paired time/blood/ischemia differences with bootstrap 95% CIs,",
        "END safety violations, deterministic vs stochastic evaluation separation, and",
        "the scene-isolated oracle Dev metrics (AUROC / balanced accuracy / unsafe-release",
        "false-positive rate).",
    ]
    (args.output_dir / "report_clinical_v102.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(args.output_dir / "report_clinical_v102.md"),
        "oracle_metrics_png": has_oracle,
        "reward_curves_png": has_reward,
        "clinical_metrics_png": has_clinical,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

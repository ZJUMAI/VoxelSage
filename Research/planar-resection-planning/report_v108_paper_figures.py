"""Generate the paper figures and paired statistics for the v10.8 C4 controller.

The paper name ``C4`` refers to the frozen behavior-cloned ranker with lazy
exact verification.  Its experiment shards retain the implementation label
``C4L``.  Statistics are always recomputed from the 256 per-scene shards.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


SIM = Path(__file__).resolve().parent
RESULTS = SIM / "results" / "clinical_window_v10_8_lazy_shield"
SHARDS = RESULTS / "shards"
PUBLICATION = RESULTS / "publication_figures"
TECH_REPORT_FIGURES = SIM.parents[2] / "tech_report" / "figures"
CONTROLLERS = {"C0": "C0", "C2": "C2", "C3": "C3", "C4": "C4L", "C5": "C5"}
BOOTSTRAP_SEED = 202608170704
BOOTSTRAP_SAMPLES = 10_000


def configure_font() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    chosen = "Times New Roman" if "Times New Roman" in available else "Liberation Serif"
    plt.rcParams.update(
        {
            "font.family": chosen,
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return chosen


def load_rows(controller: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((SHARDS / CONTROLLERS[controller]).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[row["scenario_id"]] = row
    if len(rows) != 256:
        raise RuntimeError(f"Expected 256 {controller} shards, found {len(rows)}")
    return rows


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def paired_values(
    rows: dict[str, dict[str, dict[str, Any]]],
    left: str,
    right: str,
    field: str,
) -> np.ndarray:
    scene_ids = sorted(rows[right])
    return np.asarray(
        [rows[left][scene_id][field] - rows[right][scene_id][field] for scene_id in scene_ids],
        dtype=float,
    )


def paired_summary(
    rows: dict[str, dict[str, dict[str, Any]]], left: str, right: str, seed: int
) -> dict[str, Any]:
    delta_time = paired_values(rows, left, right, "elapsed_minutes")
    delta_blood = paired_values(rows, left, right, "realized_episode_B_ml")
    return {
        "left": left,
        "right": right,
        "n": int(len(delta_time)),
        "mean_delta_time_min": float(delta_time.mean()),
        "delta_time_95_ci": bootstrap_ci(delta_time, seed),
        "cohen_dz_time": float(delta_time.mean() / delta_time.std(ddof=1)),
        "time_wins_ties_losses": [
            int((delta_time < -1e-9).sum()),
            int((np.abs(delta_time) <= 1e-9).sum()),
            int((delta_time > 1e-9).sum()),
        ],
        "mean_delta_blood_ml": float(delta_blood.mean()),
        "delta_blood_95_ci": bootstrap_ci(delta_blood, seed + 100),
        "max_delta_blood_ml": float(delta_blood.max()),
    }


def save_figure(fig: plt.Figure, name: str) -> None:
    PUBLICATION.mkdir(parents=True, exist_ok=True)
    TECH_REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = PUBLICATION / f"{name}.{suffix}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        shutil.copy2(output, TECH_REPORT_FIGURES / output.name)
    plt.close(fig)


def figure_replication_effects(rows: dict[str, dict[str, dict[str, Any]]]) -> None:
    controllers = ["C2", "C3", "C4", "C5"]
    labels = ["Myopic + exact check", "Full-rollout teacher", "BC + lazy exact", "BC, no shield"]
    time_values = [paired_values(rows, controller, "C0", "elapsed_minutes") for controller in controllers]
    blood_values = [
        paired_values(rows, controller, "C0", "realized_episode_B_ml")
        for controller in controllers
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    x = np.arange(len(labels))
    for panel, (ax, values, ylabel) in enumerate(
        (
            (axes[0], time_values, "Paired time difference (min)"),
            (axes[1], blood_values, "Paired simulated blood difference (mL)"),
        )
    ):
        means = [float(value.mean()) for value in values]
        intervals = [
            bootstrap_ci(value, BOOTSTRAP_SEED + panel * 100 + index)
            for index, value in enumerate(values)
        ]
        errors = np.asarray(
            [
                [mean - interval[0] for mean, interval in zip(means, intervals)],
                [interval[1] - mean for mean, interval in zip(means, intervals)],
            ]
        )
        ax.errorbar(x, means, yerr=errors, fmt="o", capsize=3, color="#2f5d8a")
        ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Time relative to serpentine baseline")
    axes[1].set_title("Simulated blood relative to serpentine baseline")
    fig.tight_layout()
    save_figure(fig, "replication_controller_effects")


def figure_shield_ablation(rows: dict[str, dict[str, dict[str, Any]]]) -> None:
    delta_c4 = paired_values(rows, "C4", "C0", "realized_episode_B_ml")
    delta_c5 = paired_values(rows, "C5", "C0", "realized_episode_B_ml")
    scene_ids = sorted(rows["C0"])
    overrun_c4 = sum(
        rows["C4"][scene_id]["realized_episode_B_ml"]
        > rows["C4"][scene_id]["budget_ml"] + 1e-9
        for scene_id in scene_ids
    )
    overrun_c5 = sum(
        rows["C5"][scene_id]["realized_episode_B_ml"]
        > rows["C5"][scene_id]["budget_ml"] + 1e-9
        for scene_id in scene_ids
    )
    margin = 16.07054347826075
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.2))
    axes[0].boxplot(
        [delta_c4, delta_c5],
        tick_labels=["BC + lazy exact", "BC, no shield"],
        showfliers=True,
    )
    axes[0].axhline(margin, color="#a33b3b", linestyle="--", linewidth=1, label="Frozen margin")
    axes[0].set_ylabel("Paired simulated blood difference (mL)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    counts = [overrun_c4, overrun_c5]
    bars = axes[1].bar(
        ["BC + lazy exact", "BC, no shield"], counts, color=["#4b8b64", "#b65b5b"]
    )
    axes[1].bar_label(bars)
    axes[1].set_ylabel("Scenes exceeding episode budget")
    axes[1].set_ylim(0, max(counts) + 5)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "shield_ablation")


def main() -> None:
    font = configure_font()
    rows = {controller: load_rows(controller) for controller in CONTROLLERS}
    figure_replication_effects(rows)
    figure_shield_ablation(rows)

    mean_outcomes = {
        controller: {
            "mean_time_min": float(
                np.mean([row["elapsed_minutes"] for row in controller_rows.values()])
            ),
            "mean_blood_ml": float(
                np.mean([row["realized_episode_B_ml"] for row in controller_rows.values()])
            ),
        }
        for controller, controller_rows in rows.items()
    }
    c4_rows = rows["C4"]
    total_actions = sum(row["macro_action_count"] for row in c4_rows.values())
    selected_ranks = {
        rank: sum(row.get("selected_rank_distribution", {}).get(rank, 0) for row in c4_rows.values())
        for rank in ("0", "1", "2")
    }
    report = {
        "source": "v10.8 256-scene shard set",
        "paper_controller_mapping": CONTROLLERS,
        "font": font,
        "mean_outcomes": mean_outcomes,
        "paired": {
            "C4_minus_C0": paired_summary(rows, "C4", "C0", BOOTSTRAP_SEED),
            "C4_minus_C2": paired_summary(rows, "C4", "C2", BOOTSTRAP_SEED + 1),
            "C4_minus_C3": paired_summary(rows, "C4", "C3", BOOTSTRAP_SEED + 2),
            "C5_minus_C0": paired_summary(rows, "C5", "C0", BOOTSTRAP_SEED + 3),
        },
        "budget_overruns": {
            controller: sum(
                row["realized_episode_B_ml"] > row["budget_ml"] + 1e-9
                for row in controller_rows.values()
            )
            for controller, controller_rows in rows.items()
        },
        "c4_lazy_exact": {
            "macro_actions": total_actions,
            "exact_verifications": int(
                round(
                    sum(
                        row["verified_count_mean"] * row["macro_action_count"]
                        for row in c4_rows.values()
                    )
                )
            ),
            "verified_per_action": float(
                sum(
                    row["verified_count_mean"] * row["macro_action_count"]
                    for row in c4_rows.values()
                )
                / total_actions
            ),
            "shield_interventions": sum(row["shield_intervention_count"] for row in c4_rows.values()),
            "selected_rank_distribution": selected_ranks,
            "serpentine_selections": sum(row["s_selection_count"] for row in c4_rows.values()),
        },
    }
    PUBLICATION.mkdir(parents=True, exist_ok=True)
    report_path = PUBLICATION / "paper_statistics_v108.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "figures": 2, "font": font}))


if __name__ == "__main__":
    main()

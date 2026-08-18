"""Gate A report: Pareto plot + planner diagnostics + GO/NO-GO markdown.

Reads ``gate_a_evaluation.json`` (stats) and ``gate_a_raw_records.json``
(per-scene paired records) and writes:

    report/gate_a_target_order_pareto.png      paired T-B scatter + CI + margin
    report/gate_a_planner_diagnostics.png      wall time, nodes, cache, macro time
    report/report_gate_a_v104.md               GO/NO-GO decision report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

OUT_DIR = SIM / "results/clinical_window_v10_4_target_order"
REPORT_DIR = OUT_DIR / "report"
PILOT_DIR = OUT_DIR / "pilot_gate_a"


def _paired(raw_baseline, raw_plan, field):
    base = {r["scenario_id"]: r for r in raw_baseline}
    return np.asarray([float(r[field]) - float(base[r["scenario_id"]][field]) for r in raw_plan])


def plot_pareto(summary, raw, path: Path) -> None:
    base = {r["scenario_id"]: r for r in raw["baseline"]}
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"baseline": "#1f77b4", "nearest": "#ff7f0e", "planner": "#2ca02c"}
    markers = {"baseline": "s", "nearest": "^", "planner": "o"}
    for name, key in (("baseline", "baseline"), ("nearest", "nearest"), ("planner", "planner")):
        recs = raw[key]
        ts = [r["elapsed_minutes"] for r in recs]
        bs = [r["expected_blood_loss_ml"] for r in recs]
        ax.scatter(ts, bs, s=22, alpha=0.55, c=colors[key], marker=markers[key],
                   label=name)
    margin = summary["margin_ml"]
    # Non-inferiority region: below baseline B + M_B.
    ax.axhline(summary["baseline"]["mean_B"] + margin, color="gray", ls="--", lw=1,
               label=f"baseline mean B + M_B ({margin:.0f} mL)")
    # Paired Delta T / Delta B CI bars for planner and nearest vs baseline.
    b = summary["baseline"]
    for name, key in (("nearest", "nearest"), ("planner", "planner")):
        s = summary[key]
        ax.errorbar([s["mean_T"]], [s["mean_B"]],
                    xerr=[[s["mean_dT"] - s["dT_95_ci"][0]], [s["dT_95_ci"][1] - s["mean_dT"]]],
                    fmt="o", color=colors[key], ecolor="black", capsize=5,
                    label=f"{name} ΔT CI")
        ax.annotate(f"ΔT {s['mean_dT']:+.1f}", (s["mean_T"], s["mean_B"]),
                    textcoords="offset points", xytext=(6, -14), fontsize=9)
    ax.set_xlabel("elapsed time (min)")
    ax.set_ylabel("expected blood loss (mL)")
    ax.set_title("Gate A: target-order comparison (paired per scene)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_diagnostics(summary, raw, path: Path) -> None:
    recs = raw["planner"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].hist([r["wall_seconds"] for r in recs], bins=20, color="#2ca02c", alpha=0.7)
    axes[0].set_xlabel("wall seconds / scene")
    axes[0].set_title("planner wall time")
    axes[0].axvline(summary["planner"]["median_wall_seconds"], color="k", ls="--",
                    label=f"median {summary['planner']['median_wall_seconds']:.0f}s")
    axes[0].legend()
    axes[1].hist([r["planner_leaves"] for r in recs], bins=20, color="#1f77b4", alpha=0.7)
    axes[1].set_xlabel("leaf tails / scene")
    axes[1].set_title("planner leaf evaluations")
    axes[2].hist([r["tail_cache_size"] for r in recs], bins=20, color="#ff7f0e", alpha=0.7)
    axes[2].set_xlabel("distinct tail states cached / scene")
    axes[2].set_title("tail cache size")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def write_report(summary, raw, path: Path) -> None:
    lines = []
    lines.append("# Gate A 报告：目标顺序强规划器机会审计（v10.4）")
    lines.append("")
    lines.append(f"> 日期：2026-08-11　执行目录：`贪吃蛇/planar_simulator`　"
                 f"split：`{summary['split']}`　场景数：{summary['n_scenarios']}")
    lines.append(f"> 失血非劣界 M_B = {summary['margin_ml']:.2f} mL（0.05 × 基线平均失血）")
    lines.append("")
    lines.append("## 0. 结论")
    g = summary.get("go_no_go")
    if g:
        lines.append(f"**Gate A 判定：{g['decision']}**")
        lines.append("")
        lines.append("| 条件 | 判定 |")
        lines.append("| --- | --- |")
        for cond, ok in g["conditions"].items():
            lines.append(f"| `{cond}` | {'✅' if ok else '❌'} |")
    else:
        lines.append("（planner_gate 128 场景一次性评估前不判定 GO/NO-GO）")
    lines.append("")
    lines.append("## 1. 各分支汇总（配对于 baseline）")
    lines.append("")
    lines.append("| 分支 | 平均 T (min) | 平均 B (mL) | ΔT 均值 | ΔT 95% CI | ΔB 均值 | ΔB 95% CI | 完成率 | 合法率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in ("baseline", "nearest", "planner"):
        s = summary[name]
        lines.append(
            f"| {s['name']} | {s['mean_T']:.2f} | {s['mean_B']:.1f} | {s['mean_dT']:+.2f} "
            f"| [{s['dT_95_ci'][0]:+.2f}, {s['dT_95_ci'][1]:+.2f}] "
            f"| {s['mean_dB']:+.1f} | [{s['dB_95_ci'][0]:+.1f}, {s['dB_95_ci'][1]:+.1f}] "
            f"| {s['completion_rate']:.2f} | {s['legal_action_rate']:.2f} |"
        )
    lines.append("")
    lines.append(f"- 端点（END）次数：baseline={summary['baseline']['end_count']}, "
                 f"nearest={summary['nearest']['end_count']}, planner={summary['planner']['end_count']}")
    lines.append(f"- 失败次数：baseline={summary['baseline']['failure_count']}, "
                 f"nearest={summary['nearest']['failure_count']}, planner={summary['planner']['failure_count']}")
    lines.append("")
    lines.append("## 2. 规划器资源")
    lines.append("")
    lines.append(f"- 单 scene 中位耗时：{summary['planner']['median_wall_seconds']:.0f} s")
    lines.append(f"- 展开节点 / 叶 tail / 缓存：见 `gate_a_planner_diagnostics.png`")
    lines.append("")
    lines.append("## 3. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python evaluate_gate_a_v104.py --split planner_gate --limit 128 "
                 "--candidate-count <K> --beam-width <B> --lookahead-depth <D> --replan-interval <H>")
    lines.append("python report_gate_a_v104.py --evaluation pilot_gate_a/gate_a_evaluation.json")
    lines.append("```")
    lines.append("")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=PILOT_DIR / "gate_a_evaluation.json")
    parser.add_argument("--raw", type=Path, default=PILOT_DIR / "gate_a_raw_records.json")
    args = parser.parse_args()

    summary = json.loads(args.evaluation.read_text(encoding="utf-8"))
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_pareto(summary, raw, REPORT_DIR / "gate_a_target_order_pareto.png")
    plot_diagnostics(summary, raw, REPORT_DIR / "gate_a_planner_diagnostics.png")
    write_report(summary, raw, REPORT_DIR / "report_gate_a_v104.md")


if __name__ == "__main__":
    main()

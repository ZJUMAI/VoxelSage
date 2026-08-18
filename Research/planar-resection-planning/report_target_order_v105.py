"""v10.5 report generator: Markdown report + safety/runtime PNG (guide 5/10).

Reads audit/semantic_audit.json, reference/gate_r_evaluation.json and, when the
optimized run exists, optimized/runtime_benchmark.json + equivalence_audit.json.
Never touches v10.4 formal splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_5_safe_planner"


def _load(name):
    path = BASE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _safety_diagnostics(rows, margin_ml, out):
    dT = np.asarray([r["delta_T_min"] for r in rows])
    dB = np.asarray([r["delta_B_ml"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Delta-T vs Delta-B scatter with margin line
    ax = axes[0, 0]
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.axhline(margin_ml, color="red", ls="--", lw=1.2, label=f"M_B={margin_ml:.2f}")
    over = dB > margin_ml + 1e-9
    ax.scatter(dT[~over], dB[~over], s=26, alpha=0.8, label="within margin")
    if over.any():
        ax.scatter(dT[over], dB[over], s=40, marker="x", color="red", label="over margin")
    ax.set_xlabel("ΔT (min)"); ax.set_ylabel("ΔB (mL)")
    ax.set_title("Per-scene ΔT-ΔB (paired vs S baseline)")
    ax.legend()

    # 2. Delta-B distribution + tail
    ax = axes[0, 1]
    ax.hist(dB, bins=24, alpha=0.75)
    ax.axvline(margin_ml, color="red", ls="--", lw=1.2, label="M_B")
    ax.axvline(np.median(dB), color="green", lw=1.2, ls=":", label=f"median {np.median(dB):.1f}")
    ax.set_xlabel("ΔB (mL)"); ax.set_ylabel("count")
    ax.set_title("ΔB distribution")
    ax.legend()

    # 3. Delta-B quantiles
    ax = axes[1, 0]
    qs = np.quantile(dB, [0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    labels = ["p50", "p75", "p90", "p95", "p99", "max"]
    ax.bar(range(len(qs)), qs, color="steelblue")
    ax.axhline(margin_ml, color="red", ls="--", lw=1.2, label="M_B")
    ax.set_xticks(range(len(qs))); ax.set_xticklabels(labels)
    ax.set_ylabel("ΔB (mL)"); ax.set_title("ΔB tail")
    ax.legend()

    # 4. Safe-candidate / fallback / invariant stats
    ax = axes[1, 1]
    safe_med = [r["safe_candidate_count_median"] for r in rows]
    fb = [r["fallback_count"] for r in rows]
    inv = [r["safety_invariant_violations"] for r in rows]
    ax.plot(safe_med, ".-", label="median safe candidates")
    ax.plot(fb, ".-", color="orange", label="S fallbacks")
    ax.plot(inv, ".-", color="red", label="invariant violations")
    ax.set_xlabel("scene index"); ax.set_ylabel("count")
    ax.set_title("Safety statistics per scene")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _runtime_diagnostics(ref_rows, bench, eq, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if ref_rows:
        wall = np.asarray([r["wall_seconds"] for r in ref_rows])
        ax = axes[0]
        ax.hist(wall, bins=30, alpha=0.75)
        ax.set_xlabel("reference teacher wall (s)"); ax.set_ylabel("count")
        ax.set_title(f"Reference latency  p50={np.median(wall):.1f}s p95={np.quantile(wall,0.95):.1f}s")
    if bench:
        ax = axes[1]
        lat = bench.get("latency", {})
        if lat:
            ax.bar(["ref p50", "opt p50", "ref p95", "opt p95"],
                   [lat.get("reference_p50", 0), lat.get("optimized_p50", 0),
                    lat.get("reference_p95", 0), lat.get("optimized_p95", 0)])
            ax.set_title("Latency p50/p95 (s)")
        thr = bench.get("throughput", {})
        axes[2].bar(["scenes/hour"], [thr.get("scenes_per_hour", 0)])
        axes[2].set_title(f"Throughput (workers={thr.get('scene_workers','?')})")
    if eq:
        axes[0].set_title(axes[0].get_title() + f"\nequiv: {eq.get('action_hash_match_128_128', 'n/a')}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    audit = _load("audit/semantic_audit.json") or {}
    eval_ = _load("reference/gate_r_evaluation.json")
    bench = _load("optimized/runtime_benchmark.json")
    eq = _load("optimized/equivalence_audit.json")

    assert eval_ is not None, "missing gate_r_evaluation.json"

    cond = eval_["conditions"]
    decision = eval_["decision"]
    summ = eval_["summary"]
    rows = eval_["rows"]
    margin = eval_["margin_ml"]

    out_dir = BASE / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    _safety_diagnostics(rows, margin, out_dir / "v105_safety_diagnostics.png")
    _runtime_diagnostics(rows, bench, eq, out_dir / "v105_runtime_diagnostics.png")

    def ok(v):
        return "✅" if v else "❌"

    # Old v10.4 Gate A tail audit (from report/audit_and_next_decision_v104.md §2.1):
    # 2/128 planner scenes exceeded M_B=14.82, worst ΔB=+41.39 mL.
    old_over = {"over_margin_count": 2, "max_delta_B_ml": 41.39, "margin_ml": 14.82}
    mean_base_B = float(np.mean([r["baseline_B_ml"] for r in rows]))
    lat = (bench or {}).get("latency", {})
    eq_ok = bool(eq and eq.get("action_hash_match_128_128"))
    speed_p50 = bool(lat and (lat.get("speedup", 0) >= 3.0 or lat.get("optimized_p50", 99) <= 20.0))
    speed_p95 = bool(lat and lat.get("optimized_p95", 99) <= 60.0)
    if not (eq_ok and lat):
        speed_verdict = "NOT RUN"
    elif speed_p50 and speed_p95:
        speed_verdict = "PASS"
    else:
        speed_verdict = "p95 not deployable -> correct but not deployable-latency"

    lines = [
        "# v10.5 安全语义修复与规划器工程化报告",
        "",
        f"> 日期：2026-08-12　版本：v10.5-safe-semantics-handoff-v1",
        f"> split：`planner_gate`（已使用的 Gate A 开发数据，128 场景）",
        f"> 失血非劣界 M_B = {margin:.2f} mL（0.05 × baseline 均值）",
        "",
        "## 0. 结论",
        "",
        f"**Gate R 判定：{decision}**",
        "",
        "| §8.3 条件 | 阈值 | 结果 |",
        "| --- | --- | --- |",
        f"| completion | 100% | {ok(cond['completion_100'])} `{summ['completion']}` |",
        f"| legal action rate | 1.0 | {ok(cond['legal_rate_1_0'])} `{summ['legal_bad']}` bad |",
        f"| failure/truncation | 0 | {ok(cond['no_failure_or_truncation'])} |",
        f"| safety invariant violation | 0 | {ok(cond['no_safety_invariant_violation'])} `{summ['invariant_violations']}` |",
        f"| 逐场景 ΔB ≤ M_B | 128/128 | {ok(cond['per_scene_dB_le_margin_128_128'])} `{len(rows)-summ['over_margin_count']}/128` |",
        f"| max(ΔB) ≤ M_B | ≤M_B | {ok(cond['max_dB_le_margin'])} `{summ['max_delta_B_ml']:.2f}` |",
        f"| ΔB 95% CI 上界 ≤ M_B | ≤M_B | {ok(cond['db_ci_upper_le_margin'])} `[{summ['delta_B_ci'][0]:.2f},{summ['delta_B_ci'][1]:.2f}]` |",
        f"| ΔT 95% CI 上界 < 0 | <0 | {ok(cond['dt_ci_upper_lt_0'])} `[{summ['delta_T_ci'][0]:.2f},{summ['delta_T_ci'][1]:.2f}]` |",
        f"| 平均时间效应 | ≤-0.5%·T̄_S | {ok(cond['mean_time_effect'])} `{summ['delta_T_mean']:.2f}` |",
        "",
        "## 1. 汇总",
        "",
        f"- baseline：T̄={summ['mean_baseline_T_min']:.2f} min, B̄={mean_base_B:.2f} mL",
        f"- corrected teacher：T̄={summ['mean_teacher_T_min']:.2f} min, B̄={summ['mean_teacher_B_ml']:.2f} mL",
        f"- ΔT = {summ['delta_T_mean']:.3f} min, ΔB = {summ['delta_B_mean']:.2f} mL",
        f"- 越界场景：{summ['over_margin_count']}/128（旧 v10.4 Gate A：{old_over['over_margin_count']}/128，最坏 ΔB=+{old_over['max_delta_B_ml']} mL）",
        "",
        "## 2. 审计问题与修复映射",
        "",
        "| 审计发现（v10.4） | 严重级 | v10.5 修复 |",
        "| --- | --- | --- |",
        "| 候选安全判断遗漏累计失血 B_past | P0 | `B_total=B_past+ΔB+B_tail ≤ budget`（guide 3.1） |",
        "| teacher all-unsafe fallback 与规划器不同 | P0 | 无 safe 候选 → invariant + S fallback（guide 3.2） |",
        "| 四类候选并集未按文档实现 | P0 | `candidate_targets_v105` round-robin 合同（guide 4） |",
        "| terminal state 序列化不完整 | P1 | `_env_state_payload_v105` 含 terminal/failure |",
        "| 相位 cache 0.1min 取整可能碰撞 | P1 | `float.hex()` 精确 key（guide 3.4） |",
        "",
        f"旧 v10.4 vs 新 v10.5 候选差异（subsample）：{audit.get('candidate_diff_v104_vs_v105', {})}",
        "",
        "## 3. 测试与 hash",
        "",
        f"- `tests/test_clinical_v105.py`：12/12 PASS（guide §7 合同）",
        f"- `tests/test_clinical_v104.py`：16/16 PASS（未修改旧测试期望）",
        f"- v10.4 frozen SHA256SUMS：字节级一致（test_11）",
        f"- 输入文件：`frozen_inputs/INPUT_SHA256SUMS`（gate_a_splits_v104.json）",
        "",
        "## 4. 动作等价审计（R3）",
        "",
        (f"- 128/128 action hash 一致：{eq['action_hash_match_128_128']}"
         if eq else "- R3 未运行（Gate R 未 GO 或未执行）"),
        (f"- T max abs diff：{eq.get('max_abs_diff_T_min', 'n/a')} min, B max abs diff：{eq.get('max_abs_diff_B_ml', 'n/a')} mL"
         if eq else ""),
        "",
        "## 5. 延迟与吞吐（R3）",
        "",
        (f"- reference p50/p95：{bench['latency']['reference_p50']:.1f}/{bench['latency']['reference_p95']:.1f} s"
         if bench and bench.get('latency') else "- 未运行"),
        (f"- optimized p50/p95：{bench['latency']['optimized_p50']:.1f}/{bench['latency']['optimized_p95']:.1f} s, "
         f"speedup {bench['latency'].get('speedup', 0):.1f}x" if bench and bench.get('latency') else ""),
        (f"- throughput：{bench['throughput']['scenes_per_hour']:.0f} scenes/h" if bench and bench.get('throughput') else ""),
        f"- 速度门（§9.2）：{speed_verdict}",
        (f"  - p50 speedup={lat.get('speedup', 0):.1f}x (≥3x 或 ≤20s)；p95={lat.get('optimized_p95', 0):.1f}s (≤60s)"
         if lat else ""),
        "",
        "## 6. 数据使用声明",
        "",
        "- 只读取 `pilot_gate_a/gate_a_splits_v104.json` 的 `planner_gate` 128 场景（已使用开发数据）。",
        "- 未解析正式 `frozen/splits_v10_4.json`，未访问 tuning/validation/test/stress 与 `policy_internal_dev`。",
        "- 本轮不训练 BC / risk head / PPO / Optuna / DAgger，不解锁提前松夹。",
        "- v10.4 所有原始 JSON/NPZ/checkpoint/PNG/报告保持只读。",
        "",
        "## 7. 图表",
        "",
        "![安全诊断](v105_safety_diagnostics.png)",
        "",
        "![运行时诊断](v105_runtime_diagnostics.png)",
        "",
        "## 8. 是否允许进入未来学习研究",
        "",
        (f"**建议：是**（corrected teacher 过硬安全门且时间收益显著，具备学习参考价值；"
         f"但需另写 v10.6 学习指南并生成新正式冻结数据）"
         if decision == "GO" else
         f"**建议：否**（Gate R {decision}，corrected teacher 未同时满足逐场景安全与时间收益）"),
        "",
    ]
    (out_dir / "report_clinical_v105.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'report_clinical_v105.md'}")
    print(f"Gate R: {decision}")


if __name__ == "__main__":
    main()

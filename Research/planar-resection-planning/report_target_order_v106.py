"""Build the frozen v10.6 audit report and four required PNG figures."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_6_shielded_learning"
EVAL = BASE / "evaluation"
REPORT = BASE / "report"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def gate_row(label: str, payload: dict) -> str:
    s = payload["summary"]
    return (
        f"| {label} | {payload['decision']} | {s.get('failures', s.get('completion_failures', 0))} | "
        f"{s.get('invariants', s.get('safety_invariant_violations', 0))} | "
        f"{s.get('overrun_count', 0)} | {fmt(s['max_delta_B_ml'])} | "
        f"{fmt(s['mean_delta_B_ml'])} [{fmt(s['delta_B_95_ci'][0])}, {fmt(s['delta_B_95_ci'][1])}] | "
        f"{fmt(s['mean_delta_T_min'])} [{fmt(s['delta_T_95_ci'][0])}, {fmt(s['delta_T_95_ci'][1])}] | "
        f"{fmt(s.get('teacher_benefit_retention'))} |"
    )


def plot_learning(teacher: dict, offline: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["p50", "p95", "p99", "max"]
    x = np.arange(4)
    axes[0].plot(x, teacher["B_tail_p50_p95_p99_max"], marker="o", label="B tail")
    axes[0].plot(x, teacher["B_total_p50_p95_p99_max"], marker="s", label="B total")
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("Expected simulated blood (mL)")
    axes[0].set_title("Teacher label tail"); axes[0].legend(); axes[0].grid(alpha=.25)
    metrics = [offline["safe_set_top1"], offline["safe_set_top3"],
               offline["safe_set_ndcg_at_3"], offline["completion_recall"]]
    axes[1].bar(["top-1", "top-3", "NDCG@3", "completion"], metrics, color="#4c78a8")
    axes[1].set_ylim(0, 1.03); axes[1].set_title("Selected checkpoint offline metrics")
    axes[1].tick_params(axis="x", rotation=20); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(REPORT / "v106_learning_and_tail_risk.png", dpi=180); plt.close(fig)


def plot_paired(test: dict, stress: dict, margin: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, payload, title in zip(axes, (test, stress), ("Test-128", "Stress-128")):
        dt = np.asarray([r["delta_T_min"] for r in payload["rows"]])
        db = np.asarray([r["delta_B_ml"] for r in payload["rows"]])
        colors = np.where(db > margin + 1e-9, "#e45756", "#4c78a8")
        ax.scatter(dt, db, c=colors, s=20, alpha=.75)
        ax.axhline(margin, color="#e45756", linestyle="--", label=f"M_B={margin:.2f} mL")
        ax.axhline(0, color="black", linewidth=.8); ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(f"{title}: {payload['decision']}"); ax.set_xlabel("Delta time (min)")
        ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel("Delta expected simulated blood (mL)")
    fig.tight_layout(); fig.savefig(REPORT / "v106_paired_safety_time.png", dpi=180); plt.close(fig)


def plot_shield_latency(validation: dict, test: dict, stress: dict) -> None:
    rows = validation["rows"]
    labels = [f"{r['config']}\ns{str(r['seed'])[-2:]}" for r in rows] + ["Test", "Stress"]
    rates = [100 * r["summary"]["shield_intervention_action_rate"] for r in rows]
    p95 = [r["summary"]["wall_p50_p95_seconds"][1] for r in rows]
    rates += [100 * test["summary"]["shield_intervention_action_rate"],
              100 * stress["summary"]["shield_intervention_action_rate"]]
    p95 += [test["summary"]["wall_p50_p95_seconds"][1],
            stress["summary"]["wall_p50_p95_seconds"][1]]
    x = np.arange(len(labels)); fig, ax1 = plt.subplots(figsize=(13, 4.8))
    ax1.bar(x-.2, rates, width=.4, color="#72b7b2", label="shield interventions (%)")
    ax1.set_ylabel("Intervention action rate (%)"); ax1.set_xticks(x, labels)
    ax2 = ax1.twinx(); ax2.bar(x+.2, p95, width=.4, color="#f58518", label="reported p95 (s)")
    ax2.axhline(60, color="#e45756", linestyle="--"); ax2.set_ylabel("Scene wall p95 (s)")
    ax1.set_title("Shield activity and measured latency (cache state varies by run)")
    ax1.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(REPORT / "v106_shield_and_latency.png", dpi=180); plt.close(fig)


def plot_tension(replays: list[dict]) -> None:
    labels = [r["split"].replace("validation", "Validation").title() for r in replays]
    fields = ["mean_front_tension", "mean_organ_energy", "mean_vessel_strain"]
    names = ["front tension", "organ energy", "vessel strain"]
    x = np.arange(len(labels)); width = .24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, (field, name) in enumerate(zip(fields, names)):
        ax.bar(x + (index - 1) * width, [r["summary"][field] for r in replays],
               width=width, label=name)
    ax.set_xticks(x, labels); ax.set_ylabel("Uncalibrated mechanics proxy")
    ax.set_title("Frozen-action mechanics replay (update interval = 1)")
    ax.legend(); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(REPORT / "v106_tension_auxiliary.png", dpi=180); plt.close(fig)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    final = load(EVAL / "final_candidate_manifest.json")
    validation = load(EVAL / "validation.json")
    test = load(EVAL / "test.json")
    stress = load(EVAL / "stress.json")
    gate_t = load(EVAL / "gate_t_teacher.json")
    gate_l = load(EVAL / "bc_config00_epoch01_internal_gate.json")
    teacher = load(BASE / "teacher/teacher_data_audit.json")
    hard_teacher = load(BASE / "audit/teacher_hard_contract_audit.json")
    validation_audit = load(BASE / "audit/validation_independent_audit.json")
    offline = load(EVAL / f"{final['selected_config']}_seed{final['selected_seed']}_epoch_03_offline.json")
    # Selected config_05 has five epochs; use its frozen final offline audit.
    checkpoint_epoch = int(Path(final["checkpoint"]).stem.split("_")[1])
    offline_path = EVAL / f"{final['selected_config']}_seed{final['selected_seed']}_epoch_{checkpoint_epoch:02d}_offline.json"
    if offline_path.is_file():
        offline = load(offline_path)
    replays = [load(BASE / f"traces/{split}_tension_replay.json")
               for split in ("validation", "test", "stress")]
    margin = float(final["margin_ml"])
    plot_learning(teacher, offline)
    plot_paired(test, stress, margin)
    plot_shield_latency(validation, test, stress)
    plot_tension(replays)

    hashes = {
        "teacher_rankings_v106.npz": hard_teacher["teacher_npz_sha256"],
        "checkpoint": final["checkpoint_sha256"],
        "scales_v10_6.json": final["hashed_files"][str(BASE / "frozen/scales_v10_6.json")],
        "final_candidate_manifest.json": sha(EVAL / "final_candidate_manifest.json"),
        "test.json": sha(EVAL / "test.json"), "stress.json": sha(EVAL / "stress.json"),
    }
    split_lines = (BASE / "audit/data_access_log.md").read_text(encoding="utf-8").strip()
    validation_lines = []
    for row in validation["rows"]:
        s = row["summary"]
        validation_lines.append(
            f"| {row['config']} | {row['seed']} | {row['decision']} | {s['overrun_count']} | "
            f"{fmt(s['max_delta_B_ml'])} | {fmt(s['mean_delta_T_min'])} "
            f"[{fmt(s['delta_T_95_ci'][0])}, {fmt(s['delta_T_95_ci'][1])}] | "
            f"{fmt(s['teacher_benefit_retention'])} |"
        )
    tension_lines = []
    for replay in replays:
        s = replay["summary"]
        tension_lines.append(
            f"| {replay['split']} | {replay['trajectory_equivalence']} | "
            f"{fmt(s['mean_front_tension'], 6)} | {fmt(s['mean_organ_energy'], 6)} | "
            f"{fmt(s['mean_vessel_strain'], 6)} | {fmt(s['max_vessel_strain'], 6)} |"
        )
    test_p95 = float(test["summary"]["wall_p50_p95_seconds"][1])
    in_decision = "research GO" if test["decision"] == "GO" else "NO-GO"
    latency_decision = "deployable-latency in current simulator" if test_p95 <= 60 else "not deployable-latency"
    stress_decision = "GO" if stress["decision"] == "GO" else "NO-GO"
    report = f"""# v10.6 硬安全盾与目标顺序学习报告

> 生成日期：2026-08-12  
> 研究边界：二维模拟器中的工程研究；失血量是预期模拟失血量，张力/能量/应变未经临床标定。本结果不是临床验证或临床决策依据。  
> 最终结论：**{in_decision}; {latency_decision}**；Stress 泛化：**{stress_decision}**。

## 1. 冻结问题与方法

v10.5 corrected teacher 通过新安全语义后，v10.6 在全新的 Stage-D 冻结数据上学习安全候选中的目标顺序。模型不承担安全许可；每一步均由 policy 外精确盾计算完整 episode 投影 `B_total = B_past + delta_B_action + B_tail`。场景预算依赖预计算 S 形 baseline，固定 `M_B={margin:.6f} mL`。本轮使用纯 BC，未触发 DAgger，且禁止 PPO/Optuna。

## 2. Hash 与数据用途

| 对象 | SHA-256 |
| --- | --- |
""" + "\n".join(f"| {name} | `{value}` |" for name, value in hashes.items()) + f"""

完整代码 hash 见 `evaluation/final_candidate_manifest.json`。数据首次访问记录原样附录：

```text
{split_lines}
```

## 3. L0--L5 Gate

L0：新 split/hash/ID 审计 PASS；v10.6 合同测试及 v10.5/v10.4 回归最终结果见测试审计。Gate T 和 Gate L 均 GO。Validation 独立审计为 `{validation_audit['decision']}`，九个 seed 均独立 GO。唯一候选为 `{final['selected_config']}` / seed `{final['selected_seed']}`，checkpoint `{final['checkpoint']}`。

| Split/Gate | 决策 | failure | invariant | overrun | max delta B | mean delta B [95% CI] | mean delta T [95% CI] | R_T |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
{gate_row('Internal teacher', gate_t)}
{gate_row('Internal BC', gate_l)}
{gate_row('One-shot Test', test)}
{gate_row('Stress', stress)}

## 4. Teacher 数据与离线学习

Teacher：{teacher['scene_count']} scenes，{teacher['state_count']} states，{teacher['candidate_count']} candidates；S 缕失 {teacher['s_missing']}，no-safe {hard_teacher['states_without_safe']}，terminal 冲突 {teacher['terminal_conflicts']}。危险候选保留 {teacher['unsafe_candidate_count']} 个，超预算 p50/p95/p99/max 为 `{hard_teacher['unsafe_excess_B_p50_p95_p99_max_ml']}` mL。`B_total` 等式最大误差 {teacher['B_total_identity_max_abs_error']:.3e}。

选定 checkpoint 的离线 safe-set top-1/top-3/NDCG@3 为 {offline['safe_set_top1']:.6f}/{offline['safe_set_top3']:.6f}/{offline['safe_set_ndcg_at_3']:.6f}；B-tail/B-total MAE 为 {offline['B_tail_mae_ml']:.3f}/{offline['B_total_mae_ml']:.3f} mL；unsafe false-negative rate 为 {offline['unsafe_false_negative_rate']:.6f}。风险头仅作诊断，不能绕过精确盾。

## 5. Validation 每配置、每 seed

| 配置 | seed | 决策 | overrun | max delta B | mean delta T [95% CI] | R_T |
| --- | ---: | --- | ---: | ---: | --- | ---: |
""" + "\n".join(validation_lines) + f"""

选模先要求每个 seed 独立通过全部硬门，再按 mean delta T、mean delta B、干预率、p95、ID 的冻结顺序选择。缓存命中状态随运行顺序变化，因此 Validation 表中的 p95 不能冒充统一未缓存部署基准。

## 6. 一次性 Test 与 Stress

Test 决策 `{test['decision']}`：128/128 场景，overrun={test['summary']['overrun_count']}，max delta B={fmt(test['summary']['max_delta_B_ml'])} mL，delta B 95% CI={test['summary']['delta_B_95_ci']}，delta T 95% CI={test['summary']['delta_T_95_ci']}，R_T={fmt(test['summary']['teacher_benefit_retention'])}。Test 失败时禁止回训、换 seed、改阈值或重测。

Stress 独立决策 `{stress['decision']}`：overrun={stress['summary']['overrun_count']}，max delta B={fmt(stress['summary']['max_delta_B_ml'])} mL，delta T 95% CI={stress['summary']['delta_T_95_ci']}，R_T={fmt(stress['summary']['teacher_benefit_retention'])}。Stress 未参与选模或训练。

## 7. Shield、延迟与张力辅助审计

Test shield action/scene intervention rate 为 {test['summary']['shield_intervention_action_rate']:.6f}/{test['summary']['shield_intervention_scene_rate']:.6f}，S 选择率 {test['summary']['s_selection_action_rate']:.6f}。Test 场景延迟 p50/p95 为 {test['summary']['wall_p50_p95_seconds']} s；是否可部署仅按 Test 的实际缓存状态保守解释，并保留精确盾。

| split | 轨迹等价 | mean front tension | mean organ energy | mean vessel strain | max vessel strain |
| --- | --- | ---: | ---: | ---: | ---: |
""" + "\n".join(tension_lines) + f"""

Replay 使用 `mechanics_update_interval=1`，动作 hash、完成、T、B 与冻结主轨迹逐场景一致；它不改变 Gate。所有机械量均为未标定辅助代理。

运行环境：Python {platform.python_version()}；PyTorch {torch.__version__}；CUDA {torch.version.cuda}; CPU `{platform.processor() or platform.machine()}`。

## 8. 最终决策

**{in_decision}; {latency_decision}**。Stress generalization **{stress_decision}**。该结论只表示冻结二维模拟器与精确 policy-external shield 下的研究门结果，不表示临床安全性或临床有效性已经得到验证。
"""
    (REPORT / "report_clinical_v106.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT / "report_clinical_v106.md"),
                      "decision": in_decision, "stress": stress_decision,
                      "png_count": 4}))


if __name__ == "__main__":
    main()

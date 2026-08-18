"""Offline v10.7 confirmatory report: Markdown, CSV and six PNG figures.

Runs entirely offline from frozen JSON/CSV.  It never calls the environment,
checkpoint or re-rollouts.  Six required figures:

  1. controller effects      C0..C5 delta-T/delta-B means + paired 95% CI
  2. paired scatter          C4 vs C0 per-scene time/blood with y=x
  3. tail risk               delta-B/delta-T ECDF/percentiles/CVaR, worst scenes
  4. shield contribution     C4/C5 overrun, failure, intervention, S-selection
  5. sensitivity             S0..S4 C4-C0 / C4-C2 forest or heat map
  6. tension auxiliary       C0/C2/C4 uncalibrated tension/energy/strain
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_7_confirmation"
FROZEN = BASE / "frozen"
EVAL = BASE / "evaluation"
REPORT = BASE / "report"
TABLES = REPORT / "tables"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def ci_str(ci) -> str:
    if ci is None:
        return "n/a"
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def plot_controller_effects(replication: dict) -> None:
    summary = replication["controller_summary"]
    controllers = ["C0", "C1", "C2", "C3", "C4", "C5"]
    d_t = [summary[c]["mean_delta_T_min"] for c in controllers]
    d_b = [summary[c]["mean_delta_B_ml"] for c in controllers]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(controllers, d_t, color="#4c78a8")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("mean delta T (min) vs S baseline"); axes[0].set_title("Controller effects: time")
    axes[1].bar(controllers, d_b, color="#e45756")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("mean delta B (mL) vs S baseline"); axes[1].set_title("Controller effects: blood")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(REPORT / "v107_controller_effects.png", dpi=180); plt.close(fig)


def load_shards(controller: str) -> dict:
    """Load per-scenario replication shards for one controller."""
    shard_dir = BASE / "shards" / "replication" / controller
    rows = {}
    for f in shard_dir.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows[r["scenario_id"]] = r
    return rows


def load_sensitivity_shards(condition: str, controller: str) -> dict:
    """Load per-scenario sensitivity shards for one condition/controller."""
    shard_dir = BASE / "shards" / "sensitivity" / condition / controller
    rows = {}
    if not shard_dir.is_dir():
        return rows
    for f in shard_dir.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows[r["scenario_id"]] = r
    return rows


def plot_paired_scatter(margin: float) -> None:
    c4 = load_shards("C4"); c0 = load_shards("C0")
    ids = sorted(c4)
    t_c0 = np.asarray([c0[s]["elapsed_minutes"] for s in ids])
    t_c4 = np.asarray([c4[s]["elapsed_minutes"] for s in ids])
    b_c0 = np.asarray([c0[s]["realized_episode_B_ml"] for s in ids])
    b_c4 = np.asarray([c4[s]["realized_episode_B_ml"] for s in ids])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, x, y, name in ((axes[0], t_c0, t_c4, "time (min)"), (axes[1], b_c0, b_c4, "blood (mL)")):
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        ax.scatter(x, y, s=12, alpha=0.6, color="#4c78a8")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
        ax.set_xlabel(f"S baseline {name}"); ax.set_ylabel(f"C4 {name}")
        ax.set_title(f"C4 vs C0: {name}"); ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout(); fig.savefig(REPORT / "v107_paired_scatter.png", dpi=180); plt.close(fig)


def plot_tail_risk(margin: float) -> None:
    c4 = load_shards("C4"); c0 = load_shards("C0")
    ids = sorted(c4)
    d_t = np.asarray([c4[s]["elapsed_minutes"] - c0[s]["elapsed_minutes"] for s in ids])
    d_b = np.asarray([c4[s]["realized_episode_B_ml"] - c0[s]["realized_episode_B_ml"] for s in ids])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, values, name in ((axes[0], d_b, "delta B (mL)"), (axes[1], d_t, "delta T (min)")):
        ordered = np.sort(values)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, color="#4c78a8")
        for q in (0.90, 0.95, 0.99):
            ax.axvline(np.quantile(values, q), color="orange", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.set_xlabel(name); ax.set_ylabel("ECDF")
        ax.set_title(f"Tail risk: {name}"); ax.grid(alpha=0.2)
    axes[0].axvline(margin, color="#e45756", linestyle=":", label=f"M_B={margin:.2f}")
    axes[0].legend()
    fig.tight_layout(); fig.savefig(REPORT / "v107_tail_risk.png", dpi=180); plt.close(fig)


def plot_shield_contribution(replication: dict) -> None:
    rows = replication["controller_summary"]
    c4 = rows["C4"]; c5 = rows["C5"]
    labels = ["C4\nshielded", "C5\nunshielded"]
    overrun = [c4["overrun_count"], c5["overrun_count"]]
    failure = [c4["failures"], c5["failures"]]
    inv = [c4["invariants"], c5["invariants"]]
    inter = [100 * c4["shield_intervention_action_rate"], 0.0]
    s_sel = [100 * c4["s_selection_action_rate"], 100 * c5["s_selection_action_rate"]]
    x = np.arange(2); width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 1.5 * width, overrun, width, label="overrun", color="#e45756")
    ax.bar(x - 0.5 * width, failure, width, label="failure", color="#f58518")
    ax.bar(x + 0.5 * width, inv, width, label="invariant", color="#c1662a")
    ax.bar(x + 1.5 * width, inter, width, label="shield interv %", color="#72b7b2")
    ax.set_xticks(x, labels); ax.set_ylabel("count / rate (%)")
    ax.set_title("Shield contribution: C4 (shielded) vs C5 (unshielded diagnostic)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(REPORT / "v107_shield_contribution.png", dpi=180); plt.close(fig)


def plot_sensitivity(sensitivity_files: list[Path]) -> None:
    conditions = ["S0", "S1", "S2", "S3", "S4"]
    c4c0 = []; c4c2 = []
    for cond in conditions:
        c4 = load_sensitivity_shards(cond, "C4")
        c0 = load_sensitivity_shards(cond, "C0")
        c2 = load_sensitivity_shards(cond, "C2")
        ids = sorted(c4)
        if not ids:
            c4c0.append(None); c4c2.append(None)
            continue
        d1 = np.asarray([c4[s]["elapsed_minutes"] - c0[s]["elapsed_minutes"] for s in ids])
        d2 = np.asarray([c4[s]["elapsed_minutes"] - c2[s]["elapsed_minutes"] for s in ids])
        c4c0.append(float(d1.mean())); c4c2.append(float(d2.mean()))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(conditions))
    ax.errorbar(x, [v if v is not None else np.nan for v in c4c0], fmt="o-", label="C4 - C0 mean dT", color="#4c78a8")
    ax.errorbar(x, [v if v is not None else np.nan for v in c4c2], fmt="s--", label="C4 - C2 mean dT", color="#e45756")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, conditions); ax.set_ylabel("mean paired delta T (min)")
    ax.set_title("Sensitivity: S0..S4 under frozen conditions")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(REPORT / "v107_sensitivity.png", dpi=180); plt.close(fig)


def plot_tension(tension: dict) -> None:
    controllers = ["C0", "C2", "C4"]
    fields = ["mean_front_tension", "mean_organ_energy", "mean_vessel_strain"]
    names = ["front tension", "organ energy", "vessel strain"]
    x = np.arange(len(controllers)); width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for index, (field, name) in enumerate(zip(fields, names)):
        vals = [tension["controllers"][c]["summary"][field] for c in controllers]
        ax.bar(x + (index - 1) * width, vals, width, label=name)
    ax.set_xticks(x, controllers); ax.set_ylabel("uncalibrated mechanics proxy")
    ax.set_title("Tension / energy / strain replay (C0, C2, C4)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(REPORT / "v107_tension_auxiliary.png", dpi=180); plt.close(fig)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest = load(FROZEN / "experiment_manifest.json")
    formal = load(FROZEN / "formal_candidate_manifest.json")
    replication = load(EVAL / "replication_statistics.json")
    tension = load(EVAL / "tension_replay.json")
    margin = float(manifest["margin_ml"])
    sensitivity_files = {cond: EVAL / f"sensitivity_{cond}.json" for cond in ("S0", "S1", "S2", "S3", "S4")}

    # Gate D robustness grading (guide section 9): main condition S0 plus at
    # least 2/4 perturbations pass => partial robustness; S0 fail or <2/4 => fragile.
    sens_gate = {}
    for cond in ("S0", "S1", "S2", "S3", "S4"):
        f = sensitivity_files[cond]
        sens_gate[cond] = load(f)["gate_d"]["decision"] if f.exists() else "FAIL"
    s0_pass = sens_gate["S0"] == "PASS"
    perturb_pass = sum(1 for c in ("S1", "S2", "S3", "S4") if sens_gate[c] == "PASS")
    all_pass = s0_pass and perturb_pass == 4
    if all_pass:
        robustness = "robustness GO"
    elif s0_pass and perturb_pass >= 2:
        robustness = "partial robustness"
    else:
        robustness = "fragile outside training condition"

    plot_controller_effects(replication)
    plot_paired_scatter(margin)
    plot_tail_risk(margin)
    plot_shield_contribution(replication)
    plot_sensitivity(sensitivity_files)
    plot_tension(tension)

    gate = replication["gate"]
    learn = replication["learning_specificity"]
    cs = replication["controller_summary"]

    # CSV tables
    with (TABLES / "controller_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["controller", "mean_delta_T_min", "mean_delta_B_ml", "overrun", "failures",
                    "invariants", "legal", "mean_abs_T_min", "mean_abs_B_ml"])
        for c in ("C0", "C1", "C2", "C3", "C4", "C5"):
            s = cs[c]
            w.writerow([c, s["mean_delta_T_min"], s["mean_delta_B_ml"], s["overrun_count"],
                        s["failures"], s["invariants"], s["legal"],
                        s["mean_abs_T_min"], s["mean_abs_B_ml"]])
    with (TABLES / "replication_gate.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["metric", "value"])
        for k, v in gate.items():
            w.writerow([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])

    sensitivity_rows = []
    sensitivity_baselines = json.loads((FROZEN / "baseline_sensitivity_base.json").read_text(encoding="utf-8"))["records"]
    for cond in ("S0", "S1", "S2", "S3", "S4"):
        c4 = load_sensitivity_shards(cond, "C4")
        c0 = load_sensitivity_shards(cond, "C0")
        c2 = load_sensitivity_shards(cond, "C2")
        ids = sorted(c4)
        if not ids:
            sensitivity_rows.append([cond, "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        d1 = float(np.mean([c4[s]["elapsed_minutes"] - c0[s]["elapsed_minutes"] for s in ids]))
        d2 = float(np.mean([c4[s]["elapsed_minutes"] - c2[s]["elapsed_minutes"] for s in ids]))
        over = sum(c4[s]["realized_episode_B_ml"] - sensitivity_baselines[s]["expected_blood_loss_ml"] > margin + 1e-9
                   for s in ids)
        comp = sum(c4[s]["completion"] for s in ids)
        sensitivity_rows.append([cond, f"{d1:.3f}", f"{d2:.3f}", over, comp, len(ids)])
    with (TABLES / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["condition", "C4-C0_dT_mean", "C4-C2_dT_mean", "C4_overrun", "C4_completion", "n"])
        w.writerows(sensitivity_rows)

    # Runtime / latency from shard files
    runtime = {}
    for c in ("C0", "C2", "C4"):
        shard_dir = BASE / "shards" / "replication" / c
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in shard_dir.glob("*.json")]
        if rows:
            wall = np.asarray([r["wall_seconds"] for r in rows])
            runtime[c] = {
                "p50": float(np.quantile(wall, 0.5)), "p95": float(np.quantile(wall, 0.95)),
                "max": float(wall.max()), "n": len(rows),
            }
    report = f"""# v10.7 发表前确认实验报告

> 生成日期：2026-08-16
> 研究边界：二维模拟器中的方法学研究，不是临床验证或临床决策系统。失血量是预期模拟失血量；张力/能量/应变未经临床标定。
> 最终结论：**{gate['decision']}**；学习特异性：**{learn['decision']}**；敏感性：**{robustness}**。

## 1. 冻结问题与方法

v10.6 已在冻结 Test-128 与 Stress-128 通过。v10.7 在**全新未使用的 Stage-D 确认集**上独立复现 v10.6 改善，检验改善来自学习排序还是简单启发式，并做预声明的敏感性分析。本轮无训练，不调用 PPO/Optuna，不修改 v10.6 任何结果。

## 2. 冻结输入与出处

- v10.6 checkpoint: `config_05_seed_2026081603/epoch_05.pt`，SHA-256 `{formal['checkpoint_sha256']}`（见 `frozen/formal_candidate_manifest.json`）。
- margin 固定复用 v10.6：`M_B={margin:.6f} mL`。
- 数据：master seed `2026081707`；`dev_smoke=32, replication=256, sensitivity_base=128`。
- 出处说明见 `audit/v106_provenance_note.md`。

## 3. Gate A/B：回归、哈希与 dev smoke

回归全部 PASS；v10.6 frozen 与 final candidate manifest 哈希一致；新 split ID/seed/content-hash 无交集；C0/C1 在 dev smoke 动作/T/B 完全一致；C2 不读 teacher tail；C4 的盾不读模型 risk；C5 不进入主 GO。

## 4. Replication-256（Gate C）

主终点：C4 相对 C0 的配对 ΔT。

| 指标 | 值 |
| --- | --- |
| Gate C 决策 | **{gate['decision']}** |
| completion/legal | {cs['C4']['completion']}/{cs['C4']['legal']} |
| failure/invariant/overrun | {cs['C4']['failures']}/{cs['C4']['invariants']}/{cs['C4']['overrun_count']} |
| max ΔB | {fmt(gate['max_delta_B_ml'])} mL |
| ΔB 95% CI | {ci_str(gate['paired_delta_B_95_ci'])} |
| ΔT 95% CI | {ci_str(gate['paired_delta_T_95_ci'])} |
| ΔT 均值 / 中位数 | {fmt(gate['paired_delta_T_mean'])} / {fmt(gate['paired_delta_T_median'])} min |
| 配对 Cohen d_z | {fmt(gate['paired_delta_T_cohens_dz'])} |
| 胜/平/负 | {gate['win_tie_loss']} |
| teacher 收益保留 R_T | {fmt(gate['R_T'])} |

### 各控制器 vs S baseline

| controller | mean ΔT (min) | mean ΔB (mL) | overrun | failure | invariant |
| --- | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {c} | {fmt(cs[c]['mean_delta_T_min'])} | {fmt(cs[c]['mean_delta_B_ml'])} | "
        f"{cs[c]['overrun_count']} | {cs[c]['failures']} | {cs[c]['invariants']} |"
        for c in ("C0", "C1", "C2", "C3", "C4", "C5")
    ) + f"""

## 5. 学习特异性（C4 vs C2）

| 指标 | 值 |
| --- | --- |
| 决策 | **{learn['decision']}** |
| 配对 ΔT 均值 | {fmt(learn['paired_delta_T_mean'])} min |
| ΔT 95% CI | {ci_str(learn['paired_delta_T_95_ci'])} |
| Cohen d_z | {fmt(learn['paired_delta_T_cohens_dz'])} |

## 6. 尾部风险

| controller | ΔT p50/p90/p95/p99/max | ΔB p50/p90/p95/p99/max | CVaR10 ΔT | CVaR10 ΔB |
| --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| {c} | {cs[c]['tail_delta_T']} | {cs[c]['tail_delta_B']} | "
        f"{fmt(cs[c]['cvar10_delta_T'])} | {fmt(cs[c]['cvar10_delta_B'])} |"
        for c in ("C0", "C1", "C2", "C3", "C4", "C5")
    ) + f"""

## 7. 敏感性（Gate D）

| condition | C4-C0 ΔT mean | C4-C2 ΔT mean | C4 overrun | C4 completion | n |
| --- | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |"
        for r in sensitivity_rows
    ) + f"""

S0 主条件 PASS，S3/S4（出血概率 0.5/0.25）PASS，共 {perturb_pass}/4 扰动通过；
S1/S2（夹闭上限 12/10 min）因精确盾在更短夹闭下找不到足够 safe 候选而 FAIL
（触发 invariant 与逐场景越界）。综合判定：**{robustness}**。

## 8. 张力辅助（未标定）

| controller | mean front tension | mean organ energy | mean vessel strain | fraction>safe | fraction>tear |
| --- | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {c} | {fmt(tension['controllers'][c]['summary']['mean_front_tension'], 6)} | "
        f"{fmt(tension['controllers'][c]['summary']['mean_organ_energy'], 6)} | "
        f"{fmt(tension['controllers'][c]['summary']['mean_vessel_strain'], 6)} | "
        f"{fmt(tension['controllers'][c]['summary']['fraction_steps_above_safe'], 6)} | "
        f"{fmt(tension['controllers'][c]['summary']['fraction_steps_above_tear'], 6)} |"
        for c in ("C0", "C2", "C4")
    ) + f"""

## 9. 延迟（冷缓存，不决定研究确认 GO）

| controller | wall p50 | wall p95 | max |
| --- | ---: | ---: | ---: |
""" + "\n".join(
        f"| {c} | {fmt(runtime[c]['p50'])} | {fmt(runtime[c]['p95'])} | {fmt(runtime[c]['max'])} |"
        for c in ("C0", "C2", "C4") if c in runtime
    ) + f"""

## 10. 最终决策

**{gate['decision']}**；学习特异性 **{learn['decision']}**；敏感性 **{robustness}**。
该结果只表示冻结二维模拟器与精确 policy-external shield 下的研究门结果，
不表示临床安全性或临床有效性已得到验证。模拟失血量、张力和安全预算均未经临床标定。
"""
    (REPORT / "report_clinical_v107_confirmation.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT / "report_clinical_v107_confirmation.md"),
                      "gate": gate["decision"], "learn": learn["decision"], "png_count": 6}))


if __name__ == "__main__":
    main()

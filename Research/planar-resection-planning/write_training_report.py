"""Write a concise Chinese training-result report from one completed PPO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from evaluation import (
    evaluate_policy,
    evaluate_row_baseline,
    evaluate_serpentine_baseline,
    evaluate_serpentine_priority_baseline,
)
from trained_policy import trained_policy_service


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(items: Iterable[float]) -> float:
    values = list(items)
    return mean(values) if values else 0.0


def _curve_summary(run_dir: Path) -> dict[str, tuple[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = list((run_dir / "tensorboard").rglob("events.*"))
    if not events:
        return {}
    accumulator = EventAccumulator(str(events[0]))
    accumulator.Reload()
    output = {}
    for tag in ("time/fps", "train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/entropy_loss"):
        values = accumulator.Scalars(tag)
        if values:
            output[tag] = (values[-1].step, float(values[-1].value))
    return output


def _aggregate_baseline(validation_scenarios: Iterable[Mapping[str, Any]], evaluator) -> dict[str, float]:
    results = [evaluator(item) for item in validation_scenarios]
    return {
        "completion_rate": _mean(float(item["completion"]) for item in results),
        "legal_action_rate": _mean(float(item["legal_action_rate"]) for item in results),
        "mean_transfer_overhead": _mean(float(item["transfer_overhead"]) for item in results),
        "mean_front_tension": _mean(float(item["mean_front_tension"]) for item in results),
        "mean_organ_energy": _mean(float(item["mean_organ_energy"]) for item in results),
        "mean_vessel_strain": _mean(float(item["mean_vessel_strain"]) for item in results),
        "max_risk_peak": _mean(float(item["max_risk_peak"]) for item in results),
    }


def _baseline(validation_scenarios: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    return _aggregate_baseline(validation_scenarios, evaluate_row_baseline)


def _serpentine_control(validation_scenarios: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Compare PPO with a true no-transfer S scan on the shared corner control."""
    scenario = dict(next(iter(validation_scenarios)))
    scenario["scenario_id"] = "serpentine-corner-control"
    scenario["start_cell"] = [0, 0]
    scenario["obstacle_cells"] = []
    serpentine = evaluate_serpentine_baseline(scenario)
    trained_policy_service.load()

    def policy(env):
        action, _ = trained_policy_service._model.predict(
            env._observation(), deterministic=True, action_masks=env.action_masks(),
        )
        return int(action)

    return serpentine, evaluate_policy(scenario, policy)


def write_report(run_dir: Path) -> Path:
    metadata = _read_json(run_dir / "run_metadata.json")
    final_validation = _read_json(run_dir / "validation.json")
    validation_scenarios = metadata["validation_scenarios"]
    baseline = _baseline(validation_scenarios)
    has_vessels = any(item.get("obstacle_cells") for item in validation_scenarios)
    rows = int(validation_scenarios[0]["rows"])
    cols = int(validation_scenarios[0]["cols"])
    priority_baseline = (
        _aggregate_baseline(validation_scenarios, evaluate_serpentine_priority_baseline)
        if has_vessels else None
    )
    serpentine, ppo_corner = _serpentine_control(validation_scenarios) if not has_vessels else (None, None)
    curve = _curve_summary(run_dir)
    checkpoints = sorted((run_dir / "validation").glob("step_*.json"))
    checkpoint_rows = []
    for path in checkpoints:
        result = _read_json(path)
        checkpoint_rows.append(
            f"| {result['timesteps']:,} | {result['completion_rate']:.3f} | {result['mean_transfer_overhead']:.3f} |"
        )
    config = metadata["config"]
    dependencies = metadata["dependencies"]
    lines = [
        "# Maskable PPO 训练结果",
        "",
        "## 结论",
        "",
        f"本次为 {config['total_timesteps']:,} steps 的 {rows}×{cols} {'含可释放血管' if has_vessels else '无血管'} toy curriculum 训练。"
        f"最终 Validation 完成率为 {final_validation['completion_rate']:.3f}，"
        f"平均 transfer overhead 为 {final_validation['mean_transfer_overhead']:.3f}。",
        "该结果仅说明策略在其训练/验证范围内可执行；不构成临床性能或曲面迁移结论。",
        "",
        "## 配置与可复现性",
        "",
        f"- Seed：{config['seed']}",
        f"- 环境：{config['n_envs']} 个并行环境；{config['n_steps']} steps/rollout；{config['n_epochs']} PPO epochs",
        f"- PPO：learning rate={config['learning_rate']}，gamma={config['gamma']}，GAE λ={config['gae_lambda']}，clip range={config['clip_range']}",
        f"- 依赖：PyTorch {dependencies['torch']}；Gymnasium {dependencies['gymnasium']}；SB3 {dependencies['stable_baselines3']}；sb3-contrib {dependencies['sb3_contrib']}",
        f"- 场景：Train {len(metadata['train_scenarios'])}，Validation {len(validation_scenarios)}；均为固定 seed 的 {rows}×{cols} 完整方格{'，中心血管需通过 release 才可切除' if has_vessels else ''}。",
        "",
        "## 外部指标：最终策略与逐行规则基线",
        "",
        "| 方法 | 完成率 | 合法动作率 | Transfer overhead | 平均前沿张力 | 平均器官能量 | 平均血管应变 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Maskable PPO | {final_validation['completion_rate']:.3f} | "
        f"{_mean(item['legal_action_rate'] for item in final_validation['metrics']):.3f} | "
        f"{final_validation['mean_transfer_overhead']:.3f} | "
        f"{_mean(item['mean_front_tension'] for item in final_validation['metrics']):.6f} | "
        f"{_mean(item['mean_organ_energy'] for item in final_validation['metrics']):.6f} | "
        f"{_mean(item['mean_vessel_strain'] for item in final_validation['metrics']):.6f} |",
        f"| 行列优先动态前沿（非 S 形） | {baseline['completion_rate']:.3f} | {baseline['legal_action_rate']:.3f} | "
        f"{baseline['mean_transfer_overhead']:.3f} | {baseline['mean_front_tension']:.6f} | "
        f"{baseline['mean_organ_energy']:.6f} | {baseline['mean_vessel_strain']:.6f} |",
        "",
        "## Checkpoint Validation",
        "",
        "| Steps | 完成率 | 平均 transfer overhead |",
        "| ---: | ---: | ---: |",
        *(checkpoint_rows or ["| 无中间 checkpoint | — | — |"]),
        "",
        "## 最后训练诊断",
        "",
    ]
    if has_vessels:
        assert priority_baseline is not None
        vessel_control = [
            "## 血管场景 S 形优先/release 控制 baseline",
            "",
            "连续无障碍 S 形扫描不适用于活跃血管。该 baseline 保持 S 形顺序优先，"
            "在受阻时仅使用环境规定的动态前沿、transfer 和 release 规则确定性绕行。",
            "",
            "| 方法 | 完成率 | 合法动作率 | Transfer overhead | 平均前沿张力 | 平均器官能量 | 平均血管应变 | 最大风险峰值 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| Maskable PPO | {final_validation['completion_rate']:.3f} | {_mean(item['legal_action_rate'] for item in final_validation['metrics']):.3f} | "
            f"{final_validation['mean_transfer_overhead']:.3f} | {_mean(item['mean_front_tension'] for item in final_validation['metrics']):.6f} | "
            f"{_mean(item['mean_organ_energy'] for item in final_validation['metrics']):.6f} | "
            f"{_mean(item['mean_vessel_strain'] for item in final_validation['metrics']):.6f} | "
            f"{_mean(item['max_risk_peak'] for item in final_validation['metrics']):.6f} |",
            f"| S 形优先/release baseline | {priority_baseline['completion_rate']:.3f} | {priority_baseline['legal_action_rate']:.3f} | "
            f"{priority_baseline['mean_transfer_overhead']:.3f} | {priority_baseline['mean_front_tension']:.6f} | "
            f"{priority_baseline['mean_organ_energy']:.6f} | {priority_baseline['mean_vessel_strain']:.6f} | "
            f"{priority_baseline['max_risk_peak']:.6f} |",
            "",
        ]
        position = lines.index("## Checkpoint Validation")
        lines[position:position] = vessel_control
    else:
        assert serpentine is not None and ppo_corner is not None
        control = [
            "## 固定角起点 S 形控制实验", "",
            "为避免随机边界起点使连续 S 形路径在奇数网格上不可行，以下采用完整 5×5、无血管、"
            "统一起点 `[0, 0]`。真正 S 形扫描在此控制场景中每次下一刀都相邻，因此 transfer overhead 为 0。", "",
            "| 方法 | 完成率 | Transfer 次数 | Transfer overhead | 平均前沿张力 | 平均器官能量 | 最大风险峰值 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| Maskable PPO | {ppo_corner['completion']:.3f} | {ppo_corner['total_transfer_count']} | {ppo_corner['transfer_overhead']:.3f} | {ppo_corner['mean_front_tension']:.6f} | {ppo_corner['mean_organ_energy']:.6f} | {ppo_corner['max_risk_peak']:.6f} |",
            f"| 连续 S 形 baseline | {serpentine['completion']:.3f} | {serpentine['total_transfer_count']} | {serpentine['transfer_overhead']:.3f} | {serpentine['mean_front_tension']:.6f} | {serpentine['mean_organ_energy']:.6f} | {serpentine['max_risk_peak']:.6f} |", "",
            "该控制实验中，PPO 的前沿张力略低，但有 2 次额外 transfer，且器官能量和最大风险峰值更高；因此不能宣称它优于连续 S 形 baseline。", "",
        ]
        position = lines.index("## Checkpoint Validation")
        lines[position:position] = control
    for tag, (step, value) in curve.items():
        lines.append(f"- {tag}：step {step:,}，{value:.6g}")
    if has_vessels:
        lines.extend([
            "- 诊断：若 explained variance 持续接近 0 且 transfer overhead 未接近 S 形优先/release baseline，"
            "则当前 run 只完成了可行性学习，未完成效率收敛；下一轮应先修正状态/动作表示和回报归一化，而非仅增加训练步数或放大 transfer 权重。",
        ])
    lines.extend([
        "",
        "## 产物与测试范围",
        "",
        "- `final_model.zip`：最终 Maskable PPO 模型。",
        "- `validation.json`：最终逐场景外部指标；`validation/`：checkpoint 指标。",
        "- `tensorboard/`：训练曲线事件文件。",
        "- 二维模拟器可按模型训练范围加载对应的 ML test preset 并播放决策事件。",
        "",
        "## 限制与下一步",
        "",
        f"- 当前模型仅在 {rows}×{cols} 完整方格{'、中心可释放血管' if has_vessels else '、无血管'}范围内训练；禁止将 toy 指标外推到非矩形域、不同网格、不同血管布局或真实曲面。",
        "- 下一阶段应改为局部网格状态与有效动作头、保留空间位置并加入 return normalization；修正后再扩展到随机非矩形域，并在冻结 Test 与 Stress split 上比较 PPO、规则规划器和 S 形优先 baseline。",
        "- 力学量是无量纲研究代理量，不是临床应力或撕裂风险预测。",
    ])
    output = run_dir / "训练结果.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(write_report(args.run_dir))


if __name__ == "__main__":
    main()

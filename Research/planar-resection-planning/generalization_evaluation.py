"""Frozen-policy generalization evaluation for the 7x7 spatial-v3 model."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from environment import (
    CANVAS_SIZE,
    PlanarResectionEnv,
    local_grid_action_masks,
    local_grid_observation,
)
from evaluation import (
    evaluate_policy,
    row_major_frontier_policy,
    serpentine_priority_policy,
)
from planner import (
    boundary_cells,
    is_connected,
    neighbors4,
    plan_resection,
)

GENERATOR_VERSION = 1
TEST_BASE_SEED = 2026073001
STRESS_BASE_SEED = 2026073002
DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "results"
    / "ppo_spatial_v3_mixed_random_teacher_seed2026072901_100k_retry2"
    / "final_model.zip"
)
POLICY_NAMES = ("spatial_v3", "serpentine_priority", "row_major", "rule_planner")
SCALAR_METRICS = (
    "transfer_overhead",
    "total_reward",
    "mean_front_tension",
    "mean_organ_energy",
    "mean_vessel_strain",
    "max_risk_peak",
)

_WORKER_MODEL = None


def _cell_list(cells: Iterable[tuple[int, int]]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(cells)]


def _full_domain() -> set[tuple[int, int]]:
    return {(row, col) for row in range(7) for col in range(7)}


def _irregular_domain(rng: random.Random, *, severe: bool) -> set[tuple[int, int]]:
    """Remove exposed cells while retaining a connected, hole-free 7x7 domain."""
    domain = _full_domain()
    removal_target = rng.randint(8, 12) if severe else rng.randint(3, 7)
    protected = {(3, 3)}
    removed = 0
    for _ in range(200):
        if removed >= removal_target:
            break
        exposed = sorted(
            cell for cell in boundary_cells(domain)
            if cell not in protected
            and sum(nxt in domain for nxt in neighbors4(cell)) <= (2 if severe else 1)
        )
        if not exposed:
            exposed = sorted(boundary_cells(domain) - protected)
        candidate = rng.choice(exposed)
        reduced = domain - {candidate}
        if len(reduced) >= 32 and is_connected(reduced):
            domain = reduced
            removed += 1
    return domain


def _grow_component(
    rng: random.Random,
    allowed: set[tuple[int, int]],
    size: int,
    *,
    blocked: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    blocked = blocked or set()
    starts = sorted(allowed - blocked)
    for _ in range(100):
        component = {rng.choice(starts)}
        while len(component) < size:
            frontier = sorted({
                nxt
                for cell in component
                for nxt in neighbors4(cell)
                if nxt in allowed and nxt not in blocked and nxt not in component
            })
            if not frontier:
                break
            component.add(rng.choice(frontier))
        if len(component) == size:
            return component
    raise RuntimeError("Could not grow requested vessel component")


def _separated(first: set[tuple[int, int]], second: set[tuple[int, int]]) -> bool:
    return all(
        max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
        for a in first for b in second
    )


def _choose_obstacles(
    rng: random.Random,
    domain: set[tuple[int, int]],
    category: str,
) -> set[tuple[int, int]]:
    internal = domain - boundary_cells(domain)
    if category in {"shifted_single_rectangle", "irregular_single", "concave_single"}:
        candidates = sorted(internal - {(3, 3)})
        return {rng.choice(candidates or sorted(internal))}
    if category in {"compact_vessel_rectangle", "irregular_compact"}:
        return _grow_component(rng, internal, rng.randint(2, 3))
    if category == "large_vessel_rectangle":
        return _grow_component(rng, internal, 4)
    if category in {"multiple_vessels_rectangle", "concave_multiple"}:
        first = _grow_component(rng, internal, 1)
        for _ in range(100):
            second = _grow_component(rng, internal, 1, blocked=first)
            if _separated(first, second):
                return first | second
        raise RuntimeError("Could not place separated vessel components")
    raise ValueError(f"Unknown category: {category}")


def _category_sequence(count: int, categories: Sequence[str]) -> list[str]:
    return [categories[index % len(categories)] for index in range(count)]


def _make_scenario(split: str, index: int, category: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    severe = category.startswith("concave")
    domain = _irregular_domain(rng, severe=severe) if (
        category.startswith("irregular") or severe
    ) else _full_domain()
    for _ in range(200):
        obstacles = _choose_obstacles(rng, domain, category)
        if obstacles <= (domain - boundary_cells(domain)) and is_connected(domain - obstacles):
            break
    else:
        raise RuntimeError(f"Could not create feasible scenario for {category}")
    starts = sorted(boundary_cells(domain) - obstacles)
    start = rng.choice(starts)
    scenario = {
        "scenario_id": f"generalization-{split}-{index:04d}",
        "split": split,
        "category": category,
        "seed": seed,
        "rows": 7,
        "cols": 7,
        "domain_cells": _cell_list(domain),
        "obstacle_cells": _cell_list(obstacles),
        "start_cell": list(start),
    }
    PlanarResectionEnv(scenario=scenario).reset()
    return scenario


def generate_generalization_splits(
    *, test_count: int = 120, stress_count: int = 80,
) -> dict[str, Any]:
    if test_count <= 0 or stress_count <= 0:
        raise ValueError("test_count and stress_count must be positive")
    test_categories = (
        "shifted_single_rectangle",
        "compact_vessel_rectangle",
        "irregular_single",
        "irregular_compact",
    )
    stress_categories = (
        "multiple_vessels_rectangle",
        "large_vessel_rectangle",
        "concave_single",
        "concave_multiple",
    )
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, count, base_seed, categories in (
        ("test", test_count, TEST_BASE_SEED, test_categories),
        ("stress", stress_count, STRESS_BASE_SEED, stress_categories),
    ):
        splits[split] = [
            _make_scenario(
                split,
                index,
                category,
                base_seed + index * 7919,
            )
            for index, category in enumerate(_category_sequence(count, categories))
        ]
    return {
        "generator_version": GENERATOR_VERSION,
        "base_seeds": {"test": TEST_BASE_SEED, "stress": STRESS_BASE_SEED},
        "counts": {"test": test_count, "stress": stress_count},
        "splits": splits,
    }


def generate_training_scenarios(*, count: int = 256, seed: int = 2026073101) -> list[dict[str, Any]]:
    """Generate a seed-fixed mixed-layout pool for the next training run."""
    if count <= 0:
        raise ValueError("count must be positive")
    categories = (
        "shifted_single_rectangle", "compact_vessel_rectangle",
        "irregular_single", "irregular_compact", "multiple_vessels_rectangle",
        "large_vessel_rectangle", "concave_single", "concave_multiple",
    )
    return [
        _make_scenario("train", index, categories[index % len(categories)], seed + index * 7919)
        for index in range(count)
    ]


def _worker_init(model_path: str) -> None:
    global _WORKER_MODEL
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.distributions.Distribution.set_default_validate_args(False)
    _WORKER_MODEL = MaskablePPO.load(model_path, device="cpu")


def _spatial_policy(env: PlanarResectionEnv) -> int:
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model is not initialized")
    local_action, _ = _WORKER_MODEL.predict(
        local_grid_observation(env, 7),
        deterministic=True,
        action_masks=local_grid_action_masks(env, 7),
    )
    row, col = divmod(int(local_action), 7)
    return row * CANVAS_SIZE + col


def _planner_policy(scenario: Mapping[str, Any]) -> Callable[[PlanarResectionEnv], int]:
    result = plan_resection(
        rows=int(scenario["rows"]),
        cols=int(scenario["cols"]),
        domain_cells=scenario["domain_cells"],
        obstacle_cells=scenario["obstacle_cells"],
        start_cell=scenario["start_cell"],
    )
    cuts = [
        event["cell"]
        for event in result["events"]
        if event["action"] == "cut" and event.get("reason") != "start"
    ]
    cursor = 0

    def policy(env: PlanarResectionEnv) -> int:
        nonlocal cursor
        if cursor >= len(cuts):
            legal = np.flatnonzero(env.action_masks())
            if not len(legal):
                raise RuntimeError("Rule planner exhausted after environment termination")
            return int(legal[0])
        row, col = cuts[cursor]
        cursor += 1
        return int(row) * CANVAS_SIZE + int(col)

    return policy


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "failure_reason": result["failure_reason"],
        "completion": bool(result["completion"]),
        "legal_action_rate": float(result["legal_action_rate"]),
        "release_rule_correct": bool(result["release_rule_correct"]),
        "total_transfer_count": int(result["total_transfer_count"]),
        **{name: float(result[name]) for name in SCALAR_METRICS},
    }


def _evaluate_one(scenario: Mapping[str, Any]) -> dict[str, Any]:
    policies: dict[str, Callable[[PlanarResectionEnv], int]] = {
        "spatial_v3": _spatial_policy,
        "serpentine_priority": serpentine_priority_policy,
        "row_major": row_major_frontier_policy,
        "rule_planner": _planner_policy(scenario),
    }
    results: dict[str, Any] = {}
    for name, policy in policies.items():
        try:
            results[name] = _compact_result(evaluate_policy(scenario, policy))
        except Exception as exc:  # Preserve a failed case instead of aborting the full audit.
            results[name] = {
                "status": "exception",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "completion": False,
                "legal_action_rate": 0.0,
                "release_rule_correct": False,
                "total_transfer_count": 0,
                **{metric: float("nan") for metric in SCALAR_METRICS},
            }
    return {
        "scenario_id": scenario["scenario_id"],
        "split": scenario["split"],
        "category": scenario["category"],
        "results": results,
    }


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, resamples: int = 10_000,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 500):
        width = min(500, resamples - start)
        indices = rng.integers(0, len(array), size=(width, len(array)))
        means[start:start + width] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        items = [record["results"][policy] for record in records]
        metrics: dict[str, Any] = {
            "completion_rate": mean(float(item["completion"]) for item in items),
            "legal_action_rate": mean(float(item["legal_action_rate"]) for item in items),
            "release_rule_correct_rate": mean(float(item["release_rule_correct"]) for item in items),
        }
        for metric in SCALAR_METRICS:
            values = _finite(item[metric] for item in items)
            metrics[f"mean_{metric}"] = mean(values) if values else float("nan")
            metrics[f"{metric}_ci95"] = _bootstrap_mean_ci(
                values, seed=20260731 + POLICY_NAMES.index(policy) * 101 + SCALAR_METRICS.index(metric),
            )
        summary[policy] = metrics

    paired: dict[str, Any] = {}
    for baseline in ("serpentine_priority", "row_major", "rule_planner"):
        differences = []
        wins = ties = losses = 0
        for record in records:
            candidate = record["results"]["spatial_v3"]
            reference = record["results"][baseline]
            if not candidate["completion"] or not reference["completion"]:
                continue
            difference = float(candidate["transfer_overhead"]) - float(reference["transfer_overhead"])
            differences.append(difference)
            if difference < -1e-12:
                wins += 1
            elif difference > 1e-12:
                losses += 1
            else:
                ties += 1
        paired[baseline] = {
            "paired_count": len(differences),
            "mean_transfer_overhead_difference": mean(differences) if differences else float("nan"),
            "difference_ci95": _bootstrap_mean_ci(
                differences, seed=20260801 + POLICY_NAMES.index(baseline),
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
    summary["paired_vs_spatial_v3"] = paired
    return summary


def summarize(
    scenarios: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": {}, "by_category": {}}
    for split in ("test", "stress"):
        split_records = [record for record in records if record["split"] == split]
        output["overall"][split] = _summarize_records(split_records)
        categories = sorted({record["category"] for record in split_records})
        output["by_category"][split] = {
            category: _summarize_records(
                [record for record in split_records if record["category"] == category]
            )
            for category in categories
        }
    test = output["overall"]["test"]
    stress = output["overall"]["stress"]
    candidate_test = test["spatial_v3"]
    candidate_stress = stress["spatial_v3"]
    output["acceptance"] = {
        "test_completion_at_least_0_99": candidate_test["completion_rate"] >= 0.99,
        "test_legal_rate_is_1": candidate_test["legal_action_rate"] == 1.0,
        "test_release_correct_rate_is_1": candidate_test["release_rule_correct_rate"] == 1.0,
        "test_transfer_within_10pct_serpentine": (
            candidate_test["mean_transfer_overhead"]
            <= 1.10 * test["serpentine_priority"]["mean_transfer_overhead"]
        ),
        "test_transfer_within_10pct_rule_planner": (
            candidate_test["mean_transfer_overhead"]
            <= 1.10 * test["rule_planner"]["mean_transfer_overhead"]
        ),
        "stress_completion_at_least_0_95": candidate_stress["completion_rate"] >= 0.95,
    }
    output["acceptance"]["passed"] = all(output["acceptance"].values())
    output["scenario_counts"] = scenarios["counts"]
    return output


def _fmt(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "—"


def write_report(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    summary = payload["summary"]
    category_labels = {
        "shifted_single_rectangle": "平移的单矩形血管",
        "compact_vessel_rectangle": "紧凑矩形血管",
        "irregular_single": "单个不规则血管",
        "irregular_compact": "紧凑不规则血管",
        "multiple_vessels_rectangle": "多个分离矩形血管",
        "large_vessel_rectangle": "大面积矩形血管",
        "concave_single": "凹形域单血管",
        "concave_multiple": "凹形域多血管",
    }
    lines = [
        "# Spatial-v3 冻结模型泛化测试",
        "",
        "本测试只进行冻结模型推理，没有更新模型参数。",
        "",
        "## 总体结果",
        "",
        "| Split | 方法 | 完成率 | Transfer overhead | 平均总奖励 | 平均前沿张力 | 平均血管应变 | 最大风险峰值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "spatial_v3": "Spatial-v3 PPO",
        "serpentine_priority": "S 形优先/release",
        "row_major": "行列优先",
        "rule_planner": "规则规划器",
    }
    for split in ("test", "stress"):
        for policy in POLICY_NAMES:
            item = summary["overall"][split][policy]
            lines.append(
                f"| {split.title()} | {labels[policy]} | {item['completion_rate']:.3f} | "
                f"{_fmt(item['mean_transfer_overhead'])} | {_fmt(item['mean_total_reward'])} | "
                f"{_fmt(item['mean_mean_front_tension'])} | {_fmt(item['mean_mean_vessel_strain'])} | "
                f"{_fmt(item['mean_max_risk_peak'])} |"
            )
    lines.extend([
        "",
        "## 配对 Transfer 比较",
        "",
        "差值定义为 Spatial-v3 减去规则方法；负值表示 Spatial-v3 的额外回穿更少。",
        "",
        "| Split | 对照 | 平均差值 | 95% bootstrap CI | 胜/平/负 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for split in ("test", "stress"):
        paired = summary["overall"][split]["paired_vs_spatial_v3"]
        for baseline in ("serpentine_priority", "row_major", "rule_planner"):
            item = paired[baseline]
            ci = item["difference_ci95"]
            lines.append(
                f"| {split.title()} | {labels[baseline]} | {_fmt(item['mean_transfer_overhead_difference'])} | "
                f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {item['wins']}/{item['ties']}/{item['losses']} |"
            )
    lines.extend([
        "",
        "## 分类结果",
        "",
        "下表只列路径效率；差值为 Spatial-v3 减去 S 形，正值表示模型回穿更多。",
        "",
        "| Split | 场景类别 | 数量 | Spatial-v3 | S 形 | 规则规划器 | 相对 S 形差值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for split in ("test", "stress"):
        for category, category_summary in summary["by_category"][split].items():
            model_value = category_summary["spatial_v3"]["mean_transfer_overhead"]
            serpentine_value = category_summary["serpentine_priority"]["mean_transfer_overhead"]
            planner_value = category_summary["rule_planner"]["mean_transfer_overhead"]
            count = category_summary["paired_vs_spatial_v3"]["serpentine_priority"]["paired_count"]
            lines.append(
                f"| {split.title()} | {category_labels.get(category, category)} | {count} | "
                f"{_fmt(model_value)} | {_fmt(serpentine_value)} | {_fmt(planner_value)} | "
                f"{_fmt(model_value - serpentine_value)} |"
            )
    lines.extend([
        "",
        "## 指标解释",
        "",
        "- Transfer overhead（额外回穿率）：$O_{transfer}=N_{transfer}/N_{target}$，即自动移动经过已切除区域的次数除以目标网格数，越小表示切割路径越连续、重复回穿越少。",
        "- 平均总奖励：一次完整任务中各步训练奖励之和，越高表示越符合当前奖励函数，但它不是独立的外部质量指标。",
        "- 平均前沿张力：切割过程中当前切割前沿所承受的平均无量纲张力代理量，越低通常表示前沿受力更温和。",
        "- 平均器官能量：切割过程中组织形变能的平均无量纲代理量，越低表示整体形变负担更小。",
        "- 平均血管应变：血管邻域在切割过程中的平均无量纲应变代理量，越低表示对血管的牵拉更小。",
        "- 最大风险峰值：整条轨迹中单步综合风险代理量的最大值，越低表示最危险瞬间更温和。",
        "",
        "## 结论与下一步",
        "",
        "- Spatial-v3 在当前冻结场景中保持完成、合法动作和 release 规则正确；这只支持 7×7 人工布局范围内的研究性泛化结论。",
        "- Test 集相对 S 形的平均 transfer 差值为 -0.0044，95% bootstrap CI 为 [-0.0152, 0.0054]。区间包含 0，不能据此声称策略在路径效率上优于或劣于 S 形基线。",
        "- 相对规则规划器的 Test transfer 差值为 +0.0227，95% bootstrap CI 为 [-0.0111, 0.0562]，同样不能下显著优劣结论。平均血管应变低于规则规划器是待进一步验证的模拟代理优势方向，不能等同于临床安全性。",
        "- 下一轮应在独立 Validation 集上按路径效率非劣和血管风险改善的预先定义规则选择候选；新模型仍须通过冻结测试后才能替换模拟器默认模型。",
    ])
    lines.extend([
        "",
        "## 验收门槛",
        "",
    ])
    for name, value in summary["acceptance"].items():
        if name != "passed":
            lines.append(f"- {name}: {'通过' if value else '未通过'}")
    lines.extend([
        "",
        f"**总体判定：{'通过' if summary['acceptance']['passed'] else '未通过'}。**",
        "",
        "## 限制",
        "",
        "- 所有场景仍为 7×7，测试的是血管位置、组件和域形状泛化，不代表尺寸泛化。",
        "- 力学量是无量纲研究代理量，不是临床应力或安全性结论。",
        "- 详细逐场景结果和分类汇总见同目录 `results.json`。",
    ])
    path = output_dir / "泛化测试报告.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(
    *,
    output_dir: Path,
    model_path: Path,
    test_count: int,
    stress_count: int,
    workers: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    scenarios = generate_generalization_splits(
        test_count=test_count,
        stress_count=stress_count,
    )
    (output_dir / "scenarios.json").write_text(
        json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    items = [scenario for split in ("test", "stress") for scenario in scenarios["splits"][split]]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(model_path),),
    ) as executor:
        records = list(executor.map(_evaluate_one, items, chunksize=1))
    summary = summarize(scenarios, records)
    payload = {
        "model_path": str(model_path),
        "workers": workers,
        "generator_version": GENERATOR_VERSION,
        "summary": summary,
        "records": records,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--test-count", type=int, default=120)
    parser.add_argument("--stress-count", type=int, default=80)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    payload = run(
        output_dir=args.output_dir,
        model_path=args.model_path,
        test_count=args.test_count,
        stress_count=args.stress_count,
        workers=args.workers,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "summary": payload["summary"]["overall"],
        "acceptance": payload["summary"]["acceptance"],
    }, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

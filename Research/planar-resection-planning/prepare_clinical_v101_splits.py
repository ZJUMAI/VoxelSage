"""Freeze leak-safe v10.1 splits: every split regenerated as pure Stage D.

v10 记录：Train split 取自 curriculum D（stage a/b/c/d 混合）与纯 stage d 的
Validation 分布错配导致两次 Stage 1B NO-GO。v10.1 的全部 split（train/tuning/
validation/test/stress）都用 ``generate_clinical_stage_pool(stage="d")`` 重新
生成，使用与 v10 Validation 相同的生成器与参数，并保证 ID、seed 跨 split 无重叠。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from clinical_window_scenarios import generate_clinical_stage_pool


V101_SPLIT_SEEDS = {
    "train": 2026082201,
    "tuning": 2026082202,
    "validation": 2026082203,
    "test": 2026082204,
    "stress": 2026082205,
}
V101_SPLIT_COUNTS = {
    "train": 256,
    "tuning": 32,
    "validation": 64,
    "test": 64,
    "stress": 64,
}
V101_SPLIT_USES = {
    "train": "BC, timing oracle, PPO",
    "tuning": "Optuna 唯一调参集",
    "validation": "多 seed 确认与模型选择（v10.1 全新，未受反馈污染）",
    "test": "权重冻结后只评估一次",
    "stress": "最后稳健性评估，不反向调参",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split file: {args.output}")
    splits: dict[str, list[dict]] = {}
    for name, count in V101_SPLIT_COUNTS.items():
        splits[name] = generate_clinical_stage_pool(
            stage="d",
            count=count,
            seed=V101_SPLIT_SEEDS[name],
            split=f"v10.1-{name}",
        )
    scenario_ids = [item["scenario_id"] for values in splits.values() for item in values]
    scenario_seeds = [int(item["seed"]) for values in splits.values() for item in values]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise RuntimeError("v10.1 scenario IDs overlap")
    if len(scenario_seeds) != len(set(scenario_seeds)):
        raise RuntimeError("v10.1 scenario seeds overlap")
    # 每个 split 内部也应无重叠（生成器本身保证，双保险）
    for name, values in splits.items():
        local_ids = [item["scenario_id"] for item in values]
        local_seeds = [int(item["seed"]) for item in values]
        if len(local_ids) != len(set(local_ids)) or len(local_seeds) != len(set(local_seeds)):
            raise RuntimeError(f"v10.1 split {name!r} has overlapping IDs/seeds internally")
    payload = {
        "version": "clinical-v101-splits-v1",
        "stage": "d",
        "generator": "clinical_window_planar_resection v1 make_clinical_scenario(stage='d')",
        "vessel_count_range": "4-8",
        "vessel_size_range": "1-4 cells",
        "counts": {name: len(values) for name, values in splits.items()},
        "base_seeds": V101_SPLIT_SEEDS,
        "uses": V101_SPLIT_USES,
        "note": "All splits regenerated independently; v10 validation not reused (feedback-contaminated)",
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "counts": payload["counts"],
        "total_scenarios": len(scenario_ids),
        "unique_ids": len(set(scenario_ids)),
        "unique_seeds": len(set(scenario_seeds)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

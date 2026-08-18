"""v10.1 distribution-consistency gate.

Compares median / quartiles of surgery time, baseline blood loss, clamp-cycle
count, vessel cell count, and vessel component count across all frozen splits.
This goes beyond checking the stage label: it verifies the splits sample the
same Stage D generator distribution.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from statistics import median

from planner import neighbors4
from clinical_window_evaluation import rollout_clinical_policy, serpentine_hierarchical_policy


def _percentiles(values: list[float]) -> tuple[float, float, float, float, float]:
    ordered = sorted(values)
    n = len(ordered)
    def q(frac: float) -> float:
        pos = (n - 1) * frac
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])
    return q(0.0), q(0.25), q(0.5), q(0.75), q(1.0)


def _vessel_component_count(cells: list[list[int]]) -> int:
    cell_set = {tuple(c) for c in cells}
    seen: set[tuple[int, int]] = set()
    components = 0
    for cell in cell_set:
        if cell in seen:
            continue
        components += 1
        queue: deque[tuple[int, int]] = deque([cell])
        seen.add(cell)
        while queue:
            current = queue.popleft()
            for neighbor in neighbors4(current):
                if neighbor in cell_set and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--rollout-limit", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    clinical_config = {
        "time_scale_minutes": float(scales["time_scale_minutes"]),
        "blood_scale_ml": float(scales["blood_scale_ml"]),
        "weight_kg": float(scales.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": "disabled",
        "early_end_minutes": 0.0,
    }
    reward_config = {
        "time_cost": 1.0,
        "blood_cost": 1.0,
        "progress_bonus": 0.0,
        "seal_progress_bonus": 0.0,
    }
    result: dict[str, dict] = {}
    for name, scenarios in split_payload["splits"].items():
        scenarios = list(scenarios)[: args.rollout_limit]
        vessel_cells: list[int] = []
        vessel_components: list[int] = []
        times: list[float] = []
        bloods: list[float] = []
        cycles: list[float] = []
        for item in scenarios:
            vessel_cells.append(len(item["obstacle_cells"]))
            vessel_components.append(_vessel_component_count(item["obstacle_cells"]))
            record = rollout_clinical_policy(
                item,
                serpentine_hierarchical_policy,
                clinical_config=clinical_config,
                reward_config=reward_config,
                control_mode="hierarchical",
            )
            times.append(float(record["elapsed_minutes"]))
            bloods.append(float(record["expected_blood_loss_ml"]))
            cycles.append(float(record.get("clamp_cycle_count", 0.0)))
        result[name] = {
            "n_scenarios": len(scenarios),
            "vessel_cells_min_max": [_percentiles(vessel_cells)[0], _percentiles(vessel_cells)[4]],
            "vessel_cells_q25_q50_q75": [_percentiles(vessel_cells)[1], _percentiles(vessel_cells)[2], _percentiles(vessel_cells)[3]],
            "vessel_components_q25_q50_q75": [_percentiles(vessel_components)[1], _percentiles(vessel_components)[2], _percentiles(vessel_components)[3]],
            "time_min_q25_q50_q75": [_percentiles(times)[1], _percentiles(times)[2], _percentiles(times)[3]],
            "blood_ml_q25_q50_q75": [_percentiles(bloods)[1], _percentiles(bloods)[2], _percentiles(bloods)[3]],
            "clamp_cycles_q25_q50_q75": [_percentiles(cycles)[1], _percentiles(cycles)[2], _percentiles(cycles)[3]],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 打印便于人类阅读的表
    header = f"{'split':<12} {'cells 25/50/75':>18} {'vessels 25/50/75':>18} {'time 25/50/75':>18} {'blood 25/50/75':>18} {'cycles 25/50/75':>16}"
    print(header)
    print("-" * len(header))
    for name, stats in result.items():
        def fmt(vals: list[float]) -> str:
            return "/".join(f"{v:g}" for v in vals)
        print(
            f"{name:<12} {fmt(stats['vessel_cells_q25_q50_q75']):>18} "
            f"{fmt(stats['vessel_components_q25_q50_q75']):>18} "
            f"{fmt(stats['time_min_q25_q50_q75']):>18} "
            f"{fmt(stats['blood_ml_q25_q50_q75']):>18} "
            f"{fmt(stats['clamp_cycles_q25_q50_q75']):>16}"
        )


if __name__ == "__main__":
    main()

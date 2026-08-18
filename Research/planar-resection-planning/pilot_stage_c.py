"""Stage-C pure-size timing pilot (17-24 x 17-32, up to 6 obstacles).

Runs a handful of pure-C train scenarios and reports, per scenario:

- rows / cols / valid-domain cell count / obstacle-component count
- wall time of the serpentine and planner teacher rollouts separately
- which teacher the filter selects (planner only when it dominates both proxies)
- demonstration step count, completion, transfer overhead, mean strain

It is a timing-and-scoping pilot only: it produces no model and no reusable
cache.  If a single scenario routinely approaches the 120 s subprocess
timeout, do not generate a single-file 128/192 cache without a resumable
sharded cache first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from evaluation import serpentine_priority_policy
from planner import vessel_components
from variable_scenarios import generate_stage_pool
from variable_teacher import _planner_selector, _rollout


def _cell_set(cells: list[list[int]]) -> set[tuple[int, int]]:
    return {tuple(cell) for cell in cells}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("c", "d"), default="c")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026073301)
    parser.add_argument("--output", type=Path, default=Path("results/stage_c_pilot.json"))
    args = parser.parse_args()

    scenarios = generate_stage_pool(stage=args.stage, count=args.count, seed=args.seed, split="train")
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        domain = _cell_set(scenario["domain_cells"])
        obstacles = _cell_set(scenario["obstacle_cells"])
        components = vessel_components(obstacles, domain)

        start = perf_counter()
        serpentine_trace = _rollout(scenario, serpentine_priority_policy)
        serpentine_seconds = perf_counter() - start

        start = perf_counter()
        planner_trace = _rollout(scenario, _planner_selector(scenario))
        planner_seconds = perf_counter() - start

        use_planner = (
            planner_trace["completion"]
            and planner_trace["transfer_overhead"] <= serpentine_trace["transfer_overhead"]
            and planner_trace["mean_vessel_strain"] <= serpentine_trace["mean_vessel_strain"]
        )
        trace = planner_trace if use_planner else serpentine_trace
        records.append({
            "scenario_id": scenario["scenario_id"],
            "seed": scenario["seed"],
            "rows": scenario["rows"],
            "cols": scenario["cols"],
            "domain_cell_count": len(domain),
            "obstacle_count": len(obstacles),
            "obstacle_component_count": len(components),
            "start_cell": scenario["start_cell"],
            "serpentine_seconds": round(serpentine_seconds, 3),
            "planner_seconds": round(planner_seconds, 3),
            "selected_teacher": "planner" if use_planner else "serpentine",
            "completion": trace["completion"],
            "demonstration_steps": len(trace["observations"]),
            "transfer_overhead": trace["transfer_overhead"],
            "mean_vessel_strain": trace["mean_vessel_strain"],
        })

    times_serpentine = [r["serpentine_seconds"] for r in records]
    times_planner = [r["planner_seconds"] for r in records]
    steps = [r["demonstration_steps"] for r in records]
    summary = {
        "stage": args.stage,
        "count": len(records),
        "seed": args.seed,
        "completion_rate": sum(float(r["completion"]) for r in records) / len(records),
        "planner_selected_rate": sum(float(r["selected_teacher"] == "planner") for r in records) / len(records),
        "rows_range": [min(r["rows"] for r in records), max(r["rows"] for r in records)],
        "cols_range": [min(r["cols"] for r in records), max(r["cols"] for r in records)],
        "obstacle_components_range": [min(r["obstacle_component_count"] for r in records), max(r["obstacle_component_count"] for r in records)],
        "serpentine_seconds": {
            "mean": round(sum(times_serpentine) / len(times_serpentine), 3),
            "max": round(max(times_serpentine), 3),
            "total": round(sum(times_serpentine), 3),
        },
        "planner_seconds": {
            "mean": round(sum(times_planner) / len(times_planner), 3),
            "max": round(max(times_planner), 3),
            "total": round(sum(times_planner), 3),
        },
        "both_rollouts_per_scenario_seconds": {
            "mean": round(sum(a + b for a, b in zip(times_serpentine, times_planner)) / len(records), 3),
            "max": round(max(a + b for a, b in zip(times_serpentine, times_planner)), 3),
        },
        "demonstration_steps": {
            "mean": round(sum(steps) / len(steps), 1),
            "max": max(steps),
        },
    }
    _write(args.output, {"summary": summary, "records": records})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n输出: {args.output}")
    print(f"逐场景记录 {len(records)} 条，见 records 字段。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the documented Pilot-100 fixed-weight search reproducibly.

The script writes scenario definitions, per-weight metric records, a frozen
summary, and full event sequences for the selected candidate.  It is safe to
re-run: completed candidate files are reused unless ``--no-resume`` is set.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from metrics import normalized_objective, validate_and_measure
from planner import plan_resection
from scenarios import PILOT_BASE_SEED, generate_pilot_scenarios

BASELINE_WEIGHTS = {"distance": 1.0, "vessel_risk": 0.0, "shape": 0.0, "exposure": 0.0}
DEFAULT_WEIGHTS = {"distance": 1.0, "vessel_risk": 2.0, "shape": 1.0, "exposure": 0.75}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CURRENT_DIR.parents[1], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _weight_key(weights: Mapping[str, float]) -> str:
    return f"shape_{weights['shape']:.2f}_exposure_{weights['exposure']:.2f}"


def _candidate_weights() -> List[Dict[str, float]]:
    values = []
    for shape_index in range(9):
        for exposure_index in range(7):
            values.append({
                "distance": 1.0,
                "vessel_risk": 2.0,
                "shape": shape_index * 0.5,
                "exposure": exposure_index * 0.5,
            })
    if DEFAULT_WEIGHTS not in values:
        values.append(dict(DEFAULT_WEIGHTS))
    return values


def _work_items(scenarios: Sequence[Mapping[str, object]]) -> Iterable[Tuple[Mapping[str, object], Mapping[str, object]]]:
    for scenario in scenarios:
        for start in scenario["starts"]:  # type: ignore[index]
            yield scenario, start


def _evaluate(item: Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, float], bool]) -> Dict[str, Any]:
    scenario, start, weights, include_events = item
    started = time.perf_counter()
    result = plan_resection(
        rows=int(scenario["rows"]),
        cols=int(scenario["cols"]),
        domain_cells=scenario["domain_cells"],  # type: ignore[arg-type]
        obstacle_cells=scenario["obstacle_cells"],  # type: ignore[arg-type]
        start_cell=start["cell"],  # type: ignore[arg-type]
        weights=weights,
    )
    metric = validate_and_measure(result)
    record: Dict[str, Any] = {
        "scenario_id": scenario["scenario_id"],
        "scenario_type": scenario["scenario_type"],
        "scoring_eligible": bool(scenario["scoring_eligible"]),
        "expected_status": scenario["expected_status"],
        "seed": scenario["seed"],
        "start_label": start["label"],
        "start_cell": start["cell"],
        "weights": dict(weights),
        "status": result["status"],
        "failure_reason": result["failure_reason"],
        "planning_time_ms": round((time.perf_counter() - started) * 1000, 3),
        **metric,
    }
    if include_events:
        record["result"] = result
    return record


def _run_records(
    scenarios: Sequence[Mapping[str, object]],
    weights: Mapping[str, float],
    workers: int,
    *,
    include_events: bool = False,
) -> List[Dict[str, Any]]:
    items = [(scenario, start, weights, include_events) for scenario, start in _work_items(scenarios)]
    if workers <= 1:
        return [_evaluate(item) for item in items]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_evaluate, items, chunksize=1))


def _record_index(records: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(record["scenario_id"]), str(record["start_label"])): record for record in records}


def _aggregate(
    records: Sequence[Mapping[str, Any]],
    baselines: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    objectives: List[float] = []
    eligible_total = 0
    eligible_hard_failures = 0
    stress_total = 0
    stress_expected_partial = 0
    all_successes = 0
    planning_times = []
    for record in records:
        planning_times.append(float(record["planning_time_ms"]))
        if record["status"] == "ok":
            all_successes += 1
        if record["scoring_eligible"]:
            eligible_total += 1
            baseline = baselines[(str(record["scenario_id"]), str(record["start_label"]))]
            objective = normalized_objective(record, baseline)
            if objective is None:
                eligible_hard_failures += 1
            else:
                objectives.append(objective)
        else:
            stress_total += 1
            if record["status"] == record["expected_status"]:
                stress_expected_partial += 1
    ordered = sorted(objectives)
    p90 = ordered[max(0, int(len(ordered) * 0.9) - 1)] if ordered else None
    return {
        "weights": dict(records[0]["weights"]) if records else None,
        "run_count": len(records),
        "all_success_rate": all_successes / len(records) if records else 0.0,
        "eligible_run_count": eligible_total,
        "eligible_hard_failures": eligible_hard_failures,
        "eligible_success_rate": (eligible_total - eligible_hard_failures) / eligible_total if eligible_total else 0.0,
        "objective_count": len(objectives),
        "mean_objective": mean(objectives) if objectives else None,
        "median_objective": ordered[len(ordered) // 2] if ordered else None,
        "p90_objective": p90,
        "stress_run_count": stress_total,
        "stress_expected_status_rate": stress_expected_partial / stress_total if stress_total else None,
        "mean_planning_time_ms": mean(planning_times) if planning_times else None,
    }


def _rank(summary: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        float(summary["eligible_hard_failures"]),
        float(summary["mean_objective"]) if summary["mean_objective"] is not None else float("inf"),
        float(summary["p90_objective"]) if summary["p90_objective"] is not None else float("inf"),
    )


def _ensure_scenarios(path: Path, count: int, base_seed: int, resume: bool) -> Dict[str, Any]:
    if resume and path.exists():
        payload = _load_json(path)
        if payload.get("scenario_count") == count and payload.get("base_seed") == base_seed:
            return payload
    payload = generate_pilot_scenarios(count=count, base_seed=base_seed)
    _write_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the planar-simulator Pilot-100 weight search")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "results" / "pilot_100")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scenario-count", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=PILOT_BASE_SEED)
    parser.add_argument("--max-candidates", type=int, default=None, help="Debug only: limit candidates after deterministic ordering")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.scenario_count != 100:
        raise SystemExit("Formal Pilot requires exactly --scenario-count 100")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    output_dir = args.output_dir.resolve()
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)
    resume = not args.no_resume
    scenarios_payload = _ensure_scenarios(output_dir / "scenarios.json", args.scenario_count, args.base_seed, resume)
    scenarios: List[Mapping[str, object]] = scenarios_payload["scenarios"]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_name": "pilot_100_fixed_weight_search",
        "scenario_count": args.scenario_count,
        "starts_per_scenario": 3,
        "workers": args.workers,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "baseline_weights": BASELINE_WEIGHTS,
        "fixed_vessel_risk": 2.0,
    }
    _write_json(output_dir / "metadata.json", metadata)

    baseline_path = output_dir / "baseline.json"
    if resume and baseline_path.exists():
        baseline_records = _load_json(baseline_path)["records"]
        print(f"Reusing baseline: {len(baseline_records)} scenario-start runs", flush=True)
    else:
        print("Running distance-only baseline…", flush=True)
        baseline_records = _run_records(scenarios, BASELINE_WEIGHTS, args.workers)
        _write_json(baseline_path, {"weights": BASELINE_WEIGHTS, "records": baseline_records})
    baseline_index = _record_index(baseline_records)

    candidates = _candidate_weights()
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]
    summaries: List[Dict[str, Any]] = []
    for number, weights in enumerate(candidates, start=1):
        key = _weight_key(weights)
        path = candidate_dir / f"{key}.json"
        if resume and path.exists():
            payload = _load_json(path)
            records = payload["records"]
            print(f"[{number}/{len(candidates)}] Reusing {key}", flush=True)
        else:
            print(f"[{number}/{len(candidates)}] Running {key}", flush=True)
            records = _run_records(scenarios, weights, args.workers)
            _write_json(path, {"weights": weights, "records": records})
        summary = _aggregate(records, baseline_index)
        summary["candidate_key"] = key
        summaries.append(summary)
        _write_json(output_dir / "progress.json", {"completed_candidates": summaries, "metadata": metadata})

    ranked = sorted(summaries, key=_rank)
    if not ranked:
        raise RuntimeError("No candidate results were produced")
    best = ranked[0]
    best_weights = best["weights"]
    print(f"Selected {best['candidate_key']} with objective {best['mean_objective']}", flush=True)

    selected_path = output_dir / "selected_paths.json"
    if resume and selected_path.exists():
        selected_payload = _load_json(selected_path)
        if selected_payload.get("weights") == best_weights:
            selected_records = selected_payload["records"]
        else:
            selected_records = _run_records(scenarios, best_weights, args.workers, include_events=True)
            _write_json(selected_path, {"weights": best_weights, "records": selected_records})
    else:
        selected_records = _run_records(scenarios, best_weights, args.workers, include_events=True)
        _write_json(selected_path, {"weights": best_weights, "records": selected_records})

    final = {
        "metadata": metadata,
        "scenario_file": "scenarios.json",
        "baseline_file": "baseline.json",
        "candidate_directory": "candidates",
        "selected_paths_file": "selected_paths.json",
        "candidate_count": len(summaries),
        "selected_candidate": best,
        "all_candidates_ranked": ranked,
        "selected_path_run_count": len(selected_records),
        "selection_rule": ["fewest eligible hard failures", "lowest mean objective", "lowest p90 objective"],
    }
    _write_json(output_dir / "summary.json", final)
    print(f"Pilot complete: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

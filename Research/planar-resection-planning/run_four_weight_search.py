#!/usr/bin/env python3
"""Run the resumable documented four-weight coarse grid search.

The distance weight is fixed to 1.0, so this searches the remaining vessel
risk, shape, and exposure weights over the documented 9 x 9 x 7 coarse grid.
Results are kept separate from the historical Pilot-100 directory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from run_pilot import (  # noqa: E402
    BASELINE_WEIGHTS,
    _aggregate,
    _ensure_scenarios,
    _record_index,
    _run_records,
)
from scenarios import PILOT_BASE_SEED  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    """Atomically publish a complete JSON record for safe restart reuse."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
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


def _coarse_candidates() -> list[dict[str, float]]:
    """Return the documented 9 x 9 x 7 grid in deterministic order."""
    return [
        {"distance": 1.0, "vessel_risk": risk * 0.5, "shape": shape * 0.5, "exposure": exposure * 0.5}
        for risk in range(9)
        for shape in range(9)
        for exposure in range(7)
    ]


def _weight_key(weights: Mapping[str, float]) -> str:
    return (
        f"risk_{weights['vessel_risk']:.2f}_shape_{weights['shape']:.2f}_"
        f"exposure_{weights['exposure']:.2f}"
    )


def _rank(summary: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    """Apply the documented failure, objective, P90, then time ordering."""
    return (
        float(summary["eligible_hard_failures"]),
        float(summary["mean_objective"]) if summary["mean_objective"] is not None else float("inf"),
        float(summary["p90_objective"]) if summary["p90_objective"] is not None else float("inf"),
        float(summary["mean_planning_time_ms"]) if summary["mean_planning_time_ms"] is not None else float("inf"),
    )


def _complete_candidate(payload: Any, weights: Mapping[str, float], expected_runs: int) -> bool:
    """Only reuse records that have the requested weights and every run."""
    return (
        isinstance(payload, dict)
        and payload.get("weights") == dict(weights)
        and isinstance(payload.get("records"), list)
        and len(payload["records"]) == expected_runs
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the documented Pilot-100 four-weight coarse search")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "results" / "four_weight_search_coarse")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scenario-count", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=PILOT_BASE_SEED)
    parser.add_argument("--max-candidates", type=int, default=None, help="Debug only: limit deterministic grid ordering")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.scenario_count != 100:
        raise SystemExit("Formal search requires exactly --scenario-count 100")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise SystemExit("--max-candidates must be positive")

    output_dir = args.output_dir.resolve()
    pilot_output = (CURRENT_DIR / "results" / "pilot_100").resolve()
    if output_dir == pilot_output:
        raise SystemExit("--output-dir must not overwrite results/pilot_100")
    candidate_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)
    resume = not args.no_resume

    scenarios_payload = _ensure_scenarios(output_dir / "scenarios.json", args.scenario_count, args.base_seed, resume)
    scenarios: Sequence[Mapping[str, object]] = scenarios_payload["scenarios"]
    expected_runs = args.scenario_count * 3
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_name": "pilot_100_four_weight_coarse_grid",
        "scenario_count": args.scenario_count,
        "starts_per_scenario": 3,
        "workers": args.workers,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "baseline_weights": BASELINE_WEIGHTS,
        "fixed_distance_weight": 1.0,
        "coarse_grid": {
            "vessel_risk": {"start": 0.0, "stop": 4.0, "step": 0.5},
            "shape": {"start": 0.0, "stop": 4.0, "step": 0.5},
            "exposure": {"start": 0.0, "stop": 3.0, "step": 0.5},
            "candidate_count": 567,
        },
    }
    _write_json(output_dir / "metadata.json", metadata)

    baseline_path = output_dir / "baseline.json"
    if resume and baseline_path.exists():
        baseline_payload = _load_json(baseline_path)
        if _complete_candidate(baseline_payload, BASELINE_WEIGHTS, expected_runs):
            baseline_records = baseline_payload["records"]
            print(f"Reusing baseline: {len(baseline_records)} scenario-start runs", flush=True)
        else:
            baseline_records = _run_records(scenarios, BASELINE_WEIGHTS, args.workers)
            _write_json(baseline_path, {"weights": BASELINE_WEIGHTS, "records": baseline_records})
    else:
        print("Running distance-only baseline…", flush=True)
        baseline_records = _run_records(scenarios, BASELINE_WEIGHTS, args.workers)
        _write_json(baseline_path, {"weights": BASELINE_WEIGHTS, "records": baseline_records})
    baseline_index = _record_index(baseline_records)

    candidates = _coarse_candidates()
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]
    summaries: List[Dict[str, Any]] = []
    for number, weights in enumerate(candidates, start=1):
        key = _weight_key(weights)
        candidate_path = candidate_dir / f"{key}.json"
        payload = _load_json(candidate_path) if resume and candidate_path.exists() else None
        if _complete_candidate(payload, weights, expected_runs):
            records = payload["records"]
            print(f"[{number}/{len(candidates)}] Reusing {key}", flush=True)
        else:
            print(f"[{number}/{len(candidates)}] Running {key}", flush=True)
            records = _run_records(scenarios, weights, args.workers)
            _write_json(candidate_path, {"weights": weights, "records": records})
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
    selected_payload = _load_json(selected_path) if resume and selected_path.exists() else None
    if _complete_candidate(selected_payload, best_weights, expected_runs):
        selected_records = selected_payload["records"]
    else:
        selected_records = _run_records(scenarios, best_weights, args.workers, include_events=True)
        _write_json(selected_path, {"weights": best_weights, "records": selected_records})

    _write_json(output_dir / "summary.json", {
        "metadata": metadata,
        "scenario_file": "scenarios.json",
        "baseline_file": "baseline.json",
        "candidate_directory": "candidates",
        "selected_paths_file": "selected_paths.json",
        "candidate_count": len(summaries),
        "selected_candidate": best,
        "all_candidates_ranked": ranked,
        "selected_path_run_count": len(selected_records),
        "selection_rule": [
            "fewest eligible hard failures", "lowest mean objective", "lowest p90 objective", "lowest mean planning time",
        ],
    })
    print(f"Four-weight coarse search complete: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

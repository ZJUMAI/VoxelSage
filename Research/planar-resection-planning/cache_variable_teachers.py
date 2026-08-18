"""Create an auditable teacher cache for one variable-size curriculum stage.

``--isolated`` replays every scenario in its own subprocess so a native solver
exit cannot kill the whole cache.  The isolated mode is:

- **parallel**: ``--workers`` controls how many replay subprocesses run at once
  (default 1, matching the original sequential behaviour);
- **resumable**: ``--shard-dir`` points at a persistent shard directory; on
  restart already-completed ``teacher_*.npz`` shards are skipped and only
  missing scenarios are replayed;
- **timeout-aware**: ``--timeout`` sets the per-scenario subprocess budget
  (default 120 s; Stage C pure scenarios measure ~175 s for both teacher
  rollouts, so a larger budget is required there).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from variable_scenarios import CURRICULUM_RANGES, generate_curriculum_train_pool, generate_stage_pool
from variable_teacher import load_teacher_cache, write_teacher_cache


def _replay_one(arg: tuple) -> dict[str, object] | None:
    """Replay one scenario in a fresh subprocess.  Returns a failure dict or None."""
    script, index, scenario, scenario_path, shard_path, timeout = arg
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--scenario-json", str(scenario_path), "--output", str(shard_path)],
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "index": index, "scenario_id": scenario["scenario_id"], "seed": scenario["seed"],
            "shape": [scenario["rows"], scenario["cols"]], "returncode": "timeout",
            "stderr": f"subprocess exceeded {timeout}s timeout",
        }
    if completed.returncode != 0 or not shard_path.exists():
        return {
            "index": index, "scenario_id": scenario["scenario_id"], "seed": scenario["seed"],
            "shape": [scenario["rows"], scenario["cols"]], "returncode": completed.returncode,
            "stderr": completed.stderr[-1000:],
        }
    return None


def _valid_shard(path: Path) -> bool:
    """A shard is resumable only if it loads under the cache contract."""
    try:
        load_teacher_cache(path)
        return True
    except Exception:
        return False


def _write_isolated_cache(
    output: Path, scenarios: list[dict], script: Path, *,
    workers: int = 1, timeout: int = 120, shard_dir: Path | None = None,
) -> dict[str, object]:
    """Cache each replay in a child process, optionally parallel and resumable."""
    temporary: tempfile.TemporaryDirectory | None = None
    if shard_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="variable_teacher_")
        shard_dir = Path(temporary.name)
    else:
        shard_dir = Path(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    work: list[tuple] = []
    for index, scenario in enumerate(scenarios):
        scenario_path = shard_dir / f"scenario_{index:04d}.json"
        shard_path = shard_dir / f"teacher_{index:04d}.npz"
        if shard_path.exists() and _valid_shard(shard_path):
            continue  # resumable: this scenario was already completed
        if shard_path.exists():
            shard_path.unlink()  # stale partial shard from an interrupted run
        scenario_path.write_text(json.dumps(scenario, ensure_ascii=False), encoding="utf-8")
        work.append((script, index, scenario, scenario_path, shard_path, timeout))

    if work:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for failure in executor.map(_replay_one, work):
                if failure is not None:
                    failures.append(failure)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    summaries: list[dict[str, float]] = []
    for index, scenario in enumerate(scenarios):
        shard_path = shard_dir / f"teacher_{index:04d}.npz"
        if not (shard_path.exists() and _valid_shard(shard_path)):
            if not any(f["index"] == index for f in failures):
                failures.append({
                    "index": index, "scenario_id": scenario["scenario_id"], "seed": scenario["seed"],
                    "shape": [scenario["rows"], scenario["cols"]], "returncode": None,
                    "stderr": "shard missing or invalid after replay",
                })
            continue
        obs, act, mask, summary = load_teacher_cache(shard_path)
        observations.append(obs)
        actions.append(act)
        masks.append(mask)
        summaries.append(summary)

    if not observations:
        raise RuntimeError("All isolated teacher replays failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=np.concatenate(observations), actions=np.concatenate(actions), masks=np.concatenate(masks),
        summary=np.asarray(json.dumps({"successful_episode_count": len(summaries), "failed_episode_count": len(failures)}, ensure_ascii=False)),
    )
    failure_path = output.with_suffix(".failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if temporary is not None:
        temporary.cleanup()
    return {
        "successful_episode_count": len(summaries), "failed_episode_count": len(failures),
        "failure_file": str(failure_path),
        "workers": workers, "timeout": timeout,
        "shard_dir": str(shard_dir) if shard_dir is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=tuple(CURRICULUM_RANGES), default="a")
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026073001)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--scenario-json", type=Path)
    parser.add_argument(
        "--current-stage-only", action="store_true",
        help="Generate pure current-stage train scenarios via generate_stage_pool "
             "(split='train') instead of the A/B/.../current mixed curriculum. "
             "Use this for timing scoping pilots; do not use it for real C/D caches.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent replay subprocesses (isolated mode only).")
    parser.add_argument("--timeout", type=int, default=120, help="Per-scenario subprocess timeout in seconds.")
    parser.add_argument(
        "--shard-dir", type=Path,
        help="Persistent shard directory enabling resume (isolated mode only).",
    )
    args = parser.parse_args()
    if args.scenario_json is not None:
        scenario = json.loads(args.scenario_json.read_text(encoding="utf-8"))
        summary = write_teacher_cache(args.output, [scenario])
    else:
        if args.current_stage_only:
            scenarios = generate_stage_pool(
                stage=args.stage, count=args.scenarios, seed=args.seed, split="train",
            )
        else:
            scenarios = generate_curriculum_train_pool(stage=args.stage, count=args.scenarios, seed=args.seed)
        summary = (
            _write_isolated_cache(
                args.output, scenarios, Path(__file__).resolve(),
                workers=args.workers, timeout=args.timeout, shard_dir=args.shard_dir,
            )
            if args.isolated else write_teacher_cache(args.output, scenarios)
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Run reproducible stage-A PPO replications and summarize them.

The script is intentionally serial: the mechanics workers already use all CPU
resources needed for stable PPO collection, and the target GPU is supplied by
the caller (for example ``CUDA_VISIBLE_DEVICES=6``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FROZEN_SEED = 2026074201


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env={
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/mpl_codex",
        "PYTHONUNBUFFERED": "1",
    }, check=True)


def _load_summary(path: Path) -> dict[str, float]:
    return {key: float(value) for key, value in json.loads(path.read_text(encoding="utf-8"))["summary"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026073102, 2026073103, 2026073104])
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pyth = sys.executable
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        cache = RESULTS / f"teacher_cache_stage_a_seed{seed}_128_isolated.npz"
        run_dir = RESULTS / f"variable_spatial_stage_a_seed{seed}_25k"
        evaluation = RESULTS / f"variable_eval_stage_a_ppo_seed{seed}.json"
        if not cache.exists():
            _run([pyth, "-u", "cache_variable_teachers.py", "--output", str(cache), "--stage", "a", "--scenarios", "128", "--seed", str(seed), "--isolated"])
        if not (run_dir / "final_model.zip").exists():
            _run([pyth, "-u", "train_variable_masked_ppo.py", "--output-dir", str(run_dir), "--teacher-cache", str(cache), "--stage", "a", "--timesteps", str(args.timesteps), "--train-scenarios", "128", "--n-envs", "8", "--n-steps", "256", "--batch-size", "256", "--bc-epochs", "10", "--seed", str(seed), "--device", args.device])
        if not evaluation.exists():
            _run([pyth, "-u", "evaluate_variable_policy.py", "--method", "ppo", "--stage", "a", "--count", "128", "--seed", str(FROZEN_SEED), "--workers", str(args.workers), "--model-path", str(run_dir / "final_model.zip"), "--output", str(evaluation)])
        rows.append({"seed": seed, **_load_summary(evaluation)})

    aggregate = {
        "frozen_seed": FROZEN_SEED,
        "replications": rows,
        "mean": {key: sum(row[key] for row in rows) / len(rows) for key in rows[0] if key != "seed"},
    }
    output = RESULTS / "variable_eval_stage_a_replications.json"
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

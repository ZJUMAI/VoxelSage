"""Freeze leak-safe v10 train/tuning/validation/test scenario splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from clinical_window_scenarios import generate_clinical_stage_pool


V10_SPLIT_SEEDS = {
    "tuning": 2026081202,
    "validation": 2026081203,
    "test": 2026081204,
    "stress": 2026081205,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tuning-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--stress-count", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split file: {args.output}")
    counts = {
        "tuning": args.tuning_count,
        "validation": args.validation_count,
        "test": args.test_count,
        "stress": args.stress_count,
    }
    if any(value <= 0 for value in counts.values()):
        parser.error("all split counts must be positive")
    source = json.loads(args.train_source.read_text(encoding="utf-8"))
    train = list(source["splits"]["train"])
    splits = {"train": train}
    for name, count in counts.items():
        splits[name] = generate_clinical_stage_pool(
            stage="d", count=count, seed=V10_SPLIT_SEEDS[name], split=f"v10-{name}"
        )
    scenario_ids = [item["scenario_id"] for values in splits.values() for item in values]
    scenario_seeds = [int(item["seed"]) for values in splits.values() for item in values]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise RuntimeError("v10 scenario IDs overlap")
    if len(scenario_seeds) != len(set(scenario_seeds)):
        raise RuntimeError("v10 scenario seeds overlap")
    source_hash = hashlib.sha256(args.train_source.read_bytes()).hexdigest()
    payload = {
        "version": "clinical-v10-splits-v1",
        "stage": "d",
        "train_source": str(args.train_source.resolve()),
        "train_source_sha256": source_hash,
        "counts": {name: len(values) for name, values in splits.items()},
        "base_seeds": V10_SPLIT_SEEDS,
        "policy": {
            "tuning": "Optuna only",
            "validation": "model selection and multi-seed confirmation",
            "test": "one-time final evaluation",
            "stress": "one-time robustness evaluation",
        },
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

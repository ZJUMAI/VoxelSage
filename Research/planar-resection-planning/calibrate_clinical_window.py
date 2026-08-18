"""Create frozen Train-only reward scales for clinical-window PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_window_evaluation import calibrate_global_scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--weight-kg", type=float, default=70.0)
    parser.add_argument("--bleeding-probability", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen scale file: {args.output}")
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scenarios = list(payload["splits"]["train"])
    if args.limit > 0:
        scenarios = scenarios[:args.limit]
    result = calibrate_global_scales(
        scenarios,
        clinical_config={
            "weight_kg": args.weight_kg,
            "bleeding_probability": args.bleeding_probability,
        },
    )
    result.update({
        "split_file": str(args.splits.resolve()),
        "weight_kg": args.weight_kg,
        "bleeding_probability": args.bleeding_probability,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "time_scale_minutes": result["time_scale_minutes"],
        "blood_scale_ml": result["blood_scale_ml"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()


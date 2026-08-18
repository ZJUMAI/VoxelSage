"""Drive the five frozen sensitivity conditions (C0/C2/C4 only).

Sensitivity stage runs only C0, C2, C4 per guide Section 7, on the shared
128 geometric scenes of `sensitivity_base`, once per condition S0..S4.
No teacher (C3) and no unshielded (C5) to bound compute and avoid post-hoc
comparison expansion.  It never retrains or changes the frozen implementation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_7_confirmation"
FROZEN = BASE / "frozen"

CONDITIONS = ("S0", "S1", "S2", "S3", "S4")
SENSITIVITY_CONTROLLERS = ("C0", "C2", "C4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-workers", type=int, default=12)
    parser.add_argument("--leaf-workers", type=int, default=3)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()

    conditions = [c for c in args.conditions.split(",") if c in CONDITIONS]
    for condition in conditions:
        cmd = [
            sys.executable, "evaluate_confirmation_v107.py",
            "--split", "sensitivity_base",
            "--controllers", ",".join(SENSITIVITY_CONTROLLERS),
            "--condition", condition,
            "--scene-workers", str(args.scene_workers),
            "--leaf-workers", str(args.leaf_workers),
        ]
        print(f"=== sensitivity {condition} ===", flush=True)
        subprocess.run(cmd, cwd=SIM, check=True)
        # Aggregate each condition immediately.
        agg = [
            sys.executable, "aggregate_confirmation_v107.py",
            "--split", "sensitivity_base", "--condition", condition,
        ]
        subprocess.run(agg, cwd=SIM, check=True)
    print("=== sensitivity complete ===", flush=True)


if __name__ == "__main__":
    main()

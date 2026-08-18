"""Run the frozen 3-config x 3-seed v10.6 Validation matrix."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_6_shielded_learning"


def main() -> None:
    manifest = json.loads((BASE / "training_plan_manifest.json").read_text(encoding="utf-8"))
    configs = {row["id"]: row for row in manifest["bc_configurations"]}
    selected = json.loads((BASE / "evaluation/tuning_selection.json").read_text(encoding="utf-8"))[
        "selected_for_validation"
    ]
    seeds = [int(value) for value in manifest["training_seeds"]]
    rows = []
    cache = BASE / "evaluation/shield_cache/validation"
    for config_id in selected:
        config = configs[config_id]
        for seed in seeds:
            checkpoint = (BASE / "runs/bc" / f"{config_id}_seed_{seed}"
                          / f"epoch_{int(config['epochs']):02d}.pt")
            output = BASE / "evaluation" / f"validation_{config_id}_seed{seed}.json"
            if not output.is_file():
                subprocess.run([
                    sys.executable, "evaluate_target_order_v106.py",
                    "--checkpoint", str(checkpoint),
                    "--split-file", str(BASE / "frozen/split_validation.json"),
                    "--baseline-file", str(BASE / "frozen/baseline_validation.json"),
                    "--teacher-gate", str(BASE / "evaluation/validation_teacher_reference.json"),
                    "--output", str(output), "--scene-workers", "12", "--leaf-workers", "3",
                    "--shield-cache-dir", str(cache),
                ], cwd=SIM, check=True)
            result = json.loads(output.read_text(encoding="utf-8"))
            row = {"config": config_id, "seed": seed, "checkpoint": str(checkpoint),
                   "evaluation": str(output), "decision": result["decision"],
                   "conditions": result["conditions"], "summary": result["summary"]}
            rows.append(row)
            matrix = {
                "version": "v10.6-validation-matrix-v1", "rows": rows,
                "all_completed_so_far_go": all(item["decision"] == "GO" for item in rows),
            }
            (BASE / "evaluation/validation.json").write_text(
                json.dumps(matrix, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

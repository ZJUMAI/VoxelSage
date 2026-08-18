"""Evaluate all frozen v10.6 BC configurations on Tuning-64 sequentially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_6_shielded_learning"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=11)
    parser.add_argument("--scene-workers", type=int, default=12)
    parser.add_argument("--leaf-workers", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads((BASE / "training_plan_manifest.json").read_text(encoding="utf-8"))
    seed = int(manifest["training_seeds"][0])
    cache = BASE / "evaluation/shield_cache/tuning"
    rows = []
    existing = BASE / "evaluation/tuning_config_00.json"
    if existing.is_file():
        result = json.loads(existing.read_text(encoding="utf-8"))
        rows.append({"config": "config_00", "evaluation": str(existing),
                     "decision": result["decision"], "summary": result["summary"]})
    for config in manifest["bc_configurations"]:
        number = int(config["id"].split("_")[1])
        if not args.start <= number <= args.end:
            continue
        checkpoint = (BASE / "runs/bc" / f"{config['id']}_seed_{seed}"
                      / f"epoch_{int(config['epochs']):02d}.pt")
        output = BASE / "evaluation" / f"tuning_{config['id']}.json"
        if not output.is_file():
            subprocess.run([
                sys.executable, "evaluate_target_order_v106.py",
                "--checkpoint", str(checkpoint),
                "--split-file", str(BASE / "frozen/split_tuning.json"),
                "--baseline-file", str(BASE / "frozen/baseline_tuning.json"),
                "--teacher-gate", str(BASE / "evaluation/tuning_teacher_reference.json"),
                "--output", str(output), "--scene-workers", str(args.scene_workers),
                "--leaf-workers", str(args.leaf_workers), "--shield-cache-dir", str(cache),
            ], cwd=SIM, check=True)
        result = json.loads(output.read_text(encoding="utf-8"))
        row = {"config": config["id"], "checkpoint": str(checkpoint),
               "evaluation": str(output), "decision": result["decision"],
               "conditions": result["conditions"], "summary": result["summary"]}
        rows.append(row)
        audit = BASE / "evaluation/tuning_all_configs.json"
        audit.write_text(json.dumps({
            "version": "v10.6-tuning-all-configs-v1", "rows": rows,
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

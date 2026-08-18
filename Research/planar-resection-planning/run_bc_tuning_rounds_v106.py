"""Run frozen v10.6 BC configurations one audited epoch at a time."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


SIM = Path(__file__).resolve().parent
BASE = SIM / "results/clinical_window_v10_6_shielded_learning"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=SIM, check=True)


def audit_passes(path: Path) -> tuple[bool, dict]:
    row = json.loads(path.read_text(encoding="utf-8"))
    numeric = (
        "safe_set_top1", "safe_set_top3", "safe_set_ndcg_at_3", "B_tail_mae_ml",
        "B_total_mae_ml", "unsafe_false_negative_rate", "completion_recall",
    )
    conditions = {
        "full_state_coverage": row["n_states"] == 171401,
        "full_candidate_coverage": row["n_candidates"] == 1017114,
        "all_summary_metrics_finite": all(math.isfinite(float(row[name])) for name in numeric),
        "ranking_probabilities_bounded": all(
            0.0 <= float(row[name]) <= 1.0
            for name in ("safe_set_top1", "safe_set_top3", "safe_set_ndcg_at_3")
        ),
        "completion_recall_bounded": 0.0 <= float(row["completion_recall"]) <= 1.0,
    }
    return all(conditions.values()), conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--configs", default=None, help="Comma-separated frozen config IDs.")
    args = parser.parse_args()
    manifest = json.loads((BASE / "training_plan_manifest.json").read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else manifest["training_seeds"][0])
    if seed not in [int(value) for value in manifest["training_seeds"]]:
        raise RuntimeError("seed is not in the pre-frozen training seed list")
    log_path = BASE / "audit" / f"bc_round_decisions_seed_{seed}.json"
    decisions = []
    if log_path.is_file():
        decisions = json.loads(log_path.read_text(encoding="utf-8"))["rounds"]
    for config in manifest["bc_configurations"]:
        number = int(config["id"].split("_")[1])
        selected = set(args.configs.split(",")) if args.configs else None
        if selected is not None and config["id"] not in selected:
            continue
        if selected is None and not args.start <= number <= args.end:
            continue
        output_dir = BASE / "runs/bc" / f"{config['id']}_seed_{seed}"
        for epoch in range(1, int(config["epochs"]) + 1):
            checkpoint = output_dir / f"epoch_{epoch:02d}.pt"
            audit_path = (BASE / "evaluation" /
                          f"{config['id']}_seed{seed}_epoch{epoch:02d}_offline.json")
            if not checkpoint.is_file():
                command = [
                    sys.executable, "train_target_order_v106.py", "--epochs", "1",
                    "--device", args.device, "--output-dir", str(output_dir),
                    "--seed", str(seed), "--lr", str(config["lr"]),
                    "--batch-size", str(config["batch_size"]), "--hidden", "96", "--spatial", "32",
                ]
                if epoch > 1:
                    command += ["--resume-checkpoint", str(output_dir / f"epoch_{epoch - 1:02d}.pt")]
                run(command)
            if not audit_path.is_file():
                run([
                    sys.executable, "audit_target_order_model_v106.py",
                    "--checkpoint", str(checkpoint), "--device", args.device,
                    "--output", str(audit_path),
                ])
            passed, conditions = audit_passes(audit_path)
            decision = {
                "config": config["id"], "epoch": epoch,
                "checkpoint": str(checkpoint), "offline_audit": str(audit_path),
                "conditions": conditions, "decision": "GO" if passed else "NO-GO",
            }
            decision["seed"] = seed
            decisions = [row for row in decisions if not (
                row["config"] == config["id"] and row["epoch"] == epoch
                and int(row.get("seed", manifest["training_seeds"][0])) == seed
            )]
            decisions.append(decision)
            log_path.write_text(json.dumps({
                "version": "v10.6-bc-round-decisions-v1", "rounds": decisions,
            }, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(decision), flush=True)
            if not passed:
                raise RuntimeError(f"offline hard audit failed: {config['id']} epoch {epoch}")


if __name__ == "__main__":
    main()

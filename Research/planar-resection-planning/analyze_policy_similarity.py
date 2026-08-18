"""Quantify how close the variable-size stage-C PPO is to the S-form baseline.

Runs both the frozen C model and the S-form priority policy on the same frozen
stage-C scenarios and compares their cut sequences at three granularities:

- step-aligned cell agreement: fraction of steps where both policies choose the
  same domain cell at the same index
- transfer-structure agreement: whether a transfer event occurs at the same
  step index (position in the cut sequence)
- order-free cell set: the union of cells cut (identical when both complete)

Also reports the classic transfer-overhead gap per scenario.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from environment import PlanarResectionEnv, VariableGridScenarioPoolEnv
from evaluation import serpentine_priority_policy
from variable_scenarios import generate_stage_pool


def _cut_cells(events) -> list[tuple[int, int]]:
    return [tuple(event["cell"]) for event in events if event["action"] == "cut"]


def _transfer_indices(events) -> set[int]:
    return {index for index, event in enumerate(events) if event["action"] == "transfer"}


def _trace_cells_and_events(build_env, step_fn) -> tuple[list, list]:
    """Run a policy via a generic step loop and return (cut_cells, events)."""
    env = build_env()
    _, _ = env.reset()
    env_sim = getattr(env, "base_env", env)  # VariableGrid wrapper -> inner sim
    while not env_sim.terminated and not env_sim.truncated:
        action = step_fn(env, env_sim)
        _, _, _, _, _ = env.step(action)
    return _cut_cells(env_sim.events), list(env_sim.events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="c")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026074203)
    parser.add_argument("--model", type=Path,
                        default=Path("results/variable_spatial_stage_c_from_b_seed2026073301_75k/final_model.zip"))
    parser.add_argument("--output", type=Path, default=Path("results/variable_policy_serpentine_similarity.json"))
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    import variable_policy  # noqa: F401  register the custom policy class

    model = MaskablePPO.load(str(args.model), device="cpu")

    scenarios = generate_stage_pool(stage=args.stage, count=args.count, seed=args.seed, split="frozen")
    records: list[dict] = []
    for scenario in scenarios:
        # S-form priority policy (deterministic)
        serp_cells, serp_events = _trace_cells_and_events(
            lambda: PlanarResectionEnv(scenario=scenario),
            lambda env, sim: serpentine_priority_policy(sim),
        )
        # PPO model on the padded variable-grid wrapper (inline loop to keep obs)
        venv = VariableGridScenarioPoolEnv([scenario], seed=0)
        obs, _ = venv.reset()
        while not venv.base_env.terminated and not venv.base_env.truncated:
            masks = venv.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, _, _, _ = venv.step(int(action))
        ppo_cells = _cut_cells(venv.base_env.events)
        ppo_events = list(venv.base_env.events)

        n = min(len(serp_cells), len(ppo_cells))
        aligned = sum(1 for a, b in zip(serp_cells, ppo_cells) if a == b) / n if n else 0.0
        serp_trans = _transfer_indices(serp_events)
        ppo_trans = _transfer_indices(ppo_events)
        common = len(serp_trans & ppo_trans) / max(len(ppo_trans), 1)
        coverage = (len(ppo_cells) / max(len(serp_cells), 1))

        serp_overhead = sum(e["action"] == "transfer" for e in serp_events) / max(len(serp_cells), 1)
        ppo_overhead = sum(e["action"] == "transfer" for e in ppo_events) / max(len(ppo_cells), 1)

        records.append({
            "scenario_id": scenario["scenario_id"],
            "rows": scenario["rows"], "cols": scenario["cols"],
            "serp_cut": len(serp_cells), "ppo_cut": len(ppo_cells),
            "step_aligned_agreement": round(aligned, 4),
            "transfer_index_agreement": round(common, 4),
            "coverage_ratio": round(coverage, 4),
            "serp_transfer_overhead": round(serp_overhead, 4),
            "ppo_transfer_overhead": round(ppo_overhead, 4),
            "ppo_vs_serp_transfer": round(ppo_overhead / serp_overhead, 4) if serp_overhead else None,
        })

    summary = {
        "stage": args.stage, "count": len(records), "seed": args.seed,
        "mean_step_aligned_agreement": round(mean(r["step_aligned_agreement"] for r in records), 4),
        "mean_transfer_index_agreement": round(mean(r["transfer_index_agreement"] for r in records), 4),
        "mean_coverage_ratio": round(mean(r["coverage_ratio"] for r in records), 4),
        "mean_ppo_transfer_overhead": round(mean(r["ppo_transfer_overhead"] for r in records), 4),
        "mean_serp_transfer_overhead": round(mean(r["serp_transfer_overhead"] for r in records), 4),
        "mean_ppo_vs_serp_transfer": round(mean(r["ppo_vs_serp_transfer"] for r in records), 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

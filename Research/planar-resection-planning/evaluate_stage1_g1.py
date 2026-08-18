"""Stage 1 G1 evaluation on Probe-64 (decision-maker mandate).

For each Probe scene:
  * baseline rollout (frozen BC + always-continue 15/5) records the target
    sequence and the paired-control record;
  * the trained Stage-1 model is rolled out deterministically; every
    release-legal decision contributes a (model release probability, v2-safe
    label, delta_blood, delta_ischemia, advantage) sample so AUROC /
    balanced accuracy / release recall / unsafe-release FPR are computed from
    REAL decisions against REAL counterfactuals.

Scene-level paired differences (blood / ischemia / time / transfer) get
scene-bootstrap 95% CIs.  The G1 decision applies every mandate gate.

Usage (scene-sliced for parallelism):
    python evaluate_stage1_g1.py --model .../clamp_oracle_model.zip \
        --bc-model ... --splits ... --scales ... \
        --output-dir ... --scenario-start N --scenario-count M --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_target_conditioned_environment import (  # noqa: E402
    CLAMP_RELEASE,
    TargetConditionedClampEnv,
)
from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy  # noqa: E402
from evaluate_clinical_v102 import make_target_conditioned_ppo_selector  # noqa: E402

import train_target_conditioned_clamp_oracle as m  # noqa: E402


def _make_prob_fn(model, device):
    import torch

    def prob_fn(env) -> float:
        obs = env._observation()
        with torch.no_grad():
            x = torch.as_tensor(obs[None].astype(np.float32), device=device)
            features = model.policy.extract_features(x)
            latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, x)
            logits = model.policy.action_net.scorer(fused)
            probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].cpu())

    return prob_fn


def evaluate_scene(
    scenario,
    *,
    model,
    prob_fn,
    bc_policy,
    clinical_config,
    reward_config,
    ischemia_cost,
    ischemia_scale,
    epsilon_ischemia,
    advantage_margin,
    device,
) -> dict:
    base = m.rollout_baseline_episode(
        scenario,
        target_selector=bc_policy.select_target,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=ischemia_cost,
        ischemia_scale_minutes=ischemia_scale,
        bc_target_sha256=bc_policy.checkpoint_sha256,
    )
    targets = base["target_sequence"]
    env = TargetConditionedClampEnv(
        scenario=scenario,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=ischemia_cost,
        ischemia_scale_minutes=ischemia_scale,
        target_selector=bc_policy.select_target,
        safe_release_mask=True,
    )
    env.reset()
    selector = make_target_conditioned_ppo_selector(model)
    samples = []
    model_targets = []
    rewards: list[float] = []
    reward_terms: dict[str, float] = {}
    proposed = illegal = unsafe_release = 0
    while not env.terminated and not env.truncated:
        mask = env.action_masks()
        model_targets.append(int(env.planned_target_index))
        action = selector(env)
        prob = prob_fn(env)
        proposed += 1
        if not mask[action]:
            illegal += 1
            if action == CLAMP_RELEASE:
                unsafe_release += 1
        if mask[CLAMP_RELEASE]:
            advantage, details = m.counterfactual_release_advantage(
                env,
                time_cost=float(reward_config["time_cost"]),
                blood_cost=float(reward_config["blood_cost"]),
                ischemia_cost=ischemia_cost,
                time_scale=float(clinical_config["time_scale_minutes"]),
                blood_scale=float(clinical_config["blood_scale_ml"]),
                ischemia_scale=ischemia_scale,
                target_sequence=targets,
            )
            label, _reg, db, di = m._stage1_label_and_reg(
                advantage, details,
                epsilon_ischemia=epsilon_ischemia,
                blood_scale=float(clinical_config["blood_scale_ml"]),
                ischemia_scale=ischemia_scale,
            )
            samples.append({
                "prob": prob, "label": label,
                "delta_blood": db, "delta_ischemia": di,
                "advantage": advantage,
            })
        _, reward, _, _, info = env.step(action, build_obs=True)
        rewards.append(float(reward))
        for key, value in info.get("reward_terms", {}).items():
            reward_terms[key] = reward_terms.get(key, 0.0) + float(value)
    model_rec = m._episode_record(
        env,
        scenario_id=scenario.get("scenario_id"),
        policy="stage1_model_det",
        bc_target_sha256=bc_policy.checkpoint_sha256,
        legal_rate=(proposed - illegal) / proposed if proposed else 1.0,
        rewards=rewards,
        reward_terms=reward_terms,
    )
    model_rec["unsafe_end_count"] = unsafe_release
    return {
        "baseline": base,
        "model": model_rec,
        "samples": samples,
        "model_targets": model_targets,
        "target_sequence_identical": model_targets == targets,
    }


def g1_decision(result: dict) -> dict:
    baseline_records = result["baseline_records"]
    model_records = result["model_records"]
    fields = result["fields"]
    metrics = result["classification"]
    n = len(model_records)

    baseline_comp = all(bool(r["completion"]) for r in baseline_records)
    model_comp = all(bool(r["completion"]) for r in model_records)
    baseline_legal = all(float(r["legal_action_rate"]) >= 1.0 - 1e-9 for r in baseline_records)
    model_legal = all(float(r["legal_action_rate"]) >= 1.0 - 1e-9 for r in model_records)
    unsafe_end = sum(int(r.get("unsafe_end_count", 0)) for r in model_records)
    det_end_total = sum(1 for r in model_records if int(r.get("early_end_count", 0)) > 0)
    release_scenario_fraction = det_end_total / n if n else 0.0
    max_scene_delta_blood = max(result["per_scene"]["blood"].values()) if result["per_scene"]["blood"] else float("inf")
    target_frozen = all(bool(r.get("target_sequence_identical", True)) for r in result["scene_details"])
    b = fields["blood"]
    i = fields["ischemia"]
    t = fields["time"]
    tr = fields["transfer"]

    # Time / transfer must differ from the paired control only by float noise.
    time_max_abs = max(abs(v) for v in result["per_scene"]["time"].values()) if result["per_scene"]["time"] else 0.0
    transfer_max_abs = max(abs(v) for v in result["per_scene"]["transfer"].values()) if result["per_scene"]["transfer"] else 0.0
    tol = 1e-6
    checks = {
        "auroc_ge_075": metrics["auroc"] >= 0.75,
        "balanced_acc_ge_070": metrics["balanced_accuracy"] >= 0.70,
        "release_recall_ge_050": metrics["release_recall"] >= 0.50,
        "unsafe_fpr_le_005": metrics["unsafe_release_false_positive_rate"] <= 0.05,
        "baseline_completion_100": baseline_comp,
        "model_completion_100": model_comp,
        "baseline_legal_100": baseline_legal,
        "model_legal_100": model_legal,
        "unsafe_end_0": unsafe_end == 0,
        "det_end_nonzero": det_end_total > 0,
        "release_scenarios_ge_5pct": release_scenario_fraction >= 0.05,
        "probe_no_scene_blood_worse": max_scene_delta_blood <= 0,
        "mean_ischemia_negative": i["mean_difference"] < 0,
        "ci_upper_ischemia_negative": i["bootstrap_95_ci"][1] < 0,
        "time_float_only": time_max_abs <= tol,
        "transfer_float_only": transfer_max_abs <= tol,
        "target_sequence_frozen": target_frozen,
        "target_hash_present": bool(baseline_records and baseline_records[0].get("bc_target_sha256")),
    }
    go = all(checks.values())
    return {
        "decision": "GO" if go else "NO-GO",
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "max_scene_delta_blood_ml": float(max_scene_delta_blood),
        "max_abs_delta_time_min": float(time_max_abs),
        "max_abs_delta_transfer": float(transfer_max_abs),
        "release_scenario_fraction": float(release_scenario_fraction),
        "det_end_total": det_end_total,
    }


def merge_and_decide(parts_dir: Path, output_dir: Path, *, bootstrap_samples: int, seed: int, threshold: float) -> None:
    """Merge scene-sliced G1 parts, recompute stats from records, decide G1."""
    part_files = sorted(parts_dir.glob("g1_part_*.json"))
    if not part_files:
        raise SystemExit(f"No g1_part_*.json found in {parts_dir}")
    details = []
    for path in part_files:
        details.extend(json.loads(path.read_text(encoding="utf-8"))["scene_details"])
    if len({d["baseline"]["scenario_id"] for d in details}) != len(details):
        raise SystemExit("G1 merge has duplicate scenario IDs")
    baseline_records = [d["baseline"] for d in details]
    model_records = [d["model"] for d in details]

    samples = [s for d in details for s in d["samples"]]
    if not samples:
        raise SystemExit("No release-legal samples across Probe; cannot compute classification")
    y_true = np.asarray([s["label"] for s in samples], dtype=int)
    y_prob = np.asarray([s["prob"] for s in samples], dtype=float)
    audit = [
        {"release_legal": True, "delta_blood": s["delta_blood"],
         "delta_ischemia": s["delta_ischemia"], "advantage": s["advantage"]}
        for s in samples
    ]
    metrics = m.evaluate_stage1_metrics(y_true, y_prob, audit, threshold=threshold)

    fields = {
        "blood": "expected_blood_loss_ml",
        "ischemia": "total_clamped_minutes",
        "time": "elapsed_minutes",
        "transfer": "transfer_overhead",
    }
    per_scene = {key: {} for key in fields}
    for d in details:
        sid = str(d["baseline"]["scenario_id"])
        for key, field in fields.items():
            per_scene[key][sid] = float(d["model"][field]) - float(d["baseline"][field])
    rng = np.random.default_rng(seed)
    fields_stat = {}
    for key in fields:
        values = np.asarray(list(per_scene[key].values()), dtype=float)
        idx = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        boot = values[idx].mean(axis=1)
        fields_stat[key] = {
            "mean_difference": float(values.mean()),
            "n_improved": int((values < -1e-9).sum()),
            "n_equal": int((np.abs(values) <= 1e-9).sum()),
            "n_worsened": int((values > 1e-9).sum()),
            "bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    result = {
        "baseline_records": baseline_records,
        "model_records": model_records,
        "fields": fields_stat,
        "per_scene": per_scene,
        "classification": metrics,
        "scene_details": details,
    }
    decision = g1_decision(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "g1_decision.json").write_text(
        json.dumps({"decision": decision, "classification": metrics, "fields": fields_stat},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "g1_records.json").write_text(
        json.dumps({"baseline_records": baseline_records, "model_records": model_records},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": "g1_merge",
        "n_scenarios": len(details),
        "decision": decision["decision"],
        "failed_checks": decision["failed_checks"],
        "auroc": round(metrics["auroc"], 4),
        "balanced_acc": round(metrics["balanced_accuracy"], 4),
        "release_recall": round(metrics["release_recall"], 4),
        "unsafe_fpr": round(metrics["unsafe_release_false_positive_rate"], 4),
        "delta_blood": fields_stat["blood"],
        "delta_ischemia": fields_stat["ischemia"],
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bc-model", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario-start", type=int, default=0)
    parser.add_argument("--scenario-count", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026090201)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--merge-dir", type=Path,
                        help="merge g1_part_*.json and decide G1 (skips collection)")
    args = parser.parse_args()

    if args.merge_dir is not None:
        merge_and_decide(args.merge_dir, args.output_dir,
                         bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                         threshold=args.threshold)
        return

    from sb3_contrib import MaskablePPO
    from clinical_target_conditioned_policy import TargetConditionedClampPolicy  # noqa: F401

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if not split_payload.get("frozen"):
        raise RuntimeError("Probe G1 requires the frozen split file")
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    clinical_config = {
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": "threshold",
        "early_end_minutes": 10.0,
    }
    reward_config = {
        "time_cost": 1.0, "blood_cost": 1.0,
        "completion_bonus": 5.0, "failure_penalty": 10.0,
        "invalid_action_penalty": 10.0,
    }
    ischemia_scale = float(scale_payload["ischemia_scale_minutes"])

    scenarios = list(split_payload["splits"]["probe"])[:64]
    scenarios = scenarios[args.scenario_start : args.scenario_start + args.scenario_count]
    if not scenarios:
        raise SystemExit("no probe scenes in slice")

    model = MaskablePPO.load(str(args.model), device=args.device)
    bc_policy = FrozenBCMacroTargetPolicy(args.bc_model, device=args.device)
    prob_fn = _make_prob_fn(model, args.device)

    scene_details = []
    for scenario in scenarios:
        detail = evaluate_scene(
            scenario,
            model=model, prob_fn=prob_fn, bc_policy=bc_policy,
            clinical_config=clinical_config, reward_config=reward_config,
            ischemia_cost=1.0, ischemia_scale=ischemia_scale,
            epsilon_ischemia=1e-6, advantage_margin=1e-6,
            device=args.device,
        )
        scene_details.append(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "scenario_start": args.scenario_start,
        "n_scenarios": len(scenarios),
        "bc_parameter_sha256": bc_policy.parameter_sha256(),
        "bc_checkpoint_sha256": bc_policy.checkpoint_sha256,
        "scene_details": scene_details,
    }
    out_path = args.output_dir / f"g1_part_{args.scenario_start:03d}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "scenario_start": args.scenario_start,
        "n_scenarios": len(scenarios),
        "model_comp": sum(1 for d in scene_details if d["model"]["completion"]),
        "target_identical": sum(1 for d in scene_details if d["target_sequence_identical"]),
        "output": str(out_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

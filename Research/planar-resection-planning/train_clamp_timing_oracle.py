"""Pretrain the v10 clamp head with deterministic counterfactual rollouts.

For each sampled legal release state, compare two branches under the same
mechanical S target policy: release now versus continue clamping.  Only the
clamp head is optimized; target selection and the spatial encoder stay frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from clinical_hierarchical_environment import (
    CLAMP_CONTINUE,
    CLAMP_RELEASE,
    ClinicalHierarchicalResectionEnv,
)
from clinical_window_evaluation import serpentine_hierarchical_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _objective(env, *, time_cost: float, blood_cost: float) -> float:
    return (
        time_cost * env.elapsed_minutes / env.clinical_config["time_scale_minutes"]
        + blood_cost
        * env.expected_blood_loss_ml
        / env.clinical_config["blood_scale_ml"]
    )


def _finish_with_serpentine(env: ClinicalHierarchicalResectionEnv) -> None:
    while not env.terminated and not env.truncated:
        env.step(serpentine_hierarchical_policy(env))


def counterfactual_release_advantage(
    env: ClinicalHierarchicalResectionEnv,
    target_action: int,
    *,
    time_cost: float,
    blood_cost: float,
) -> tuple[float, dict[str, float]]:
    """Return ``continue_cost - release_cost`` from one identical state."""
    continue_env = copy.deepcopy(env)
    release_env = copy.deepcopy(env)
    continue_env.step(np.asarray([CLAMP_CONTINUE, target_action], dtype=np.int64))
    release_env.step(np.asarray([CLAMP_RELEASE, target_action], dtype=np.int64))
    _finish_with_serpentine(continue_env)
    _finish_with_serpentine(release_env)
    continue_cost = _objective(
        continue_env, time_cost=time_cost, blood_cost=blood_cost
    )
    release_cost = _objective(release_env, time_cost=time_cost, blood_cost=blood_cost)
    return continue_cost - release_cost, {
        "continue_cost": continue_cost,
        "release_cost": release_cost,
        "continue_time": continue_env.elapsed_minutes,
        "release_time": release_env.elapsed_minutes,
        "continue_blood": continue_env.expected_blood_loss_ml,
        "release_blood": release_env.expected_blood_loss_ml,
    }


def collect_oracle_examples(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    max_examples: int,
    sample_every: int,
    seed: int,
    advantage_margin: float,
) -> tuple[list[np.ndarray], list[int], list[dict[str, Any]]]:
    rng = random.Random(seed)
    ordered = list(scenarios)
    rng.shuffle(ordered)
    observations: list[np.ndarray] = []
    labels: list[int] = []
    audit: list[dict[str, Any]] = []
    candidate_index = 0
    for scenario in ordered:
        env = ClinicalHierarchicalResectionEnv(
            scenario=scenario,
            clinical_config=clinical_config,
            reward_config=reward_config,
            mechanics_update_interval=0,
            safe_release_mask=True,
        )
        env.reset()
        while not env.terminated and not env.truncated:
            teacher_action = serpentine_hierarchical_policy(env)
            if env.action_masks()[CLAMP_RELEASE]:
                candidate_index += 1
                if candidate_index % sample_every == 0:
                    advantage, details = counterfactual_release_advantage(
                        env,
                        int(teacher_action[1]),
                        time_cost=float(reward_config["time_cost"]),
                        blood_cost=float(reward_config["blood_cost"]),
                    )
                    label = int(advantage > advantage_margin)
                    observations.append(env._observation().astype(np.float16))
                    labels.append(label)
                    audit.append({
                        "scenario_id": scenario.get("scenario_id"),
                        "elapsed_minutes": env.elapsed_minutes,
                        "clamp_elapsed_minutes": env.phase_elapsed_minutes,
                        "target_action": int(teacher_action[1]),
                        "release_advantage": advantage,
                        "label": label,
                        **details,
                    })
                    if len(observations) >= max_examples:
                        return observations, labels, audit
            env.step(teacher_action)
    return observations, labels, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario-limit", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=1024)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--time-cost", type=float, default=1.0)
    parser.add_argument("--blood-cost", type=float, default=1.0)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--early-end-mode", choices=("threshold", "full"), default="threshold")
    parser.add_argument("--early-end-minutes", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026081001)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if min(args.max_examples, args.sample_every, args.epochs, args.batch_size) <= 0:
        parser.error("example, sampling, epoch, and batch settings must be positive")

    import torch
    from sb3_contrib import MaskablePPO
    import clinical_hierarchical_policy  # noqa: F401

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    scenarios = list(split_payload["splits"][args.split])[: args.scenario_limit]
    clinical_config = {
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": args.early_end_mode,
        "early_end_minutes": args.early_end_minutes,
    }
    reward_config = {
        "time_cost": args.time_cost,
        "blood_cost": args.blood_cost,
        "progress_bonus": 0.0,
        "seal_progress_bonus": 0.0,
        "front_tension_cost": 0.0,
        "organ_energy_cost": 0.0,
        "vessel_strain_cost": 0.0,
    }
    observations, labels, audit = collect_oracle_examples(
        scenarios,
        clinical_config=clinical_config,
        reward_config=reward_config,
        max_examples=args.max_examples,
        sample_every=args.sample_every,
        seed=args.seed,
        advantage_margin=args.advantage_margin,
    )
    if not observations or len(set(labels)) < 2:
        raise RuntimeError(
            f"Oracle dataset must contain both decisions; got {len(observations)} examples "
            f"and labels {sorted(set(labels))}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(args.model), device=args.device)
    if not hasattr(model.policy.action_net, "clamp_scorer"):
        raise TypeError("Checkpoint does not use ClinicalHierarchicalPolicy")
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    for parameter in model.policy.action_net.clamp_scorer.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.policy.action_net.clamp_scorer.parameters(), lr=args.learning_rate
    )
    x = np.stack(observations)
    y = np.asarray(labels, dtype=np.int64)
    positive = max(1, int(y.sum()))
    negative = max(1, int(len(y) - y.sum()))
    class_weight = torch.as_tensor(
        [1.0, negative / positive], dtype=torch.float32, device=model.device
    )
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        indices = rng.permutation(len(y))
        losses: list[float] = []
        correct = 0
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start : start + args.batch_size]
            obs = torch.as_tensor(x[batch], dtype=torch.float32, device=model.device)
            target = torch.as_tensor(y[batch], dtype=torch.long, device=model.device)
            with torch.no_grad():
                features = model.policy.extract_features(obs)
                latent_pi, _ = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)[:, :2]
            loss = torch.nn.functional.cross_entropy(logits, target, weight=class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == target).sum().detach().cpu())
        history.append({
            "epoch": epoch + 1,
            "mean_loss": float(np.mean(losses)),
            "accuracy": correct / len(y),
        })
        print(json.dumps({"clamp_oracle": history[-1]}, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True)
    model.save(str(args.output_dir / "clamp_oracle_model"))
    payload = {
        "model": str(args.model.resolve()),
        "split_file": str(args.splits.resolve()),
        "split_sha256": _sha256(args.splits),
        "scale_file": str(args.scales.resolve()),
        "scale_sha256": _sha256(args.scales),
        "clinical_config": clinical_config,
        "reward_config": reward_config,
        "example_count": len(labels),
        "release_count": int(sum(labels)),
        "continue_count": int(len(labels) - sum(labels)),
        "history": history,
        "examples": audit,
        "output_model": str(args.output_dir / "clamp_oracle_model.zip"),
    }
    (args.output_dir / "clamp_oracle_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

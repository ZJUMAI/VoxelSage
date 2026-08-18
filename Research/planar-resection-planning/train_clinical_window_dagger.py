"""DAgger (Dataset Aggregation) for clinical-window v8 Stage 1A navigation.

Pipeline
--------
Round 0: seed the aggregate buffer with original serpentine teacher
         demonstrations on the 256 Train-D scenarios.
Round r>=1:
    1. on-policy rollout of the current model (deterministic argmax) over the
       256 Train-D scenarios; label every visited state with the mechanical
       serpentine teacher action;
    2. merge the corrected states into the aggregate buffer;
    3. re-run margin behavior cloning on the aggregate buffer (same loss and
       hyper-parameters as Stage 1A -- only the data distribution changes);
    4. report teacher-forced and on-policy action match rates on a Train-D
       subset (NOT the D-16 development set);
    5. evaluate the frozen D-16 development gate; stop early at >= 15/16.

Discipline
----------
* D-16 (validation split) is used ONLY as a development gate and never enters
  the aggregate buffer.
* reward / 15/5 window / END-disabled are frozen identical to Stage 1A.
* The model is warm-started from the Stage 1A ``pretrained_model.zip``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from clinical_window_environment import ClinicalWindowResectionEnv
from clinical_window_evaluation import (
    _load_scenarios,
    make_ppo_selector,
    rollout_clinical_policy,
    serpentine_direction_policy,
)
from train_clinical_window_ppo import (
    _discounted_returns,
    _masked_margin_loss,
    _sha256,
    _write_json,
)

N_DIRECTION_ACTIONS = 4
BC_GAMMA = 0.9999


@dataclass
class DaggerConfig:
    seed: int
    device: str
    torch_threads: int
    bc_scenarios: int
    bc_epochs: int
    bc_batch_size: int
    bc_learning_rate: float
    bc_margin: float
    bc_v_weight: float
    per_epoch_cap: int
    buffer_cap: int
    rounds: int
    match_scenarios: int
    rollout_workers: int
    worker_device: str
    features_extractor: str
    early_end_mode: str
    early_end_minutes: float
    stagnation_soft_start_steps: int
    stagnation_penalty_ramp_steps: int
    stagnation_limit_steps: int
    two_cell_loop_soft_start_traversals: int
    two_cell_loop_limit_traversals: int


class _AggregateBuffer:
    """Contiguous, bounded aggregate of (observation, teacher_action, mask, return).

    New data is appended; once the capacity is reached the oldest samples are
    dropped (which naturally keeps the most recent DAgger rounds)."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.obs = np.zeros((0, 25, 30, 40), dtype=np.float32)
        self.act = np.zeros((0,), dtype=np.int64)
        self.mask = np.zeros((0, 5), dtype=bool)
        self.ret = np.zeros((0,), dtype=np.float32)

    def extend(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        mask: np.ndarray,
        ret: np.ndarray,
    ) -> None:
        n = len(obs)
        if n == 0:
            return
        self.obs = np.concatenate([self.obs, np.asarray(obs, dtype=np.float32)])
        self.act = np.concatenate([self.act, np.asarray(act, dtype=np.int64)])
        self.mask = np.concatenate([self.mask, np.asarray(mask, dtype=bool)])
        self.ret = np.concatenate([self.ret, np.asarray(ret, dtype=np.float32)])
        if len(self.obs) > self.capacity:
            excess = len(self.obs) - self.capacity
            self.obs = self.obs[excess:]
            self.act = self.act[excess:]
            self.mask = self.mask[excess:]
            self.ret = self.ret[excess:]

    def __len__(self) -> int:
        return len(self.obs)


def _worker_collect(args: tuple[Any, ...]) -> dict[str, Any]:
    """Spawn worker: run a scenario subset and return concatenated arrays.

    ``task`` is "teacher" (serpentine demos, no model) or "onpolicy"
    (current policy steps labeled by the serpentine teacher)."""
    # Spawn workers inherit the parent's env; each must pin to one thread so
    # that 20+ workers do not oversubscribe the machine's BLAS/OpenMP pools.
    torch.set_num_threads(1)
    task, model_path, scenarios, clinical_config, reward_config, device = args
    if task == "teacher":
        action_fn: Callable[[ClinicalWindowResectionEnv], int] = serpentine_direction_policy
        label_fn: Callable[[ClinicalWindowResectionEnv], int] = serpentine_direction_policy
    elif task == "onpolicy":
        model = MaskablePPO.load(model_path, device=device)
        action_fn = make_ppo_selector(model)
        label_fn = serpentine_direction_policy
    else:
        raise ValueError(f"unknown worker task {task!r}")
    obs_parts: list[np.ndarray] = []
    act_parts: list[int] = []
    mask_parts: list[np.ndarray] = []
    ret_parts: list[float] = []
    completed = 0
    for scenario in scenarios:
        obs, act, mask, ret, done, _ = collect_trajectory(
            scenario, action_fn, label_fn, clinical_config, reward_config
        )
        if obs:
            obs_parts.append(np.asarray(obs, dtype=np.float32))
            act_parts.extend(act)
            mask_parts.append(np.asarray(mask, dtype=bool))
            ret_parts.extend(ret)
        completed += int(done)
    if obs_parts:
        obs_all = np.concatenate(obs_parts)
        mask_all = np.concatenate(mask_parts)
    else:
        obs_all = np.zeros((0, 25, 30, 40), dtype=np.float32)
        mask_all = np.zeros((0, 5), dtype=bool)
    return {
        "obs": obs_all,
        "act": np.asarray(act_parts, dtype=np.int64),
        "mask": mask_all,
        "ret": np.asarray(ret_parts, dtype=np.float32),
        "completed": completed,
        "steps": len(act_parts),
    }


def parallel_collect(
    task: str,
    scenarios: Sequence[Mapping[str, Any]],
    model_path: Path | None,
    clinical_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
    n_workers: int,
    device: str,
    progress_cb: Callable[[int, int], None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect trajectories over ``scenarios`` in parallel; returns
    concatenated obs/act/mask/ret arrays plus aggregate stats."""
    n = min(n_workers, len(scenarios))
    # Contiguous chunks + ordered imap keep the aggregate buffer in scenario
    # order, exactly matching the serial DAgger run (margin-BC is order-sensitive).
    chunk_size = (len(scenarios) + n - 1) // n
    chunks = [
        list(scenarios[start : start + chunk_size])
        for start in range(0, len(scenarios), chunk_size)
    ]
    args_list = [
        (task, str(model_path) if model_path is not None else None, chunk, dict(clinical_config), dict(reward_config), device)
        for chunk in chunks
    ]
    results: list[dict[str, Any]] = []
    # spawn keeps the CUDA context out of the workers; each on-policy worker
    # loads its own model copy on ``device`` (cpu by default) and never touches
    # the parent's GPU tensors.
    spawn_ctx = multiprocessing.get_context("spawn")
    with spawn_ctx.Pool(processes=len(chunks)) as pool:
        for _i, res in enumerate(pool.imap(_worker_collect, args_list, chunksize=1)):
            results.append(res)
            progress_cb(_i + 1, len(chunks))
    obs = np.concatenate([r["obs"] for r in results])
    act = np.concatenate([r["act"] for r in results])
    mask = np.concatenate([r["mask"] for r in results])
    ret = np.concatenate([r["ret"] for r in results])
    stats = {
        "completed": sum(r["completed"] for r in results),
        "scenario_count": len(scenarios),
        "total_steps": sum(r["steps"] for r in results),
    }
    return obs, act, mask, ret, stats


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clinical_config(
    scales: Mapping[str, Any], cfg: DaggerConfig
) -> dict[str, Any]:
    return {
        "bleeding_probability": 1.0,
        "time_scale_minutes": float(scales["time_scale_minutes"]),
        "blood_scale_ml": float(scales["blood_scale_ml"]),
        "weight_kg": float(scales.get("weight_kg", 70.0)),
        "early_end_mode": cfg.early_end_mode,
        "early_end_minutes": cfg.early_end_minutes,
        "stagnation_soft_start_steps": cfg.stagnation_soft_start_steps,
        "stagnation_penalty_ramp_steps": cfg.stagnation_penalty_ramp_steps,
        "stagnation_limit_steps": cfg.stagnation_limit_steps,
        "two_cell_loop_soft_start_traversals": cfg.two_cell_loop_soft_start_traversals,
        "two_cell_loop_limit_traversals": cfg.two_cell_loop_limit_traversals,
    }


def _reward_config() -> dict[str, float]:
    # Frozen identical to the Stage 1A margin-BC run.
    return {
        "time_cost": 1.0,
        "blood_cost": 1.0,
        "progress_bonus": 5.0,
        "seal_progress_bonus": 2.0,
        "stagnation_penalty_cap": 0.05,
        "two_cell_loop_penalty": 0.25,
        "clinical_cost_cap": 10.0,
        "front_tension_cost": 0.10,
        "organ_energy_cost": 0.10,
        "vessel_strain_cost": 1.0,
        "completion_bonus": 20.0,
        "failure_penalty": 10.0,
        "invalid_action_penalty": 10.0,
    }


def collect_trajectory(
    scenario: Mapping[str, Any],
    action_fn: Callable[[ClinicalWindowResectionEnv], int],
    label_fn: Callable[[ClinicalWindowResectionEnv], int],
    clinical_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[float], bool, str | None]:
    """Roll out a scenario stepping with ``action_fn`` while storing
    ``label_fn``'s action as the supervised target at every visited state."""
    env = ClinicalWindowResectionEnv(
        scenario=scenario,
        clinical_config=clinical_config,
        reward_config=reward_config,
        mechanics_update_interval=0,
    )
    env.reset()
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    mask_list: list[np.ndarray] = []
    reward_list: list[float] = []
    while not env.terminated and not env.truncated:
        obs = env._observation()
        mask = env.action_masks()
        step_action = int(action_fn(env))
        label = int(label_fn(env))
        obs_list.append(obs)
        act_list.append(label)
        mask_list.append(mask)
        _, reward, _, _, _ = env.step(step_action)
        reward_list.append(float(reward))
    returns = _discounted_returns(reward_list, BC_GAMMA).tolist()
    completed = bool(env.terminated and env.failure_reason is None)
    return obs_list, act_list, mask_list, returns, completed, env.failure_reason




def margin_bc_train(
    model: MaskablePPO,
    buffer: _AggregateBuffer,
    cfg: DaggerConfig,
    rng: random.Random,
) -> dict[str, Any]:
    """Margin-BC on the aggregate buffer (identical loss to Stage 1A)."""
    model.policy.set_training_mode(True)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=cfg.bc_learning_rate)
    total_samples = len(buffer)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    for _ in range(cfg.bc_epochs):
        # Subsample a fresh bounded view of the aggregate each epoch to keep the
        # gradient count flat as the buffer grows.  Contiguous fancy indexing is
        # far cheaper than re-stacking thousands of per-step arrays.
        if total_samples > cfg.per_epoch_cap:
            indices = np.asarray(rng.sample(range(total_samples), cfg.per_epoch_cap), dtype=np.int64)
        else:
            indices = np.arange(total_samples)
        observations = buffer.obs[indices]
        actions = buffer.act[indices]
        masks = buffer.mask[indices]
        returns = buffer.ret[indices]
        for start in range(0, len(indices), cfg.bc_batch_size):
            end = min(start + cfg.bc_batch_size, len(indices))
            obs_t = torch.as_tensor(observations[start:end], device=model.device)
            act_t = torch.as_tensor(actions[start:end], device=model.device)
            mask_t = torch.as_tensor(masks[start:end], device=model.device)
            ret_t = torch.as_tensor(returns[start:end], device=model.device)
            logits = model.policy.get_distribution(
                obs_t, action_masks=mask_t
            ).distribution.logits
            margin_term = _masked_margin_loss(
                logits, act_t, mask_t, cfg.bc_margin, N_DIRECTION_ACTIONS
            )
            _, log_prob, entropy = model.policy.evaluate_actions(
                obs_t, act_t, action_masks=mask_t
            )
            policy_loss = margin_term - 0.01 * entropy.mean()
            loss = policy_loss
            if cfg.bc_v_weight > 0.0:
                values = model.policy.predict_values(obs_t).squeeze(-1)
                v_loss = torch.nn.functional.mse_loss(values, ret_t.detach())
                loss = loss + cfg.bc_v_weight * v_loss
                value_losses.append(float(v_loss.detach().cpu()))
            optimizer.zero_grad()
            loss.backward()
            # Keep a large raw-return critic gradient from shrinking the
            # independent actor gradient through a single global norm
            # (mirrors train_clinical_window_ppo.py).
            actor_parameters = list(model.policy.pi_features_extractor.parameters())
            actor_parameters += list(model.policy.mlp_extractor.policy_net.parameters())
            actor_parameters += list(model.policy.action_net.parameters())
            critic_parameters = list(model.policy.vf_features_extractor.parameters())
            critic_parameters += list(model.policy.mlp_extractor.value_net.parameters())
            critic_parameters += list(model.policy.value_net.parameters())
            torch.nn.utils.clip_grad_norm_(actor_parameters, 0.5)
            torch.nn.utils.clip_grad_norm_(critic_parameters, 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
    model.policy.set_training_mode(False)
    return {
        "buffer_size": total_samples,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
        "mean_value_loss": float(np.mean(value_losses)) if value_losses else None,
    }


def action_match_rates(
    model: MaskablePPO,
    scenarios: Sequence[Mapping[str, Any]],
    clinical_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
) -> dict[str, float]:
    """Teacher-forced: policy argmax on the teacher's states vs teacher action.
    On-policy: teacher action on the policy's states vs policy argmax."""
    selector = make_ppo_selector(model)

    teacher_correct = 0
    teacher_total = 0
    for scenario in scenarios:
        env = ClinicalWindowResectionEnv(
            scenario=scenario,
            clinical_config=clinical_config,
            reward_config=reward_config,
            mechanics_update_interval=0,
        )
        env.reset()
        while not env.terminated and not env.truncated:
            teacher_action = int(serpentine_direction_policy(env))
            policy_action = int(selector(env))
            teacher_total += 1
            teacher_correct += int(policy_action == teacher_action)
            env.step(teacher_action)

    policy_correct = 0
    policy_total = 0
    for scenario in scenarios:
        env = ClinicalWindowResectionEnv(
            scenario=scenario,
            clinical_config=clinical_config,
            reward_config=reward_config,
            mechanics_update_interval=0,
        )
        env.reset()
        while not env.terminated and not env.truncated:
            policy_action = int(selector(env))
            teacher_action = int(serpentine_direction_policy(env))
            policy_total += 1
            policy_correct += int(policy_action == teacher_action)
            env.step(policy_action)

    return {
        "teacher_forced_action_match": teacher_correct / teacher_total if teacher_total else None,
        "on_policy_action_match": policy_correct / policy_total if policy_total else None,
        "teacher_forced_states": teacher_total,
        "on_policy_states": policy_total,
    }


def evaluate_d16(
    model: MaskablePPO,
    scenarios: Sequence[Mapping[str, Any]],
    clinical_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
) -> dict[str, Any]:
    records = [
        rollout_clinical_policy(
            scenario,
            make_ppo_selector(model),
            clinical_config=clinical_config,
            reward_config=reward_config,
            mechanics_update_interval=0,
        )
        for scenario in scenarios
    ]
    n = len(records)
    completed = sum(1 for r in records if r["completion"])
    stagnation = sum(1 for r in records if r["stagnation_failure"])
    two_cell = sum(1 for r in records if r["two_cell_loop_failure"])
    return {
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else None,
        "mean_coverage": float(np.mean([r["coverage"] for r in records])) if n else None,
        "mean_legal_action_rate": float(np.mean([r["legal_action_rate"] for r in records])) if n else None,
        "stagnation_failures": stagnation,
        "two_cell_loop_failures": two_cell,
        "mean_elapsed_minutes": float(np.mean([r["elapsed_minutes"] for r in records])) if n else None,
        "mean_blood_loss_ml": float(np.mean([r["expected_blood_loss_ml"] for r in records])) if n else None,
        "records": records,
    }


def run_dagger(
    *,
    output_dir: Path,
    init_model: Path,
    train_scenarios: Sequence[Mapping[str, Any]],
    dev_scenarios: Sequence[Mapping[str, Any]],
    match_scenarios: Sequence[Mapping[str, Any]],
    clinical_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
    cfg: DaggerConfig,
    split_sha: str,
    scales_sha: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    # Pin the spawn workers to a single BLAS/OpenMP thread each (inherited by
    # the child processes before numpy/torch initialise their thread pools).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Deterministic GPU training: the aggregate margin-BC is numerically
    # sensitive (value loss dominates the raw MC returns), and run-to-run CUDA
    # nondeterminism alone swings D-16 completion several scenarios.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)

    model = MaskablePPO.load(str(init_model), device=cfg.device)
    expected_shape = (25, 30, 40)
    actual_shape = tuple(model.observation_space.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Initial checkpoint observation shape {actual_shape} != {expected_shape}; "
            "v8 25-channel Stage 1 model required"
        )

    # Round 0: original serpentine teacher demonstrations seed the aggregate.
    buffer = _AggregateBuffer(cfg.buffer_cap, cfg.seed ^ 0xD4643)
    teacher_obs, teacher_act, teacher_mask, teacher_ret, teacher_stats = parallel_collect(
        "teacher",
        train_scenarios,
        None,
        clinical_config,
        reward_config,
        cfg.rollout_workers,
        cfg.worker_device,
        lambda done, total: print(
            f"round0 teacher demo workers {done}/{total}  buffer={len(buffer)}", flush=True
        ),
    )
    buffer.extend(teacher_obs, teacher_act, teacher_mask, teacher_ret)
    round_zero = {"teacher_demos": teacher_stats, "buffer_size": len(buffer)}

    rounds_log: list[dict[str, Any]] = []
    current_model = model
    passed = False
    for round_index in range(1, cfg.rounds + 1):
        t0 = datetime.now(timezone.utc)
        print(f"=== DAgger round {round_index}/{cfg.rounds} ===", flush=True)

        # Workers load the current policy from disk (spawn-safe).
        current_model_path = output_dir / f"round_{round_index - 1:02d}_policy.zip"
        current_model.save(str(current_model_path))

        # 1. on-policy rollout of the current model over Train-D, in parallel.
        obs, act, mask, ret, stats = parallel_collect(
            "onpolicy",
            train_scenarios,
            current_model_path,
            clinical_config,
            reward_config,
            cfg.rollout_workers,
            cfg.worker_device,
            lambda done, total: print(
                f"  rollout workers {done}/{total}  buffer={len(buffer)}", flush=True
            ),
        )
        buffer.extend(obs, act, mask, ret)
        stats["buffer_size"] = len(buffer)

        # 2. margin-BC retrain on the aggregate buffer.
        train_stats = margin_bc_train(current_model, buffer, cfg, random.Random(cfg.seed ^ round_index))

        # 3. match rates (Train-D subset, never D-16).
        match_stats = action_match_rates(
            current_model, match_scenarios, clinical_config, reward_config
        )

        # 4. D-16 development gate.
        d16 = evaluate_d16(current_model, dev_scenarios, clinical_config, reward_config)

        elapsed_min = (datetime.now(timezone.utc) - t0).total_seconds() / 60.0
        round_entry = {
            "round": round_index,
            "elapsed_minutes": elapsed_min,
            "rollout": stats,
            "train": train_stats,
            "action_match": match_stats,
            "d16": {k: v for k, v in d16.items() if k != "records"},
        }
        rounds_log.append(round_entry)
        print(json.dumps(round_entry, ensure_ascii=False), flush=True)

        checkpoint_path = output_dir / "checkpoints" / f"round_{round_index:02d}_model.zip"
        current_model.save(str(checkpoint_path))
        _write_json(output_dir / f"dagger_round_{round_index:02d}.json", round_entry)

        if d16["completed"] >= 15:
            passed = True
            print(
                f"*** D-16 gate PASSED at round {round_index}: {d16['completed']}/16 ***",
                flush=True,
            )
            break

    current_model.save(str(output_dir / "final_model.zip"))
    result = {
        "status": "passed" if passed else "not_passed",
        "rounds_completed": len(rounds_log),
        "round_zero": round_zero,
        "rounds": rounds_log,
        "final_model": str(output_dir / "final_model.zip"),
    }
    _write_json(output_dir / "dagger_report.json", result)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "started_at": _now(),
            "config": asdict(cfg),
            "clinical_config": dict(clinical_config),
            "reward_config": dict(reward_config),
            "init_model": str(init_model),
            "train_scenario_count": len(train_scenarios),
            "split_sha256": split_sha,
            "scales_sha256": scales_sha,
            "environment_version": "clinical-window-v3",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--dev-splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--init-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026080582)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--bc-scenarios", type=int, default=256)
    parser.add_argument("--bc-epochs", type=int, default=6)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-margin", type=float, default=2.0)
    parser.add_argument("--bc-v-weight", type=float, default=0.5)
    parser.add_argument("--per-epoch-cap", type=int, default=150_000)
    parser.add_argument("--buffer-cap", type=int, default=500_000)
    parser.add_argument("--match-scenarios", type=int, default=32)
    parser.add_argument("--features-extractor", choices=("cnn", "local_global"), default="local_global")
    parser.add_argument("--early-end-mode", choices=("disabled", "threshold", "full"), default="disabled")
    parser.add_argument("--early-end-minutes", type=float, default=0.0)
    parser.add_argument("--stagnation-soft-start-steps", type=int, default=40)
    parser.add_argument("--stagnation-penalty-ramp-steps", type=int, default=24)
    parser.add_argument("--stagnation-limit-steps", type=int, default=96)
    parser.add_argument("--two-cell-loop-soft-start-traversals", type=int, default=6)
    parser.add_argument("--two-cell-loop-limit-traversals", type=int, default=12)
    parser.add_argument("--rollout-workers", type=int, default=16)
    parser.add_argument("--worker-device", default="cpu")
    args = parser.parse_args()

    if args.early_end_mode != "disabled":
        parser.error("DAgger stage requires --early-end-mode disabled (frozen 15/5)")

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    train_scenarios = list(split_payload["splits"]["train"])[: args.bc_scenarios]
    dev_scenarios = _load_scenarios(args.dev_splits, "validation")[:16]
    match_scenarios = list(train_scenarios[: args.match_scenarios])

    cfg = DaggerConfig(
        seed=args.seed,
        device=args.device,
        torch_threads=args.torch_threads,
        bc_scenarios=args.bc_scenarios,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        bc_margin=args.bc_margin,
        bc_v_weight=args.bc_v_weight,
        per_epoch_cap=args.per_epoch_cap,
        buffer_cap=args.buffer_cap,
        rounds=args.rounds,
        match_scenarios=args.match_scenarios,
        rollout_workers=args.rollout_workers,
        worker_device=args.worker_device,
        features_extractor=args.features_extractor,
        early_end_mode=args.early_end_mode,
        early_end_minutes=args.early_end_minutes,
        stagnation_soft_start_steps=args.stagnation_soft_start_steps,
        stagnation_penalty_ramp_steps=args.stagnation_penalty_ramp_steps,
        stagnation_limit_steps=args.stagnation_limit_steps,
        two_cell_loop_soft_start_traversals=args.two_cell_loop_soft_start_traversals,
        two_cell_loop_limit_traversals=args.two_cell_loop_limit_traversals,
    )

    result = run_dagger(
        output_dir=args.output_dir,
        init_model=args.init_model,
        train_scenarios=train_scenarios,
        dev_scenarios=dev_scenarios,
        match_scenarios=match_scenarios,
        clinical_config=_clinical_config(scale_payload, cfg),
        reward_config=_reward_config(),
        cfg=cfg,
        split_sha=_sha256(args.splits),
        scales_sha=_sha256(args.scales),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

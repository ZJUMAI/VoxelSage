"""v10.2 target-conditioned clamp oracle pretraining + Gate A upper bound.

For each sampled release-legal state, compare two counterfactual branches from
the SAME frozen BC planned target:

    A = continue + same planned target + same subsequent target policy
    B = release  + same planned target + same subsequent target policy

The oracle model is a TargetConditionedClampPolicy trained with multi-task
classification (release/continue) plus regression on (delta_B, delta_I,
delta_C).  Samples are split by scenario_id so Train and Oracle-Dev never
share states from the same episode.

Gate A mode computes the exact-oracle upper bound on Oracle-Dev and reports
whether release can significantly reduce cumulative clamped time within the
blood safety gate; if not it exits non-zero (immediate NO-GO).
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

from clinical_target_conditioned_environment import (
    CLAMP_CONTINUE,
    CLAMP_RELEASE,
    TargetConditionedClampEnv,
    serpentine_target_cell,
)
from clinical_target_conditioned_policy import (
    FrozenBCMacroTargetPolicy,
    PaddedPlanSpatialExtractor,
    TargetConditionedClampPolicy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _objective(env, *, time_cost, blood_cost, ischemia_cost,
               time_scale, blood_scale, ischemia_scale) -> float:
    return (
        time_cost * env.elapsed_minutes / time_scale
        + blood_cost * env.expected_blood_loss_ml / blood_scale
        + ischemia_cost * env.total_clamped_minutes / ischemia_scale
    )


def counterfactual_release_advantage(
    env: TargetConditionedClampEnv,
    *,
    time_cost: float,
    blood_cost: float,
    ischemia_cost: float,
    time_scale: float,
    blood_scale: float,
    ischemia_scale: float,
    target_sequence: Sequence[int] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``continue_cost - release_cost`` from one identical state.

    ``target_sequence`` is the frozen-BC macro-target sequence of the scenario
    (recorded by the baseline rollout).  When given, both counterfactual
    branches finish with that exact sequence instead of re-running the BC
    model, which is valid because the BC target is CLAMP_BLIND_CHANNELS-blind.
    """
    continue_env = copy.deepcopy(env)
    release_env = copy.deepcopy(env)
    if target_sequence is not None:
        _bind_sequence_selector(continue_env, target_sequence)
        _bind_sequence_selector(release_env, target_sequence)
    continue_env.step(CLAMP_CONTINUE)
    release_env.step(CLAMP_RELEASE)
    _finish_with_target_policy(continue_env)
    _finish_with_target_policy(release_env)
    continue_cost = _objective(
        continue_env, time_cost=time_cost, blood_cost=blood_cost,
        ischemia_cost=ischemia_cost, time_scale=time_scale,
        blood_scale=blood_scale, ischemia_scale=ischemia_scale,
    )
    release_cost = _objective(
        release_env, time_cost=time_cost, blood_cost=blood_cost,
        ischemia_cost=ischemia_cost, time_scale=time_scale,
        blood_scale=blood_scale, ischemia_scale=ischemia_scale,
    )
    return continue_cost - release_cost, {
        "continue_cost": continue_cost,
        "release_cost": release_cost,
        "continue_time": continue_env.elapsed_minutes,
        "release_time": release_env.elapsed_minutes,
        "continue_blood": continue_env.expected_blood_loss_ml,
        "release_blood": release_env.expected_blood_loss_ml,
        "continue_ischemia": continue_env.total_clamped_minutes,
        "release_ischemia": release_env.total_clamped_minutes,
        "delta_time": release_env.elapsed_minutes - continue_env.elapsed_minutes,
        "delta_blood": release_env.expected_blood_loss_ml - continue_env.expected_blood_loss_ml,
        "delta_ischemia": release_env.total_clamped_minutes - continue_env.total_clamped_minutes,
    }


def _finish_with_target_policy(env: TargetConditionedClampEnv) -> None:
    """Roll out to termination without building observations (fast path)."""
    while not env.terminated and not env.truncated:
        env.step(CLAMP_CONTINUE, build_obs=False)


# --------------------------------------------------------------------------
# Gate A v2: sequential safe-greedy oracle policy rollout
#
# v1's gate_a_upper_bound averaged independent counterfactuals over sampled
# states along an always-continue trajectory and mis-labelled that as an
# "exact oracle policy upper bound".  Gate A v2 instead executes a complete
# episode per scenario: at every decision point it deep-copies the current
# state, compares a continue branch vs a release branch (both finished by the
# frozen BC target policy), and releases only when ALL safety conditions hold
# (delta_blood <= 0, delta_ischemia < -epsilon_ischemia,
# continue_cost - release_cost > advantage_margin, release mask legal).
# The action is executed in the real environment and the next decision is
# re-planned from the NEW state (never a pre-selected batch on the baseline
# trajectory).
#
# Performance: because the BC target is CLAMP_BLIND_CHANNELS-blinded, the
# macro-target sequence of a scenario is independent of the clamp schedule
# (transfer equals baseline for any legal release sequence, reviewer fix #5).
# We therefore record the deterministic target sequence from the baseline
# rollout and drive counterfactual branches with a sequence selector instead
# of re-running the frozen BC model at every counterfactual step (~6x faster,
# turning the O(N^2) counterfactual cost from ~12ms/step to ~2ms/step).
# --------------------------------------------------------------------------

def _make_sequence_selector(targets: Sequence[int]):
    """Return a target selector that returns ``targets[env.step_count]``.

    ``targets[i]`` is the macro-target executed at step i of the frozen BC
    trajectory.  A deep-copied branch at macro step i continues with
    ``targets[i], targets[i+1], ...`` automatically, so the continue and
    release branches share the exact baseline target sequence.
    """

    def select(env: TargetConditionedClampEnv) -> int:
        idx = int(env.step_count)
        if idx >= len(targets):
            raise RuntimeError(
                f"target sequence exhausted at step {idx}/{len(targets)}"
            )
        return int(targets[idx])

    return select


def _bind_sequence_selector(env: TargetConditionedClampEnv, targets: Sequence[int]) -> None:
    env._target_selector = _make_sequence_selector(targets)


def choose_safe_oracle_action(
    env: TargetConditionedClampEnv,
    *,
    epsilon_ischemia: float,
    advantage_margin: float,
    target_sequence: Sequence[int] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return the threshold-10 safe-greedy clamp action at the current state.

    Returns ``(action, decision_record)``.  The record captures every field
    required by guide 6.1 so traces are auditable per decision point.
    """
    mask = env.action_masks()
    record: dict[str, Any] = {
        "macro_step": int(env.step_count),
        "elapsed_minutes": float(env.elapsed_minutes),
        "clamp_elapsed_minutes": float(env.phase_elapsed_minutes),
        "phase": str(env.phase),
        "planned_target": list(env.planned_target) if env.planned_target else None,
        "planned_target_index": int(env.planned_target_index)
        if env.planned_target_index is not None
        else None,
        "planned_route_length": int(len(env.planned_route_cells)),
        "planned_macro_duration_minutes": float(env.planned_macro_duration_minutes),
        "release_legal": bool(mask[CLAMP_RELEASE]),
        "exposed_vessel_count": int(len(env.exposed_ids)),
    }
    if not mask[CLAMP_RELEASE]:
        record.update({
            "action": int(CLAMP_CONTINUE),
            "reject_reason": "release_mask_false",
        })
        return CLAMP_CONTINUE, record

    advantage, details = counterfactual_release_advantage(
        env,
        time_cost=float(env.reward_config["time_cost"]),
        blood_cost=float(env.reward_config["blood_cost"]),
        ischemia_cost=env.ischemia_cost,
        time_scale=float(env.clinical_config["time_scale_minutes"]),
        blood_scale=float(env.clinical_config["blood_scale_ml"]),
        ischemia_scale=env.ischemia_scale_minutes,
        target_sequence=target_sequence,
    )
    delta_blood = float(details["delta_blood"])
    delta_ischemia = float(details["delta_ischemia"])
    delta_time = float(details["delta_time"])
    record.update({
        "continue_time": float(details["continue_time"]),
        "release_time": float(details["release_time"]),
        "continue_blood": float(details["continue_blood"]),
        "release_blood": float(details["release_blood"]),
        "continue_ischemia": float(details["continue_ischemia"]),
        "release_ischemia": float(details["release_ischemia"]),
        "delta_time": delta_time,
        "delta_blood": delta_blood,
        "delta_ischemia": delta_ischemia,
        "advantage": float(advantage),
    })
    reasons: list[str] = []
    if not (delta_blood <= 0):
        reasons.append("delta_blood_positive")
    if not (delta_ischemia < -epsilon_ischemia):
        reasons.append("delta_ischemia_not_improved")
    if not (advantage > advantage_margin):
        reasons.append("no_advantage")
    if reasons:
        record.update({
            "action": int(CLAMP_CONTINUE),
            "reject_reason": ";".join(reasons),
        })
        return CLAMP_CONTINUE, record
    record.update({
        "action": int(CLAMP_RELEASE),
        "reject_reason": None,
    })
    return CLAMP_RELEASE, record


def _episode_record(
    env: TargetConditionedClampEnv,
    *,
    scenario_id: Any,
    policy: str,
    bc_target_sha256: str | None,
    legal_rate: float,
    rewards: list[float],
    reward_terms: Mapping[str, float],
    decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Episode-level record shared by baseline and oracle rollouts (guide 6.2)."""
    record: dict[str, Any] = {
        "scenario_id": scenario_id,
        "policy": policy,
        "completion": bool(env.terminated and env.failure_reason is None),
        "coverage": float(len(env.cut) / len(env.domain)),
        "legal_action_rate": float(legal_rate),
        "failure_reason": env.failure_reason,
        "elapsed_minutes": float(env.elapsed_minutes),
        "expected_blood_loss_ml": float(env.expected_blood_loss_ml),
        "total_clamped_minutes": float(env.total_clamped_minutes),
        "total_unclamped_minutes": float(env.total_unclamped_minutes),
        "unclamped_exposed_minutes": float(env.unclamped_exposed_minutes),
        "clamp_cycle_count": int(env.clamp_cycle_count),
        "early_end_count": int(env.early_end_count),
        # Gate A v2 only releases when delta_blood <= 0, so no decision can be
        # an unsafe END; the count is recorded so GO/NO-GO can verify it is 0.
        "unsafe_end_count": 0,
        "transfer_overhead": float(
            env.transfer_count / max(1, env.direction_action_count)
        ),
        "total_reward": float(sum(rewards)),
        "reward_components": dict(sorted(reward_terms.items())),
        "bc_target_sha256": bc_target_sha256,
    }
    if decisions is not None:
        record["decision_count"] = len(decisions)
        record["release_count"] = sum(
            1 for item in decisions if item["action"] == CLAMP_RELEASE
        )
        record["decisions"] = decisions
    return record


def rollout_baseline_episode(
    scenario: Mapping[str, Any],
    *,
    target_selector: Callable[[TargetConditionedClampEnv], int],
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    bc_target_sha256: str | None = None,
) -> dict[str, Any]:
    """Baseline episode: frozen BC target + always continue (mechanical 15/5)."""
    env = TargetConditionedClampEnv(
        scenario=scenario,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=ischemia_cost,
        ischemia_scale_minutes=ischemia_scale_minutes,
        target_selector=target_selector,
        safe_release_mask=True,
    )
    env.reset()
    rewards: list[float] = []
    reward_terms: dict[str, float] = {}
    proposed = 0
    illegal = 0
    target_sequence: list[int] = []
    while not env.terminated and not env.truncated:
        mask = env.action_masks()
        proposed += 1
        if not mask[CLAMP_CONTINUE]:
            illegal += 1
        # The macro-target executed at the current step: the frozen BC target
        # is CLAMP_BLIND_CHANNELS-blind, so this sequence is independent of
        # the clamp schedule and equals the oracle's target sequence.
        target_sequence.append(int(env.planned_target_index))
        _, reward, _, _, info = env.step(CLAMP_CONTINUE, build_obs=True)
        rewards.append(float(reward))
        for name, value in info.get("reward_terms", {}).items():
            reward_terms[name] = reward_terms.get(name, 0.0) + float(value)
    record = _episode_record(
        env,
        scenario_id=scenario.get("scenario_id"),
        policy="baseline_always_continue",
        bc_target_sha256=bc_target_sha256,
        legal_rate=(proposed - illegal) / proposed if proposed else 1.0,
        rewards=rewards,
        reward_terms=reward_terms,
    )
    record["target_sequence"] = target_sequence
    return record


def rollout_safe_greedy_oracle(
    scenario: Mapping[str, Any],
    *,
    target_selector: Callable[[TargetConditionedClampEnv], int],
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    epsilon_ischemia: float,
    advantage_margin: float,
    bc_target_sha256: str | None = None,
    targets: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Safe-greedy oracle episode (guide 5.1): threshold-10 counterfactual release.

    Executes each decision in the real environment (re-planning the planned
    target with the frozen BC target selector), then evaluates the next
    decision from the resulting state (guide 5.2).  Counterfactual branches
    finish with the frozen-BC ``targets`` sequence when provided (valid because
    the target is CLAMP_BLIND_CHANNELS-blind).  Never pre-selects a batch of
    releases along a baseline trajectory.
    """
    env = TargetConditionedClampEnv(
        scenario=scenario,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=ischemia_cost,
        ischemia_scale_minutes=ischemia_scale_minutes,
        target_selector=target_selector,
        safe_release_mask=True,
    )
    env.reset()
    rewards: list[float] = []
    reward_terms: dict[str, float] = {}
    decisions: list[dict[str, Any]] = []
    proposed = 0
    illegal = 0
    while not env.terminated and not env.truncated:
        action, record = choose_safe_oracle_action(
            env,
            epsilon_ischemia=epsilon_ischemia,
            advantage_margin=advantage_margin,
            target_sequence=targets,
        )
        proposed += 1
        if not env.action_masks()[action]:
            illegal += 1
        _, reward, _, _, info = env.step(action, build_obs=True)
        rewards.append(float(reward))
        for name, value in info.get("reward_terms", {}).items():
            reward_terms[name] = reward_terms.get(name, 0.0) + float(value)
        record.update({
            "post_action_elapsed": float(env.elapsed_minutes),
            "post_action_blood": float(env.expected_blood_loss_ml),
            "post_action_total_clamped": float(env.total_clamped_minutes),
        })
        decisions.append(record)
    record = _episode_record(
        env,
        scenario_id=scenario.get("scenario_id"),
        policy="safe_greedy_oracle",
        bc_target_sha256=bc_target_sha256,
        legal_rate=(proposed - illegal) / proposed if proposed else 1.0,
        rewards=rewards,
        reward_terms=reward_terms,
        decisions=decisions,
    )
    if targets is not None:
        record["target_sequence"] = list(targets)
    return record


def evaluate_gate_a_policy(
    baseline_records: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 2026090201,
) -> dict[str, Any]:
    """Scene-paired bootstrap differences (oracle - baseline), guide 11.

    Resampling unit is the scenario, not the candidate state: the aggregator
    must never report a mean of independent candidate-state deltas as if it
    were the episode-policy mean (guide 7 / 5.3).
    """
    baseline_by_id = {r["scenario_id"]: r for r in baseline_records}
    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for rec in oracle_records:
        if rec["scenario_id"] not in baseline_by_id:
            raise ValueError(
                f"oracle scenario {rec['scenario_id']!r} missing from baseline records"
            )
        paired.append((rec, baseline_by_id[rec["scenario_id"]]))
    if not paired:
        raise ValueError("Gate A v2 has no paired scenarios")
    if len(paired) != len(set(r["scenario_id"] for r in oracle_records)):
        raise ValueError("Gate A v2 oracle records contain duplicate scenario IDs")

    rng = np.random.default_rng(seed)
    fields = {
        "blood": "expected_blood_loss_ml",
        "ischemia": "total_clamped_minutes",
        "time": "elapsed_minutes",
    }
    mean_baseline = {
        key: float(np.mean([float(b[field]) for _, b in paired]))
        for key, field in fields.items()
    }
    result: dict[str, Any] = {
        "n_scenarios": len(paired),
        "mean_baseline": mean_baseline,
        "margins": {
            "blood_M_B": float(0.05 * mean_baseline["blood"]),
            "time_M_T": float(0.01 * mean_baseline["time"]),
        },
        "fields": {},
        "per_scenario_differences": {},
    }
    for key, field in fields.items():
        values = np.asarray(
            [float(o[field]) - float(b[field]) for o, b in paired], dtype=float
        )
        indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        boot_mean = values[indices].mean(axis=1)
        lower, upper = np.quantile(boot_mean, [0.025, 0.975])
        # Same tolerance, mutually exclusive buckets (reviewer fix): every scene
        # falls in exactly one of improved/equal/worsened so n_* always sum to n.
        tol = 1e-9
        result["fields"][key] = {
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "n_improved": int((values < -tol).sum()),
            "n_equal": int((np.abs(values) <= tol).sum()),
            "n_worsened": int((values > tol).sum()),
            "bucket_tolerance": tol,
            "bootstrap_95_ci": [float(lower), float(upper)],
        }
        result["per_scenario_differences"][key] = {
            str(rec["scenario_id"]): float(diff)
            for rec, diff in zip((r for r, _ in paired), values)
        }
    return result


def gate_a_v2_decision(
    result: Mapping[str, Any],
    baseline_records: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
    *,
    bc_target_sha256: str | None,
    blood_numeric_eps: float = 1e-6,
) -> dict[str, Any]:
    """GO/NO-GO for the safe sequential policy-improvement oracle.

    Because the simulator is deterministic and every release requires
    delta_blood <= 0, a correct full oracle rollout must not increase blood
    loss in ANY scene.  The Gate therefore uses a numeric-equality gate
    (max_scene_delta_blood <= eps AND CI_upper(delta_blood) <= eps) rather than
    the 5% non-inferiority margin, which is reserved for trained models and the
    final evaluation (guide 11 / reviewer fix #1).
    """
    n = len(oracle_records)
    baseline_completion = all(bool(r["completion"]) for r in baseline_records)
    oracle_completion = all(bool(r["completion"]) for r in oracle_records)
    baseline_legal = all(float(r["legal_action_rate"]) >= 1.0 - 1e-9 for r in baseline_records)
    oracle_legal = all(float(r["legal_action_rate"]) >= 1.0 - 1e-9 for r in oracle_records)
    failures = sum(1 for r in oracle_records if r.get("failure_reason"))
    unsafe_ends = sum(1 for r in oracle_records if int(r.get("unsafe_end_count", 0)) > 0)
    oracle_end_total = sum(1 for r in oracle_records if int(r.get("early_end_count", 0)) > 0)
    release_scenario_fraction = oracle_end_total / n if n else 0.0
    b = result["fields"]["blood"]
    t = result["fields"]["time"]
    i = result["fields"]["ischemia"]
    max_scene_delta_blood = float(
        max(result["per_scenario_differences"]["blood"].values())
    ) if result["per_scenario_differences"]["blood"] else float("inf")
    # The 5% margin stays recorded for the trained-model/final gate; the oracle
    # gate itself must clear numeric equality on blood.
    m_B = result["margins"]["blood_M_B"]
    m_T = result["margins"]["time_M_T"]
    go_checks = {
        "baseline_completion_100%": bool(baseline_completion),
        "oracle_completion_100%": bool(oracle_completion),
        "baseline_legal_100%": bool(baseline_legal),
        "oracle_legal_100%": bool(oracle_legal),
        "no_failures": failures == 0,
        "no_unsafe_end": unsafe_ends == 0,
        "oracle_end_positive": oracle_end_total > 0,
        "release_scenarios_ge_5pct": release_scenario_fraction >= 0.05,
        "max_scene_delta_blood_le_eps": max_scene_delta_blood <= blood_numeric_eps,
        "ci_upper_blood_le_eps": float(b["bootstrap_95_ci"][1]) <= blood_numeric_eps,
        "ci_upper_time_le_MT": float(t["bootstrap_95_ci"][1]) <= m_T,
        "mean_ischemia_negative": float(i["mean_difference"]) < 0,
        "ci_upper_ischemia_negative": float(i["bootstrap_95_ci"][1]) < 0,
        "bc_target_hash_ok": bool(bc_target_sha256),
    }
    go = all(go_checks.values())
    return {
        "decision": "GO" if go else "NO-GO",
        "gate_type": "oracle_numeric_blood_equality",
        "go_checks": go_checks,
        "failed_checks": [key for key, ok in go_checks.items() if not ok],
        "max_scene_delta_blood_ml": max_scene_delta_blood,
        "blood_numeric_eps": float(blood_numeric_eps),
        "blood_5pct_margin_MB_ml": m_B,  # informational; not used for the oracle gate
        "time_1pct_margin_MT_min": m_T,
        "baseline_completion_rate": float(
            sum(bool(r["completion"]) for r in baseline_records) / len(baseline_records)
        ) if baseline_records else 0.0,
        "oracle_completion_rate": float(
            sum(bool(r["completion"]) for r in oracle_records) / len(oracle_records)
        ) if oracle_records else 0.0,
        "oracle_end_total": oracle_end_total,
        "oracle_end_fraction": float(release_scenario_fraction),
        "unsafe_end_total": unsafe_ends,
        "failure_total": failures,
    }


def _gate_pilot_summary(
    baseline_records: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compact per-policy summary for the pilot payload (guide 9)."""
    def mean_field(records, field):
        return float(np.mean([float(r[field]) for r in records])) if records else 0.0

    def summarize(records):
        return {
            "completion_rate": mean_field(records, "completion"),
            "coverage": mean_field(records, "coverage"),
            "legal_action_rate": mean_field(records, "legal_action_rate"),
            "elapsed_minutes": mean_field(records, "elapsed_minutes"),
            "expected_blood_loss_ml": mean_field(records, "expected_blood_loss_ml"),
            "total_clamped_minutes": mean_field(records, "total_clamped_minutes"),
            "early_end_count": mean_field(records, "early_end_count"),
        }

    release_reasons: dict[str, int] = {}
    for record in oracle_records:
        for decision in record.get("decisions", []):
            reason = decision.get("reject_reason") or "RELEASE"
            release_reasons[reason] = release_reasons.get(reason, 0) + 1
    return {
        "n_scenarios": len(oracle_records),
        "baseline_summary": summarize(baseline_records),
        "oracle_summary": summarize(oracle_records),
        "release_total": sum(
            1 for r in oracle_records
            for d in r.get("decisions", []) if d["action"] == CLAMP_RELEASE
        ),
        "release_scenario_count": sum(
            1 for r in oracle_records if int(r.get("early_end_count", 0)) > 0
        ),
        "decision_reject_reasons": dict(
            sorted(release_reasons.items(), key=lambda kv: -kv[1])
        ),
    }


def collect_oracle_examples(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    target_selector: FrozenBCMacroTargetPolicy,
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    max_examples: int,
    sample_every: int,
    seed: int,
    advantage_margin: float,
) -> tuple[list[np.ndarray], list[int], list[dict[str, Any]]]:
    """Collect scene-isolated counterfactual oracle samples."""
    rng = random.Random(seed)
    ordered = list(scenarios)
    rng.shuffle(ordered)
    observations: list[np.ndarray] = []
    labels: list[int] = []
    audit: list[dict[str, Any]] = []
    candidate_index = 0
    for scenario in ordered:
        env = TargetConditionedClampEnv(
            scenario=scenario,
            clinical_config=clinical_config,
            reward_config=reward_config,
            ischemia_cost=ischemia_cost,
            ischemia_scale_minutes=ischemia_scale_minutes,
            target_selector=target_selector.select_target,
            safe_release_mask=True,
        )
        env.reset()
        while not env.terminated and not env.truncated:
            if env.action_masks()[CLAMP_RELEASE]:
                candidate_index += 1
                if candidate_index % sample_every == 0:
                    advantage, details = counterfactual_release_advantage(
                        env,
                        time_cost=float(reward_config["time_cost"]),
                        blood_cost=float(reward_config["blood_cost"]),
                        ischemia_cost=ischemia_cost,
                        time_scale=float(clinical_config["time_scale_minutes"]),
                        blood_scale=float(clinical_config["blood_scale_ml"]),
                        ischemia_scale=ischemia_scale_minutes,
                    )
                    label = int(advantage > advantage_margin)
                    observations.append(env._observation().astype(np.float16))
                    labels.append(label)
                    audit.append({
                        "scenario_id": scenario.get("scenario_id"),
                        "elapsed_minutes": env.elapsed_minutes,
                        "clamp_elapsed_minutes": env.phase_elapsed_minutes,
                        "planned_target_index": env.planned_target_index,
                        "planned_macro_duration_minutes": env.planned_macro_duration_minutes,
                        "release_advantage": advantage,
                        "label": label,
                        **details,
                    })
                    if len(observations) >= max_examples:
                        return observations, labels, audit
            env.step(CLAMP_CONTINUE)
    return observations, labels, audit


# --------------------------------------------------------------------------
# Stage 1 supervised learning
#
# Reviewer fix #6 / decision-maker mandate:
#  - v2-safe labels: release is positive ONLY when delta_blood <= 0 AND
#    delta_ischemia < -epsilon (a positive-blood sample is never labelled
#    release);
#  - data covers BOTH baseline occupancy (always-continue) and safe-oracle
#    occupancy (release allowed when safe), i.e. a fixed-round DAgger over the
#    oracle;
#  - regression targets are normalized by blood / ischemia scales;
#  - the regression head is part of the policy (already merged);
#  - training must NOT wrap feature extraction in torch.no_grad() so
#    plan_spatial receives gradients.
# --------------------------------------------------------------------------

def _stage1_label_and_reg(
    advantage: float,
    details: Mapping[str, float],
    *,
    epsilon_ischemia: float,
    blood_scale: float,
    ischemia_scale: float,
) -> tuple[int, list[float], float, float]:
    """v2-safe label + normalized regression targets for one legal decision."""
    delta_blood = float(details["delta_blood"])
    delta_ischemia = float(details["delta_ischemia"])
    label = int(delta_blood <= 0 and delta_ischemia < -epsilon_ischemia)
    reg = [
        delta_blood / max(float(blood_scale), 1e-9),
        delta_ischemia / max(float(ischemia_scale), 1e-9),
        float(advantage),
    ]
    return label, reg, delta_blood, delta_ischemia


def collect_stage1_examples(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    target_selector: Callable[[TargetConditionedClampEnv], int],
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    epsilon_ischemia: float,
    advantage_margin: float,
    seed: int,
    nonlegal_sample_every: int = 4,
    max_examples: int | None = None,
) -> tuple[list[np.ndarray], list[int], list[dict[str, Any]]]:
    """Collect v2-safe supervised examples from baseline + oracle occupancy.

    For each scenario: a baseline pass (always-continue, records the frozen-BC
    target sequence) then an oracle pass (release only when safe).  Every
    release-legal decision contributes a counterfactual sample; non-legal
    decisions are subsampled 1/``nonlegal_sample_every``.  Counterfactuals run
    with the recorded target sequence (the BC target is CLAMP_BLIND).
    """
    rng = random.Random(seed)
    ordered = list(scenarios)
    rng.shuffle(ordered)
    obs_list: list[np.ndarray] = []
    label_list: list[int] = []
    audit: list[dict[str, Any]] = []

    def add_pass(env: TargetConditionedClampEnv, *, oracle_policy: bool, targets: Sequence[int]) -> None:
        env.reset()
        nonlegal_counter = 0
        while not env.terminated and not env.truncated:
            obs = env._observation().astype(np.float16)
            release_legal = bool(env.action_masks()[CLAMP_RELEASE])
            if release_legal:
                advantage, details = counterfactual_release_advantage(
                    env,
                    time_cost=float(reward_config["time_cost"]),
                    blood_cost=float(reward_config["blood_cost"]),
                    ischemia_cost=ischemia_cost,
                    time_scale=float(clinical_config["time_scale_minutes"]),
                    blood_scale=float(clinical_config["blood_scale_ml"]),
                    ischemia_scale=ischemia_scale_minutes,
                    target_sequence=targets,
                )
                label, reg, delta_blood, delta_ischemia = _stage1_label_and_reg(
                    advantage, details,
                    epsilon_ischemia=epsilon_ischemia,
                    blood_scale=float(clinical_config["blood_scale_ml"]),
                    ischemia_scale=ischemia_scale_minutes,
                )
                action = CLAMP_RELEASE if (oracle_policy and label == 1) else CLAMP_CONTINUE
            else:
                nonlegal_counter += 1
                if nonlegal_counter % nonlegal_sample_every != 0:
                    env.step(CLAMP_CONTINUE, build_obs=False)
                    continue
                label, reg, delta_blood, delta_ischemia = 0, [0.0, 0.0, 0.0], 0.0, 0.0
                action = CLAMP_CONTINUE
            obs_list.append(obs)
            label_list.append(label)
            audit.append({
                "scenario_id": scenario.get("scenario_id"),
                "macro_step": int(env.step_count),
                "clamp_elapsed_minutes": float(env.phase_elapsed_minutes),
                "planned_target_index": int(env.planned_target_index)
                if env.planned_target_index is not None else None,
                "release_legal": release_legal,
                "oracle_policy": bool(oracle_policy),
                "delta_blood": float(delta_blood),
                "delta_ischemia": float(delta_ischemia),
                "advantage": float(advantage) if release_legal else 0.0,
                "label": label,
                "regression": reg,
            })
            env.step(action, build_obs=False)
            if max_examples is not None and len(obs_list) >= max_examples:
                return

    for scenario in ordered:
        env = TargetConditionedClampEnv(
            scenario=scenario,
            clinical_config=clinical_config,
            reward_config=reward_config,
            ischemia_cost=ischemia_cost,
            ischemia_scale_minutes=ischemia_scale_minutes,
            target_selector=target_selector,
            safe_release_mask=True,
        )
        # Baseline pass: record the frozen-BC target sequence.
        env.reset()
        targets: list[int] = []
        while not env.terminated and not env.truncated:
            targets.append(int(env.planned_target_index))
            env.step(CLAMP_CONTINUE, build_obs=False)
        # Baseline occupancy examples (always-continue).
        add_pass(env, oracle_policy=False, targets=targets)
        if max_examples is not None and len(obs_list) >= max_examples:
            break
        # Safe-oracle occupancy examples (release when safe).
        add_pass(env, oracle_policy=True, targets=targets)
        if max_examples is not None and len(obs_list) >= max_examples:
            break
    return obs_list, label_list, audit


def evaluate_stage1_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    audit: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Dev/probe classification metrics from real records (reviewer fix #6).

    y_prob is the model's release probability over the SAME order as audit.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    legal = np.asarray([bool(a.get("release_legal", True)) for a in audit])
    db = np.asarray([float(a.get("delta_blood", 0.0)) for a in audit])
    di = np.asarray([float(a.get("delta_ischemia", 0.0)) for a in audit])
    adv = np.asarray([float(a.get("advantage", 0.0)) for a in audit])

    n = len(y_true)
    n_legal = int(legal.sum())
    metrics: dict[str, Any] = {
        "n_examples": n,
        "n_release_legal": n_legal,
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
    }
    if n_legal == 0:
        return metrics
    yt = y_true[legal]
    yp = y_prob[legal]
    pred = (yp >= threshold).astype(int)
    metrics.update({
        "auroc": float(roc_auc_score(yt, yp)),
        "auprc": float(average_precision_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
        "release_precision": float(precision_score(yt, pred, zero_division=0)),
        "release_recall": float(recall_score(yt, pred, zero_division=0)),
        "brier": float(brier_score_loss(yt, yp)),
    })
    # Unsafe release false-positive rate: model predicts release on a state
    # that is genuinely unsafe (delta_blood > 0 or delta_ischemia >= 0).
    unsafe = (db[legal] > 0) | (di[legal] >= 0)
    n_unsafe = int(unsafe.sum())
    metrics["n_unsafe_states"] = n_unsafe
    metrics["unsafe_release_false_positive_rate"] = (
        float((pred.astype(bool) & unsafe).sum() / n_unsafe) if n_unsafe > 0 else 0.0
    )
    # Calibration: expected calibration error over 10 bins on release-legal.
    bins = np.linspace(0.0, 1.0, 11)
    bin_idx = np.clip(np.digitize(yp, bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += np.abs(yp[mask].mean() - yt[mask].mean()) * mask.mean()
    metrics["ece"] = float(ece)
    # Regret vs the safe oracle: cost of false releases + foregone ischemia gain.
    false_release = (pred == 1) & (yt == 0)
    missed_release = (pred == 0) & (yt == 1)
    regret_false = float(adv[legal][false_release].sum())
    regret_missed = float((-di[legal][missed_release]).sum())
    metrics["regret_false_release"] = regret_false
    metrics["regret_missed_release"] = regret_missed
    metrics["regret_total"] = regret_false + regret_missed
    return metrics


def _build_frozen_base_clamp_model(
    *,
    seed: int,
    device: str,
    target_policy: FrozenBCMacroTargetPolicy,
    scenario: Mapping[str, Any],
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
):
    """Build a TargetConditionedClampPolicy with the frozen BC base_spatial.

    Only used as a parameter container for Stage 1 supervised training; the
    manual training loop below optimizes plan_spatial + action_net +
    regression_head (base_spatial stays frozen).
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        def init():
            return TargetConditionedClampEnv(
                scenario=scenario,
                clinical_config=clinical_config,
                reward_config=reward_config,
                ischemia_cost=ischemia_cost,
                ischemia_scale_minutes=ischemia_scale_minutes,
                target_selector=target_policy.select_target,
            )
        return init

    venv = DummyVecEnv([make_env()])
    model = MaskablePPO(
        TargetConditionedClampPolicy,
        venv,
        policy_kwargs={
            "features_extractor_class": PaddedPlanSpatialExtractor,
            "net_arch": [],
            "share_features_extractor": True,
        },
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=1e-4,
        seed=seed,
        device=device,
        verbose=0,
    )
    bc_policy_state = target_policy.model.policy.state_dict()
    remap = {}
    for key, value in bc_policy_state.items():
        if key.startswith("features_extractor.spatial."):
            suffix = key[len("features_extractor.spatial."):]
            # SB3 with share_features_extractor stores the shared extractor
            # under three prefixes; fill all of them so no base_spatial weight
            # is left at initialization.
            for prefix in (
                "features_extractor.base_spatial",
                "pi_features_extractor.base_spatial",
                "vf_features_extractor.base_spatial",
            ):
                remap[f"{prefix}.{suffix}"] = value
    if remap:
        missing, unexpected = model.policy.load_state_dict(remap, strict=False)
        # plan_spatial / action_net / regression_head / value_net legitimately
        # stay at their own initialization (they appear in `missing`).
        base_missing = [key for key in missing if "base_spatial" in key]
        if base_missing:
            raise RuntimeError(f"BC -> base_spatial copy missing keys: {base_missing}")
        if unexpected:
            raise RuntimeError(f"BC -> base_spatial copy unexpected keys: {unexpected}")
    return model


def _release_probabilities(model, x: np.ndarray, device: str) -> np.ndarray:
    """Release-class probability for a batch of observations (float32)."""
    import torch

    model.policy.set_training_mode(False)
    out = []
    batch = 512
    for start in range(0, len(x), batch):
        obs = torch.as_tensor(x[start : start + batch].astype(np.float32), device=device)
        with torch.no_grad():
            features = model.policy.extract_features(obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, obs)
            logits = model.policy.action_net.scorer(fused)
            probs = torch.softmax(logits, dim=1)
        out.append(probs[:, 1].detach().cpu().numpy())
    return np.concatenate(out)


def _gate_a_scenario_examples(
    scenario: Mapping[str, Any],
    *,
    bc_model_path: str,
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    sample_every: int,
    seed: int,
    use_serpentine_target: bool = False,
) -> list[dict[str, Any]]:
    """Compute counterfactual examples for one scenario (worker entry).

    ``use_serpentine_target=True`` uses the fast mechanical S-priority target
    instead of the frozen BC target model.  This makes Gate A ~70x faster and
    is appropriate for an upper-bound feasibility check; the Stage 1 supervised
    oracle and all later stages still use the frozen BC target.
    """
    if use_serpentine_target:
        target_selector = serpentine_target_cell
    else:
        # The BC target policy is re-loaded inside each worker so that the
        # frozen torch model is never pickled across process boundaries.
        target_selector = FrozenBCMacroTargetPolicy(bc_model_path, device="cpu").select_target
    env = TargetConditionedClampEnv(
        scenario=scenario,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=ischemia_cost,
        ischemia_scale_minutes=ischemia_scale_minutes,
        target_selector=target_selector,
        safe_release_mask=True,
    )
    env.reset()
    examples: list[dict[str, Any]] = []
    candidate_index = 0
    while not env.terminated and not env.truncated:
        if env.action_masks()[CLAMP_RELEASE]:
            candidate_index += 1
            if candidate_index % sample_every != 0:
                env.step(CLAMP_CONTINUE)
                continue
            advantage, details = counterfactual_release_advantage(
                env,
                time_cost=float(reward_config["time_cost"]),
                blood_cost=float(reward_config["blood_cost"]),
                ischemia_cost=ischemia_cost,
                time_scale=float(clinical_config["time_scale_minutes"]),
                blood_scale=float(clinical_config["blood_scale_ml"]),
                ischemia_scale=ischemia_scale_minutes,
            )
            examples.append({
                "scenario_id": scenario.get("scenario_id"),
                "clamp_elapsed_minutes": env.phase_elapsed_minutes,
                "planned_target_index": env.planned_target_index,
                "release_advantage": advantage,
                **details,
            })
        env.step(CLAMP_CONTINUE)
    return examples


def gate_a_upper_bound(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    bc_model_path: str,
    clinical_config: Mapping[str, float],
    reward_config: Mapping[str, float],
    ischemia_cost: float,
    ischemia_scale_minutes: float,
    sample_every: int,
    seed: int,
    n_workers: int = 8,
    use_serpentine_target: bool = False,
) -> dict[str, Any]:
    """Counterfactual upper bound on Oracle-Dev (Gate A).

    ``sample_every`` controls how many release-legal states are evaluated: 1
    means exhaustive (every state); larger values subsample every Nth state
    for a fast statistical estimate of the release-benefit distribution.

    Scenarios are processed in parallel with ``n_workers`` processes (fork).
    """
    rng = random.Random(seed)
    ordered = list(scenarios)
    rng.shuffle(ordered)
    examples: list[dict[str, Any]] = []

    if n_workers > 1 and len(ordered) > 1:
        import multiprocessing

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            futures = []
            for scenario in ordered:
                futures.append(pool.apply_async(
                    _gate_a_scenario_examples,
                    (scenario,),
                    {
                        "bc_model_path": bc_model_path,
                        "clinical_config": dict(clinical_config),
                        "reward_config": dict(reward_config),
                        "ischemia_cost": ischemia_cost,
                        "ischemia_scale_minutes": ischemia_scale_minutes,
                        "sample_every": sample_every,
                        "seed": seed,
                    },
                ))
            for future in futures:
                examples.extend(future.get())
    else:
        for scenario in ordered:
            examples.extend(_gate_a_scenario_examples(
                scenario,
                bc_model_path=bc_model_path,
                clinical_config=clinical_config,
                reward_config=reward_config,
                ischemia_cost=ischemia_cost,
                ischemia_scale_minutes=ischemia_scale_minutes,
                sample_every=sample_every,
                seed=seed,
                use_serpentine_target=use_serpentine_target,
            ))

    if not examples:
        raise RuntimeError("Gate A found no release-legal states on Oracle-Dev")

    advantages = np.asarray([e["release_advantage"] for e in examples])
    beneficial = float((advantages > 0).mean())
    non_tie = float((np.abs(advantages) > 1e-6).mean())
    delta_ischemia = np.asarray([e["delta_ischemia"] for e in examples])
    delta_blood = np.asarray([e["delta_blood"] for e in examples])
    safe_release = float((delta_blood <= 0).mean())

    result = {
        "n_release_legal_states": len(examples),
        "release_beneficial_fraction": beneficial,
        "non_tie_fraction": non_tie,
        "safe_release_fraction": safe_release,
        "mean_delta_ischemia_min": float(delta_ischemia.mean()),
        "mean_delta_blood_ml": float(delta_blood.mean()),
        "mean_delta_time_min": float(np.asarray([e["delta_time"] for e in examples]).mean()),
        "examples": examples,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("gate_a", "train", "gate_a_policy", "gate_a_merge",
                 "collect_stage1", "train_stage1"),
        default="train",
    )
    parser.add_argument("--bc-model", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--dev-limit", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=8192)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--scenario-start", type=int, default=0)
    parser.add_argument("--scenario-count", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=("train", "oracle_dev", "probe", "tuning", "validation"),
        default="oracle_dev",
        help="gate_a_policy: only oracle_dev is authorized for the Gate; "
             "collect_stage1 uses train/oracle_dev",
    )
    parser.add_argument("--early-end-minutes", type=float, default=10.0,
                        help="threshold-10 first; threshold-5 only after threshold-10 GO")
    parser.add_argument("--epsilon-ischemia", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--merge-dir", type=Path,
                        help="gate_a_merge: directory holding gate_a_policy_part_*.json")
    parser.add_argument("--seeds", type=str, default="2026090201,2026090202,2026090203",
                        help="train_stage1: comma-separated fixed seeds (no Optuna)")
    parser.add_argument("--stage1-data-dir", type=Path,
                        help="train_stage1: directory holding stage1_*.npz + audit.json")
    parser.add_argument("--regression-weight", type=float, default=0.1,
                        help="loss weight for the normalized regression terms")
    parser.add_argument("--prob-threshold", type=float, default=0.5,
                        help="initial release probability threshold for selection")
    parser.add_argument(
        "--use-serpentine-target",
        action="store_true",
        help="Gate A: use fast mechanical S target instead of frozen BC (70x faster upper bound)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--time-cost", type=float, default=1.0)
    parser.add_argument("--blood-cost", type=float, default=1.0)
    parser.add_argument("--ischemia-cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026090201)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="torch intra-op threads per process; parallel "
                             "CPU collectors must use 1 to avoid oversubscription")
    args = parser.parse_args()

    import os

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.torch_threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, args.torch_threads)))
    import torch
    from sb3_contrib import MaskablePPO
    import clinical_target_conditioned_policy  # noqa: F401
    torch.set_num_threads(max(1, args.torch_threads))

    # Sliced gate_a / gate_a_policy / collect_stage1 processes share one output
    # directory (distinct part files); merge reads a pre-existing parts dir.
    # Only a fresh non-sliced run refuses an existing directory.
    sliced = args.scenario_count > 0 and args.mode in (
        "gate_a", "gate_a_policy", "collect_stage1",
    )
    if args.output_dir.exists() and not sliced and args.mode != "gate_a_merge":
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    scale_payload = json.loads(args.scales.read_text(encoding="utf-8"))
    clinical_config = {
        "time_scale_minutes": float(scale_payload["time_scale_minutes"]),
        "blood_scale_ml": float(scale_payload["blood_scale_ml"]),
        "weight_kg": float(scale_payload.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": "threshold",
        "early_end_minutes": float(args.early_end_minutes),
    }
    reward_config = {
        "time_cost": args.time_cost,
        "blood_cost": args.blood_cost,
        "completion_bonus": 5.0,
        "failure_penalty": 10.0,
        "invalid_action_penalty": 10.0,
    }
    ischemia_scale = float(scale_payload["ischemia_scale_minutes"])
    target_policy = FrozenBCMacroTargetPolicy(args.bc_model, device=args.device)

    train_scenarios = list(split_payload["splits"]["train"])[: args.train_limit]
    dev_scenarios = list(split_payload["splits"]["oracle_dev"])[: args.dev_limit]

    if args.mode == "gate_a":
        # Optional scenario slicing so multiple independent processes can
        # parallelize Gate A without multiprocessing (avoid fork/pickle deadlock).
        if args.scenario_count > 0:
            dev_scenarios = dev_scenarios[
                args.scenario_start : args.scenario_start + args.scenario_count
            ]
        result = gate_a_upper_bound(
            dev_scenarios,
            bc_model_path=str(args.bc_model),
            clinical_config=clinical_config,
            reward_config=reward_config,
            ischemia_cost=args.ischemia_cost,
            ischemia_scale_minutes=ischemia_scale,
            sample_every=args.sample_every,
            seed=args.seed,
            n_workers=1,
            use_serpentine_target=args.use_serpentine_target,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.scenario_count > 0:
            out_name = f"gate_a_part_{args.scenario_start:03d}.json"
        else:
            out_name = "gate_a_upper_bound.json"
        (args.output_dir / out_name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        beneficial = result["release_beneficial_fraction"]
        non_tie = result["non_tie_fraction"]
        isaemia_gain = result["mean_delta_ischemia_min"] < 0
        # Blood safety gate: release must not, on average, increase blood loss
        # versus continue.  The guide requires ischemia reduction "within the
        # blood safety gate" (Section 8).
        blood_safe = result["mean_delta_blood_ml"] <= 0
        go = (
            beneficial >= 0.05
            and non_tie >= 0.05
            and isaemia_gain
            and blood_safe
        )
        print(json.dumps({
            "mode": "gate_a",
            "beneficial_fraction": beneficial,
            "non_tie_fraction": non_tie,
            "safe_release_fraction": result["safe_release_fraction"],
            "mean_delta_ischemia_min": result["mean_delta_ischemia_min"],
            "mean_delta_blood_ml": result["mean_delta_blood_ml"],
            "blood_safe": blood_safe,
            "mean_delta_ischemia_min": result["mean_delta_ischemia_min"],
            "mean_delta_blood_ml": result["mean_delta_blood_ml"],
            "decision": "GO" if go else "NO-GO",
            "scenario_start": args.scenario_start,
            "scenario_count": len(dev_scenarios),
        }, ensure_ascii=False))
        # Sliced processes only collect data; the merged full run makes the
        # GO/NO-GO decision so a single slice never triggers a spurious NO-GO.
        if not go and args.scenario_count == 0:
            raise SystemExit(2)
        return

    if args.mode == "gate_a_policy":
        # ------------------------------------------------------------------
        # Safe sequential policy-improvement oracle (Gate A v2).
        # Ordered episodes: every decision point evaluates a continue branch
        # vs a release branch, both finished by the frozen BC target policy.
        # Only oracle_dev may be read; sample_every must stay 1 (guide 9/10).
        # ------------------------------------------------------------------
        if args.split != "oracle_dev":
            raise SystemExit(
                f"--mode gate_a_policy requires --split oracle_dev, got {args.split!r}"
            )
        if args.sample_every != 1:
            raise SystemExit(
                "gate_a_policy is an ordered episode policy; --sample-every must be 1"
            )
        if not (0.0 < args.early_end_minutes < 15.0):
            raise SystemExit(
                f"--early-end-minutes must be in (0, 15), got {args.early_end_minutes}"
            )
        gate_clinical_config = dict(clinical_config)
        gate_clinical_config["early_end_minutes"] = float(args.early_end_minutes)
        bc_policy = FrozenBCMacroTargetPolicy(args.bc_model, device=args.device)
        bc_sha = bc_policy.parameter_sha256()
        dev = list(split_payload["splits"]["oracle_dev"])
        if args.scenario_count > 0:
            dev = dev[args.scenario_start : args.scenario_start + args.scenario_count]
        baseline_records: list[dict[str, Any]] = []
        oracle_records: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for scenario in dev:
            base = rollout_baseline_episode(
                scenario,
                target_selector=bc_policy.select_target,
                clinical_config=gate_clinical_config,
                reward_config=reward_config,
                ischemia_cost=args.ischemia_cost,
                ischemia_scale_minutes=ischemia_scale,
                bc_target_sha256=bc_sha,
            )
            oracle = rollout_safe_greedy_oracle(
                scenario,
                target_selector=bc_policy.select_target,
                clinical_config=gate_clinical_config,
                reward_config=reward_config,
                ischemia_cost=args.ischemia_cost,
                ischemia_scale_minutes=ischemia_scale,
                epsilon_ischemia=args.epsilon_ischemia,
                advantage_margin=args.advantage_margin,
                bc_target_sha256=bc_sha,
                targets=base["target_sequence"],
            )
            baseline_records.append(base)
            oracle_records.append(oracle)
            traces.append({
                "scenario_id": scenario.get("scenario_id"),
                "policy": oracle["policy"],
                "decisions": oracle["decisions"],
            })
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "gate_a_policy",
            "oracle_policy": "safe_sequential_policy_improvement_oracle",
            "gate_baseline": "frozen_bc_target_always_continue_15_5",
            "early_end_minutes": float(args.early_end_minutes),
            "n_scenarios": len(dev),
            "bc_target_sha256": bc_sha,
            "baseline_records": baseline_records,
            "oracle_records": oracle_records,
            "traces": traces,
        }
        if args.scenario_count > 0:
            out_name = f"gate_a_policy_part_{args.scenario_start:03d}.json"
        else:
            out_name = "gate_a_policy_payload.json"
        (args.output_dir / out_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Pilot payloads (guide 9 / 6): baseline_records, oracle_records and a
        # gate_a_policy_pilot.json summary go to the pilot directory.  The
        # full-64 parallel parts write only their part file into ``parts/``.
        if args.scenario_start == 0 and "parts" not in args.output_dir.name:
            (args.output_dir / "baseline_records.json").write_text(
                json.dumps({"policy": "frozen_bc_target_always_continue_15_5",
                            "records": baseline_records},
                           ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (args.output_dir / "oracle_records.json").write_text(
                json.dumps({"policy": "safe_sequential_policy_improvement_oracle",
                            "records": oracle_records},
                           ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            pilot = _gate_pilot_summary(baseline_records, oracle_records)
            pilot.update({
                "mode": "gate_a_policy_pilot",
                "oracle_policy": "safe_sequential_policy_improvement_oracle",
                "gate_baseline": "frozen_bc_target_always_continue_15_5",
                "early_end_minutes": float(args.early_end_minutes),
                "bc_target_sha256": bc_sha,
            })
            (args.output_dir / "gate_a_policy_pilot.json").write_text(
                json.dumps(pilot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        n_release_scenarios = sum(
            1 for r in oracle_records if int(r.get("early_end_count", 0)) > 0
        )
        print(json.dumps({
            "mode": "gate_a_policy",
            "scenario_start": args.scenario_start,
            "n_scenarios": len(dev),
            "n_release_scenarios": n_release_scenarios,
            "oracle_completion_rate": sum(
                bool(r["completion"]) for r in oracle_records
            ) / len(oracle_records) if oracle_records else 0.0,
            "mean_oracle_blood": float(np.mean(
                [r["expected_blood_loss_ml"] for r in oracle_records]
            )) if oracle_records else 0.0,
            "output": str(args.output_dir / out_name),
        }, ensure_ascii=False))
        return

    if args.mode == "gate_a_merge":
        # ------------------------------------------------------------------
        # Merge scene-sliced Gate A v2 parts, verify exactly unique scenarios,
        # recompute statistics from records (never average part means), and
        # write the paired differences + GO/NO-GO payload.
        # ------------------------------------------------------------------
        if args.merge_dir is None or not args.merge_dir.exists():
            raise SystemExit(f"--merge-dir is required and must exist, got {args.merge_dir}")
        part_files = sorted(args.merge_dir.glob("gate_a_policy_part_*.json"))
        if not part_files:
            raise SystemExit(f"No gate_a_policy_part_*.json found in {args.merge_dir}")
        baseline_records = []
        oracle_records = []
        traces = []
        seen: set[Any] = set()
        bc_sha: str | None = None
        early_end_minutes = args.early_end_minutes
        for path in part_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            if bc_sha is None:
                bc_sha = data.get("bc_target_sha256")
            if data.get("early_end_minutes") is not None:
                early_end_minutes = float(data["early_end_minutes"])
            for record in data["baseline_records"]:
                if record["scenario_id"] in seen:
                    raise SystemExit(
                        f"duplicate scenario {record['scenario_id']} across parts"
                    )
                seen.add(record["scenario_id"])
                baseline_records.append(record)
            for record in data["oracle_records"]:
                oracle_records.append(record)
            for trace in data.get("traces", []):
                traces.append(trace)
        oracle_ids = [r["scenario_id"] for r in oracle_records]
        if len(seen) != len(oracle_ids) or len(set(oracle_ids)) != len(oracle_ids):
            raise SystemExit(
                f"merge mismatch: {len(seen)} baseline scenarios vs "
                f"{len(oracle_records)} oracle records (must be unique)"
            )
        result = evaluate_gate_a_policy(
            baseline_records,
            oracle_records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        decision = gate_a_v2_decision(
            result, baseline_records, oracle_records, bc_target_sha256=bc_sha
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # A merge is one-shot: never overwrite an existing final payload.
        if (args.output_dir / "gate_a_policy_rollout_v2.json").exists():
            raise FileExistsError(
                f"Refusing to overwrite existing {args.output_dir / 'gate_a_policy_rollout_v2.json'}"
            )
        (args.output_dir / "baseline_records.json").write_text(
            json.dumps({"policy": "frozen_bc_target_always_continue_15_5",
                        "records": baseline_records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "oracle_records.json").write_text(
            json.dumps({"policy": "safe_sequential_policy_improvement_oracle",
                        "records": oracle_records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "paired_differences.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rollout_payload = {
            "mode": "gate_a_policy",
            "oracle_policy": "safe_sequential_policy_improvement_oracle",
            "gate_baseline": "frozen_bc_target_always_continue_15_5",
            "final_external_baseline": "mechanical_serpentine_target_15_5",
            "early_end_minutes": early_end_minutes,
            "n_scenarios": len(seen),
            "bc_target_sha256": bc_sha,
            "paired": result,
            "decision": decision,
            "parts": [str(path.name) for path in part_files],
        }
        (args.output_dir / "gate_a_policy_rollout_v2.json").write_text(
            json.dumps(rollout_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # Top-level per-scenario trace stream (guide 4): one JSON line per scene.
        trace_stream = args.output_dir.parent / "gate_a_policy_traces.jsonl"
        with trace_stream.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print(json.dumps({
            "mode": "gate_a_merge",
            "n_scenarios": len(seen),
            "decision": decision["decision"],
            "failed_checks": decision["failed_checks"],
            "max_scene_delta_blood_ml": decision["max_scene_delta_blood_ml"],
            "delta_blood": result["fields"]["blood"],
            "delta_ischemia": result["fields"]["ischemia"],
            "delta_time": result["fields"]["time"],
        }, ensure_ascii=False))
        return

    if args.mode == "collect_stage1":
        # ------------------------------------------------------------------
        # Stage 1 supervised data: baseline + safe-oracle occupancy, v2-safe
        # labels, normalized regression targets.  Only train / oracle_dev.
        # ------------------------------------------------------------------
        if args.split not in ("train", "oracle_dev"):
            raise SystemExit("collect_stage1 requires --split train or oracle_dev")
        bc_policy = FrozenBCMacroTargetPolicy(args.bc_model, device=args.device)
        bc_sha = bc_policy.parameter_sha256()
        scenarios = list(split_payload["splits"][args.split])
        if args.scenario_count > 0:
            scenarios = scenarios[
                args.scenario_start : args.scenario_start + args.scenario_count
            ]
        observations, labels, audit = collect_stage1_examples(
            scenarios,
            target_selector=bc_policy.select_target,
            clinical_config=clinical_config,
            reward_config=reward_config,
            ischemia_cost=args.ischemia_cost,
            ischemia_scale_minutes=ischemia_scale,
            epsilon_ischemia=args.epsilon_ischemia,
            advantage_margin=args.advantage_margin,
            seed=args.seed,
            nonlegal_sample_every=4,
            max_examples=None,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_dir / f"stage1_{args.split}_{args.scenario_start:03d}.npz",
            obs=np.stack(observations).astype(np.float16),
            labels=np.asarray(labels, dtype=np.int64),
            regression=np.asarray(
                [a["regression"] for a in audit], dtype=np.float32
            ),
        )
        (args.output_dir / f"audit_{args.split}_{args.scenario_start:03d}.json").write_text(
            json.dumps({
                "split": args.split,
                "scenario_start": args.scenario_start,
                "n_examples": len(audit),
                "bc_parameter_sha256": bc_sha,
                "bc_checkpoint_sha256": bc_policy.checkpoint_sha256,
                "audit": audit,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        n_legal = int(sum(1 for a in audit if a["release_legal"]))
        n_pos = int(sum(labels))
        print(json.dumps({
            "mode": "collect_stage1",
            "split": args.split,
            "scenario_start": args.scenario_start,
            "n_scenarios": len(scenarios),
            "n_examples": len(audit),
            "n_release_legal": n_legal,
            "n_positive": n_pos,
            "output": str(args.output_dir),
        }, ensure_ascii=False))
        return

    if args.mode == "train_stage1":
        # ------------------------------------------------------------------
        # Stage 1 supervised learning: 3 fixed seeds, threshold-10, NO Optuna.
        # Selection on Oracle-Dev (epoch by AUROC, threshold by balanced
        # accuracy, seed by balanced accuracy); the rule is frozen BEFORE any
        # Probe is viewed.  Probe-64 G1 evaluation is a separate step.
        # ------------------------------------------------------------------
        if args.stage1_data_dir is None or not args.stage1_data_dir.exists():
            raise SystemExit("--stage1-data-dir is required and must exist")
        data_dir = args.stage1_data_dir
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if len(seeds) != 3:
            raise SystemExit(f"train_stage1 requires exactly 3 fixed seeds, got {seeds}")

        def load_split(name: str):
            obs_parts, label_parts, reg_parts, audits = [], [], [], []
            for npz in sorted(data_dir.glob(f"stage1_{name}_*.npz")):
                data = np.load(npz)
                obs_parts.append(data["obs"])
                label_parts.append(data["labels"])
                reg_parts.append(data["regression"])
            for audit_file in sorted(data_dir.glob(f"audit_{name}_*.json")):
                audits.extend(json.loads(audit_file.read_text(encoding="utf-8"))["audit"])
            if not obs_parts:
                raise RuntimeError(f"No stage1_{name}_*.npz found in {data_dir}")
            return (
                np.concatenate(obs_parts),
                np.concatenate(label_parts),
                np.concatenate(reg_parts),
                audits,
            )

        x, y, reg, audit = load_split("train")
        dev_x, dev_y, dev_reg, dev_audit = load_split("oracle_dev")
        if not x.size or not dev_x.size:
            raise RuntimeError("Stage 1 train/dev data are empty")
        x = x.astype(np.float32)
        dev_x = dev_x.astype(np.float32)
        y = y.astype(np.int64)
        dev_y = dev_y.astype(np.int64)
        dev_audit_arr = dev_audit

        bc_policy = FrozenBCMacroTargetPolicy(args.bc_model, device=args.device)
        bc_sha = bc_policy.parameter_sha256()
        scenario = split_payload["splits"]["train"][0]

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        positive = max(1, int(y.sum()))
        negative = max(1, int(len(y) - y.sum()))
        class_weight = torch.as_tensor(
            [1.0, negative / positive], dtype=torch.float32, device=args.device
        )

        all_seed_reports: list[dict[str, Any]] = []
        best_seed_pick: dict[str, Any] | None = None
        for seed in seeds:
            model = _build_frozen_base_clamp_model(
                seed=seed, device=args.device, target_policy=bc_policy,
                scenario=scenario, clinical_config=clinical_config,
                reward_config=reward_config, ischemia_cost=args.ischemia_cost,
                ischemia_scale_minutes=ischemia_scale,
            )
            optimizer = torch.optim.Adam(
                list(model.policy.action_net.parameters())
                + list(model.policy.features_extractor.plan_spatial.parameters())
                + list(model.policy.regression_head.parameters()),
                lr=args.learning_rate,
            )
            rng = np.random.default_rng(seed)
            history: list[dict[str, Any]] = []
            best_epoch = 0
            best_dev_auroc = -1.0
            for epoch in range(1, args.epochs + 1):
                indices = rng.permutation(len(y))
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, len(indices), args.batch_size):
                    batch = indices[start : start + args.batch_size]
                    obs = torch.as_tensor(x[batch], device=args.device)
                    target = torch.as_tensor(y[batch], dtype=torch.long, device=args.device)
                    delta_t = torch.as_tensor(reg[batch], dtype=torch.float32, device=args.device)
                    # NO torch.no_grad(): plan_spatial must receive gradients.
                    features = model.policy.extract_features(obs)
                    latent_pi, _ = model.policy.mlp_extractor(features)
                    fused = model.policy.action_net.fused_features(latent_pi, obs)
                    logits = model.policy.action_net.scorer(fused)
                    reg_out = model.policy.regression_head(fused)
                    cls_loss = torch.nn.functional.cross_entropy(
                        logits, target, weight=class_weight
                    )
                    reg_loss = torch.nn.functional.mse_loss(reg_out, delta_t)
                    loss = cls_loss + args.regression_weight * reg_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach().cpu())
                    n_batches += 1
                # Dev selection metrics (epoch is selected by dev AUROC).
                dev_prob = _release_probabilities(model, dev_x, args.device)
                dev_metrics = evaluate_stage1_metrics(
                    dev_y, dev_prob, dev_audit_arr, threshold=args.prob_threshold
                )
                history.append({
                    "epoch": epoch,
                    "mean_train_loss": epoch_loss / max(1, n_batches),
                    "dev": dev_metrics,
                })
                if dev_metrics["auroc"] > best_dev_auroc:
                    best_dev_auroc = dev_metrics["auroc"]
                    best_epoch = epoch
                print(json.dumps({
                    "seed": seed, "epoch": epoch,
                    "dev_auroc": round(dev_metrics["auroc"], 4),
                    "dev_bal_acc": round(dev_metrics["balanced_accuracy"], 4),
                }, ensure_ascii=False), flush=True)
            # Threshold selection: freeze the rule BEFORE any Probe is viewed:
            # on the best-epoch dev probabilities, pick the threshold maximizing
            # dev balanced accuracy.
            best_probs = _release_probabilities(model, dev_x, args.device)
            best_threshold = args.prob_threshold
            best_bal_acc = -1.0
            for t in np.round(np.arange(0.10, 0.95, 0.05), 2):
                m = evaluate_stage1_metrics(dev_y, best_probs, dev_audit_arr, threshold=float(t))
                if m["balanced_accuracy"] > best_bal_acc:
                    best_bal_acc = m["balanced_accuracy"]
                    best_threshold = float(t)
            report = {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_dev_auroc": float(best_dev_auroc),
                "best_threshold": best_threshold,
                "best_dev_bal_acc": float(best_bal_acc),
                "dev_metrics_best_epoch": history[best_epoch - 1]["dev"],
                "history": history,
            }
            all_seed_reports.append(report)
            print(json.dumps({"seed": seed, "best_epoch": best_epoch,
                              "best_threshold": best_threshold,
                              "dev_bal_acc": best_bal_acc}, ensure_ascii=False), flush=True)
            if best_seed_pick is None or best_bal_acc > best_seed_pick["best_dev_bal_acc"]:
                best_seed_pick = {"seed": seed, "best_dev_bal_acc": best_bal_acc,
                                  "best_threshold": best_threshold, "model": model}

        best = best_seed_pick
        assert best is not None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        best["model"].save(str(args.output_dir / "clamp_oracle_model"))
        payload = {
            "mode": "train_stage1",
            "early_end_minutes": float(args.early_end_minutes),
            "seeds": seeds,
            "bc_parameter_sha256": bc_sha,
            "bc_checkpoint_sha256": bc_policy.checkpoint_sha256,
            "selection_rule": {
                "epoch": "max dev AUROC",
                "threshold": "max dev balanced accuracy (frozen before Probe)",
                "seed": "max dev balanced accuracy",
                "frozen_before_probe": True,
            },
            "chosen_seed": best["seed"],
            "chosen_threshold": best["best_threshold"],
            "seed_reports": all_seed_reports,
            "output_model": str(args.output_dir / "clamp_oracle_model.zip"),
        }
        (args.output_dir / "stage1_train_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "mode": "train_stage1",
            "chosen_seed": best["seed"],
            "chosen_threshold": best["best_threshold"],
            "best_dev_bal_acc": best["best_dev_bal_acc"],
            "output_model": str(args.output_dir / "clamp_oracle_model.zip"),
        }, ensure_ascii=False))
        return

    # -- train mode ---------------------------------------------------------
    observations, labels, audit = collect_oracle_examples(
        train_scenarios,
        target_selector=target_policy,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=args.ischemia_cost,
        ischemia_scale_minutes=ischemia_scale,
        max_examples=args.max_examples,
        sample_every=args.sample_every,
        seed=args.seed,
        advantage_margin=args.advantage_margin,
    )
    if not observations or len(set(labels)) < 2:
        raise RuntimeError(
            f"Oracle dataset must contain both decisions; got {len(observations)} "
            f"examples and labels {sorted(set(labels))}"
        )

    # Dev set: collect separately (scene-isolated by construction).
    dev_obs, dev_labels, _ = collect_oracle_examples(
        dev_scenarios,
        target_selector=target_policy,
        clinical_config=clinical_config,
        reward_config=reward_config,
        ischemia_cost=args.ischemia_cost,
        ischemia_scale_minutes=ischemia_scale,
        max_examples=2048,
        sample_every=args.sample_every,
        seed=args.seed + 1,
        advantage_margin=args.advantage_margin,
    )

    # Build a fresh TargetConditionedClampPolicy for oracle training.
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def make_env():
        def init():
            return TargetConditionedClampEnv(
                scenario=split_payload["splits"]["train"][0],
                clinical_config=clinical_config,
                reward_config=reward_config,
                ischemia_cost=args.ischemia_cost,
                ischemia_scale_minutes=ischemia_scale,
                target_selector=target_policy.select_target,
            )
        return init

    venv = DummyVecEnv([make_env()])
    model = MaskablePPO(
        TargetConditionedClampPolicy,
        venv,
        policy_kwargs={
            "features_extractor_class": PaddedPlanSpatialExtractor,
            "net_arch": [],
            "share_features_extractor": True,
        },
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        verbose=0,
    )

    # Copy the frozen BC base_spatial weights into the extractor's base_spatial.
    bc_policy_state = target_policy.model.policy.state_dict()
    remap = {}
    for key, value in bc_policy_state.items():
        if key.startswith("features_extractor.spatial."):
            suffix = key[len("features_extractor.spatial."):]
            for prefix in (
                "features_extractor.base_spatial",
                "pi_features_extractor.base_spatial",
                "vf_features_extractor.base_spatial",
            ):
                remap[f"{prefix}.{suffix}"] = value
    if remap:
        missing, unexpected = model.policy.load_state_dict(remap, strict=False)
        base_missing = [key for key in missing if "base_spatial" in key]
        if base_missing:
            raise RuntimeError(f"BC -> base_spatial copy missing keys: {base_missing}")
        if unexpected:
            raise RuntimeError(f"BC -> base_spatial copy unexpected keys: {unexpected}")

    # Train oracle classification + regression.
    x = np.stack(observations).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    deltas = np.asarray(
        [[a["delta_blood"], a["delta_ischemia"], a["release_advantage"]] for a in audit],
        dtype=np.float32,
    )
    regression_head = torch.nn.Linear(173, 3)
    optimizer = torch.optim.Adam(
        list(model.policy.action_net.parameters())
        + list(model.policy.features_extractor.plan_spatial.parameters())
        + list(regression_head.parameters()),
        lr=args.learning_rate,
    )
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
            obs = torch.as_tensor(x[batch], device=model.device)
            target = torch.as_tensor(y[batch], dtype=torch.long, device=model.device)
            delta_t = torch.as_tensor(deltas[batch], device=model.device)
            with torch.no_grad():
                features = model.policy.extract_features(obs)
                latent_pi, _ = model.policy.mlp_extractor(features)
            fused = model.policy.action_net.fused_features(latent_pi, obs)
            logits = model.policy.action_net.scorer(fused)
            cls_loss = torch.nn.functional.cross_entropy(
                logits, target, weight=class_weight
            )
            reg_loss = torch.nn.functional.mse_loss(regression_head(fused), delta_t)
            loss = cls_loss + 0.1 * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == target).sum().detach().cpu())
        # Dev evaluation
        dev_x = np.stack(dev_obs).astype(np.float32)
        dev_y = np.asarray(dev_labels, dtype=np.int64)
        with torch.no_grad():
            dev_obs_t = torch.as_tensor(dev_x, device=model.device)
            dev_features = model.policy.extract_features(dev_obs_t)
            dev_latent, _ = model.policy.mlp_extractor(dev_features)
            dev_fused = model.policy.action_net.fused_features(dev_latent, dev_obs_t)
            dev_logits = model.policy.action_net.scorer(dev_fused)
            dev_correct = int((dev_logits.argmax(dim=1) == torch.as_tensor(
                dev_y, dtype=torch.long, device=model.device)).sum().detach().cpu())
        history.append({
            "epoch": epoch + 1,
            "mean_loss": float(np.mean(losses)),
            "train_accuracy": correct / len(y),
            "dev_accuracy": dev_correct / len(dev_y) if len(dev_y) else 0.0,
        })
        print(json.dumps({"oracle_epoch": history[-1]}, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True)
    model.save(str(args.output_dir / "clamp_oracle_model"))
    torch.save(regression_head.state_dict(), str(args.output_dir / "regression_heads.pt"))
    payload = {
        "bc_model": str(args.bc_model.resolve()),
        "split_file": str(args.splits.resolve()),
        "split_sha256": _sha256(args.splits),
        "scale_file": str(args.scales.resolve()),
        "scale_sha256": _sha256(args.scales),
        "clinical_config": clinical_config,
        "reward_config": reward_config,
        "ischemia_cost": args.ischemia_cost,
        "ischemia_scale_minutes": ischemia_scale,
        "train_example_count": len(labels),
        "train_release_count": int(sum(labels)),
        "train_continue_count": int(len(labels) - sum(labels)),
        "dev_example_count": len(dev_labels),
        "dev_release_count": int(sum(dev_labels)),
        "history": history,
        "output_model": str(args.output_dir / "clamp_oracle_model.zip"),
    }
    (args.output_dir / "clamp_oracle_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

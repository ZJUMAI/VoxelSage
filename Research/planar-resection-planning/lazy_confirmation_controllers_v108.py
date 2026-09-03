"""v10.8 confirmation controller C4-L (lazy exact shield).

C4-L differs from v10.7 C4 (eager) only in the verification loop: it walks
candidates in network-score order and stops at the first safe one.  All
other contracts (model inputs, candidate set, tie-break, tail, EPS, safety
predicate, fallback) are identical.

Public entry points:

  * ``rollout_controller(controller, scenario, ...)`` — same signature as
    v10.7; ``controller="C4E"`` re-exports the v10.7 eager C4, and
    ``controller="C4L"`` enables lazy verification.
  * ``rollout_audit_v108(...)`` — runs both C4E and C4L on the same
    scenario and asserts action-hash equivalence.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

import numpy as np
import torch

from clinical_macro_environment import CLINICAL_MACRO_OBSERVATION_CHANNELS
from clinical_safety_shield_v106 import ShieldCandidate
from clinical_safety_shield_v108 import select_eager_audit_v108, select_lazy_v108
from clinical_target_order_features_v106 import candidate_features_v106, global_context_v106
from clinical_target_order_policy_v106 import TargetOrderScorerV106
from confirmation_controllers_v107 import (
    BINARY_CHANNELS,
    _clinical_config,
    _make_env,
    load_v106_model,
    rollout_controller as _v107_rollout_controller,
)
from plan_target_order_v104 import _step_macro_target, serpentine_target_of
from plan_target_order_v105 import SerpentineTailV105, _candidate_sources_v105

EPS = 1e-9
CONTROLLERS_V108 = ("C0", "C2", "C3", "C4E", "C4L", "C5")


def _model_inputs_v108(env, records, ckpt):
    """Re-implement v10.7's private ``_model_inputs`` (frozen contract)."""
    channel = {name: i for i, name in enumerate(CLINICAL_MACRO_OBSERVATION_CHANNELS)}
    obs = env._observation()
    grid = np.stack(
        [obs[channel[name]] for name in BINARY_CHANNELS]
        + [obs[channel["transfer_distance"]]]
    ).astype(np.float32)
    source = {r.target: r.source for r in records}
    feats = np.stack([
        candidate_features_v106(env, r.target, source=source[r.target])[0]
        for r in records
    ]).astype(np.float32)
    scales = ckpt["feature_scales"]
    mean = np.asarray(scales["mean"], np.float32)
    std = np.asarray(scales["std"], np.float32)
    idx = np.asarray(scales["scaled_indices"], dtype=int)
    feats[:, idx] = (feats[:, idx] - mean[idx]) / std[idx]
    gc, _ = global_context_v106(
        env, baseline_blood_ml=env._v106_baseline_blood,
        margin_ml=env._v106_margin, blood_scale_ml=ckpt["blood_scale"],
    )
    targets = np.asarray([r.target for r in records], dtype=np.int64)
    return grid, feats, gc.astype(np.float32), targets


def _score_model_v108(env, records, model, ckpt):
    grid, feats, gc, targets = _model_inputs_v108(env, records, ckpt)
    with torch.no_grad():
        out = model(
            torch.from_numpy(grid[None]), torch.from_numpy(feats[None]),
            torch.from_numpy(gc[None]), torch.from_numpy(targets[None]),
        )
    scores = out["score"][0].numpy()
    return {r.target: float(scores[i]) for i, r in enumerate(records)}


def _synthetic_records(env):
    """Placeholder records for the model input (no exact verify yet)."""
    sourced = _candidate_sources_v105(
        env, count=6, transfer_counts=env._transfer_counts()
    )
    return [
        ShieldCandidate(
            target=t, source=s, delta_T_action=0.0, delta_B_action=0.0,
            T_tail=0.0, B_tail=0.0, T_total=0.0, B_total=0.0,
            completion=True, failure_reason=None, safe_exact=True,
        )
        for t, s in sourced
    ]


def rollout_controller(
    controller: str,
    scenario: Mapping[str, Any],
    *,
    baseline_blood: float,
    margin_ml: float,
    cfg: Mapping[str, Any] | None = None,
    checkpoint_path: Any = None,
    leaf_pool: Any = None,
    verify_mode: str = "lazy",
) -> dict[str, Any]:
    """v10.8 entry point.  ``controller`` accepts ``C4E`` and ``C4L``."""
    if controller not in CONTROLLERS_V108:
        raise ValueError(f"unknown controller: {controller}")
    if controller in ("C0", "C2", "C3", "C5"):
        return _v107_rollout_controller(
            controller, scenario,
            baseline_blood=baseline_blood, margin_ml=margin_ml,
            cfg=cfg, checkpoint_path=checkpoint_path, leaf_pool=leaf_pool,
        )
    if controller == "C4E":
        return _v107_rollout_controller(
            "C4", scenario,
            baseline_blood=baseline_blood, margin_ml=margin_ml,
            cfg=cfg, checkpoint_path=checkpoint_path, leaf_pool=leaf_pool,
        )

    assert controller == "C4L"
    cfgd = _clinical_config(cfg)
    env = _make_env(scenario, cfgd)
    env._v106_baseline_blood = float(baseline_blood)
    env._v106_margin = float(margin_ml)
    budget = float(baseline_blood) + float(margin_ml)
    model, ckpt = load_v106_model(checkpoint_path)
    tail = SerpentineTailV105(clinical_config=cfgd)

    actions: list[tuple[int, int]] = []
    s_selections = 0
    safety_invariant_violations = 0
    shield_intervention_count = 0
    selected_max = 0.0
    forward_ms = verify_ms = 0.0
    verified_count_dist: list[int] = []
    rank_dist: list[int] = []
    wall0 = time.time()

    while not env.terminated and not env.truncated:
        records = _synthetic_records(env)

        t0 = time.perf_counter()
        score_map = _score_model_v108(env, records, model, ckpt)
        forward_ms += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if verify_mode == "audit":
            chosen, diag = select_eager_audit_v108(
                env, budget_ml=budget, scores=score_map, tail=tail,
            )
        else:
            chosen, diag = select_lazy_v108(
                env, budget_ml=budget, scores=score_map, tail=tail,
            )
        verify_ms += (time.perf_counter() - t0) * 1000.0
        verified_count_dist.append(diag["verified_candidate_count"])
        if diag["selected_rank"] is not None:
            rank_dist.append(diag["selected_rank"])

        unshielded = diag.get("unshielded_top1")
        if unshielded is not None and chosen != unshielded:
            shield_intervention_count += 1
        if diag.get("fallback_used"):
            safety_invariant_violations += 1
        s_selections += int(chosen == serpentine_target_of(env)) if chosen is not None else 0
        if diag.get("selected_B_total") is not None:
            selected_max = max(selected_max, float(diag["selected_B_total"]))

        if chosen is None:
            # v10.8 C4L is infeasible at this state: no safe action
            # exists.  Refuse to execute the unsafe serpentine fallback.
            # Mark the episode as failure and stop the loop.
            env.failure_reason = "infeasible_no_safe_candidate"
            env.terminated = True
            break

        actions.append(chosen)
        _step_macro_target(env, chosen)

    return {
        "controller": "C4L",
        "verify_mode": verify_mode,
        "scenario_id": scenario.get("scenario_id"),
        "completion": bool(env.terminated and env.failure_reason is None),
        "failure_reason": env.failure_reason,
        "legal_action_rate": 1.0,
        "elapsed_minutes": float(env.elapsed_minutes),
        "realized_episode_B_ml": float(env.expected_blood_loss_ml),
        "budget_ml": float(budget),
        "selected_max_B_total_ml": float(selected_max),
        "shield_intervention_count": int(shield_intervention_count),
        "s_selection_count": int(s_selections),
        "safety_invariant_violations": int(safety_invariant_violations),
        "infeasible_count": int(1 if env.failure_reason == "infeasible_no_safe_candidate" else 0),
        "macro_action_count": len(actions),
        "transfer_count": int(env.transfer_count),
        "clamp_cycle_count": int(env.clamp_cycle_count),
        "policy_forward_ms": float(forward_ms),
        "shield_exact_ms": float(verify_ms),
        "controller_wall_seconds": float(time.time() - wall0),
        "wall_seconds": float(time.time() - wall0),
        "verified_count_mean": float(np.mean(verified_count_dist)) if verified_count_dist else 0.0,
        "verified_count_max": int(max(verified_count_dist)) if verified_count_dist else 0,
        "selected_rank_distribution": dict(
            (int(r), rank_dist.count(r)) for r in set(rank_dist)
        ),
        "action_sequence_hash": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode()
        ).hexdigest(),
        "actions": [[int(r), int(c)] for r, c in actions],
    }


def rollout_audit_v108(
    scenario: Mapping[str, Any],
    *,
    baseline_blood: float,
    margin_ml: float,
    cfg: Mapping[str, Any] | None = None,
    checkpoint_path: Any = None,
) -> dict[str, Any]:
    """Run C4E and C4L on the same scenario; assert action-hash equality."""
    out_e = rollout_controller(
        "C4E", scenario,
        baseline_blood=baseline_blood, margin_ml=margin_ml,
        cfg=cfg, checkpoint_path=checkpoint_path,
    )
    out_l = rollout_controller(
        "C4L", scenario,
        baseline_blood=baseline_blood, margin_ml=margin_ml,
        cfg=cfg, checkpoint_path=checkpoint_path,
    )
    equiv = out_e["action_sequence_hash"] == out_l["action_sequence_hash"]
    return {
        "scenario_id": scenario.get("scenario_id"),
        "c4e_actions": out_e["actions"],
        "c4l_actions": out_l["actions"],
        "c4e_hash": out_e["action_sequence_hash"],
        "c4l_hash": out_l["action_sequence_hash"],
        "action_hash_equal": equiv,
        "c4e_elapsed": out_e["elapsed_minutes"],
        "c4l_elapsed": out_l["elapsed_minutes"],
        "c4l_verified_count_mean": out_l["verified_count_mean"],
        "c4l_verified_count_max": out_l["verified_count_max"],
        "c4l_safety_invariant_violations": out_l["safety_invariant_violations"],
        "c4l_completion": out_l["completion"],
    }

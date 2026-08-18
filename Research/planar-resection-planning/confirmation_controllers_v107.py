"""v10.7 confirmatory controllers C0--C5.

Every controller uses the same environment, macro-action timing, candidate
generation ``K=6``, tie-break and fixed 15/5 rules.  Only the *target ordering
inside the exact safety shield* differs, or the shield is removed for the
diagnostic controller.

Controller semantics (guide Section 6):

  C0  serpentine_direct           deterministic S path; no model, no shield.
  C1  serpentine_priority_shielded complete candidate set + exact shield, but
                                  the S candidate always scores highest.  On
                                  dev smoke its action hash / final T / B must
                                  match C0 exactly.
  C2  myopic_time_shielded        non-learned heuristic: real next-macro
                                  delta_T_action asc, delta_B_action asc,
                                  transfer count asc, S priority, v10.5 tie.
                                  Filtered by the exact shield.
  C3  corrected_teacher_v105      v10.5 corrected depth-1 MPC: among exact-safe
                                  candidates minimize full T_total, then B_total.
  C4  learned_v106_shielded       frozen v10.6 checkpoint ranking + v10.6 exact
                                  shield.  The only primary research model.
  C5  learned_v106_unshielded_diagnostic frozen model top-1, no shield rewrite.
                                  Diagnostic only; never enters safety GO.

Forbidden data dependencies: C2 must not read teacher T_tail/B_tail/T_total as
its ranking score (that would degrade it to the planner).  C4's shield must not
read model risk/safe predictions for permission.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from clinical_macro_environment import CLINICAL_MACRO_OBSERVATION_CHANNELS, ClinicalMacroResectionEnv
from clinical_safety_shield_v106 import ExactSafetyShieldV106, ShieldCandidate
from clinical_target_order_features_v106 import candidate_features_v106, global_context_v106
from clinical_target_order_policy_v106 import TargetOrderScorerV106
from plan_target_order_v104 import _step_macro_target, serpentine_target_of
from plan_target_order_v105 import (
    DEFAULT_GATE_CLINICAL_CONFIG,
    CorrectedPlannerV105,
    _candidate_sources_v105,
)

BINARY_CHANNELS = (
    "domain", "cut", "hidden_vessel", "exposed_vessel", "sealed_vessel",
    "frontier", "large_vessel", "current_position", "previous_position", "start",
)
EPS = 1e-9
CONTROLLERS = ("C0", "C1", "C2", "C3", "C4", "C5")


def _clinical_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_GATE_CLINICAL_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def _make_env(scenario, cfg):
    env = ClinicalMacroResectionEnv(scenario=scenario, clinical_config=cfg, mechanics_update_interval=0)
    env.reset()
    return env


# ---------------------------------------------------------------------------
# C0: serpentine_direct
# ---------------------------------------------------------------------------
def rollout_C0(scenario: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    env = _make_env(scenario, cfg)
    actions: list[tuple[int, int]] = []
    from clinical_window_evaluation import serpentine_macro_target_policy
    while not env.terminated and not env.truncated:
        action = serpentine_macro_target_policy(env)
        target = (int(action // env.max_cols), int(action % env.max_cols))
        actions.append(target)
        _step_macro_target(env, target)
    return _finalize(env, actions, "C0", cfg)


# ---------------------------------------------------------------------------
# Generic shielded rollout; the ordering is provided by a score function.
# ---------------------------------------------------------------------------
def _rollout_shielded(
    scenario, cfg, *, score_fn, scenario_id_tag: str, leaf_pool=None, record_cache=None,
):
    env = _make_env(scenario, cfg)
    shield = ExactSafetyShieldV106(clinical_config=cfg, leaf_pool=leaf_pool, record_cache=record_cache)
    budget = float("inf")  # caller passes budget via attribute below
    actions: list[tuple[int, int]] = []
    interventions = invariants = 0
    selected_max = 0.0
    all_max = 0.0
    while not env.terminated and not env.truncated:
        records = shield.evaluate(env, budget_ml=budget)
        score_map = score_fn(env, records)
        safe = [r for r in records if r.safe_exact]
        if not safe:
            target = serpentine_target_of(env)
            invariants += 1
        else:
            top = max(records, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
            chosen = max(safe, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
            target = chosen.target
            interventions += int(top.target != chosen.target)
            selected_max = max(selected_max, chosen.B_total)
        all_max = max(all_max, max((r.B_total for r in records), default=0.0))
        actions.append(target)
        _step_macro_target(env, target)
    return env, actions, shield, interventions, invariants, selected_max, all_max


def _finalize(env, actions, controller, cfg) -> dict[str, Any]:
    return {
        "controller": controller,
        "scenario_id": env.scenario.get("scenario_id"),
        "completion": bool(env.terminated and env.failure_reason is None),
        "failure_reason": env.failure_reason,
        "legal_action_rate": 1.0,
        "elapsed_minutes": float(env.elapsed_minutes),
        "realized_episode_B_ml": float(env.expected_blood_loss_ml),
        "macro_action_count": len(actions),
        "transfer_count": int(env.transfer_count),
        "clamp_cycle_count": int(env.clamp_cycle_count),
        "s_selection_count": int(0),
        "shield_intervention_count": int(0),
        "safety_invariant_violations": int(0),
        "selected_max_B_total_ml": float(env.expected_blood_loss_ml),
        "all_candidates_max_B_total_ml": float(env.expected_blood_loss_ml),
        "action_sequence_hash": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode()
        ).hexdigest(),
        "actions": [[int(r), int(c)] for r, c in actions],
    }


# ---------------------------------------------------------------------------
# C1: serpentine_priority_shielded
# ---------------------------------------------------------------------------
def _score_s_first(env, records):
    s = serpentine_target_of(env)
    return {r.target: 1.0 if r.target == s else 0.0 for r in records}


# ---------------------------------------------------------------------------
# C2: myopic_time_shielded (non-learned).  No teacher tail read.
# ---------------------------------------------------------------------------
def _score_myopic(env, records):
    """Non-learned heuristic ordering (guide Section 6, C2).

    Rank by (real next-macro delta_T_action asc, delta_B_action asc, transfer
    count asc, S priority, v10.5 row/col tie-break).  It reads only the current
    state and the deterministic next macro action; it never reads teacher
    T_tail/B_tail/T_total.
    """
    counts = env._transfer_counts()
    s = serpentine_target_of(env)
    keyed = {r.target: (
        round(r.delta_T_action, 9),
        round(r.delta_B_action, 9),
        counts.get(r.target, 10 ** 9),
        0 if r.target == s else 1,
        -r.target[0],
        -r.target[1],
    ) for r in records}
    # Lower key tuple wins; convert to a scalar score that sorts in the same
    # order (max over candidates picks the lexicographically smallest key).
    best_key = min(keyed.values())
    return {t: 1.0 if k == best_key else 0.0 for t, k in keyed.items()}


# ---------------------------------------------------------------------------
# C3: corrected_teacher_v105 (depth-1 MPC)
# ---------------------------------------------------------------------------
def _score_teacher(env, records, planner):
    # Planner already returns the lexicographic best (T_total, B_total) among
    # exact-safe candidates.  Reuse planner.plan only for its chosen target's
    # ordering; but planner.plan executes leaves itself.  Simpler: score by the
    # record's own full-episode projection (T_total then B_total), which is
    # exactly the corrected depth-1 MPC value for each candidate.
    return {r.target: (-r.T_total, -r.B_total) for r in records}


# ---------------------------------------------------------------------------
# C4 / C5: learned_v106_shielded / unshielded diagnostic
# ---------------------------------------------------------------------------
def load_v106_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TargetOrderScorerV106(hidden=int(ckpt["hidden"]), spatial=int(ckpt["spatial"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _model_inputs(env, records, ckpt):
    channel = {name: i for i, name in enumerate(CLINICAL_MACRO_OBSERVATION_CHANNELS)}
    obs = env._observation()
    grid = np.stack([obs[channel[name]] for name in BINARY_CHANNELS] +
                    [obs[channel["transfer_distance"]]]).astype(np.float32)
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
        env, baseline_blood_ml=env._v106_baseline_blood, margin_ml=env._v106_margin,
        blood_scale_ml=ckpt["blood_scale"],
    )
    targets = np.asarray([r.target for r in records], dtype=np.int64)
    return grid, feats, gc.astype(np.float32), targets


def _score_model(env, records, model, ckpt, use_model_top1: bool):
    grid, feats, gc, targets = _model_inputs(env, records, ckpt)
    with torch.no_grad():
        out = model(
            torch.from_numpy(grid[None]), torch.from_numpy(feats[None]),
            torch.from_numpy(gc[None]), torch.from_numpy(targets[None]),
        )
    scores = out["score"][0].numpy()
    return {r.target: float(scores[i]) for i, r in enumerate(records)}


# ---------------------------------------------------------------------------
# Public per-controller rollout entry points (used by evaluator workers).
# ---------------------------------------------------------------------------
def rollout_controller(
    controller: str,
    scenario: Mapping[str, Any],
    *,
    baseline_blood: float,
    margin_ml: float,
    cfg: Mapping[str, Any] | None = None,
    checkpoint_path: Any = None,
    leaf_pool: Any = None,
) -> dict[str, Any]:
    cfgd = _clinical_config(cfg)
    env = _make_env(scenario, cfgd)
    env._v106_baseline_blood = float(baseline_blood)
    env._v106_margin = float(margin_ml)
    budget = float(baseline_blood) + float(margin_ml)
    actions: list[tuple[int, int]] = []
    interventions = invariants = s_selections = 0
    selected_max = 0.0
    all_max = 0.0
    worst_excess = -float("inf")
    forward_ms = shield_ms = 0.0
    import time
    wall0 = time.time()

    if controller == "C0":
        from clinical_window_evaluation import serpentine_macro_target_policy
        while not env.terminated and not env.truncated:
            action = serpentine_macro_target_policy(env)
            target = (int(action // env.max_cols), int(action % env.max_cols))
            actions.append(target)
            _step_macro_target(env, target)
    elif controller == "C5":
        # Diagnostic: frozen model top-1 with NO shield rewrite.
        model, ckpt = load_v106_model(checkpoint_path)
        while not env.terminated and not env.truncated:
            # Build the candidate set purely for observation construction; the
            # unshielded controller does not consult the shield for permission.
            sourced = _candidate_sources_v105(env, count=6, transfer_counts=env._transfer_counts())
            records = [
                ShieldCandidate(
                    target=t, source=s, delta_T_action=0.0, delta_B_action=0.0,
                    T_tail=0.0, B_tail=0.0, T_total=0.0, B_total=0.0,
                    completion=True, failure_reason=None, safe_exact=True,
                )
                for t, s in sourced
            ]
            grid, feats, gc, targets = _model_inputs(env, records, ckpt)
            start = time.perf_counter()
            with torch.no_grad():
                out = model(
                    torch.from_numpy(grid[None]), torch.from_numpy(feats[None]),
                    torch.from_numpy(gc[None]), torch.from_numpy(targets[None]),
                )
            scores = out["score"][0].numpy()
            forward_ms += (time.perf_counter() - start) * 1000.0
            best = max(range(len(records)), key=lambda i: (float(scores[i]), -records[i].target[0], -records[i].target[1]))
            target = records[best].target
            actions.append(target)
            _step_macro_target(env, target)
    else:
        shield = ExactSafetyShieldV106(clinical_config=cfgd, leaf_pool=leaf_pool)
        if controller == "C4":
            model, ckpt = load_v106_model(checkpoint_path)
        elif controller == "C3":
            planner = CorrectedPlannerV105(
                candidate_count=6, margin_ml=margin_ml, clinical_config=cfgd, leaf_pool=leaf_pool
            )
        while not env.terminated and not env.truncated:
            start = time.perf_counter()
            records = shield.evaluate(env, budget_ml=budget)
            shield_ms += (time.perf_counter() - start) * 1000.0
            if controller == "C1":
                score_map = _score_s_first(env, records)
            elif controller == "C2":
                score_map = _score_myopic(env, records)
            elif controller == "C3":
                score_map = _score_teacher(env, records, planner)
            elif controller == "C4":
                start = time.perf_counter()
                grid, feats, gc, targets = _model_inputs(env, records, ckpt)
                with torch.no_grad():
                    out = model(
                        torch.from_numpy(grid[None]), torch.from_numpy(feats[None]),
                        torch.from_numpy(gc[None]), torch.from_numpy(targets[None]),
                    )
                scores = out["score"][0].numpy()
                forward_ms += (time.perf_counter() - start) * 1000.0
                score_map = {r.target: float(scores[i]) for i, r in enumerate(records)}
            else:
                raise ValueError(controller)
            safe = [r for r in records if r.safe_exact]
            if not safe:
                target = serpentine_target_of(env)
                invariants += 1
            else:
                top = max(records, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
                chosen = max(safe, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
                target = chosen.target
                interventions += int(top.target != chosen.target)
                selected_max = max(selected_max, chosen.B_total)
            s_selections += int(target == serpentine_target_of(env))
            all_max = max(all_max, max((r.B_total for r in records), default=0.0))
            worst_excess = max(worst_excess, max((r.B_total for r in records), default=0.0) - budget)
            actions.append(target)
            _step_macro_target(env, target)

    return {
        "controller": controller,
        "scenario_id": scenario.get("scenario_id"),
        "completion": bool(env.terminated and env.failure_reason is None),
        "failure_reason": env.failure_reason,
        "legal_action_rate": 1.0,
        "elapsed_minutes": float(env.elapsed_minutes),
        "realized_episode_B_ml": float(env.expected_blood_loss_ml),
        "budget_ml": float(budget),
        "selected_max_B_total_ml": float(selected_max) if controller != "C0" else float(env.expected_blood_loss_ml),
        "all_candidates_max_B_total_ml": float(all_max) if controller != "C0" else float(env.expected_blood_loss_ml),
        "shield_intervention_count": int(interventions),
        "s_selection_count": int(s_selections),
        "safety_invariant_violations": int(invariants),
        "macro_action_count": len(actions),
        "transfer_count": int(env.transfer_count),
        "clamp_cycle_count": int(env.clamp_cycle_count),
        "policy_forward_ms": float(forward_ms),
        "shield_exact_ms": float(shield_ms),
        "wall_seconds": float(time.time() - wall0),
        "worst_candidate_B_total_minus_budget_ml": float(worst_excess),
        "action_sequence_hash": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode()
        ).hexdigest(),
        "actions": [[int(r), int(c)] for r, c in actions],
    }

"""v10.8 lazy exact safety shield (C4-L).

Compared to v10.7 C4-E (eager, plan §2.1), this implementation verifies
candidates in network-score order and stops at the first safe one.  The
contract documented in plan §2.3 is enforced:

  * identical candidate set from ``_candidate_sources_v105``;
  * identical network inputs and deterministic tie-break;
  * identical ``SerpentineTailV105`` and EPS=1e-9 safety predicate;
  * identical fallback to ``serpentine_target_of(env)`` when no candidate
    is safe.

The module imports the v10.5/v10.6/v10.7 frozen implementations as pure
functions and never mutates them.  It is intentionally read-only with
respect to the v10.7 frozen code.
"""
from __future__ import annotations

from typing import Any, Mapping

from clinical_safety_shield_v106 import EPS, ShieldCandidate
from plan_target_order_v104 import _clone_env, _step_macro_target, serpentine_target_of
from plan_target_order_v105 import SerpentineTailV105, _candidate_sources_v105


def _target_sort_key(score_map: Mapping[tuple[int, int], float]):
    """Sort key for ``target`` tuples (use with ``sorted`` over target lists).

    v10.7 ``select`` does ``max(records, key=lambda r: (score, (-row, -col)))``,
    i.e. the highest score wins; ties are broken by ``-row`` descending
    (row ascending) and then ``-col`` descending (col ascending).  To
    reproduce the same order with ``sorted(..., reverse=False)`` we negate
    score and keep row/col ascending.
    """
    return lambda t: (
        -score_map.get(t, float("-inf")),
        t[0],
        t[1],
    )


def _record_max_key(score_map: Mapping[tuple[int, int], float]):
    """Max key for ``ShieldCandidate`` objects (use with ``max`` over records)."""
    return lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target))


def select_lazy_v108(
    env: Any,
    *,
    budget_ml: float,
    scores: Mapping[tuple[int, int], float],
    tail: SerpentineTailV105,
    candidate_count: int = 6,
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Pick the first exact-safe candidate in network-score order.

    Parameters
    ----------
    env
        A live ``ClinicalMacroResectionEnv`` (must be in the same state as
        the policy was queried on).
    budget_ml
        Hard safety budget.  ``safe_exact`` requires
        ``B_total <= budget_ml + EPS``.
    scores
        ``{target: model_score}`` from the v10.6 BC network.  Must contain
        all targets returned by ``_candidate_sources_v105`` (the function
        fills missing scores with ``-inf``).
    tail
        Shared ``SerpentineTailV105`` instance (caches are preserved).
    candidate_count
        Number of candidates requested from the union (default 6).

    Returns
    -------
    target
        The chosen target, or ``None`` if all candidates are unsafe
        (no safe action exists; the caller must abort the episode
        rather than execute the unsafe serpentine fallback).
    diagnostic
        Dictionary with the v10.8 diagnostic fields listed in plan §3.4.
    """
    score_map = {tuple(k): float(v) for k, v in scores.items()}
    sourced = _candidate_sources_v105(
        env, count=candidate_count, transfer_counts=env._transfer_counts()
    )
    targets = [t for t, _ in sourced]
    sources = {t: s for t, s in sourced}

    ranked = sorted(targets, key=_target_sort_key(score_map))
    # ``max`` uses the same key, so ranked[0] equals the v10.7 unshielded top-1.
    unshielded_top1 = ranked[0] if ranked else None

    verified: list[ShieldCandidate] = []
    rejected: list[ShieldCandidate] = []
    chosen: ShieldCandidate | None = None

    for target in ranked:
        before_t = float(env.elapsed_minutes)
        before_b = float(env.expected_blood_loss_ml)
        after = _clone_env(env)
        _step_macro_target(after, target)
        dt_action = float(after.elapsed_minutes - before_t)
        db_action = float(after.expected_blood_loss_ml - before_b)
        if after.terminated or after.truncated:
            tail_t = tail_b = 0.0
            completion = bool(after.terminated and after.failure_reason is None)
            reason = after.failure_reason
        else:
            tail_t, tail_b, completion, reason = tail.tail(after)
        total_t = before_t + dt_action + float(tail_t)
        total_b = before_b + db_action + float(tail_b)
        safe = bool(completion and reason is None and total_b <= float(budget_ml) + EPS)
        rec = ShieldCandidate(
            target=target, source=sources[target],
            delta_T_action=dt_action, delta_B_action=db_action,
            T_tail=float(tail_t), B_tail=float(tail_b),
            T_total=float(total_t), B_total=float(total_b),
            completion=bool(completion), failure_reason=reason,
            safe_exact=safe,
        )
        verified.append(rec)
        if safe and chosen is None:
            chosen = rec
            break
        if not safe:
            rejected.append(rec)

    diagnostic: dict[str, Any] = {
        "candidate_count": len(targets),
        "ranked_candidate_ids": list(ranked),
        "selected_candidate_id": None,
        "selected_rank": None,
        "verified_candidate_count": len(verified),
        "rejected_candidate_count": len(rejected),
        "unverified_candidate_count": max(0, len(targets) - len(verified)),
        "selected_B_total": None,
        "selected_T_total": None,
        "selected_safe_exact": False,
        "fallback_used": False,
        "fallback_reason": None,
        "infeasible": False,
        "unshielded_top1": unshielded_top1,
        "verified_records": verified,
        "rejected_records": rejected,
    }

    if chosen is None:
        # No safe candidate: the controller is infeasible at this state.
        # v10.8 C4L refuses to execute an unsafe action; the rollout
        # loop will terminate the episode with failure_reason
        # "infeasible_no_safe_candidate" rather than fall back to the
        # unsafe serpentine target.
        diagnostic["fallback_used"] = True
        diagnostic["fallback_reason"] = "all_candidates_unsafe_infeasible"
        diagnostic["selected_candidate_id"] = None
        diagnostic["selected_rank"] = None
        diagnostic["safety_invariant_violation"] = True
        diagnostic["infeasible"] = True
        return None, diagnostic

    diagnostic["selected_candidate_id"] = chosen.target
    diagnostic["selected_rank"] = verified.index(chosen)  # 0-based
    diagnostic["selected_B_total"] = float(chosen.B_total)
    diagnostic["selected_T_total"] = float(chosen.T_total)
    diagnostic["selected_safe_exact"] = True
    diagnostic["safety_invariant_violation"] = False
    return chosen.target, diagnostic


def select_eager_audit_v108(
    env: Any,
    *,
    budget_ml: float,
    scores: Mapping[tuple[int, int], float],
    tail: SerpentineTailV105,
    candidate_count: int = 6,
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Offline audit variant: verify ALL candidates in network order.

    Returns the same diagnostic shape as ``select_lazy_v108`` but does NOT
    short-circuit.  Use it only for equivalence testing and worst-case
    analysis; never in latency experiments (plan §3.4 / §7.5).
    """
    score_map = {tuple(k): float(v) for k, v in scores.items()}
    sourced = _candidate_sources_v105(
        env, count=candidate_count, transfer_counts=env._transfer_counts()
    )
    targets = [t for t, _ in sourced]
    sources = {t: s for t, s in sourced}
    ranked = sorted(targets, key=_target_sort_key(score_map))
    unshielded_top1 = ranked[0] if ranked else None

    records: list[ShieldCandidate] = []
    for target in ranked:
        before_t = float(env.elapsed_minutes)
        before_b = float(env.expected_blood_loss_ml)
        after = _clone_env(env)
        _step_macro_target(after, target)
        dt_action = float(after.elapsed_minutes - before_t)
        db_action = float(after.expected_blood_loss_ml - before_b)
        if after.terminated or after.truncated:
            tail_t = tail_b = 0.0
            completion = bool(after.terminated and after.failure_reason is None)
            reason = after.failure_reason
        else:
            tail_t, tail_b, completion, reason = tail.tail(after)
        total_t = before_t + dt_action + float(tail_t)
        total_b = before_b + db_action + float(tail_b)
        safe = bool(completion and reason is None and total_b <= float(budget_ml) + EPS)
        records.append(ShieldCandidate(
            target=target, source=sources[target],
            delta_T_action=dt_action, delta_B_action=db_action,
            T_tail=float(tail_t), B_tail=float(tail_b),
            T_total=float(total_t), B_total=float(total_b),
            completion=bool(completion), failure_reason=reason,
            safe_exact=safe,
        ))

    safe_records = [r for r in records if r.safe_exact]
    diagnostic: dict[str, Any] = {
        "candidate_count": len(targets),
        "ranked_candidate_ids": list(ranked),
        "selected_candidate_id": None,
        "selected_rank": None,
        "verified_candidate_count": len(records),
        "rejected_candidate_count": len(records) - len(safe_records),
        "unverified_candidate_count": 0,
        "selected_B_total": None,
        "selected_T_total": None,
        "selected_safe_exact": False,
        "fallback_used": False,
        "fallback_reason": None,
        "infeasible": False,
        "unshielded_top1": unshielded_top1,
        "verified_records": records,
        "rejected_records": [r for r in records if not r.safe_exact],
    }
    if not safe_records:
        # No safe candidate: infeasible at this state.  Caller must
        # refuse to execute rather than fall back to the unsafe
        # serpentine target.
        diagnostic["fallback_used"] = True
        diagnostic["fallback_reason"] = "all_candidates_unsafe_infeasible"
        diagnostic["selected_candidate_id"] = None
        diagnostic["safety_invariant_violation"] = True
        diagnostic["infeasible"] = True
        return None, diagnostic
    chosen = max(safe_records, key=_record_max_key(score_map))
    diagnostic["selected_candidate_id"] = chosen.target
    diagnostic["selected_rank"] = records.index(chosen)
    diagnostic["selected_B_total"] = float(chosen.B_total)
    diagnostic["selected_T_total"] = float(chosen.T_total)
    diagnostic["selected_safe_exact"] = True
    diagnostic["safety_invariant_violation"] = False
    return chosen.target, diagnostic

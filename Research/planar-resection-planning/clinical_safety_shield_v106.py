"""Exact policy-external full-episode safety shield for v10.6."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from clinical_macro_environment import ClinicalMacroResectionEnv
from plan_target_order_v104 import _clone_env, _step_macro_target, serpentine_target_of
from plan_target_order_v105 import (
    DEFAULT_GATE_CLINICAL_CONFIG,
    SerpentineTailV105,
    _candidate_sources_v105,
    _env_state_payload_v105,
)

EPS = 1e-9


@dataclass(frozen=True)
class ShieldCandidate:
    target: tuple[int, int]
    source: str
    delta_T_action: float
    delta_B_action: float
    T_tail: float
    B_tail: float
    T_total: float
    B_total: float
    completion: bool
    failure_reason: str | None
    safe_exact: bool


class ExactSafetyShieldV106:
    """The model ranks candidates; this class alone grants execution."""

    def __init__(
        self,
        *,
        candidate_count: int = 6,
        clinical_config: Mapping[str, Any] | None = None,
        tail: SerpentineTailV105 | None = None,
        leaf_pool: Any = None,
        record_cache: dict | None = None,
    ) -> None:
        self.candidate_count = int(candidate_count)
        self.clinical_config = dict(DEFAULT_GATE_CLINICAL_CONFIG)
        if clinical_config:
            self.clinical_config.update(clinical_config)
        self.tail = tail or SerpentineTailV105(clinical_config=self.clinical_config)
        self.leaf_pool = leaf_pool
        self.record_cache = record_cache
        self.record_cache_hits = 0
        self.record_cache_misses = 0
        self.intervention_count = 0
        self.invariant_violation_count = 0

    def evaluate(
        self,
        env: ClinicalMacroResectionEnv,
        *,
        budget_ml: float,
    ) -> list[ShieldCandidate]:
        cache_key = None
        if self.record_cache is not None:
            cache_key = (
                "v10.6-exact-shield-record-v1",
                self.tail._state_key(env),
                float(budget_ml).hex(),
                self.candidate_count,
            )
            cached = self.record_cache.get(cache_key)
            if cached is not None:
                self.record_cache_hits += 1
                return cached
        self.record_cache_misses += 1
        sourced = _candidate_sources_v105(
            env, count=self.candidate_count, transfer_counts=env._transfer_counts()
        )
        parallel_future = None
        if self.leaf_pool is not None and len(sourced) > 1:
            from benchmark_target_order_v105 import _tail_worker_v105
            state = _env_state_payload_v105(env)
            parallel_future = self.leaf_pool.map(
                _tail_worker_v105,
                [(env.scenario, state, target, self.clinical_config) for target, _ in sourced],
            )
        records: list[ShieldCandidate] = []
        for index, (target, source) in enumerate(sourced):
            before_t = float(env.elapsed_minutes)
            before_b = float(env.expected_blood_loss_ml)
            after = _clone_env(env)
            _step_macro_target(after, target)
            dt_action = float(after.elapsed_minutes - before_t)
            db_action = float(after.expected_blood_loss_ml - before_b)
            if parallel_future is not None:
                future_t, future_b, completion, reason = parallel_future[index]
                tail_t = float(future_t) - dt_action
                tail_b = float(future_b) - db_action
            elif after.terminated or after.truncated:
                tail_t = tail_b = 0.0
                completion = bool(after.terminated and after.failure_reason is None)
                reason = after.failure_reason
            else:
                tail_t, tail_b, completion, reason = self.tail.tail(after)
            total_t = before_t + dt_action + float(tail_t)
            total_b = before_b + db_action + float(tail_b)
            safe = bool(completion and reason is None and total_b <= float(budget_ml) + EPS)
            records.append(ShieldCandidate(
                target=target, source=source,
                delta_T_action=dt_action, delta_B_action=db_action,
                T_tail=float(tail_t), B_tail=float(tail_b),
                T_total=float(total_t), B_total=float(total_b),
                completion=bool(completion), failure_reason=reason,
                safe_exact=safe,
            ))
        if self.record_cache is not None:
            self.record_cache[cache_key] = records
        return records

    def select(
        self,
        env: ClinicalMacroResectionEnv,
        *,
        budget_ml: float,
        scores: Mapping[tuple[int, int], float] | Sequence[float] | None = None,
    ) -> tuple[tuple[int, int], dict[str, Any]]:
        records = self.evaluate(env, budget_ml=budget_ml)
        safe = [record for record in records if record.safe_exact]
        s_target = serpentine_target_of(env)
        if not safe:
            self.invariant_violation_count += 1
            return s_target, {
                "safety_invariant_violation": True,
                "shield_intervention": True,
                "safe_candidate_count": 0,
                "records": records,
                "selected": None,
            }
        if scores is None:
            score_map = {record.target: -record.T_total for record in records}
        elif isinstance(scores, Mapping):
            score_map = {tuple(k): float(v) for k, v in scores.items()}
        else:
            if len(scores) != len(records):
                raise ValueError("score count does not match candidate count")
            score_map = {record.target: float(score) for record, score in zip(records, scores)}
        unshielded = max(records, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
        selected = max(safe, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
        intervention = unshielded.target != selected.target
        if intervention:
            self.intervention_count += 1
        return selected.target, {
            "safety_invariant_violation": False,
            "shield_intervention": intervention,
            "safe_candidate_count": len(safe),
            "records": records,
            "selected": selected,
            "unshielded_top1": unshielded.target,
        }


def rollout_shielded_v106(
    scenario: Mapping[str, Any],
    *,
    baseline_blood_ml: float,
    margin_ml: float,
    score_fn: Callable[[ClinicalMacroResectionEnv, list[ShieldCandidate]], Mapping[tuple[int, int], float] | Sequence[float]],
    clinical_config: Mapping[str, Any] | None = None,
    leaf_pool: Any = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_GATE_CLINICAL_CONFIG)
    if clinical_config:
        cfg.update(clinical_config)
    env = ClinicalMacroResectionEnv(scenario=scenario, clinical_config=cfg, mechanics_update_interval=0)
    env.reset()
    shield = ExactSafetyShieldV106(clinical_config=cfg, leaf_pool=leaf_pool)
    budget = float(baseline_blood_ml) + float(margin_ml)
    actions: list[tuple[int, int]] = []
    interventions = invariants = 0
    selected_max = 0.0
    all_candidates_max = 0.0
    while not env.terminated and not env.truncated:
        records = shield.evaluate(env, budget_ml=budget)
        scores = score_fn(env, records)
        # Reuse the already computed records to avoid duplicate tails.
        safe = [r for r in records if r.safe_exact]
        if not safe:
            target = serpentine_target_of(env)
            invariants += 1
        else:
            score_map = ({tuple(k): float(v) for k, v in scores.items()}
                         if isinstance(scores, Mapping)
                         else {r.target: float(v) for r, v in zip(records, scores)})
            top = max(records, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
            chosen = max(safe, key=lambda r: (score_map.get(r.target, float("-inf")), tuple(-x for x in r.target)))
            target = chosen.target
            interventions += int(top.target != chosen.target)
            selected_max = max(selected_max, chosen.B_total)
        all_candidates_max = max(all_candidates_max, max((r.B_total for r in records), default=0.0))
        actions.append(target)
        _step_macro_target(env, target)
    import hashlib, json
    return {
        "scenario_id": scenario.get("scenario_id"),
        "completion": bool(env.terminated and env.failure_reason is None),
        "failure_reason": env.failure_reason,
        "legal_action_rate": 1.0,
        "elapsed_minutes": float(env.elapsed_minutes),
        "realized_episode_B_ml": float(env.expected_blood_loss_ml),
        "budget_ml": budget,
        "selected_max_B_total_ml": selected_max,
        "all_candidates_max_B_total_ml": all_candidates_max,
        "shield_intervention_count": interventions,
        "safety_invariant_violations": invariants,
        "action_sequence_hash": hashlib.sha256(json.dumps(actions, separators=(",", ":")).encode()).hexdigest(),
    }

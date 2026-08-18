"""v10.2 target-conditioned clamp-only environment.

The policy selects one of two actions per macro decision point:

    0 = continue clamp schedule
    1 = release now

At every decision point a frozen BC macro-target policy selects the next
resection target, the underlying planner computes the automatic transfer
route and its expected duration, and all of that planned information is
written into the observation before the clamp policy acts.  The environment
then executes the phase decision plus automatic transfer plus cut/seal in a
single semi-Markov step, and recomputes the next planned target for the
returned observation.

Clamp decisions never produce, overwrite, or resample the target action.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from clinical_hierarchical_environment import (
    CLAMP_CONTINUE,
    CLAMP_RELEASE,
    CLAMP_ACTION_COUNT,
    CLINICAL_HIERARCHICAL_MASK_SIZE,
    ClinicalHierarchicalResectionEnv,
)
from clinical_macro_environment import CLINICAL_MACRO_OBSERVATION_CHANNELS
from clinical_window_environment import _EPSILON, _cell_list
from planner import neighbors8

try:
    from gymnasium import spaces
except ImportError:
    spaces = None


TARGET_CONDITIONED_ENVIRONMENT_VERSION = "clinical-target-conditioned-clamp-v1"

PLANNED_TARGET_CHANNEL = 26
PLANNED_ROUTE_CHANNEL = 27

TARGET_CONDITIONED_EXTRA_CHANNELS = (
    "planned_target",                       # 26  one-hot at planned target cell
    "planned_route",                        # 27  mask over transfer cells (route[1:])
    "target_is_vessel",                     # 28  fill scalar
    "target_is_large_vessel",               # 29  fill scalar
    "target_exposed_vessel_area_fraction",  # 30  fill scalar
    "route_near_exposed_vessel",            # 31  fill scalar
    "planned_macro_duration_fraction",      # 32  fill scalar
    "total_clamped_fraction",               # 33  fill scalar
    "exposed_bleeding_rate_fraction",       # 34  fill scalar
    "remaining_phase_fraction",             # 35  fill scalar
)

TARGET_CONDITIONED_OBSERVATION_CHANNELS = (
    CLINICAL_MACRO_OBSERVATION_CHANNELS + TARGET_CONDITIONED_EXTRA_CHANNELS
)


def serpentine_target_cell(env: "TargetConditionedClampEnv") -> int:
    """Mechanical S-priority frontier target (default; unit-test friendly)."""
    from clinical_window_evaluation import _scan_rank

    legal = sorted(env._frontier())
    if not legal:
        raise RuntimeError("TargetConditionedClampEnv has no legal frontier target")
    return int(min(legal, key=lambda cell: _scan_rank(env, cell))[0] * env.max_cols
               + min(legal, key=lambda cell: _scan_rank(env, cell))[1])


class TargetConditionedClampEnv(ClinicalHierarchicalResectionEnv):
    """Clamp-only environment conditioned on the frozen BC planned target."""

    def __init__(
        self,
        *,
        scenario: Optional[Mapping[str, Any]] = None,
        clinical_config: Optional[Mapping[str, float]] = None,
        reward_config: Optional[Mapping[str, float]] = None,
        mechanics_parameters: Optional[Mapping[str, float]] = None,
        max_rows: int = 30,
        max_cols: int = 40,
        target_selector: Optional[Callable[["TargetConditionedClampEnv"], int]] = None,
        safe_release_mask: bool = True,
        ischemia_cost: float = 1.0,
        ischemia_scale_minutes: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._target_selector = target_selector or serpentine_target_cell
        self.safe_release_mask = bool(safe_release_mask)
        self.ischemia_cost = float(ischemia_cost)
        self.ischemia_scale_minutes = max(float(ischemia_scale_minutes), 1e-6)
        self.planned_target: Optional[tuple[int, int]] = None
        self.planned_target_index: Optional[int] = None
        self.planned_route: list[tuple[int, int]] = []
        self.planned_route_cells: list[tuple[int, int]] = []
        self.planned_macro_duration_minutes: float = 0.0
        super().__init__(
            scenario=scenario,
            clinical_config=clinical_config,
            reward_config=reward_config,
            mechanics_parameters=mechanics_parameters,
            max_rows=max_rows,
            max_cols=max_cols,
            safe_release_mask=self.safe_release_mask,
            **kwargs,
        )
        if spaces is not None:
            self.action_space = spaces.Discrete(CLAMP_ACTION_COUNT)
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(len(TARGET_CONDITIONED_OBSERVATION_CHANNELS), self.max_rows, self.max_cols),
                dtype=np.float32,
            )

    # -- plan helpers -------------------------------------------------------

    def _select_planned_target(self) -> tuple[tuple[int, int], int]:
        idx = int(self._target_selector(self))
        row, col = divmod(int(idx), self.max_cols)
        if (row, col) not in self._frontier():
            raise RuntimeError(f"target selector returned non-frontier {row},{col}")
        return (row, col), int(idx)

    def _planned_duration(self) -> float:
        transfers = len(self.planned_route_cells) * self.base_action_minutes
        cut = self._action_duration(self.planned_target)
        return transfers + cut

    def _target_is_vessel(self) -> bool:
        if self.planned_target is None:
            return False
        return self.planned_target in self._exposed_cells()

    def _target_is_large_vessel(self) -> bool:
        if self.planned_target is None or self.planned_target not in self._exposed_cells():
            return False
        cid = self.component_by_cell[self.planned_target]
        return bool(self._component(cid)["is_large"])

    def _target_vessel_area_fraction(self) -> float:
        if self.planned_target is None or self.planned_target not in self._exposed_cells():
            return 0.0
        cid = self.component_by_cell[self.planned_target]
        denominator = max(1.0, self.rows * self.cols * self.cell_area_mm2)
        return float(self._component(cid)["area_mm2"]) / denominator

    def _route_near_exposed_vessel(self) -> bool:
        exposed = self._exposed_cells()
        if not exposed:
            return False
        for cell in self.planned_route_cells:
            if any(neighbor in exposed for neighbor in neighbors8(cell)):
                return True
        return False

    # -- lifecycle ----------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ):
        result = super().reset(seed=seed, options=options)
        # super().reset() already called _observation() while plan fields were
        # None; recompute the plan and build the observation again.
        self.planned_target, self.planned_target_index = self._select_planned_target()
        self.planned_route = self._transfer_path(self.planned_target)
        self.planned_route_cells = self.planned_route[1:]
        self.planned_macro_duration_minutes = self._planned_duration()
        return self._observation(), self._info(events=[])

    def action_masks(self) -> np.ndarray:
        if not self._state_ready:
            raise RuntimeError("Call reset() before action_masks()")
        mask = np.zeros(CLAMP_ACTION_COUNT, dtype=bool)
        mask[CLAMP_CONTINUE] = True
        mask[CLAMP_RELEASE] = self._release_is_legal()
        return mask

    def step(self, action: int, *, build_obs: bool = True):
        if not self._state_ready:
            raise RuntimeError("Call reset() before step()")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended; call reset()")
        action = int(action)
        step_events: list[dict[str, Any]] = []
        mask = self.action_masks()
        if not 0 <= action < CLAMP_ACTION_COUNT or not mask[action]:
            self.terminated = True
            self.failure_reason = f"invalid clamp action {action}"
            reward = -self.reward_config["invalid_action_penalty"]
            obs = self._observation() if build_obs else None
            return obs, reward, True, False, self._info(
                events=step_events,
                reward_terms={"invalid_action": self.reward_config["invalid_action_penalty"]},
            )

        self.step_count += 1
        cut_count_before = len(self.cut)
        elapsed_before = self.elapsed_minutes
        blood_before = self.expected_blood_loss_ml
        ischemia_before = self.total_clamped_minutes
        target = self.planned_target
        route = self.planned_route

        # (1) Phase decision
        if action == CLAMP_RELEASE:
            self.early_end_count += 1
            self._switch_phase(
                "unclamped", step_events, reason="policy_released_before_macro_target"
            )
            step_events[-1]["clamp_decision"] = "release_now"
        step_events.append({
            "index": len(self.events) + len(step_events),
            "action": "clamp_decision",
            "clamp_decision": "continue" if action == CLAMP_CONTINUE else "release_now",
            "time_minutes": self.elapsed_minutes,
        })

        # (2) Automatic transfer over route[1:]
        for cell in route[1:]:
            source = self.current
            duration = self.base_action_minutes
            self._advance_time(duration, step_events)
            self.current = cell
            self.previous_direction_position = source
            self.transfer_count += 1
            self.direction_action_count += 1
            step_events.append({
                "index": len(self.events) + len(step_events),
                "action": "transfer",
                "cell": list(cell),
                "duration_minutes": duration,
                "time_minutes": self.elapsed_minutes,
            })

        # (3) Cut / seal the planned target
        source = self.current
        duration = self._action_duration(target)
        self._advance_time(duration, step_events)
        if target in self._exposed_cells():
            component_id = self.component_by_cell[target]
            component = self._component(component_id)
            self.exposed_ids.remove(component_id)
            self.sealed_ids.add(component_id)
            self.cut.update(component["cells"])
            step_events.append({
                "index": len(self.events) + len(step_events),
                "action": "seal_and_cut_vessel",
                "component_id": component_id,
                "cells": _cell_list(component["cells"]),
                "cross_section_cells": int(component["cross_section_cells"]),
                "area_mm2": float(component["area_mm2"]),
                "is_large": bool(component["is_large"]),
                "duration_minutes": duration,
                "time_minutes": self.elapsed_minutes,
            })
        else:
            self.cut.add(target)
            step_events.append({
                "index": len(self.events) + len(step_events),
                "action": "cut",
                "cell": list(target),
                "duration_minutes": duration,
                "time_minutes": self.elapsed_minutes,
            })
        self.current = target
        self.previous_direction_position = source
        self.direction_action_count += 1
        self._release_ready_components(step_events)
        self.no_progress_streak = 0
        self.same_edge_streak = 0
        self._last_no_progress_edge = None

        self.events.extend(step_events)
        self._update_mechanics()

        # (4) Reward: pure incremental time/blood/ischemia, no shaping
        delta_time = self.elapsed_minutes - elapsed_before
        delta_blood = self.expected_blood_loss_ml - blood_before
        delta_ischemia = self.total_clamped_minutes - ischemia_before
        reward_terms = {
            "time_cost": (
                self.reward_config["time_cost"]
                * delta_time / self.clinical_config["time_scale_minutes"]
            ),
            "blood_cost": (
                self.reward_config["blood_cost"]
                * delta_blood / self.clinical_config["blood_scale_ml"]
            ),
            "ischemia_cost": (
                self.ischemia_cost
                * delta_ischemia / self.ischemia_scale_minutes
            ),
        }
        reward = -sum(reward_terms.values())

        # (5) Termination
        if self.cut == self.domain:
            self.terminated = True
            reward += self.reward_config["completion_bonus"]
            reward_terms["completion_bonus"] = -self.reward_config["completion_bonus"]
        elif self.elapsed_minutes >= self.clinical_config["max_episode_minutes"] - _EPSILON:
            self.truncated = True
            self.failure_reason = "maximum episode time reached"
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]
        elif self.step_count >= self.max_steps:
            self.truncated = True
            self.failure_reason = f"maximum macro step count ({self.max_steps}) reached"
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]

        # (6) Compute the next planned target for the returned observation
        if not (self.terminated or self.truncated):
            self.planned_target, self.planned_target_index = self._select_planned_target()
            self.planned_route = self._transfer_path(self.planned_target)
            self.planned_route_cells = self.planned_route[1:]
            self.planned_macro_duration_minutes = self._planned_duration()
        else:
            self.planned_target = None
            self.planned_target_index = None
            self.planned_route = []
            self.planned_route_cells = []
            self.planned_macro_duration_minutes = 0.0

        obs = self._observation() if build_obs else None
        return (
            obs,
            float(reward),
            self.terminated,
            self.truncated,
            self._info(events=step_events, reward_terms=reward_terms),
        )

    # -- observations -------------------------------------------------------

    def _base_observation(self) -> np.ndarray:
        """26-channel v10.1-macro-compatible observation for the frozen BC model."""
        return ClinicalHierarchicalResectionEnv._observation(self)

    def _observation(self) -> np.ndarray:
        obs = np.zeros(
            (len(TARGET_CONDITIONED_OBSERVATION_CHANNELS), self.max_rows, self.max_cols),
            dtype=np.float32,
        )
        base = ClinicalHierarchicalResectionEnv._observation(self)
        obs[: base.shape[0]] = base

        if self.planned_target is not None:
            tr, tc = self.planned_target
            obs[PLANNED_TARGET_CHANNEL, tr, tc] = 1.0
            for row, col in self.planned_route_cells:
                obs[PLANNED_ROUTE_CHANNEL, row, col] = 1.0
            obs[28].fill(float(self._target_is_vessel()))
            obs[29].fill(float(self._target_is_large_vessel()))
            obs[30].fill(float(self._target_vessel_area_fraction()))
            obs[31].fill(float(self._route_near_exposed_vessel()))
            obs[32].fill(min(
                1.0, self.planned_macro_duration_minutes / self.clinical_config["time_scale_minutes"]
            ))
        obs[33].fill(min(
            1.0, self.total_clamped_minutes / self.clinical_config["max_episode_minutes"]
        ))
        obs[34].fill(min(
            1.0, self._expected_bleeding_rate() / max(self.reference_flow_ml_per_min, _EPSILON)
        ))
        limit = (
            self.clinical_config["max_clamp_minutes"]
            if self.phase == "clamped"
            else self.clinical_config["unclamp_minutes"]
        )
        obs[35].fill(max(0.0, limit - self.phase_elapsed_minutes) / max(limit, _EPSILON))
        return obs

    def _info(self, **kwargs: Any) -> dict[str, Any]:
        info = super()._info(**kwargs)
        if self.terminated or self.truncated:
            info["action_mask"] = np.zeros(CLAMP_ACTION_COUNT, dtype=bool)
        info.update({
            "control_mode": "target_conditioned_clamp",
            "environment_version": TARGET_CONDITIONED_ENVIRONMENT_VERSION,
            "planned_target": list(self.planned_target) if self.planned_target else None,
            "planned_target_index": self.planned_target_index,
            "planned_macro_duration_minutes": self.planned_macro_duration_minutes,
            "total_clamped_minutes": self.total_clamped_minutes,
        })
        return info

    def episode_replay(self) -> dict[str, Any]:
        replay = super().episode_replay()
        replay["environment_version"] = TARGET_CONDITIONED_ENVIRONMENT_VERSION
        replay["control_mode"] = "target_conditioned_clamp"
        replay["planned_target"] = (
            list(self.planned_target) if self.planned_target else None
        )
        return replay


class TargetConditionedScenarioPoolEnv(TargetConditionedClampEnv):
    """Seeded scenario sampler for target-conditioned MaskablePPO."""

    def __init__(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        **kwargs: Any,
    ) -> None:
        if not scenarios:
            raise ValueError("TargetConditionedScenarioPoolEnv requires scenarios")
        import random

        self._pool = [dict(item) for item in scenarios]
        self._scenario_rng = random.Random(seed)
        super().__init__(scenario=self._pool[0], **kwargs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ):
        if seed is not None:
            self._scenario_rng.seed(seed)
        options = dict(options or {})
        if "scenario" not in options:
            options["scenario"] = self._scenario_rng.choice(self._pool)
        return super().reset(seed=seed, options=options)

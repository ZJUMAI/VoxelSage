"""v10 clinical environment with factorized clamp and macro-target actions.

The action is ``[clamp_decision, target_cell]`` where clamp decision 0 means
continue and 1 means release now.  Release and target execution belong to the
same semi-Markov step, avoiding the zero-duration END step used by v9.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from clinical_macro_environment import (
    CLINICAL_MACRO_GRID_ACTIONS,
    CLINICAL_MACRO_OBSERVATION_CHANNELS,
    ClinicalMacroResectionEnv,
)
from clinical_window_environment import _EPSILON, _cell_list

try:
    from gymnasium import spaces
except ImportError:
    spaces = None


CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION = "clinical-hierarchical-window-v1"
CLAMP_CONTINUE = 0
CLAMP_RELEASE = 1
CLAMP_ACTION_COUNT = 2
CLINICAL_HIERARCHICAL_MASK_SIZE = CLAMP_ACTION_COUNT + CLINICAL_MACRO_GRID_ACTIONS
CLINICAL_HIERARCHICAL_OBSERVATION_CHANNELS = CLINICAL_MACRO_OBSERVATION_CHANNELS


class ClinicalHierarchicalResectionEnv(ClinicalMacroResectionEnv):
    """Joint clamp-control and spatial-target environment.

    ``safe_release_mask`` encodes only the hard safety rule that an exposed,
    unsealed vessel must not be reperfused.  It deliberately does not encode a
    full timing heuristic, leaving safe-window anticipation to the policy.
    """

    def __init__(self, *, safe_release_mask: bool = True, **kwargs: Any) -> None:
        self.safe_release_mask = bool(safe_release_mask)
        super().__init__(**kwargs)
        if spaces is not None:
            self.action_space = spaces.MultiDiscrete(
                np.asarray([CLAMP_ACTION_COUNT, CLINICAL_MACRO_GRID_ACTIONS], dtype=np.int64)
            )

    def _release_is_legal(self) -> bool:
        if self.safe_release_mask and self.exposed_ids:
            return False
        mode = self.clinical_config["early_end_mode"]
        if mode == "disabled":
            return False
        if mode == "threshold":
            return (
                self.phase == "clamped"
                and self.phase_elapsed_minutes
                >= float(self.clinical_config["early_end_minutes"]) - _EPSILON
                and self.cut != self.domain
            )
        return (
            self.phase == "clamped"
            and self.phase_elapsed_minutes > _EPSILON
            and self.cut != self.domain
        )

    def action_masks(self) -> np.ndarray:
        if not self._state_ready:
            raise RuntimeError("Call reset() before action_masks()")
        mask = np.zeros(CLINICAL_HIERARCHICAL_MASK_SIZE, dtype=bool)
        mask[CLAMP_CONTINUE] = True
        mask[CLAMP_RELEASE] = self._release_is_legal()
        for row, col in self._frontier():
            mask[CLAMP_ACTION_COUNT + row * self.max_cols + col] = True
        return mask

    def step(self, action):
        if not self._state_ready:
            raise RuntimeError("Call reset() before step()")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended; call reset()")
        values = np.asarray(action, dtype=np.int64).reshape(-1)
        if values.shape != (2,):
            raise ValueError(f"Hierarchical action must contain [clamp, target], got {action!r}")
        clamp_action, target_action = map(int, values)
        mask = self.action_masks()
        valid = (
            0 <= clamp_action < CLAMP_ACTION_COUNT
            and 0 <= target_action < CLINICAL_MACRO_GRID_ACTIONS
            and mask[clamp_action]
            and mask[CLAMP_ACTION_COUNT + target_action]
        )
        if not valid:
            self.terminated = True
            self.failure_reason = f"invalid hierarchical action {[clamp_action, target_action]}"
            reward = -self.reward_config["invalid_action_penalty"]
            return self._observation(), reward, True, False, self._info(
                events=[],
                reward_terms={"invalid_action": self.reward_config["invalid_action_penalty"]},
            )

        target = divmod(target_action, self.max_cols)
        route = self._transfer_path(target)
        if not route:
            self.terminated = True
            self.failure_reason = "no cut-region transfer path to selected frontier"
            reward = -self.reward_config["failure_penalty"]
            return self._observation(), reward, True, False, self._info(
                events=[],
                reward_terms={"failure_penalty": self.reward_config["failure_penalty"]},
            )

        step_events: list[dict[str, Any]] = []
        self.step_count += 1
        cut_count_before = len(self.cut)
        elapsed_before = self.elapsed_minutes
        blood_before = self.expected_blood_loss_ml
        if clamp_action == CLAMP_RELEASE:
            self.early_end_count += 1
            self._switch_phase(
                "unclamped", step_events, reason="policy_released_before_macro_target"
            )
            step_events[-1]["clamp_decision"] = "release_now"

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
        newly_cut_cells = len(self.cut) - cut_count_before
        seal_count = sum(event.get("action") == "seal_and_cut_vessel" for event in step_events)
        duration_total = self.elapsed_minutes - elapsed_before
        self.max_macro_duration_minutes = max(self.max_macro_duration_minutes, duration_total)
        blood_total = self.expected_blood_loss_ml - blood_before
        reward_terms = self._reward_terms(
            duration_total, blood_total, newly_cut_cells, int(seal_count)
        )
        reward = -sum(reward_terms.values())
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
        return (
            self._observation(),
            float(reward),
            self.terminated,
            self.truncated,
            self._info(events=step_events, reward_terms=reward_terms),
        )

    def _info(self, **kwargs: Any) -> dict[str, Any]:
        info = super()._info(**kwargs)
        if self.terminated or self.truncated:
            info["action_mask"] = np.zeros(CLINICAL_HIERARCHICAL_MASK_SIZE, dtype=bool)
        info.update({
            "control_mode": "hierarchical_clamp_target",
            "safe_release_mask": self.safe_release_mask,
            "environment_version": CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION,
        })
        return info

    def episode_replay(self) -> dict[str, Any]:
        replay = super().episode_replay()
        replay["environment_version"] = CLINICAL_HIERARCHICAL_ENVIRONMENT_VERSION
        replay["control_mode"] = "hierarchical_clamp_target"
        replay["safe_release_mask"] = self.safe_release_mask
        return replay


class ClinicalHierarchicalScenarioPoolEnv(ClinicalHierarchicalResectionEnv):
    """Seeded scenario sampler for hierarchical MaskablePPO."""

    def __init__(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        **kwargs: Any,
    ) -> None:
        if not scenarios:
            raise ValueError("ClinicalHierarchicalScenarioPoolEnv requires scenarios")
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

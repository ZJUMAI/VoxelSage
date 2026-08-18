"""High-level clinical-window environment with deterministic transfer.

The policy selects one legal frontier cell (30x40 padded grid) or END.  The
environment follows a shortest path through already-cut cells, charging every
physical transfer step its real duration and blood loss, then cuts the target.
This keeps the clinical target/timing decision in RL without asking it to
relearn low-level grid navigation.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from clinical_window_environment import (
    CLINICAL_OBSERVATION_CHANNELS,
    ClinicalWindowResectionEnv,
    _EPSILON,
    _cell_list,
)
from planner import Cell, neighbors4

try:
    from gymnasium import spaces
except ImportError:
    spaces = None


CLINICAL_MACRO_ENVIRONMENT_VERSION = "clinical-macro-window-v1"
CLINICAL_MACRO_MAX_ROWS = 30
CLINICAL_MACRO_MAX_COLS = 40
CLINICAL_MACRO_GRID_ACTIONS = CLINICAL_MACRO_MAX_ROWS * CLINICAL_MACRO_MAX_COLS
ACTION_END_CLAMP_MACRO = CLINICAL_MACRO_GRID_ACTIONS
CLINICAL_MACRO_ACTION_COUNT = CLINICAL_MACRO_GRID_ACTIONS + 1
CLINICAL_MACRO_OBSERVATION_CHANNELS = CLINICAL_OBSERVATION_CHANNELS + (
    "transfer_distance",
)


class ClinicalMacroResectionEnv(ClinicalWindowResectionEnv):
    """Select a cut target or END; execute shortest transfer deterministically."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("max_rows", CLINICAL_MACRO_MAX_ROWS)
        kwargs.setdefault("max_cols", CLINICAL_MACRO_MAX_COLS)
        super().__init__(**kwargs)
        if self.max_rows != CLINICAL_MACRO_MAX_ROWS or self.max_cols != CLINICAL_MACRO_MAX_COLS:
            raise ValueError("Clinical macro environment requires a 30x40 padded grid")
        if spaces is not None:
            self.action_space = spaces.Discrete(CLINICAL_MACRO_ACTION_COUNT)
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(
                    len(CLINICAL_MACRO_OBSERVATION_CHANNELS),
                    self.max_rows,
                    self.max_cols,
                ),
                dtype=np.float32,
            )

    def reset(self, **kwargs: Any):
        self.max_macro_duration_minutes = 0.0
        return super().reset(**kwargs)

    def _end_is_legal(self) -> bool:
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
        mask = np.zeros(CLINICAL_MACRO_ACTION_COUNT, dtype=bool)
        for row, col in self._frontier():
            mask[row * self.max_cols + col] = True
        mask[ACTION_END_CLAMP_MACRO] = self._end_is_legal()
        return mask

    def _transfer_path(self, target: Cell) -> list[Cell]:
        goals = set(neighbors4(target)) & self.cut
        queue: deque[Cell] = deque([self.current])
        parent: dict[Cell, Optional[Cell]] = {self.current: None}
        goal: Optional[Cell] = None
        while queue:
            cell = queue.popleft()
            if cell in goals:
                goal = cell
                break
            for neighbor in neighbors4(cell):
                if neighbor in self.cut and neighbor not in parent:
                    parent[neighbor] = cell
                    queue.append(neighbor)
        if goal is None:
            return []
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])  # type: ignore[arg-type]
        return list(reversed(path))

    def _transfer_counts(self) -> dict[Cell, int]:
        """Compute all frontier transfer distances from one cut-region BFS."""
        queue: deque[Cell] = deque([self.current])
        distance = {self.current: 0}
        while queue:
            cell = queue.popleft()
            for neighbor in neighbors4(cell):
                if neighbor in self.cut and neighbor not in distance:
                    distance[neighbor] = distance[cell] + 1
                    queue.append(neighbor)
        result: dict[Cell, int] = {}
        for target in self._frontier():
            candidates = [distance[cell] for cell in neighbors4(target) if cell in distance]
            if candidates:
                result[target] = min(candidates)
        return result

    def _observation(self) -> np.ndarray:
        base = super()._observation()
        transfer = np.zeros((self.max_rows, self.max_cols), dtype=np.float32)
        if self._state_ready:
            normalizer = float(self.max_rows + self.max_cols)
            for (row, col), count in self._transfer_counts().items():
                transfer[row, col] = count / normalizer
        return np.concatenate((base, transfer[None, ...]), axis=0)

    def step(self, action: int):
        if not self._state_ready:
            raise RuntimeError("Call reset() before step()")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended; call reset()")
        action = int(action)
        step_events: list[dict[str, Any]] = []
        mask = self.action_masks()
        if not 0 <= action < len(mask) or not mask[action]:
            self.terminated = True
            self.failure_reason = f"invalid macro action {action}"
            reward = -self.reward_config["invalid_action_penalty"]
            return self._observation(), reward, True, False, self._info(
                events=step_events,
                reward_terms={"invalid_action": self.reward_config["invalid_action_penalty"]},
            )

        self.step_count += 1
        cut_count_before = len(self.cut)
        elapsed_before = self.elapsed_minutes
        blood_before = self.expected_blood_loss_ml
        if action == ACTION_END_CLAMP_MACRO:
            self.early_end_count += 1
            self._switch_phase("unclamped", step_events, reason="policy_ended_clamp_early")
            step_events[-1]["action"] = "end_clamp_early"
        else:
            target = divmod(action, self.max_cols)
            route = self._transfer_path(target)
            if not route:
                self.terminated = True
                self.failure_reason = "no cut-region transfer path to selected frontier"
                reward = -self.reward_config["failure_penalty"]
                return self._observation(), reward, True, False, self._info(
                    events=step_events,
                    reward_terms={"failure_penalty": self.reward_config["failure_penalty"]},
                )

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
                    "cell": list(target),
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
        seal_count = sum(
            event.get("action") == "seal_and_cut_vessel" for event in step_events
        )
        duration_total = self.elapsed_minutes - elapsed_before
        self.max_macro_duration_minutes = max(
            self.max_macro_duration_minutes, duration_total,
        )
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
            info["action_mask"] = np.zeros(CLINICAL_MACRO_ACTION_COUNT, dtype=bool)
        info.update({
            "control_mode": "macro_target",
            "macro_action_count": self.step_count,
            "max_macro_duration_minutes": self.max_macro_duration_minutes,
            "environment_version": CLINICAL_MACRO_ENVIRONMENT_VERSION,
        })
        return info

    def episode_replay(self) -> dict[str, Any]:
        replay = super().episode_replay()
        replay["environment_version"] = CLINICAL_MACRO_ENVIRONMENT_VERSION
        replay["control_mode"] = "macro_target"
        replay["summary"]["macro_action_count"] = self.step_count
        replay["summary"]["max_macro_duration_minutes"] = self.max_macro_duration_minutes
        return replay


class ClinicalMacroScenarioPoolEnv(ClinicalMacroResectionEnv):
    """Seeded macro-environment sampler for vectorized MaskablePPO."""

    def __init__(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        **kwargs: Any,
    ) -> None:
        if not scenarios:
            raise ValueError("ClinicalMacroScenarioPoolEnv requires scenarios")
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

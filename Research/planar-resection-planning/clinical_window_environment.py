"""Clinical-window planar resection environment.

This module is intentionally separate from ``environment.py``.  The frozen v3
environment selects any legal frontier cell and inserts transfer events
automatically; this environment exposes five policy actions instead:

``up, down, left, right, end_clamp_early``.

Vessel components remain blocked until their surrounding ring has been cut.
They then become exposed.  Entering one exposed component seals and cuts the
whole cross-section in one action.  A one-cell vessel takes one base time unit;
components containing two or more cells take three units.

The bleeding quantity is a deterministic *expected simulated blood loss* used
for research.  It is not a clinically validated blood-loss predictor.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from mechanics import DEFAULT_MECHANICS, solve_tension
from planner import Cell, boundary_cells, is_connected, neighbors4, vessel_components

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Keep deterministic tests runnable without the RL stack.
    gym = None
    spaces = None


GymEnvBase = gym.Env if gym is not None else object

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_END_CLAMP_EARLY = 4
ACTION_NAMES = ("up", "down", "left", "right", "end_clamp_early")
CLINICAL_ENVIRONMENT_VERSION = "clinical-window-v3"
ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

DEFAULT_MAX_ROWS = 30
DEFAULT_MAX_COLS = 40
CLINICAL_OBSERVATION_CHANNELS = (
    "domain",
    "cut",
    "hidden_vessel",
    "exposed_vessel",
    "sealed_vessel",
    "large_vessel",
    "frontier",
    "current_position",
    "start",
    "thickness",
    "front_tension",
    "organ_energy",
    "vessel_strain",
    "expected_bleeding_rate",
    "valid_cell_mask",
    "row_coordinate",
    "column_coordinate",
    "clamped_phase",
    "clamp_elapsed_fraction",
    "unclamp_remaining_fraction",
    "elapsed_time_fraction",
    "no_progress_streak_fraction",
    "previous_position",
    "same_edge_streak_fraction",
    "clinical_cost_fraction",
)

DEFAULT_CLINICAL_CONFIG = {
    "cell_side_mm": 4.0,
    "transection_speed_cm2_per_min": 2.3,
    "max_clamp_minutes": 15.0,
    "unclamp_minutes": 5.0,
    "weight_kg": 70.0,
    "bleeding_probability": 1.0,
    # Reference exposed area = 5 cells * 16 mm^2 (4 mm cell side).  Sets
    # the unit-area bleeding rate beta = Q_ref / A_ref; see design doc
    # 临床时间窗口与模拟出血奖励模型设计.md §6.3.
    "reference_area_mm2": 80.0,
    "large_vessel_min_cells": 2.0,
    "large_vessel_time_multiplier": 3.0,
    "time_scale_minutes": 60.0,
    "blood_scale_ml": 100.0,
    "max_episode_minutes": 240.0,
    "max_steps_multiplier": 8.0,
    # Early-end control (auditable two-phase training):
    #   disabled  -> END mask always False, environment still auto 15/5
    #   threshold -> END legal only once clamped_elapsed >= early_end_minutes
    #   full      -> current clinical semantics (legal after clamp starts)
    "early_end_mode": "full",
    "early_end_minutes": 0.0,
    "stagnation_soft_start_steps": 40.0,
    "stagnation_penalty_ramp_steps": 24.0,
    "stagnation_limit_steps": 96.0,
    "two_cell_loop_soft_start_traversals": 6.0,
    "two_cell_loop_limit_traversals": 12.0,
}

DEFAULT_CLINICAL_REWARD = {
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

_EPSILON = 1e-9


def _cell(value: Sequence[int]) -> Cell:
    if len(value) != 2:
        raise ValueError(f"Cell must be [row, column], got {value!r}")
    return int(value[0]), int(value[1])


def _cell_list(cells: Iterable[Cell]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(cells)]


def _validated_config(values: Optional[Mapping[str, float]]) -> dict[str, float]:
    result = dict(DEFAULT_CLINICAL_CONFIG)
    if values:
        unknown = set(values) - set(result)
        if unknown:
            raise ValueError(f"Unknown clinical settings: {sorted(unknown)}")
        for name, value in values.items():
            if name == "early_end_mode":
                mode = str(value)
                if mode not in ("disabled", "threshold", "full"):
                    raise ValueError(
                        f"early_end_mode must be disabled|threshold|full, got {mode!r}"
                    )
                result[name] = mode
            else:
                result[name] = float(value)
    mode = result["early_end_mode"]
    threshold = result["early_end_minutes"]
    if not math.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError("early_end_minutes must be finite and non-negative")
    if mode == "threshold":
        if not (0.0 < float(threshold) < float(result["max_clamp_minutes"])):
            raise ValueError(
                f"threshold early-end requires 0 < early_end_minutes < max_clamp_minutes, "
                f"got {threshold!r}"
            )
    else:
        result["early_end_minutes"] = 0.0
    # early_end_minutes may legitimately be 0 in disabled/full modes; the
    # threshold branch above enforces 0 < threshold < max_clamp when needed.
    strictly_positive = set(result) - {"bleeding_probability", "early_end_mode", "early_end_minutes"}
    if any(not math.isfinite(result[name]) or result[name] <= 0 for name in strictly_positive):
        raise ValueError("Clinical settings other than bleeding_probability must be finite and positive")
    probability = result["bleeding_probability"]
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("bleeding_probability must be between 0 and 1")
    if result["large_vessel_min_cells"] != int(result["large_vessel_min_cells"]):
        raise ValueError("large_vessel_min_cells must be an integer")
    for name in (
        "stagnation_soft_start_steps",
        "stagnation_penalty_ramp_steps",
        "stagnation_limit_steps",
        "two_cell_loop_soft_start_traversals",
        "two_cell_loop_limit_traversals",
    ):
        if result[name] != int(result[name]):
            raise ValueError(f"{name} must be an integer")
    if result["stagnation_limit_steps"] <= result["stagnation_soft_start_steps"]:
        raise ValueError("stagnation_limit_steps must exceed stagnation_soft_start_steps")
    if (
        result["two_cell_loop_limit_traversals"]
        <= result["two_cell_loop_soft_start_traversals"]
    ):
        raise ValueError(
            "two_cell_loop_limit_traversals must exceed two_cell_loop_soft_start_traversals"
        )
    return result


def _validated_reward(values: Optional[Mapping[str, float]]) -> dict[str, float]:
    result = dict(DEFAULT_CLINICAL_REWARD)
    if values:
        unknown = set(values) - set(result)
        if unknown:
            raise ValueError(f"Unknown clinical reward settings: {sorted(unknown)}")
        result.update({name: float(value) for name, value in values.items()})
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("Clinical reward settings must be finite and non-negative")
    if result["clinical_cost_cap"] <= 0:
        raise ValueError("clinical_cost_cap must be positive")
    return result


class ClinicalWindowResectionEnv(GymEnvBase):
    """Five-action direction environment with intermittent inflow occlusion."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario: Optional[Mapping[str, Any]] = None,
        clinical_config: Optional[Mapping[str, float]] = None,
        reward_config: Optional[Mapping[str, float]] = None,
        mechanics_parameters: Optional[Mapping[str, float]] = None,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_cols: int = DEFAULT_MAX_COLS,
        max_steps: Optional[int] = None,
        mechanics_update_interval: int = 0,
    ) -> None:
        self._fixed_scenario = dict(scenario) if scenario is not None else None
        self.clinical_config = _validated_config(clinical_config)
        self.reward_config = _validated_reward(reward_config)
        self.mechanics_parameters = dict(mechanics_parameters or {})
        self.max_rows = int(max_rows)
        self.max_cols = int(max_cols)
        self.max_steps_override = int(max_steps) if max_steps is not None else None
        self.mechanics_update_interval = int(mechanics_update_interval)
        if self.mechanics_update_interval < 0:
            raise ValueError("mechanics_update_interval must be non-negative")
        if self.max_rows <= 0 or self.max_cols <= 0:
            raise ValueError("max_rows and max_cols must be positive")
        self._rng = random.Random()
        self._episode_id = 0
        self._state_ready = False
        if spaces is not None:
            self.action_space = spaces.Discrete(len(ACTION_NAMES))
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(len(CLINICAL_OBSERVATION_CHANNELS), self.max_rows, self.max_cols),
                dtype=np.float32,
            )
        else:
            self.action_space = None
            self.observation_space = None

    @property
    def cell_area_mm2(self) -> float:
        return self.clinical_config["cell_side_mm"] ** 2

    @property
    def base_action_minutes(self) -> float:
        return (self.cell_area_mm2 / 100.0) / self.clinical_config["transection_speed_cm2_per_min"]

    @property
    def reference_flow_ml_per_min(self) -> float:
        # Reference total hepatic blood flow per kg, summed as HA 3.5 + PV 13.5
        # (mL/min/kg) from Carlisle KM, et al., Gut 1992.  Engineering scale
        # that defines a repeatable simulator regime, not a patient-specific
        # predictor.  See design doc 临床时间窗口与模拟出血奖励模型设计.md §6.2.
        return 17.0 * self.clinical_config["weight_kg"]

    @property
    def bleeding_beta(self) -> float:
        return self.reference_flow_ml_per_min / self.clinical_config["reference_area_mm2"]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ):
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        options = dict(options or {})
        scenario = dict(options.get("scenario") or self._fixed_scenario or {})
        if not scenario:
            raise ValueError("reset requires a scenario at construction or in options['scenario']")
        self._load_scenario(scenario)
        self._episode_id += 1
        self.cut: set[Cell] = {self.start}
        self.current = self.start
        self.previous_direction_position = self.start
        self.hidden_ids = {int(component["id"]) for component in self.components}
        self.exposed_ids: set[int] = set()
        self.sealed_ids: set[int] = set()
        self.phase = "clamped"
        self.phase_elapsed_minutes = 0.0
        self.elapsed_minutes = 0.0
        self.total_clamped_minutes = 0.0
        self.total_unclamped_minutes = 0.0
        self.unclamped_exposed_minutes = 0.0
        self.expected_blood_loss_ml = 0.0
        self.peak_expected_bleeding_rate_ml_per_min = 0.0
        self.clamp_cycle_count = 1
        self.early_end_count = 0
        self.transfer_count = 0
        self.direction_action_count = 0
        self.no_progress_streak = 0
        self.max_no_progress_streak = 0
        self.same_edge_streak = 0
        self.max_same_edge_streak = 0
        self._last_no_progress_edge: Optional[tuple[Cell, Cell]] = None
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.failure_reason: Optional[str] = None
        self.events: list[dict[str, Any]] = [
            {"index": 0, "action": "cut", "cell": list(self.start), "time_minutes": 0.0, "reason": "start"}
        ]
        self._release_ready_components(self.events)
        self._update_mechanics(force=True)
        self._state_ready = True
        return self._observation(), self._info(events=self.events[:])

    def _load_scenario(self, scenario: Mapping[str, Any]) -> None:
        self.rows = int(scenario["rows"])
        self.cols = int(scenario["cols"])
        if not 1 <= self.rows <= self.max_rows or not 1 <= self.cols <= self.max_cols:
            raise ValueError(
                f"Scenario shape {self.rows}x{self.cols} exceeds maximum {self.max_rows}x{self.max_cols}"
            )
        self.domain = {_cell(value) for value in scenario["domain_cells"]}
        self.vessel_cells = {_cell(value) for value in scenario.get("obstacle_cells", ())}
        if not self.domain or not is_connected(self.domain):
            raise ValueError("domain_cells must be a non-empty four-connected region")
        if not self.vessel_cells <= self.domain:
            raise ValueError("All vessel cells must be inside domain_cells")
        # Boundary components use the same in-domain tissue ring, exposure,
        # sealing, and cutting rules as interior components.
        self.start = _cell(scenario["start_cell"])
        if self.start not in boundary_cells(self.domain) or self.start in self.vessel_cells:
            raise ValueError("start_cell must be a non-vessel domain boundary cell")
        raw_components = vessel_components(self.vessel_cells, self.domain)
        threshold = int(self.clinical_config["large_vessel_min_cells"])
        self.components: list[dict[str, Any]] = []
        self.component_by_cell: dict[Cell, int] = {}
        for raw in raw_components:
            component_id = int(raw["id"])
            cells = set(raw["cells"])
            item = {
                "id": component_id,
                "cells": cells,
                "ring": set(raw["ring"]),
                "cross_section_cells": len(cells),
                "area_mm2": len(cells) * self.cell_area_mm2,
                "is_large": len(cells) >= threshold,
            }
            self.components.append(item)
            for cell in cells:
                self.component_by_cell[cell] = component_id
        multiplier = int(self.clinical_config["max_steps_multiplier"])
        self.max_steps = int(self.max_steps_override or max(len(self.domain) * multiplier, len(self.domain) + 1))
        self.scenario = {
            "scenario_id": scenario.get("scenario_id"),
            "rows": self.rows,
            "cols": self.cols,
            "domain_cells": _cell_list(self.domain),
            "obstacle_cells": _cell_list(self.vessel_cells),
            "start_cell": list(self.start),
            "cell_size_mm": float(scenario.get("cell_size_mm", self.clinical_config["cell_side_mm"])),
        }

    def _component(self, component_id: int) -> dict[str, Any]:
        return self.components[int(component_id)]

    def _hidden_cells(self) -> set[Cell]:
        return {
            cell
            for component_id in self.hidden_ids
            for cell in self._component(component_id)["cells"]
        }

    def _exposed_cells(self) -> set[Cell]:
        return {
            cell
            for component_id in self.exposed_ids
            for cell in self._component(component_id)["cells"]
        }

    def _sealed_cells(self) -> set[Cell]:
        return {
            cell
            for component_id in self.sealed_ids
            for cell in self._component(component_id)["cells"]
        }

    def _unsealed_cells(self) -> set[Cell]:
        return self._hidden_cells() | self._exposed_cells()

    def _release_ready_components(self, events: list[dict[str, Any]]) -> None:
        released = False
        for component_id in sorted(tuple(self.hidden_ids)):
            component = self._component(component_id)
            ring = set(component["ring"])
            if ring and ring <= self.cut:
                self.hidden_ids.remove(component_id)
                self.exposed_ids.add(component_id)
                released = True
                events.append({
                    "index": len(self.events) + len(events),
                    "action": "expose_vessel",
                    "component_id": component_id,
                    "cells": _cell_list(component["cells"]),
                    "area_mm2": float(component["area_mm2"]),
                    "is_large": bool(component["is_large"]),
                    "time_minutes": self.elapsed_minutes,
                })

        # A boundary vessel can separate the remaining domain into two tissue
        # regions. If the current side is exhausted, the full in-domain ring is
        # unreachable until that vessel is sealed. Release only the first such
        # component touching the cut frontier; all ordinary ring releases above
        # and every interior-vessel scenario retain their original behavior.
        if released or self._frontier() or self.cut == self.domain:
            return
        domain_boundary = boundary_cells(self.domain)
        for component_id in sorted(self.hidden_ids):
            component = self._component(component_id)
            cells = set(component["cells"])
            if not cells & domain_boundary:
                continue
            if not any(
                neighbor in self.cut
                for cell in cells
                for neighbor in neighbors4(cell)
            ):
                continue
            ring = set(component["ring"])
            self.hidden_ids.remove(component_id)
            self.exposed_ids.add(component_id)
            events.append({
                "index": len(self.events) + len(events),
                "action": "expose_vessel",
                "component_id": component_id,
                "cells": _cell_list(cells),
                "area_mm2": float(component["area_mm2"]),
                "is_large": bool(component["is_large"]),
                "time_minutes": self.elapsed_minutes,
                "release_rule": "boundary_frontier_deadlock",
                "required_ring_cell_count": len(ring & self.cut),
                "full_ring_cell_count": len(ring),
            })
            break

    def _frontier(self) -> set[Cell]:
        blocked = self._hidden_cells()
        return {
            cell
            for cell in self.domain - self.cut - blocked
            if any(neighbor in self.cut for neighbor in neighbors4(cell))
        }

    def action_masks(self) -> np.ndarray:
        if not self._state_ready:
            raise RuntimeError("Call reset() before action_masks()")
        mask = np.zeros(len(ACTION_NAMES), dtype=bool)
        hidden = self._hidden_cells()
        for action, (delta_row, delta_col) in ACTION_DELTAS.items():
            target = self.current[0] + delta_row, self.current[1] + delta_col
            mask[action] = target in self.domain and target not in hidden
        end_mode = self.clinical_config["early_end_mode"]
        if end_mode == "disabled":
            end_legal = False
        elif end_mode == "threshold":
            end_legal = (
                self.phase == "clamped"
                and self.phase_elapsed_minutes
                >= float(self.clinical_config["early_end_minutes"]) - _EPSILON
                and self.cut != self.domain
            )
        else:  # full
            end_legal = (
                self.phase == "clamped"
                and self.phase_elapsed_minutes > _EPSILON
                and self.cut != self.domain
            )
        mask[ACTION_END_CLAMP_EARLY] = end_legal
        return mask

    def _expected_bleeding_rate(self) -> float:
        if self.phase != "unclamped" or not self.exposed_ids:
            return 0.0
        exposed_area = sum(float(self._component(component_id)["area_mm2"]) for component_id in self.exposed_ids)
        uncapped = (
            self.clinical_config["bleeding_probability"]
            * self.bleeding_beta
            * exposed_area
        )
        return min(self.reference_flow_ml_per_min, uncapped)

    def _switch_phase(self, phase: str, events: list[dict[str, Any]], *, reason: str) -> None:
        self.phase = phase
        self.phase_elapsed_minutes = 0.0
        if phase == "clamped":
            self.clamp_cycle_count += 1
        events.append({
            "index": len(self.events) + len(events),
            "action": "phase_change",
            "phase": phase,
            "reason": reason,
            "time_minutes": self.elapsed_minutes,
        })

    def _advance_time(self, duration_minutes: float, events: list[dict[str, Any]]) -> tuple[float, float]:
        """Advance through exact phase boundaries and integrate expected loss."""
        remaining = float(duration_minutes)
        blood_loss = 0.0
        exposed_minutes = 0.0
        while remaining > _EPSILON:
            if self.phase == "clamped":
                limit = self.clinical_config["max_clamp_minutes"]
            else:
                limit = self.clinical_config["unclamp_minutes"]
            until_boundary = max(0.0, limit - self.phase_elapsed_minutes)
            if until_boundary <= _EPSILON:
                self._switch_phase(
                    "unclamped" if self.phase == "clamped" else "clamped",
                    events,
                    reason="maximum_clamp_reached" if self.phase == "clamped" else "reperfusion_complete",
                )
                continue
            segment = min(remaining, until_boundary)
            rate = self._expected_bleeding_rate()
            if self.phase == "clamped":
                self.total_clamped_minutes += segment
            else:
                self.total_unclamped_minutes += segment
                if self.exposed_ids:
                    exposed_minutes += segment
                loss = rate * segment
                blood_loss += loss
                self.peak_expected_bleeding_rate_ml_per_min = max(
                    self.peak_expected_bleeding_rate_ml_per_min, rate,
                )
            self.elapsed_minutes += segment
            self.phase_elapsed_minutes += segment
            remaining -= segment
            if self.phase_elapsed_minutes >= limit - _EPSILON:
                old_phase = self.phase
                self._switch_phase(
                    "unclamped" if old_phase == "clamped" else "clamped",
                    events,
                    reason="maximum_clamp_reached" if old_phase == "clamped" else "reperfusion_complete",
                )
        self.expected_blood_loss_ml += blood_loss
        self.unclamped_exposed_minutes += exposed_minutes
        return blood_loss, exposed_minutes

    def _action_duration(self, target: Cell) -> float:
        if target in self._exposed_cells():
            component = self._component(self.component_by_cell[target])
            if component["is_large"]:
                return self.base_action_minutes * self.clinical_config["large_vessel_time_multiplier"]
        return self.base_action_minutes

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
            self.failure_reason = f"invalid action {action}"
            reward = -self.reward_config["invalid_action_penalty"]
            return self._observation(), reward, True, False, self._info(
                events=step_events,
                reward_terms={"invalid_action": self.reward_config["invalid_action_penalty"]},
            )

        self.step_count += 1
        cut_count_before = len(self.cut)
        if action == ACTION_END_CLAMP_EARLY:
            self.early_end_count += 1
            self._switch_phase("unclamped", step_events, reason="policy_ended_clamp_early")
            step_events[-1]["action"] = "end_clamp_early"
            duration = 0.0
            blood_loss = 0.0
        else:
            source = self.current
            delta_row, delta_col = ACTION_DELTAS[action]
            target = self.current[0] + delta_row, self.current[1] + delta_col
            duration = self._action_duration(target)
            blood_loss, _ = self._advance_time(duration, step_events)
            if target in self.cut:
                self.transfer_count += 1
                step_events.append({
                    "index": len(self.events) + len(step_events),
                    "action": "transfer",
                    "cell": list(target),
                    "duration_minutes": duration,
                    "time_minutes": self.elapsed_minutes,
                })
            elif target in self._exposed_cells():
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

        self.events.extend(step_events)
        self._update_mechanics()
        newly_cut_cells = len(self.cut) - cut_count_before
        if action != ACTION_END_CLAMP_EARLY:
            if newly_cut_cells > 0:
                self.no_progress_streak = 0
                self.same_edge_streak = 0
                self._last_no_progress_edge = None
            else:
                self.no_progress_streak += 1
                self.max_no_progress_streak = max(
                    self.max_no_progress_streak, self.no_progress_streak,
                )
                edge = tuple(sorted((source, self.current)))
                if edge == self._last_no_progress_edge:
                    self.same_edge_streak += 1
                else:
                    self.same_edge_streak = 1
                    self._last_no_progress_edge = edge
                self.max_same_edge_streak = max(
                    self.max_same_edge_streak, self.same_edge_streak,
                )
        seal_count = sum(
            1 for event in step_events if event.get("action") == "seal_and_cut_vessel"
        )
        reward_terms = self._reward_terms(duration, blood_loss, newly_cut_cells, seal_count)
        reward = -sum(reward_terms.values())
        if self.cut == self.domain:
            self.terminated = True
            reward += self.reward_config["completion_bonus"]
            reward_terms["completion_bonus"] = -self.reward_config["completion_bonus"]
        elif self.same_edge_streak >= int(
            self.clinical_config["two_cell_loop_limit_traversals"]
        ):
            self.truncated = True
            self.failure_reason = (
                "two-cell oscillation: "
                f"{int(self.clinical_config['two_cell_loop_limit_traversals'])} "
                "consecutive no-progress traversals of the same edge"
            )
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]
        elif self.no_progress_streak >= int(self.clinical_config["stagnation_limit_steps"]):
            self.truncated = True
            self.failure_reason = (
                f"stagnation: {int(self.clinical_config['stagnation_limit_steps'])} "
                "consecutive direction actions without new cut"
            )
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]
        elif self.elapsed_minutes >= self.clinical_config["max_episode_minutes"] - _EPSILON:
            self.truncated = True
            self.failure_reason = "maximum episode time reached"
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]
        elif self.step_count >= self.max_steps:
            self.truncated = True
            self.failure_reason = f"maximum step count ({self.max_steps}) reached"
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]
        return (
            self._observation(),
            float(reward),
            self.terminated,
            self.truncated,
            self._info(events=step_events, reward_terms=reward_terms),
        )

    def _zero_mechanics(self) -> None:
        self.mechanics = {
            "cells": [
                {
                    "cell": list(cell),
                    "thickness": 0.0,
                    "front_tension": 0.0,
                    "organ_energy": 0.0,
                    "vessel_strain": 0.0,
                }
                for cell in sorted(self.domain)
            ],
            "peak_front_tension": 0.0,
            "peak_organ_energy": 0.0,
            "peak_vessel_strain": 0.0,
        }
        self.mechanics_by_cell = {tuple(item["cell"]): item for item in self.mechanics["cells"]}

    def _update_mechanics(self, *, force: bool = False) -> None:
        if self.mechanics_update_interval == 0:
            self._zero_mechanics()
            return
        if not force and self.direction_action_count % self.mechanics_update_interval != 0:
            return
        self.mechanics = solve_tension(
            rows=self.rows,
            cols=self.cols,
            domain_cells=_cell_list(self.domain),
            vessel_cells=_cell_list(self._unsealed_cells()),
            cut_cells=_cell_list(self.cut),
            parameters=self.mechanics_parameters,
        )
        self.mechanics_by_cell = {tuple(item["cell"]): item for item in self.mechanics["cells"]}

    def _reward_terms(
        self,
        duration_minutes: float,
        blood_loss_ml: float,
        newly_cut_cells: int,
        seal_count: int = 0,
    ) -> dict[str, float]:
        progress_denominator = max(1, len(self.domain) - 1)
        raw_time_delta = (
            self.reward_config["time_cost"]
            * duration_minutes / self.clinical_config["time_scale_minutes"]
        )
        raw_blood_delta = (
            self.reward_config["blood_cost"]
            * blood_loss_ml / self.clinical_config["blood_scale_ml"]
        )
        raw_after = (
            self.reward_config["time_cost"]
            * self.elapsed_minutes / self.clinical_config["time_scale_minutes"]
            + self.reward_config["blood_cost"]
            * self.expected_blood_loss_ml / self.clinical_config["blood_scale_ml"]
        )
        raw_delta = raw_time_delta + raw_blood_delta
        raw_before = max(0.0, raw_after - raw_delta)
        cost_cap = self.reward_config["clinical_cost_cap"]
        bounded_delta = min(cost_cap, raw_after) - min(cost_cap, raw_before)
        clinical_factor = bounded_delta / raw_delta if raw_delta > _EPSILON else 1.0
        terms = {
            "time_cost": raw_time_delta * clinical_factor,
            "blood_cost": raw_blood_delta * clinical_factor,
            # Reward terms use a cost-sign convention: negative entries become
            # positive reward when step() applies ``-sum(reward_terms)``.
            "progress_bonus": -self.reward_config["progress_bonus"]
            * newly_cut_cells / progress_denominator,
            "front_tension_cost": self.reward_config["front_tension_cost"]
            * float(self.mechanics["peak_front_tension"]),
            "organ_energy_cost": self.reward_config["organ_energy_cost"]
            * float(self.mechanics["peak_organ_energy"]),
            "vessel_strain_cost": self.reward_config["vessel_strain_cost"]
            * float(self.mechanics["peak_vessel_strain"]),
        }
        soft_start = int(self.clinical_config["stagnation_soft_start_steps"])
        if self.no_progress_streak > soft_start:
            ramp = int(self.clinical_config["stagnation_penalty_ramp_steps"])
            terms["stagnation_cost"] = self.reward_config["stagnation_penalty_cap"] * min(
                1.0, (self.no_progress_streak - soft_start) / ramp,
            )
        loop_soft_start = int(
            self.clinical_config["two_cell_loop_soft_start_traversals"]
        )
        if self.same_edge_streak >= loop_soft_start:
            terms["two_cell_loop_cost"] = self.reward_config["two_cell_loop_penalty"]
        # Normalize over the scenario's component count so sealing all vessels
        # always contributes the same bounded shaping reward.
        if seal_count > 0:
            terms["seal_progress_bonus"] = (
                -self.reward_config["seal_progress_bonus"]
                * seal_count
                / max(1, len(self.components))
            )
        return terms

    def _observation(self) -> np.ndarray:
        observation = np.zeros(
            (len(CLINICAL_OBSERVATION_CHANNELS), self.max_rows, self.max_cols),
            dtype=np.float32,
        )
        channels = {name: index for index, name in enumerate(CLINICAL_OBSERVATION_CHANNELS)}
        hidden = self._hidden_cells() if self._state_ready else set()
        exposed = self._exposed_cells() if self._state_ready else set()
        sealed = self._sealed_cells() if self._state_ready else set()
        frontier = self._frontier() if self._state_ready else set()
        rate_normalizer = max(self.reference_flow_ml_per_min, _EPSILON)
        for cell in self.domain:
            row, col = cell
            observation[channels["domain"], row, col] = 1.0
            observation[channels["valid_cell_mask"], row, col] = 1.0
            if cell in self.cut:
                observation[channels["cut"], row, col] = 1.0
            if cell in hidden:
                observation[channels["hidden_vessel"], row, col] = 1.0
            if cell in exposed:
                observation[channels["exposed_vessel"], row, col] = 1.0
            if cell in sealed:
                observation[channels["sealed_vessel"], row, col] = 1.0
            if cell in frontier:
                observation[channels["frontier"], row, col] = 1.0
            component_id = self.component_by_cell.get(cell)
            if component_id is not None and self._component(component_id)["is_large"]:
                observation[channels["large_vessel"], row, col] = 1.0
            item = self.mechanics_by_cell.get(cell, {}) if self._state_ready else {}
            observation[channels["thickness"], row, col] = min(
                float(item.get("thickness", 0.0)) / DEFAULT_MECHANICS["thickness_max"], 10.0,
            )
            observation[channels["front_tension"], row, col] = min(
                float(item.get("front_tension", 0.0)), 10.0,
            )
            observation[channels["organ_energy"], row, col] = min(
                float(item.get("organ_energy", 0.0)), 10.0,
            )
            observation[channels["vessel_strain"], row, col] = min(
                float(item.get("vessel_strain", 0.0)) / DEFAULT_MECHANICS["tear_vessel_strain"], 10.0,
            )
        for component_id in self.exposed_ids:
            component = self._component(component_id)
            component_rate = (
                self.clinical_config["bleeding_probability"]
                * self.bleeding_beta
                * float(component["area_mm2"])
            )
            per_cell = min(1.0, component_rate / rate_normalizer) / len(component["cells"])
            for row, col in component["cells"]:
                observation[channels["expected_bleeding_rate"], row, col] = per_cell
        if self._state_ready:
            observation[channels["current_position"], self.current[0], self.current[1]] = 1.0
            observation[
                channels["previous_position"],
                self.previous_direction_position[0],
                self.previous_direction_position[1],
            ] = 1.0
            observation[channels["start"], self.start[0], self.start[1]] = 1.0
        row_coordinates = np.broadcast_to(
            np.arange(self.max_rows, dtype=np.float32)[:, None] / max(1, self.max_rows - 1),
            (self.max_rows, self.max_cols),
        )
        column_coordinates = np.broadcast_to(
            np.arange(self.max_cols, dtype=np.float32)[None, :] / max(1, self.max_cols - 1),
            (self.max_rows, self.max_cols),
        )
        observation[channels["row_coordinate"]] = row_coordinates
        observation[channels["column_coordinate"]] = column_coordinates
        observation[channels["clamped_phase"]].fill(float(self.phase == "clamped") if self._state_ready else 1.0)
        observation[channels["clamp_elapsed_fraction"]].fill(
            min(1.0, self.phase_elapsed_minutes / self.clinical_config["max_clamp_minutes"])
            if self._state_ready and self.phase == "clamped" else 0.0
        )
        observation[channels["unclamp_remaining_fraction"]].fill(
            max(0.0, 1.0 - self.phase_elapsed_minutes / self.clinical_config["unclamp_minutes"])
            if self._state_ready and self.phase == "unclamped" else 0.0
        )
        observation[channels["elapsed_time_fraction"]].fill(
            min(1.0, self.elapsed_minutes / self.clinical_config["max_episode_minutes"])
            if self._state_ready else 0.0
        )
        observation[channels["no_progress_streak_fraction"]].fill(
            min(
                1.0,
                self.no_progress_streak / self.clinical_config["stagnation_limit_steps"],
            )
            if self._state_ready else 0.0
        )
        observation[channels["same_edge_streak_fraction"]].fill(
            min(
                1.0,
                self.same_edge_streak
                / self.clinical_config["two_cell_loop_limit_traversals"],
            )
            if self._state_ready else 0.0
        )
        raw_clinical_cost = (
            self.reward_config["time_cost"]
            * self.elapsed_minutes / self.clinical_config["time_scale_minutes"]
            + self.reward_config["blood_cost"]
            * self.expected_blood_loss_ml / self.clinical_config["blood_scale_ml"]
        ) if self._state_ready else 0.0
        observation[channels["clinical_cost_fraction"]].fill(
            min(1.0, raw_clinical_cost / self.reward_config["clinical_cost_cap"])
        )
        return observation

    def _info(
        self,
        *,
        events: list[dict[str, Any]],
        reward_terms: Optional[Mapping[str, float]] = None,
    ) -> dict[str, Any]:
        return {
            "events": events,
            "action_mask": self.action_masks()
            if not (self.terminated or self.truncated)
            else np.zeros(len(ACTION_NAMES), dtype=bool),
            "phase": self.phase,
            "phase_elapsed_minutes": self.phase_elapsed_minutes,
            "elapsed_minutes": self.elapsed_minutes,
            "total_clamped_minutes": self.total_clamped_minutes,
            "total_unclamped_minutes": self.total_unclamped_minutes,
            "unclamped_exposed_minutes": self.unclamped_exposed_minutes,
            "expected_blood_loss_ml": self.expected_blood_loss_ml,
            "expected_bleeding_rate_ml_per_min": self._expected_bleeding_rate(),
            "peak_expected_bleeding_rate_ml_per_min": self.peak_expected_bleeding_rate_ml_per_min,
            "clamp_cycle_count": self.clamp_cycle_count,
            "early_end_count": self.early_end_count,
            "transfer_count": self.transfer_count,
            "direction_action_count": self.direction_action_count,
            "no_progress_streak": self.no_progress_streak,
            "max_no_progress_streak": self.max_no_progress_streak,
            "same_edge_streak": self.same_edge_streak,
            "max_same_edge_streak": self.max_same_edge_streak,
            "coverage": len(self.cut) / len(self.domain),
            "cut_count": len(self.cut),
            "hidden_component_ids": sorted(self.hidden_ids),
            "exposed_component_ids": sorted(self.exposed_ids),
            "sealed_component_ids": sorted(self.sealed_ids),
            "peak_front_tension": float(self.mechanics["peak_front_tension"]),
            "peak_organ_energy": float(self.mechanics["peak_organ_energy"]),
            "peak_vessel_strain": float(self.mechanics["peak_vessel_strain"]),
            "reward_terms": dict(reward_terms or {}),
            "failure_reason": self.failure_reason,
        }

    def episode_replay(self) -> dict[str, Any]:
        if not self._state_ready:
            raise RuntimeError("Call reset() before episode_replay()")
        return {
            "environment_version": CLINICAL_ENVIRONMENT_VERSION,
            "episode_id": self._episode_id,
            "scenario": self.scenario,
            "clinical_config": self.clinical_config,
            "reward_config": self.reward_config,
            "mechanics_parameters": self.mechanics_parameters,
            "mechanics_update_interval": self.mechanics_update_interval,
            "max_steps": self.max_steps,
            "events": self.events,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "failure_reason": self.failure_reason,
            "summary": {
                "elapsed_minutes": self.elapsed_minutes,
                "expected_blood_loss_ml": self.expected_blood_loss_ml,
                "peak_expected_bleeding_rate_ml_per_min": self.peak_expected_bleeding_rate_ml_per_min,
                "total_clamped_minutes": self.total_clamped_minutes,
                "total_unclamped_minutes": self.total_unclamped_minutes,
                "unclamped_exposed_minutes": self.unclamped_exposed_minutes,
                "clamp_cycle_count": self.clamp_cycle_count,
                "early_end_count": self.early_end_count,
                "transfer_count": self.transfer_count,
                "no_progress_streak": self.no_progress_streak,
                "max_no_progress_streak": self.max_no_progress_streak,
                "same_edge_streak": self.same_edge_streak,
                "max_same_edge_streak": self.max_same_edge_streak,
            },
        }

    def write_replay(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.episode_replay(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class ClinicalWindowScenarioPoolEnv(ClinicalWindowResectionEnv):
    """Seeded scenario sampler suitable for vectorized MaskablePPO."""

    def __init__(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        **kwargs: Any,
    ) -> None:
        if not scenarios:
            raise ValueError("ClinicalWindowScenarioPoolEnv requires at least one scenario")
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

"""RL environment contract for sequential planar resection.

The environment deliberately keeps the planning semantics in one place: an
action chooses the *next cut cell* only, while movement inside the existing
cut region is emitted as automatic ``transfer`` events.  It has no hard
Gymnasium dependency so that simulator tests remain runnable before the PPO
stack is installed.  When Gymnasium is installed, its ``observation_space``
and ``action_space`` attributes are populated for direct use with wrappers.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from mechanics import DEFAULT_MECHANICS, solve_tension
from planner import Cell, boundary_cells, is_connected, neighbors4, vessel_components

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Keep the simulator usable before the PPO dependencies exist.
    gym = None
    spaces = None

GymEnvBase = gym.Env if gym is not None else object

CANVAS_SIZE = 50
OBSERVATION_CHANNELS = (
    "domain", "cut", "vessel", "released_vessel", "frontier",
    "current_position", "start", "thickness", "normal_tension",
    "shear_tension", "front_tension", "organ_energy", "vessel_strain",
    "tip", "valid_cell_mask",
)
LOCAL_OBSERVATION_CHANNELS = OBSERVATION_CHANNELS + ("row_coordinate", "column_coordinate")
VARIABLE_GRID_ROWS = 30
VARIABLE_GRID_COLS = 40
VARIABLE_OBSERVATION_CHANNELS = OBSERVATION_CHANNELS + (
    "row_coordinate", "column_coordinate", "transfer_distance",
)
DEFAULT_REWARD = {
    "transfer_cost": 1.0,
    "lookahead_transfer_cost": 0.0,
    "tension_cost": 0.10,
    "organ_energy_cost": 0.10,
    "vessel_strain_cost": 1.0,
    "completion_bonus": 25.0,
    "failure_penalty": 25.0,
    "invalid_action_penalty": 25.0,
}


def _cell(value: Sequence[int]) -> Cell:
    if len(value) != 2:
        raise ValueError(f"Cell must be [row, column], got {value!r}")
    return int(value[0]), int(value[1])


def _cell_list(cells: Iterable[Cell]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(cells)]


class PlanarResectionEnv(GymEnvBase):
    """Fixed-canvas sequential-resection environment.

    ``reset()`` returns ``(observation, info)`` and ``step()`` returns the
    Gymnasium five-tuple.  Observations are ``float32`` arrays in CHW order
    with shape ``(15, 50, 50)``.  ``action_masks()`` returns a boolean vector
    of length 2500; action ``row * 50 + col`` addresses a canvas cell.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario: Optional[Mapping[str, Any]] = None,
        mechanics_parameters: Optional[Mapping[str, float]] = None,
        reward_config: Optional[Mapping[str, float]] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self._fixed_scenario = dict(scenario) if scenario is not None else None
        self.mechanics_parameters = dict(mechanics_parameters or {})
        self.reward_config = self._validated_reward(reward_config)
        self.max_steps_override = max_steps
        self._rng = random.Random()
        self._episode_id = 0
        self._state_ready = False
        if spaces is not None:
            self.action_space = spaces.Discrete(CANVAS_SIZE * CANVAS_SIZE)
            self.observation_space = spaces.Box(
                low=0.0, high=np.finfo(np.float32).max,
                shape=(len(OBSERVATION_CHANNELS), CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32,
            )
        else:
            self.action_space = None
            self.observation_space = None

    @staticmethod
    def _validated_reward(values: Optional[Mapping[str, float]]) -> Dict[str, float]:
        result = dict(DEFAULT_REWARD)
        if values:
            unknown = set(values) - set(result)
            if unknown:
                raise ValueError(f"Unknown reward settings: {sorted(unknown)}")
            result.update({name: float(value) for name, value in values.items()})
        if any(not math.isfinite(value) or value < 0 for value in result.values()):
            raise ValueError("All reward settings must be finite and non-negative")
        return result

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        """Start an episode from a supplied scenario (or construction scenario).

        The scenario schema is ``rows``, ``cols``, ``domain_cells``,
        ``obstacle_cells`` and ``start_cell``.  A Pilot scenario may instead
        provide ``starts``; in that case ``options={'start_index': 0}`` selects
        the entry.  The start cell is an initial cut, matching ``plan_resection``.
        """
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        options = dict(options or {})
        scenario = dict(options.get("scenario") or self._fixed_scenario or {})
        if not scenario:
            raise ValueError("reset requires a scenario at construction or in options['scenario']")
        self._load_scenario(scenario, int(options.get("start_index", 0)))
        self._episode_id += 1
        self.events = [{"index": 0, "action": "cut", "cell": list(self.start), "reason": "start"}]
        self.cut = {self.start}
        self.current = self.start
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.failure_reason: Optional[str] = None
        self._release_ready_components(self.events)
        self._update_mechanics()
        self._state_ready = True
        return self._observation(), self._info(events=self.events[:])

    def _load_scenario(self, scenario: Mapping[str, Any], start_index: int) -> None:
        self.rows, self.cols = int(scenario["rows"]), int(scenario["cols"])
        if not 1 <= self.rows <= CANVAS_SIZE or not 1 <= self.cols <= CANVAS_SIZE:
            raise ValueError("rows and cols must each be between 1 and 50")
        self.domain = {_cell(value) for value in scenario["domain_cells"]}
        self.obstacles = {_cell(value) for value in scenario.get("obstacle_cells", ())}
        if not self.domain or not is_connected(self.domain):
            raise ValueError("domain_cells must be a non-empty four-connected region")
        if not self.obstacles <= self.domain:
            raise ValueError("All obstacle cells must be inside domain_cells")
        boundary = boundary_cells(self.domain)
        if self.obstacles & boundary:
            raise ValueError("Obstacle cells cannot lie on the domain boundary")
        if "start_cell" in scenario:
            self.start = _cell(scenario["start_cell"])
        else:
            starts = scenario.get("starts", ())
            if not 0 <= start_index < len(starts):
                raise ValueError("start_index does not select a scenario start")
            self.start = _cell(starts[start_index]["cell"])
        if self.start not in boundary or self.start in self.obstacles:
            raise ValueError("start_cell must be a non-obstacle domain boundary cell")
        self.components = vessel_components(self.obstacles, self.domain)
        self.active_ids = {int(component["id"]) for component in self.components}
        self.released_ids: Set[int] = set()
        self.max_steps = int(self.max_steps_override or len(self.domain))
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.scenario = {
            "rows": self.rows, "cols": self.cols, "domain_cells": _cell_list(self.domain),
            "obstacle_cells": _cell_list(self.obstacles), "start_cell": list(self.start),
        }

    def _active_obstacles(self) -> Set[Cell]:
        return {cell for component in self.components if int(component["id"]) in self.active_ids
                for cell in component["cells"]}

    def _released_vessels(self) -> Set[Cell]:
        return {cell for component in self.components if int(component["id"]) in self.released_ids
                for cell in component["cells"]}

    def _release_ready_components(self, events: list[dict[str, Any]]) -> None:
        for component in self.components:
            component_id = int(component["id"])
            ring = set(component["ring"])
            if component_id in self.active_ids and ring and ring <= self.cut:
                self.active_ids.remove(component_id)
                self.released_ids.add(component_id)
                events.append({"index": len(self.events) + len(events) - len(events), "action": "release",
                               "component_id": component_id, "cells": _cell_list(component["cells"]),
                               "ring": _cell_list(ring)})

    def _frontier(self) -> Set[Cell]:
        traversable = self.domain - self.cut - self._active_obstacles()
        return {cell for cell in traversable if any(nxt in self.cut for nxt in neighbors4(cell))}

    def action_masks(self) -> np.ndarray:
        """Return the current legal-cut mask; calling it does not mutate state."""
        if not self._state_ready:
            raise RuntimeError("Call reset() before action_masks()")
        mask = np.zeros(CANVAS_SIZE * CANVAS_SIZE, dtype=bool)
        for row, col in self._frontier():
            mask[row * CANVAS_SIZE + col] = True
        return mask

    def _transfer_path(self, target: Cell) -> list[Cell]:
        goals = set(neighbors4(target)) & self.cut
        queue: deque[Cell] = deque([self.current])
        parent: Dict[Cell, Optional[Cell]] = {self.current: None}
        goal: Optional[Cell] = None
        while queue:
            cell = queue.popleft()
            if cell in goals:
                goal = cell
                break
            for nxt in neighbors4(cell):
                if nxt in self.cut and nxt not in parent:
                    parent[nxt] = cell
                    queue.append(nxt)
        if goal is None:
            return []
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])  # type: ignore[arg-type]
        return list(reversed(path))

    def _next_frontier_transfer_count(self) -> int:
        """Return the shortest cut-region route to any next frontier cell."""
        frontier = self._frontier()
        if not frontier:
            return 0
        routes = [self._transfer_path(target) for target in frontier]
        valid_lengths = [len(route) - 1 for route in routes if route]
        return min(valid_lengths) if valid_lengths else 0

    def step(self, action: int):
        if not self._state_ready:
            raise RuntimeError("Call reset() before step()")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended; call reset()")
        action = int(action)
        row, col = divmod(action, CANVAS_SIZE)
        target = (row, col)
        step_events: list[dict[str, Any]] = []
        mask = self.action_masks()
        if not 0 <= action < len(mask) or not mask[action]:
            self.terminated = True
            self.failure_reason = f"invalid action {action}; action must select the current legal frontier"
            reward = -self.reward_config["invalid_action_penalty"]
            return self._observation(), reward, True, False, self._info(events=step_events, reward_terms={"invalid_action": -reward})

        route = self._transfer_path(target)
        if not route:
            self.terminated = True
            self.failure_reason = "No transfer path through cut cells to selected frontier cell"
            reward = -self.reward_config["failure_penalty"]
            return self._observation(), reward, True, False, self._info(events=step_events, reward_terms={"failure": -reward})
        for cell in route[1:]:
            step_events.append({"index": len(self.events) + len(step_events), "action": "transfer", "cell": list(cell)})
        step_events.append({"index": len(self.events) + len(step_events), "action": "cut", "cell": list(target)})
        self.cut.add(target)
        self.current = target
        self.step_count += 1
        self.events.extend(step_events)
        release_events: list[dict[str, Any]] = []
        self._release_ready_components(release_events)
        for event in release_events:
            event["index"] = len(self.events)
            self.events.append(event)
            step_events.append(event)
        self._update_mechanics()
        reward_terms = self._reward_terms(len(route) - 1)
        next_transfer_count = self._next_frontier_transfer_count()
        reward_terms["lookahead_transfer_cost"] = (
            self.reward_config["lookahead_transfer_cost"] * next_transfer_count
        )
        reward = -sum(reward_terms.values())
        if self.cut == self.domain:
            self.terminated = True
            reward += self.reward_config["completion_bonus"]
            reward_terms["completion_bonus"] = -self.reward_config["completion_bonus"]
        elif not self._frontier():
            self.terminated = True
            self.failure_reason = "No dynamic-frontier candidate remains"
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]
        elif self.step_count >= self.max_steps:
            self.truncated = True
            self.failure_reason = f"maximum step count ({self.max_steps}) reached"
            reward -= self.reward_config["failure_penalty"]
            reward_terms["failure_penalty"] = self.reward_config["failure_penalty"]
        return self._observation(), float(reward), self.terminated, self.truncated, self._info(events=step_events, reward_terms=reward_terms)

    def _update_mechanics(self) -> None:
        self.mechanics = solve_tension(rows=self.rows, cols=self.cols, domain_cells=_cell_list(self.domain),
                                       vessel_cells=_cell_list(self._active_obstacles()), cut_cells=_cell_list(self.cut),
                                       parameters=self.mechanics_parameters)
        self.mechanics_by_cell = {tuple(item["cell"]): item for item in self.mechanics["cells"]}

    def _reward_terms(self, transfer_count: int) -> Dict[str, float]:
        front = [self.mechanics_by_cell[cell]["front_tension"] for cell in self._frontier()]
        return {
            "transfer_cost": self.reward_config["transfer_cost"] * transfer_count,
            "tension_cost": self.reward_config["tension_cost"] * (max(front) if front else 0.0),
            "organ_energy_cost": self.reward_config["organ_energy_cost"] * float(self.mechanics["peak_organ_energy"]),
            "vessel_strain_cost": self.reward_config["vessel_strain_cost"] * float(self.mechanics["peak_vessel_strain"]),
        }

    def _observation(self) -> np.ndarray:
        observation = np.zeros((len(OBSERVATION_CHANNELS), CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        channels = {name: index for index, name in enumerate(OBSERVATION_CHANNELS)}
        frontier = self._frontier() if self._state_ready else set()
        active = self._active_obstacles() if self._state_ready else set()
        released = self._released_vessels() if self._state_ready else set()
        scale = {"thickness": DEFAULT_MECHANICS["thickness_max"], "normal_tension": 1.0,
                 "shear_tension": 1.0, "front_tension": 1.0, "organ_energy": 1.0,
                 "vessel_strain": DEFAULT_MECHANICS["tear_vessel_strain"]}
        for cell in self.domain:
            row, col = cell
            observation[channels["domain"], row, col] = 1.0
            observation[channels["valid_cell_mask"], row, col] = 1.0
            item = self.mechanics_by_cell.get(cell, {}) if self._state_ready else {}
            for name, divisor in scale.items():
                observation[channels[name], row, col] = min(float(item.get(name, 0.0)) / divisor, 10.0)
            if cell in self.cut: observation[channels["cut"], row, col] = 1.0
            if cell in active: observation[channels["vessel"], row, col] = 1.0
            if cell in released: observation[channels["released_vessel"], row, col] = 1.0
            if cell in frontier: observation[channels["frontier"], row, col] = 1.0
            if item.get("is_tip"): observation[channels["tip"], row, col] = 1.0
        observation[channels["current_position"], self.current[0], self.current[1]] = 1.0
        observation[channels["start"], self.start[0], self.start[1]] = 1.0
        return observation

    def _info(self, *, events: list[dict[str, Any]], reward_terms: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        return {"events": events, "action_mask": self.action_masks() if not (self.terminated or self.truncated) else np.zeros(CANVAS_SIZE ** 2, dtype=bool),
                "coverage": len(self.cut) / len(self.domain), "cut_count": len(self.cut),
                "transfer_count": sum(event["action"] == "transfer" for event in self.events),
                "released_component_ids": sorted(self.released_ids), "failure_reason": self.failure_reason,
                "peak_vessel_strain": float(self.mechanics["peak_vessel_strain"]),
                "peak_front_tension": float(self.mechanics["peak_front_tension"]),
                "peak_organ_energy": float(self.mechanics["peak_organ_energy"]),
                "reward_terms": dict(reward_terms or {})}

    def episode_replay(self) -> Dict[str, Any]:
        """Return a JSON-serializable replay record for deterministic inspection."""
        if not self._state_ready:
            raise RuntimeError("Call reset() before episode_replay()")
        return {"environment_version": 1, "episode_id": self._episode_id, "scenario": self.scenario,
                "mechanics_parameters": self.mechanics_parameters, "reward_config": self.reward_config,
                "max_steps": self.max_steps, "events": self.events, "terminated": self.terminated,
                "truncated": self.truncated, "failure_reason": self.failure_reason}

    def write_replay(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.episode_replay(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ScenarioPoolEnv(PlanarResectionEnv):
    """A Gymnasium environment sampling exclusively from a frozen scenario pool.

    It is module-level (rather than defined in a CLI entry point) so that
    ``SubprocVecEnv(..., start_method='spawn')`` can pickle it correctly.
    """

    def __init__(self, scenarios: Sequence[Mapping[str, Any]], *, seed: int, **kwargs: Any) -> None:
        if not scenarios:
            raise ValueError("ScenarioPoolEnv requires at least one scenario")
        self._pool = [dict(item) for item in scenarios]
        self._scenario_rng = random.Random(seed)
        super().__init__(scenario=self._pool[0], **kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        if seed is not None:
            self._scenario_rng.seed(seed)
        options = dict(options or {})
        scenario_index = int(options.pop("scenario_index", self._scenario_rng.randrange(len(self._pool))))
        if not 0 <= scenario_index < len(self._pool):
            raise ValueError("scenario_index outside fixed scenario pool")
        options["scenario"] = self._pool[scenario_index]
        return super().reset(seed=seed, options=options)


def local_grid_observation(env: PlanarResectionEnv, grid_size: int) -> np.ndarray:
    """Crop the fixed canvas and append explicit row/column coordinates."""
    if (env.rows, env.cols) != (grid_size, grid_size):
        raise ValueError(f"Local-grid observation requires a {grid_size}x{grid_size} scenario")
    cropped = env._observation()[:, :grid_size, :grid_size]
    coordinates = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    row_channel = np.broadcast_to(coordinates[:, None], (grid_size, grid_size))
    column_channel = np.broadcast_to(coordinates[None, :], (grid_size, grid_size))
    return np.concatenate(
        (cropped, row_channel[None, ...], column_channel[None, ...]), axis=0,
    ).astype(np.float32, copy=False)


def local_grid_action_masks(env: PlanarResectionEnv, grid_size: int) -> np.ndarray:
    """Map the fixed-canvas legal mask onto the active local grid."""
    canvas_mask = env.action_masks().reshape(CANVAS_SIZE, CANVAS_SIZE)
    return canvas_mask[:grid_size, :grid_size].reshape(-1).copy()


def local_to_canvas_action(action: int, grid_size: int) -> int:
    """Map one local-grid action index to the fixed-canvas action index."""
    action = int(action)
    if not 0 <= action < grid_size * grid_size:
        raise ValueError(f"Local action must be between 0 and {grid_size * grid_size - 1}")
    row, col = divmod(action, grid_size)
    return row * CANVAS_SIZE + col


def variable_grid_observation(
    env: PlanarResectionEnv,
    *,
    max_rows: int = VARIABLE_GRID_ROWS,
    max_cols: int = VARIABLE_GRID_COLS,
) -> np.ndarray:
    """Return a size-preserving, zero-padded observation for variable grids.

    The first 15 channels are the canonical simulator state.  Coordinates are
    normalized against the *maximum* canvas so a 4 mm cell retains its physical
    scale across cases.  ``transfer_distance`` is the shortest cut-region route
    from the current position to each legal frontier cell, divided by
    ``max_rows + max_cols``; non-frontier cells are zero.  This exposes the
    long-range movement cost without changing the environment transition.
    """
    if not 1 <= env.rows <= max_rows or not 1 <= env.cols <= max_cols:
        raise ValueError(
            f"Scenario shape {env.rows}x{env.cols} exceeds variable-grid maximum "
            f"{max_rows}x{max_cols}"
        )
    base = env._observation()[:, :max_rows, :max_cols]
    row_coordinates = np.broadcast_to(
        np.arange(max_rows, dtype=np.float32)[:, None] / max(1, max_rows - 1),
        (max_rows, max_cols),
    )
    column_coordinates = np.broadcast_to(
        np.arange(max_cols, dtype=np.float32)[None, :] / max(1, max_cols - 1),
        (max_rows, max_cols),
    )
    transfer_distance = np.zeros((max_rows, max_cols), dtype=np.float32)
    for cell in env._frontier():
        route = env._transfer_path(cell)
        if route:
            row, col = cell
            transfer_distance[row, col] = (len(route) - 1) / float(max_rows + max_cols)
    return np.concatenate(
        (base, row_coordinates[None, ...], column_coordinates[None, ...], transfer_distance[None, ...]),
        axis=0,
    ).astype(np.float32, copy=False)


def variable_grid_action_masks(
    env: PlanarResectionEnv,
    *,
    max_rows: int = VARIABLE_GRID_ROWS,
    max_cols: int = VARIABLE_GRID_COLS,
) -> np.ndarray:
    """Map canonical 50x50 legal actions onto a padded variable-grid mask."""
    if not 1 <= env.rows <= max_rows or not 1 <= env.cols <= max_cols:
        raise ValueError(
            f"Scenario shape {env.rows}x{env.cols} exceeds variable-grid maximum "
            f"{max_rows}x{max_cols}"
        )
    result = np.zeros((max_rows, max_cols), dtype=bool)
    canvas_mask = env.action_masks().reshape(CANVAS_SIZE, CANVAS_SIZE)
    result[:env.rows, :env.cols] = canvas_mask[:env.rows, :env.cols]
    return result.reshape(-1)


def variable_to_canvas_action(action: int, *, max_rows: int = VARIABLE_GRID_ROWS, max_cols: int = VARIABLE_GRID_COLS) -> int:
    """Map a padded variable-grid action index to the canonical 50x50 action."""
    action = int(action)
    if not 0 <= action < max_rows * max_cols:
        raise ValueError(f"Variable-grid action must be between 0 and {max_rows * max_cols - 1}")
    row, col = divmod(action, max_cols)
    return row * CANVAS_SIZE + col


class LocalGridScenarioPoolEnv(GymEnvBase):
    """Compact training view over :class:`ScenarioPoolEnv`.

    The simulator keeps its stable 50x50 API, while PPO sees only the active
    grid and emits only active-grid actions.  This removes unused logits and
    preserves exact spatial structure without changing simulator semantics.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        *,
        grid_size: int,
        seed: int,
        **kwargs: Any,
    ) -> None:
        if not 1 <= grid_size <= CANVAS_SIZE:
            raise ValueError("grid_size must be between 1 and 50")
        for scenario in scenarios:
            if (int(scenario["rows"]), int(scenario["cols"])) != (grid_size, grid_size):
                raise ValueError("Every local-grid scenario must match grid_size")
        self.grid_size = int(grid_size)
        self.base_env = ScenarioPoolEnv(scenarios, seed=seed, **kwargs)
        if spaces is not None:
            self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(len(LOCAL_OBSERVATION_CHANNELS), self.grid_size, self.grid_size),
                dtype=np.float32,
            )
        else:
            self.action_space = None
            self.observation_space = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        _, info = self.base_env.reset(seed=seed, options=options)
        info = dict(info)
        info["action_mask"] = self.action_masks()
        return local_grid_observation(self.base_env, self.grid_size), info

    def step(self, action: int):
        canvas_action = local_to_canvas_action(action, self.grid_size)
        _, reward, terminated, truncated, info = self.base_env.step(canvas_action)
        info = dict(info)
        info["action_mask"] = (
            np.zeros(self.grid_size * self.grid_size, dtype=bool)
            if terminated or truncated else self.action_masks()
        )
        return (
            local_grid_observation(self.base_env, self.grid_size),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        return local_grid_action_masks(self.base_env, self.grid_size)

    def close(self) -> None:
        close = getattr(self.base_env, "close", None)
        if close is not None:
            close()


class VariableGridScenarioPoolEnv(GymEnvBase):
    """Padded 30x40 training view that preserves each scenario's true shape.

    Unlike :class:`LocalGridScenarioPoolEnv`, scenarios may have different
    heights and widths.  PPO always receives a 30x40 tensor and 1,200 action
    positions; padding is non-actionable through both ``valid_cell_mask`` and
    ``action_masks()``.  The underlying state transitions remain exclusively
    owned by :class:`PlanarResectionEnv`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        max_rows: int = VARIABLE_GRID_ROWS,
        max_cols: int = VARIABLE_GRID_COLS,
        **kwargs: Any,
    ) -> None:
        if not scenarios:
            raise ValueError("VariableGridScenarioPoolEnv requires at least one scenario")
        if not 1 <= max_rows <= CANVAS_SIZE or not 1 <= max_cols <= CANVAS_SIZE:
            raise ValueError("Variable-grid maximum dimensions must be between 1 and 50")
        for scenario in scenarios:
            rows, cols = int(scenario["rows"]), int(scenario["cols"])
            if not 1 <= rows <= max_rows or not 1 <= cols <= max_cols:
                raise ValueError(
                    f"Scenario shape {rows}x{cols} exceeds variable-grid maximum {max_rows}x{max_cols}"
                )
        self.max_rows = int(max_rows)
        self.max_cols = int(max_cols)
        self.base_env = ScenarioPoolEnv(scenarios, seed=seed, **kwargs)
        if spaces is not None:
            self.action_space = spaces.Discrete(self.max_rows * self.max_cols)
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(len(VARIABLE_OBSERVATION_CHANNELS), self.max_rows, self.max_cols),
                dtype=np.float32,
            )
        else:
            self.action_space = None
            self.observation_space = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        _, info = self.base_env.reset(seed=seed, options=options)
        info = dict(info)
        info["action_mask"] = self.action_masks()
        return variable_grid_observation(self.base_env, max_rows=self.max_rows, max_cols=self.max_cols), info

    def step(self, action: int):
        canvas_action = variable_to_canvas_action(action, max_rows=self.max_rows, max_cols=self.max_cols)
        _, reward, terminated, truncated, info = self.base_env.step(canvas_action)
        info = dict(info)
        info["action_mask"] = (
            np.zeros(self.max_rows * self.max_cols, dtype=bool)
            if terminated or truncated else self.action_masks()
        )
        return (
            variable_grid_observation(self.base_env, max_rows=self.max_rows, max_cols=self.max_cols),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        return variable_grid_action_masks(self.base_env, max_rows=self.max_rows, max_cols=self.max_cols)

    def close(self) -> None:
        close = getattr(self.base_env, "close", None)
        if close is not None:
            close()

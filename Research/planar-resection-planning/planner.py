"""Pure-Python planning core for the planar resection simulator.

The module deliberately has no dependency on the medical-imaging pipeline.
Cells are represented as ``(row, column)`` integer pairs.  Cutting and travel
use four-neighbour adjacency; vessel components and their release rings use
eight-neighbour adjacency.
"""

from __future__ import annotations

import math
import random
import secrets
from collections import deque
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Cell = Tuple[int, int]

DEFAULT_WEIGHTS = {
    "distance": 1.0,
    "vessel_risk": 2.0,
    "shape": 1.0,
    "exposure": 0.75,
}
RISK_DECAY_CELLS = 2.0


def _cell(value: Sequence[int]) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Cell must be [row, column], got {value!r}")
    return int(value[0]), int(value[1])


def _cell_list(cells: Iterable[Cell]) -> List[List[int]]:
    return [[r, c] for r, c in sorted(cells)]


def neighbors4(cell: Cell) -> Iterable[Cell]:
    r, c = cell
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(cell: Cell) -> Iterable[Cell]:
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def boundary_cells(domain: Set[Cell]) -> Set[Cell]:
    """Return the one-cell four-neighbour boundary of a domain."""
    return {cell for cell in domain if any(nxt not in domain for nxt in neighbors4(cell))}


def is_connected(domain: Set[Cell]) -> bool:
    if not domain:
        return False
    seen = {next(iter(domain))}
    queue = deque(seen)
    while queue:
        cur = queue.popleft()
        for nxt in neighbors4(cur):
            if nxt in domain and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen == domain


def _fill_holes(domain: Set[Cell], rows: int, cols: int) -> Set[Cell]:
    outside: Set[Cell] = set()
    queue: deque[Cell] = deque()
    for r in range(rows):
        for c in (0, cols - 1):
            if (r, c) not in domain and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if (r, c) not in domain and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))
    while queue:
        cur = queue.popleft()
        for nxt in neighbors4(cur):
            r, c = nxt
            if 0 <= r < rows and 0 <= c < cols and nxt not in domain and nxt not in outside:
                outside.add(nxt)
                queue.append(nxt)
    holes = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in domain and (r, c) not in outside
    }
    return domain | holes


def _is_rectangle(domain: Set[Cell]) -> bool:
    if not domain:
        return False
    rs = [cell[0] for cell in domain]
    cs = [cell[1] for cell in domain]
    return len(domain) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def generate_domain(
    seed: Optional[int] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> Dict[str, object]:
    """Generate a deterministic, smooth-edged, irregular grid domain.

    The outline is sampled from a low-frequency radial function instead of
    growing a random cell frontier.  Keeping only broad harmonics produces
    organic lobes without the one-cell noise and saw-tooth bays created by
    frontier growth.
    """
    actual_seed = int(seed) if seed is not None else secrets.randbits(31)
    rng = random.Random(actual_seed)
    actual_rows = int(rows) if rows is not None else rng.randint(18, 30)
    actual_cols = int(cols) if cols is not None else rng.randint(18, 30)
    if not 10 <= actual_rows <= 50 or not 10 <= actual_cols <= 50:
        raise ValueError("rows and cols must both be between 10 and 50")

    # The ellipse leaves a visible margin.  Low-frequency perturbations make
    # each seed distinct while ensuring the continuous outline remains smooth.
    center_r = (actual_rows - 1) / 2 + rng.uniform(-0.04, 0.04) * actual_rows
    center_c = (actual_cols - 1) / 2 + rng.uniform(-0.04, 0.04) * actual_cols
    radius_r = (actual_rows - 2) / 2 * rng.uniform(0.78, 0.92)
    radius_c = (actual_cols - 2) / 2 * rng.uniform(0.78, 0.92)
    harmonics = [
        (frequency, rng.uniform(-0.075, 0.075), rng.uniform(0.0, 2 * math.pi))
        for frequency in (2, 3, 4)
    ]

    domain: Set[Cell] = set()
    for r in range(1, actual_rows - 1):
        for c in range(1, actual_cols - 1):
            y = (r - center_r) / radius_r
            x = (c - center_c) / radius_c
            angle = math.atan2(y, x)
            smooth_radius = 1.0 + sum(
                amplitude * math.cos(frequency * angle + phase)
                for frequency, amplitude, phase in harmonics
            )
            if math.hypot(y, x) <= smooth_radius:
                domain.add((r, c))

    domain = _fill_holes(domain, actual_rows, actual_cols)
    if _is_rectangle(domain):
        # This is extremely unlikely with a perturbed radial outline. Removing
        # one corner preserves connectivity while making the guarantee exact.
        corner = min(domain)
        candidate = domain - {corner}
        if is_connected(candidate):
            domain = candidate

    return {
        "seed": actual_seed,
        "rows": actual_rows,
        "cols": actual_cols,
        "domain_cells": _cell_list(domain),
        "boundary_cells": _cell_list(boundary_cells(domain)),
        "cell_count": len(domain),
        "fill_ratio": round(len(domain) / (actual_rows * actual_cols), 6),
    }


def vessel_components(obstacles: Set[Cell], domain: Set[Cell]) -> List[Dict[str, object]]:
    """Split obstacles by 8-connectivity and compute each external 8-ring."""
    remaining = set(obstacles)
    components: List[Dict[str, object]] = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        queue = deque([seed])
        remaining.remove(seed)
        while queue:
            cur = queue.popleft()
            for nxt in neighbors8(cur):
                if nxt in remaining:
                    remaining.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        ring = {
            nxt
            for cell in component
            for nxt in neighbors8(cell)
            if nxt in domain and nxt not in obstacles
        }
        components.append({
            "id": len(components),
            "cells": component,
            "ring": ring,
        })
    return components


def _shortest_path(start: Cell, goal: Cell, allowed: Set[Cell]) -> List[Cell]:
    if start == goal:
        return [start]
    queue = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while queue:
        cur = queue.popleft()
        for nxt in neighbors4(cur):
            if nxt not in allowed or nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == goal:
                path = [goal]
                while path[-1] != start:
                    previous = parent[path[-1]]
                    if previous is None:
                        break
                    path.append(previous)
                return list(reversed(path))
            queue.append(nxt)
    return []


def _shortest_path_tree(start: Cell, allowed: Set[Cell]) -> Dict[Cell, Optional[Cell]]:
    """Build one BFS tree for every transfer destination in ``allowed``.

    The planner previously ran a separate BFS for each candidate frontier cell.
    A single tree has identical shortest-path semantics while making batch weight
    searches practical.
    """
    queue = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while queue:
        cur = queue.popleft()
        for nxt in neighbors4(cur):
            if nxt in allowed and nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return parent


def _path_from_tree(goal: Cell, parent: Mapping[Cell, Optional[Cell]]) -> List[Cell]:
    if goal not in parent:
        return []
    path = [goal]
    while parent[path[-1]] is not None:
        previous = parent[path[-1]]
        if previous is None:
            break
        path.append(previous)
    return list(reversed(path))


def _validated_weights(values: Optional[Mapping[str, float]]) -> Dict[str, float]:
    result = dict(DEFAULT_WEIGHTS)
    if values:
        unknown = set(values) - set(result)
        if unknown:
            raise ValueError(f"Unknown weight names: {sorted(unknown)}")
        for name, value in values.items():
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"Weight '{name}' must be a finite non-negative number")
            result[name] = number
    return result


def plan_resection(
    *,
    rows: int,
    cols: int,
    domain_cells: Sequence[Sequence[int]],
    obstacle_cells: Sequence[Sequence[int]],
    start_cell: Sequence[int],
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Plan a complete dynamic-frontier sequence for one planar simulation."""
    rows, cols = int(rows), int(cols)
    if not 1 <= rows <= 50 or not 1 <= cols <= 50:
        raise ValueError("rows and cols must both be between 1 and 50")
    domain = {_cell(value) for value in domain_cells}
    obstacles = {_cell(value) for value in obstacle_cells}
    start = _cell(start_cell)
    if not domain:
        raise ValueError("domain_cells cannot be empty")
    if any(not (0 <= r < rows and 0 <= c < cols) for r, c in domain):
        raise ValueError("domain_cells contains a cell outside rows/cols")
    if not is_connected(domain):
        raise ValueError("domain_cells must form one four-connected region")
    if not obstacles <= domain:
        raise ValueError("All obstacle cells must be inside the domain")
    outer_boundary = boundary_cells(domain)
    invalid_obstacles = obstacles & outer_boundary
    if invalid_obstacles:
        raise ValueError(f"Obstacle cells cannot lie on the domain boundary: {sorted(invalid_obstacles)}")
    if start not in outer_boundary:
        raise ValueError("start_cell must lie on the domain boundary")
    if start in obstacles:
        raise ValueError("start_cell cannot be an obstacle")

    actual_weights = _validated_weights(weights)
    components = vessel_components(obstacles, domain)
    active_ids = {int(component["id"]) for component in components}
    released_ids: Set[int] = set()
    cut: Set[Cell] = {start}
    current = start
    events: List[Dict[str, object]] = [{
        "index": 0,
        "action": "cut",
        "cell": list(start),
        "score": None,
        "reason": "start",
    }]

    def active_obstacles() -> Set[Cell]:
        return {
            cell
            for component in components
            if int(component["id"]) in active_ids
            for cell in component["cells"]  # type: ignore[union-attr]
        }

    def release_ready_components() -> bool:
        changed = False
        for component in components:
            component_id = int(component["id"])
            ring: Set[Cell] = component["ring"]  # type: ignore[assignment]
            if component_id in active_ids and ring and ring <= cut:
                active_ids.remove(component_id)
                released_ids.add(component_id)
                events.append({
                    "index": len(events),
                    "action": "release",
                    "component_id": component_id,
                    "cells": _cell_list(component["cells"]),  # type: ignore[arg-type]
                    "ring": _cell_list(ring),
                })
                changed = True
        return changed

    while cut != domain:
        release_ready_components()
        blocked = active_obstacles()
        traversable_uncut = domain - cut - blocked
        frontier = {
            cell for cell in traversable_uncut
            if any(nxt in cut for nxt in neighbors4(cell))
        }
        if not frontier:
            break

        candidate_data: Dict[Cell, Dict[str, object]] = {}
        raw_distances: List[int] = []
        transfer_tree = _shortest_path_tree(current, cut)
        for candidate in sorted(frontier):
            best_route: Optional[List[Cell]] = None
            for entry in sorted(set(neighbors4(candidate)) & cut):
                route = _path_from_tree(entry, transfer_tree)
                if route and (
                    best_route is None
                    or len(route) < len(best_route)
                    or (len(route) == len(best_route) and route < best_route)
                ):
                    best_route = route
            if best_route is None:
                continue
            distance = len(best_route)  # transfer edges plus final cut edge
            raw_distances.append(distance)
            candidate_data[candidate] = {"route": best_route, "distance": distance}

        if not candidate_data:
            break
        min_distance, max_distance = min(raw_distances), max(raw_distances)
        distance_span = max_distance - min_distance
        blocked = active_obstacles()

        for candidate, data in candidate_data.items():
            raw_distance = int(data["distance"])
            distance_score = (
                (raw_distance - min_distance) / distance_span if distance_span else 0.0
            )
            if blocked:
                nearest_vessel = min(
                    math.hypot(candidate[0] - cell[0], candidate[1] - cell[1])
                    for cell in blocked
                )
                risk_score = math.exp(-(max(1.0, nearest_vessel) - 1.0) / RISK_DECAY_CELLS)
            else:
                nearest_vessel = None
                risk_score = 0.0
            cut_neighbors = sum(nxt in cut for nxt in neighbors4(candidate))
            shape_score = (4 - cut_neighbors) / 3.0
            new_exposure = sum(
                nxt in traversable_uncut
                and nxt not in frontier
                and nxt != candidate
                for nxt in neighbors4(candidate)
            )
            exposure_score = new_exposure / 3.0
            score = (
                actual_weights["distance"] * distance_score
                + actual_weights["vessel_risk"] * risk_score
                + actual_weights["shape"] * shape_score
                - actual_weights["exposure"] * exposure_score
            )
            data.update({
                "score": score,
                "distance_score": distance_score,
                "risk_score": risk_score,
                "nearest_vessel": nearest_vessel,
                "shape_score": shape_score,
                "exposure_score": exposure_score,
                "new_exposure": new_exposure,
            })

        target = min(
            candidate_data,
            key=lambda cell: (
                round(float(candidate_data[cell]["score"]), 12),
                int(candidate_data[cell]["distance"]),
                -int(candidate_data[cell]["new_exposure"]),
                cell[0] * cols + cell[1],
            ),
        )
        selected = candidate_data[target]
        route: List[Cell] = selected["route"]  # type: ignore[assignment]
        for transfer_cell in route[1:]:
            events.append({
                "index": len(events),
                "action": "transfer",
                "cell": list(transfer_cell),
            })
        events.append({
            "index": len(events),
            "action": "cut",
            "cell": list(target),
            "score": round(float(selected["score"]), 8),
            "score_terms": {
                "distance": round(float(selected["distance_score"]), 8),
                "vessel_risk": round(float(selected["risk_score"]), 8),
                "shape": round(float(selected["shape_score"]), 8),
                "exposure": round(float(selected["exposure_score"]), 8),
            },
        })
        cut.add(target)
        current = target

    release_ready_components()
    uncovered = domain - cut
    active_at_end = sorted(active_ids)
    status = "ok" if not uncovered else "partial"
    failure_reason = None
    if uncovered:
        failure_reason = (
            f"No dynamic-frontier candidate remains; {len(uncovered)} cells are uncovered "
            f"and {len(active_at_end)} vessel components remain active."
        )

    serialized_components = [{
        "id": int(component["id"]),
        "cells": _cell_list(component["cells"]),  # type: ignore[arg-type]
        "ring": _cell_list(component["ring"]),  # type: ignore[arg-type]
    } for component in components]
    return {
        "status": status,
        "rows": rows,
        "cols": cols,
        "domain_cells": _cell_list(domain),
        "boundary_cells": _cell_list(outer_boundary),
        "obstacle_cells": _cell_list(obstacles),
        "start_cell": list(start),
        "weights": actual_weights,
        "risk_decay_cells": RISK_DECAY_CELLS,
        "components": serialized_components,
        "events": events,
        "event_count": len(events),
        "cut_count": len(cut),
        "transfer_count": sum(event["action"] == "transfer" for event in events),
        "release_count": sum(event["action"] == "release" for event in events),
        "coverage": round(len(cut) / len(domain), 8),
        "uncovered_cells": _cell_list(uncovered),
        "active_component_ids": active_at_end,
        "released_component_ids": sorted(released_ids),
        "failure_reason": failure_reason,
    }


def serpentine_priority_resection(
    *,
    rows: int,
    cols: int,
    domain_cells: Sequence[Sequence[int]],
    obstacle_cells: Sequence[Sequence[int]],
    start_cell: Sequence[int],
) -> Dict[str, object]:
    """Execute the fast S-order dynamic-frontier baseline.

    This is the procedural equivalent of ``serpentine_priority_policy``.  It
    deliberately does not instantiate :class:`PlanarResectionEnv`: that
    environment recomputes the optional 2.5D mechanics model after every cut,
    while this baseline's action rule never consumes mechanics.  The surgical
    grid semantics are preserved: cuts can only be selected from the current
    frontier, movement through cut tissue is emitted as transfer events, and a
    vessel component is released only after its full ring is cut.
    """
    rows, cols = int(rows), int(cols)
    if not 1 <= rows <= 50 or not 1 <= cols <= 50:
        raise ValueError("rows and cols must both be between 1 and 50")
    domain = {_cell(value) for value in domain_cells}
    obstacles = {_cell(value) for value in obstacle_cells}
    start = _cell(start_cell)
    canvas = {(row, col) for row in range(rows) for col in range(cols)}
    if not domain or not domain <= canvas:
        raise ValueError("domain_cells must be a non-empty region inside rows/cols")
    if not is_connected(domain):
        raise ValueError("domain_cells must form one four-connected region")
    if not obstacles <= domain:
        raise ValueError("All obstacle cells must be inside the domain")
    outer_boundary = boundary_cells(domain)
    if obstacles & outer_boundary:
        raise ValueError("Obstacle cells cannot lie on the domain boundary")
    if start not in outer_boundary or start in obstacles:
        raise ValueError("start_cell must be a non-obstacle domain boundary cell")

    components = vessel_components(obstacles, domain)
    active_ids = {int(component["id"]) for component in components}
    released_ids: Set[int] = set()
    cut: Set[Cell] = {start}
    current = start
    events: List[Dict[str, object]] = [{
        "index": 0,
        "action": "cut",
        "cell": list(start),
        "reason": "start",
    }]

    def active_obstacles() -> Set[Cell]:
        return {
            cell
            for component in components
            if int(component["id"]) in active_ids
            for cell in component["cells"]  # type: ignore[union-attr]
        }

    def release_ready_components() -> None:
        for component in components:
            component_id = int(component["id"])
            ring: Set[Cell] = component["ring"]  # type: ignore[assignment]
            if component_id in active_ids and ring and ring <= cut:
                active_ids.remove(component_id)
                released_ids.add(component_id)
                events.append({
                    "index": len(events),
                    "action": "release",
                    "component_id": component_id,
                    "cells": _cell_list(component["cells"]),  # type: ignore[arg-type]
                    "ring": _cell_list(ring),
                })

    def scan_rank(cell: Cell) -> Tuple[int, int, int]:
        row, col = cell
        scan_col = col if row % 2 == 0 else cols - 1 - col
        return row * cols + scan_col, row, col

    while cut != domain:
        release_ready_components()
        blocked = active_obstacles()
        frontier = {
            cell for cell in domain - cut - blocked
            if any(neighbor in cut for neighbor in neighbors4(cell))
        }
        if not frontier:
            break
        target = min(frontier, key=scan_rank)
        tree = _shortest_path_tree(current, cut)
        routes = [
            _path_from_tree(entry, tree)
            for entry in sorted(set(neighbors4(target)) & cut)
        ]
        routes = [route for route in routes if route]
        if not routes:
            break
        route = min(routes, key=lambda item: (len(item), item))
        for transfer_cell in route[1:]:
            events.append({
                "index": len(events),
                "action": "transfer",
                "cell": list(transfer_cell),
            })
        events.append({
            "index": len(events),
            "action": "cut",
            "cell": list(target),
        })
        cut.add(target)
        current = target

    release_ready_components()
    uncovered = domain - cut
    serialized_components = [{
        "id": int(component["id"]),
        "cells": _cell_list(component["cells"]),  # type: ignore[arg-type]
        "ring": _cell_list(component["ring"]),  # type: ignore[arg-type]
    } for component in components]
    return {
        "status": "ok" if not uncovered else "partial",
        "policy": "serpentine-priority",
        "rows": rows,
        "cols": cols,
        "domain_cells": _cell_list(domain),
        "boundary_cells": _cell_list(outer_boundary),
        "obstacle_cells": _cell_list(obstacles),
        "start_cell": list(start),
        "components": serialized_components,
        "events": events,
        "event_count": len(events),
        "cut_count": len(cut),
        "transfer_count": sum(event["action"] == "transfer" for event in events),
        "release_count": sum(event["action"] == "release" for event in events),
        "coverage": round(len(cut) / len(domain), 8),
        "uncovered_cells": _cell_list(uncovered),
        "active_component_ids": sorted(active_ids),
        "released_component_ids": sorted(released_ids),
        "failure_reason": (
            None if not uncovered
            else f"No dynamic-frontier candidate remains; {len(uncovered)} cells are uncovered."
        ),
    }

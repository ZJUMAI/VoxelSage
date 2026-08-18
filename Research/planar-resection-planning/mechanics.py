"""Deterministic 2.5D tissue mechanics for the planar sandbox.

The browser chooses cells on a 2D resection plane.  The mechanics model gives
each cell an upper and lower 3D node, solves small-deformation equilibrium,
and projects normal tension, shear, organ energy, and vessel strain back to
the 2D grid.  Cutting opens the local upper/lower cohesive connection and
removes cross-surface links that would bridge an already opened interface.

This is a dimensionless research surrogate, not a calibrated clinical solver.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from planner import Cell


DEFAULT_MECHANICS = {
    "organ_stiffness": 1.0,
    "vessel_stiffness": 3.0,
    "attachment_stiffness": 1.5,
    "diagonal_factor": 0.5,
    "normal_stiffness": 1.0,
    "shear_stiffness": 0.5,
    "support_stiffness": 0.35,
    "prestrain": 0.08,
    "lateral_prestrain": 0.03,
    "thickness_min": 0.6,
    "thickness_max": 1.8,
    "thickness_gamma": 1.4,
    "safe_vessel_strain": 0.12,
    "tear_vessel_strain": 0.25,
}

EDGE_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DOF_PER_NODE = 3


def _cell(value: Sequence[int]) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Cell must be [row, column], got {value!r}")
    return int(value[0]), int(value[1])


def _cell_list(cells: Iterable[Cell]) -> List[List[int]]:
    return [[row, col] for row, col in sorted(cells)]


def _parameters(values: Mapping[str, float] | None) -> Dict[str, float]:
    result = dict(DEFAULT_MECHANICS)
    if values:
        unknown = set(values) - set(result)
        if unknown:
            raise ValueError(f"Unknown mechanics parameters: {sorted(unknown)}")
        for name, raw in values.items():
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Mechanics parameter '{name}' must be finite and non-negative")
            result[name] = value
    for name in (
        "organ_stiffness", "vessel_stiffness", "attachment_stiffness",
        "normal_stiffness", "shear_stiffness", "support_stiffness",
    ):
        if result[name] <= 0:
            raise ValueError(f"Mechanics parameter '{name}' must be positive")
    if not 0 <= result["prestrain"] < 1:
        raise ValueError("prestrain must be in [0, 1)")
    if not 0 <= result["lateral_prestrain"] < 1:
        raise ValueError("lateral_prestrain must be in [0, 1)")
    if result["thickness_min"] <= 0 or result["thickness_max"] < result["thickness_min"]:
        raise ValueError("thickness_max must be at least thickness_min, and both must be positive")
    if result["thickness_gamma"] <= 0:
        raise ValueError("thickness_gamma must be positive")
    if result["tear_vessel_strain"] <= result["safe_vessel_strain"]:
        raise ValueError("tear_vessel_strain must be greater than safe_vessel_strain")
    return result


def _edge_stiffness(first: Cell, second: Cell, vessels: Set[Cell], diagonal: bool, p: Mapping[str, float]) -> float:
    if first in vessels and second in vessels:
        stiffness = p["vessel_stiffness"]
    elif first in vessels or second in vessels:
        stiffness = p["attachment_stiffness"]
    else:
        stiffness = p["organ_stiffness"]
    return float(stiffness * (p["diagonal_factor"] if diagonal else 1.0))


def _build_edges(active: Set[Cell], vessels: Set[Cell], p: Mapping[str, float]):
    edges = []
    for first in sorted(active):
        for dr, dc in EDGE_OFFSETS:
            second = first[0] + dr, first[1] + dc
            if second in active:
                edges.append((first, second, _edge_stiffness(first, second, vessels, dr != 0 and dc != 0, p)))
    return edges


def _thickness_field(domain: Set[Cell], p: Mapping[str, float]) -> Dict[Cell, float]:
    boundary = {
        cell for cell in domain
        if any((cell[0] + dr, cell[1] + dc) not in domain for dr, dc in CARDINAL_OFFSETS)
    }
    depth = {cell: 0 for cell in boundary}
    queue = deque(boundary)
    while queue:
        cell = queue.popleft()
        for dr, dc in CARDINAL_OFFSETS:
            neighbor = cell[0] + dr, cell[1] + dc
            if neighbor in domain and neighbor not in depth:
                depth[neighbor] = depth[cell] + 1
                queue.append(neighbor)
    max_depth = max(depth.values(), default=0)
    return {
        cell: p["thickness_min"] + (p["thickness_max"] - p["thickness_min"])
        * ((depth.get(cell, 0) / max(1, max_depth)) ** p["thickness_gamma"])
        for cell in domain
    }


def _node_index(cell_index: Mapping[Cell, int], cell: Cell, side: int) -> int:
    return 2 * cell_index[cell] + side


def _dof(node: int, component: int) -> int:
    return DOF_PER_NODE * node + component


def _add_vector_spring(
    matrix: Dict[Tuple[int, int], float],
    first_node: int,
    second_node: int,
    stiffness: float,
    rest_error: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """Add an isotropic small-deformation spring with initial rest error."""
    for component in range(DOF_PER_NODE):
        first = _dof(first_node, component)
        second = _dof(second_node, component)
        matrix[(first, first)] = matrix.get((first, first), 0.0) + stiffness
        matrix[(second, second)] = matrix.get((second, second), 0.0) + stiffness
        matrix[(first, second)] = matrix.get((first, second), 0.0) - stiffness
        matrix[(second, first)] = matrix.get((second, first), 0.0) - stiffness
        rhs[first] += stiffness * rest_error[component]
        rhs[second] -= stiffness * rest_error[component]


def _initial_position(cell: Cell, side: int, thickness: Mapping[Cell, float]) -> np.ndarray:
    row, col = cell
    z = thickness[cell] / 2.0 if side == 0 else -thickness[cell] / 2.0
    return np.asarray([float(col), float(row), z], dtype=float)


def _front_and_tips(domain: Set[Cell], cut: Set[Cell]) -> Tuple[Set[Cell], Set[Cell]]:
    front = {
        cell for cell in domain - cut
        if any((cell[0] + dr, cell[1] + dc) in cut for dr, dc in CARDINAL_OFFSETS)
    }
    cut_tips = {
        cell for cell in cut
        if sum((cell[0] + dr, cell[1] + dc) in cut for dr, dc in CARDINAL_OFFSETS) <= 1
    }
    return front, cut_tips


def solve_tension(
    *,
    rows: int,
    cols: int,
    domain_cells: Sequence[Sequence[int]],
    vessel_cells: Sequence[Sequence[int]] = (),
    cut_cells: Sequence[Sequence[int]] = (),
    parameters: Mapping[str, float] | None = None,
    anchor_cells: Sequence[Sequence[int]] = (),
    tractions: Sequence[Mapping[str, Sequence[float]]] = (),
) -> Dict[str, object]:
    """Solve one 2.5D quasi-static state without user-applied forces."""
    if anchor_cells or tractions:
        raise ValueError("2.5D model does not accept anchor_cells or tractions; use vessel_cells and cut_cells only")
    rows, cols = int(rows), int(cols)
    if not 1 <= rows <= 50 or not 1 <= cols <= 50:
        raise ValueError("rows and cols must both be between 1 and 50")
    domain = {_cell(cell) for cell in domain_cells}
    vessels = {_cell(cell) for cell in vessel_cells}
    cut = {_cell(cell) for cell in cut_cells}
    if not domain:
        raise ValueError("domain_cells cannot be empty")
    if any(not (0 <= row < rows and 0 <= col < cols) for row, col in domain):
        raise ValueError("domain_cells contains a cell outside rows/cols")
    if not vessels <= domain or not cut <= domain:
        raise ValueError("Vessel and cut cells must be inside the domain")
    if cut & vessels:
        raise ValueError("Vessel cells cannot be cut in the tension sandbox")
    p = _parameters(parameters)
    thickness = _thickness_field(domain, p)
    cells = sorted(domain)
    cell_index = {cell: index for index, cell in enumerate(cells)}
    front_cells, tip_cells = _front_and_tips(domain, cut)
    node_count = 2 * len(cells)
    total_dofs = node_count * DOF_PER_NODE
    matrix_values: Dict[Tuple[int, int], float] = {}
    rhs = np.zeros(total_dofs, dtype=float)
    boundary = {
        cell for cell in domain
        if any((cell[0] + dr, cell[1] + dc) not in domain for dr, dc in CARDINAL_OFFSETS)
    }
    springs: List[Dict[str, object]] = []

    def add_spring(first: Cell, second: Cell, first_side: int, second_side: int,
                   stiffness: float, kind: str, rest_scale: float = 1.0) -> None:
        first_position = _initial_position(first, first_side, thickness)
        second_position = _initial_position(second, second_side, thickness)
        initial_vector = second_position - first_position
        rest_vector = initial_vector * rest_scale
        rest_error = initial_vector - rest_vector
        first_node = _node_index(cell_index, first, first_side)
        second_node = _node_index(cell_index, second, second_side)
        _add_vector_spring(matrix_values, first_node, second_node, stiffness, rest_error, rhs)
        springs.append({
            "from": first, "to": second, "first_side": first_side, "second_side": second_side,
            "stiffness": stiffness, "kind": kind, "rest_vector": rest_vector,
            "initial_vector": initial_vector,
        })

    for first, second, stiffness in _build_edges(domain, vessels, p):
        # Upper and lower surface material remains connected after a cut.
        add_spring(first, second, 0, 0, stiffness, "lateral", 1.0 - p["lateral_prestrain"])
        add_spring(first, second, 1, 1, stiffness, "lateral", 1.0 - p["lateral_prestrain"])
        # These oblique links carry cross-surface shear/bending. They are
        # removed if they would bridge an opened cut interface.
        if first not in cut and second not in cut:
            cross_stiffness = p["shear_stiffness"] * (p["diagonal_factor"] if first[0] != second[0] and first[1] != second[1] else 1.0)
            add_spring(first, second, 0, 1, cross_stiffness, "cross", 1.0 - p["lateral_prestrain"])
            add_spring(first, second, 1, 0, cross_stiffness, "cross", 1.0 - p["lateral_prestrain"])

    for cell in cells:
        if cell not in cut:
            add_spring(cell, cell, 0, 1, p["normal_stiffness"] / thickness[cell], "normal", 1.0 - p["prestrain"])
        if cell in boundary:
            for side in (0, 1):
                node = _node_index(cell_index, cell, side)
                for component in range(DOF_PER_NODE):
                    matrix_values[(_dof(node, component), _dof(node, component))] = (
                        matrix_values.get((_dof(node, component), _dof(node, component)), 0.0)
                        + p["support_stiffness"]
                    )

    rows_index, cols_index, values = zip(*((row, col, value) for (row, col), value in matrix_values.items()))
    matrix = coo_matrix((values, (rows_index, cols_index)), shape=(total_dofs, total_dofs)).tocsr()
    try:
        displacement = spsolve(matrix, rhs)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("2.5D mechanical equilibrium could not be solved") from exc
    if not np.all(np.isfinite(displacement)):
        raise ValueError("2.5D mechanical equilibrium produced non-finite displacement")

    organ_energy = {cell: 0.0 for cell in cells}
    vessel_strain = {cell: 0.0 for cell in vessels}
    normal_tension = {cell: 0.0 for cell in cells}
    normal_strain = {cell: 0.0 for cell in cells}
    normal_opening = {cell: 0.0 for cell in cells}
    shear_tension = {cell: 0.0 for cell in cells}
    shear_strain = {cell: 0.0 for cell in cells}
    interface_tension = {cell: 0.0 for cell in cells}
    serialized_springs: List[Dict[str, object]] = []

    for spring in springs:
        first = spring["from"]
        second = spring["to"]
        assert isinstance(first, tuple) and isinstance(second, tuple)
        first_side = int(spring["first_side"])
        second_side = int(spring["second_side"])
        first_node = _node_index(cell_index, first, first_side)
        second_node = _node_index(cell_index, second, second_side)
        first_u = displacement[first_node * DOF_PER_NODE:(first_node + 1) * DOF_PER_NODE]
        second_u = displacement[second_node * DOF_PER_NODE:(second_node + 1) * DOF_PER_NODE]
        delta = second_u - first_u
        initial_vector = np.asarray(spring["initial_vector"], dtype=float)
        rest_vector = np.asarray(spring["rest_vector"], dtype=float)
        extension_vector = initial_vector + delta - rest_vector
        rest_length = max(float(np.linalg.norm(rest_vector)), 1e-12)
        strain = max(0.0, float(np.linalg.norm(initial_vector + delta) - np.linalg.norm(rest_vector)) / rest_length)
        tension = float(spring["stiffness"]) * strain
        energy = 0.5 * float(spring["stiffness"]) * float(np.dot(extension_vector, extension_vector))
        organ_energy[first] += 0.5 * energy
        organ_energy[second] += 0.5 * energy
        vessel_edge = first in vessels or second in vessels
        if vessel_edge:
            vessel_strain[first] = max(vessel_strain.get(first, 0.0), strain)
            vessel_strain[second] = max(vessel_strain.get(second, 0.0), strain)
        kind = str(spring["kind"])
        if kind == "normal":
            normal_strain[first] = strain
            normal_tension[first] = tension
            normal_opening[first] = max(0.0, float(delta[2]))
        elif kind == "cross":
            tangential = float(np.linalg.norm(delta[:2]))
            tangential_tension = float(spring["stiffness"]) * tangential / rest_length
            shear_strain[first] = max(shear_strain[first], tangential / rest_length)
            shear_strain[second] = max(shear_strain[second], tangential / rest_length)
            shear_tension[first] = max(shear_tension[first], tangential_tension)
            shear_tension[second] = max(shear_tension[second], tangential_tension)
        elif kind == "lateral" and (first in cut) != (second in cut):
            interface_tension[first] = max(interface_tension[first], tension)
            interface_tension[second] = max(interface_tension[second], tension)
        serialized_springs.append({
            "from": list(first), "to": list(second), "kind": kind,
            "side": [first_side, second_side], "strain": round(strain, 8),
            "tension": round(tension, 8), "vessel_edge": vessel_edge,
        })

    front_tension = {
        cell: math.sqrt(normal_tension[cell] ** 2 + shear_tension[cell] ** 2 + interface_tension[cell] ** 2)
        if cell in front_cells else 0.0
        for cell in cells
    }
    peak_vessel_strain = max(vessel_strain.values(), default=0.0)
    vessel_status = (
        "tear-risk" if peak_vessel_strain >= p["tear_vessel_strain"] else
        "warning" if peak_vessel_strain >= p["safe_vessel_strain"] else "safe"
    )
    organ_values = [organ_energy[cell] for cell in cells if cell not in vessels]
    count = max(1, math.ceil(len(organ_values) * 0.05)) if organ_values else 0
    cvar = sum(sorted(organ_values, reverse=True)[:count]) / count if count else 0.0
    serialized_cells = []
    for cell in cells:
        node_displacements = []
        for side in (0, 1):
            node = _node_index(cell_index, cell, side)
            node_displacements.append([
                round(float(value), 8)
                for value in displacement[node * DOF_PER_NODE:(node + 1) * DOF_PER_NODE]
            ])
        serialized_cells.append({
            "cell": list(cell), "thickness": round(thickness[cell], 8),
            "normal_tension": round(normal_tension[cell], 8),
            "normal_strain": round(normal_strain[cell], 8),
            "normal_opening": round(normal_opening[cell], 8),
            "shear_tension": round(shear_tension[cell], 8),
            "shear_strain": round(shear_strain[cell], 8),
            "interface_tension": round(interface_tension[cell], 8),
            "front_tension": round(front_tension[cell], 8),
            "organ_energy": round(organ_energy[cell], 8),
            "vessel_strain": round(vessel_strain.get(cell, 0.0), 8),
            "is_vessel": cell in vessels, "is_cut": cell in cut,
            "is_front": cell in front_cells, "is_tip": cell in tip_cells,
            "displacement": node_displacements,
        })
    return {
        "status": "ok", "model": "2.5d-front-tension-v2",
        "rows": rows, "cols": cols,
        "domain_cells": _cell_list(domain), "vessel_cells": _cell_list(vessels), "cut_cells": _cell_list(cut),
        "front_cells": _cell_list(front_cells), "tip_cells": _cell_list(tip_cells),
        "parameters": p, "thickness": {f"{r},{c}": round(value, 8) for (r, c), value in thickness.items()},
        "active_cell_count": len(domain), "spring_count": len(springs),
        "unstable_cell_count": 0, "unstable_cells": [],
        "max_displacement": round(float(np.max(np.abs(displacement))), 8),
        "peak_normal_tension": round(max(normal_tension.values(), default=0.0), 8),
        "peak_shear_tension": round(max(shear_tension.values(), default=0.0), 8),
        "peak_front_tension": round(max(front_tension.values(), default=0.0), 8),
        "peak_normal_opening": round(max(normal_opening.values(), default=0.0), 8),
        "peak_vessel_strain": round(peak_vessel_strain, 8),
        "peak_vessel_tension": round(max((spring["tension"] for spring in serialized_springs if spring["vessel_edge"]), default=0.0), 8),
        "vessel_status": vessel_status,
        "peak_organ_energy": round(max(organ_values, default=0.0), 8),
        "organ_cvar95_energy": round(cvar, 8),
        "cells": serialized_cells, "springs": serialized_springs,
    }

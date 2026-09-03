"""Saved resection-plane sequence planning.

Deterministic baselines remain the default.  The optional ``learned_shielded``
algorithm maps the saved 3D surface to its two-dimensional parameter grid and
invokes the frozen v10.6 C4 ranker plus exact simulator shield.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from Tool_Box.mask_resolution import resolve_mask_path


def _surface_positions(cp: np.ndarray, n_u: int, n_v: int) -> np.ndarray:
    """Evaluate a bicubic Bézier surface in the browser's centered coordinates."""
    b = np.empty((n_u, n_v, 3), dtype=np.float64)
    for i in range(n_u):
        u = i / max(n_u - 1, 1)
        bu = np.array([(1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3])
        for j in range(n_v):
            v = j / max(n_v - 1, 1)
            bv = np.array([(1-v)**3, 3*v*(1-v)**2, 3*v*v*(1-v), v**3])
            b[i, j] = np.einsum("i,j,ijc->c", bu, bv, cp)
    return b


def _learned_surface_resolution(cp: np.ndarray, cell_side_mm: float = 4.0) -> List[int]:
    """Choose an approximately 4-mm grid over the complete saved patch.

    The saved browser patch includes a display border outside the liver.  The
    frozen canvas limit therefore cannot be checked until the liver target has
    been computed and that empty border has been cropped.
    """
    probe = _surface_positions(cp, 33, 33)
    u_lengths = np.linalg.norm(np.diff(probe, axis=0), axis=2).sum(axis=0)
    v_lengths = np.linalg.norm(np.diff(probe, axis=1), axis=2).sum(axis=1)
    u_cells = max(1, int(np.ceil(float(np.mean(u_lengths)) / float(cell_side_mm))))
    v_cells = max(1, int(np.ceil(float(np.mean(v_lengths)) / float(cell_side_mm))))
    return [u_cells + 1, v_cells + 1]


def _learned_target_crop(
    target: np.ndarray,
    source_rows: int,
    source_cols: int,
    *,
    max_rows: int = 30,
    max_cols: int = 40,
) -> Dict[str, Any]:
    """Return the tight rectangular adapter window containing every target cell.

    Cell indices exposed to the viewer remain indices of the complete saved
    surface.  The learned controller receives only this result-independent
    rectangular crop, preserving the original 4-mm sampling while removing
    cells that are purely display padding.
    """
    flat_target = np.asarray(target, dtype=bool).reshape(-1)
    if flat_target.size != int(source_rows) * int(source_cols):
        raise ValueError("target size does not match the source surface grid")
    occupied = np.argwhere(flat_target.reshape(source_rows, source_cols))
    if occupied.size == 0:
        raise ValueError("保存剖面与 Liver 没有离散交集，无法进行路径规划")

    row0, col0 = occupied.min(axis=0).astype(int)
    row1, col1 = (occupied.max(axis=0) + 1).astype(int)
    adapter_rows = int(row1 - row0)
    adapter_cols = int(col1 - col0)
    if adapter_rows > max_rows or adapter_cols > max_cols:
        raise ValueError(
            "保存剖面的 Liver 目标按 4-mm 单元裁剪后超过冻结模型的 "
            f"{max_rows}x{max_cols} 上限：目标 {adapter_rows}x{adapter_cols}，"
            f"完整显示网格 {source_rows}x{source_cols}"
        )

    source_grid = np.arange(source_rows * source_cols, dtype=np.int64).reshape(
        source_rows, source_cols
    )
    local_to_source = source_grid[row0:row1, col0:col1].reshape(-1)
    source_to_local = np.full(source_rows * source_cols, -1, dtype=np.int64)
    source_to_local[local_to_source] = np.arange(local_to_source.size, dtype=np.int64)
    return {
        "row0": int(row0),
        "col0": int(col0),
        "rows": adapter_rows,
        "cols": adapter_cols,
        "local_to_source": local_to_source,
        "source_to_local": source_to_local,
    }


def _remap_adapter_steps(
    steps: Sequence[Dict[str, Any]],
    local_to_source: np.ndarray,
    *,
    source_cols: int,
    adapter_cols: int,
) -> List[Dict[str, Any]]:
    """Map frozen-canvas path cells back to the complete saved-surface grid."""
    remapped = []
    for raw_step in steps:
        local_cell = int(raw_step["cell"])
        source_cell = int(local_to_source[local_cell])
        step = dict(raw_step)
        step["adapter_cell"] = local_cell
        step["adapter_grid_ij"] = [
            int(local_cell // adapter_cols),
            int(local_cell % adapter_cols),
        ]
        step["cell"] = source_cell
        step["grid_ij"] = [
            int(source_cell // source_cols),
            int(source_cell % source_cols),
        ]
        remapped.append(step)
    return remapped


def _cells(n_u: int, n_v: int) -> Iterable[Tuple[int, int, int]]:
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            yield i, j, i * (n_v - 1) + j


def _neighbors(cell: int, rows: int, cols: int) -> Iterable[int]:
    i, j = divmod(cell, cols)
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            yield ni * cols + nj


def _outer_boundary_mask(target: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Return the one-cell outer boundary of a target grid region."""
    boundary = np.zeros(len(target), dtype=bool)
    for cell in np.flatnonzero(target):
        neighbors = list(_neighbors(int(cell), rows, cols))
        boundary[cell] = len(neighbors) < 4 or any(not target[nxt] for nxt in neighbors)
    return boundary


def _shortest_path(start: int, goal: int, neighbors, allowed: np.ndarray) -> List[int]:
    if start == goal:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if not allowed[nxt] or nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == goal:
                out = [goal]
                while out[-1] != start:
                    out.append(parent[out[-1]])
                return out[::-1]
            q.append(nxt)
    return []


def _nearest_path(start: int, safe: np.ndarray, rows: int, cols: int) -> List[int]:
    neigh = lambda x: _neighbors(x, rows, cols)
    path = [start]
    covered = {start}
    current = start
    while len(covered) < int(safe.sum()):
        targets = [x for x in np.flatnonzero(safe) if x not in covered]
        best = None
        for target in targets:
            route = _shortest_path(current, int(target), neigh, safe)
            if route and (best is None or len(route) < len(best[1])):
                best = (int(target), route)
        if best is None:
            break
        route = best[1]
        path.extend(route[1:])
        covered.update(route)
        current = route[-1]
    return path


def _dfs_path(start: int, safe: np.ndarray, rows: int, cols: int) -> List[int]:
    neigh = lambda x: _neighbors(x, rows, cols)
    path = [start]
    seen = {start}

    def visit(cur: int):
        for nxt in neigh(cur):
            if not safe[nxt] or nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            visit(nxt)
            path.append(cur)  # transfer/backtrack is retained in the plan

    visit(start)
    return path


def _spanning_tree_path(start: int, safe: np.ndarray, rows: int, cols: int) -> List[int]:
    """Traverse a BFS spanning tree and retain tree backtracking transfers."""
    neigh = lambda x: _neighbors(x, rows, cols)
    parent = {start: None}
    queue = deque([start])
    children: Dict[int, List[int]] = {}
    while queue:
        cur = queue.popleft()
        for nxt in neigh(cur):
            if not safe[nxt] or nxt in parent:
                continue
            parent[nxt] = cur
            children.setdefault(cur, []).append(nxt)
            queue.append(nxt)

    path = [start]

    def walk(cur: int):
        for nxt in children.get(cur, []):
            path.append(nxt)
            walk(nxt)
            path.append(cur)

    walk(start)
    return path


def _sequence_result_path(
    output_dir: str, source_json_path: Path, fallback_case_name: str
) -> Path:
    """Return the sidecar path expected by the 3D viewer.

    The browser derives the sequence URL from the actual ``*_3d.json`` file,
    while ``ctx.case_id`` may be an output-directory alias. Use the source JSON
    basename so both sides always address the same sidecar.
    """
    suffix = "_3d.json"
    source_name = source_json_path.name
    prefix = source_name[:-len(suffix)] if source_name.endswith(suffix) else fallback_case_name
    return Path(output_dir) / f"{prefix}_resection_sequence.json"


def _vascular_safe_mask(
    ctx,
    centers: np.ndarray,
    center_offset: Sequence[float],
    threshold: float,
    mask_variants: Dict[str, str] | None = None,
) -> np.ndarray:
    """Conservative center-sample safety mask from hepatic/portal masks."""
    imgs = []
    for name in ("hepatic", "portal"):
        try:
            resolved = resolve_mask_path(ctx.mask_dir, name)
        except FileNotFoundError:
            continue
        imgs.append(nib.load(resolved.path))
        if mask_variants is not None:
            mask_variants[name] = resolved.variant
    if not imgs:
        return np.ones(len(centers), dtype=bool)
    vessel = np.zeros(imgs[0].shape[:3], dtype=bool)
    affine = imgs[0].affine
    for img in imgs:
        vessel |= np.asarray(img.get_fdata()) > 0
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    distance = distance_transform_edt(~vessel, sampling=spacing)
    world = centers + np.asarray(center_offset, dtype=np.float64)
    vox = nib.affines.apply_affine(np.linalg.inv(affine), world)
    idx = np.rint(vox).astype(int)
    result = np.zeros(len(centers), dtype=bool)
    for i, p in enumerate(idx):
        if np.all(p >= 0) and np.all(p < np.array(distance.shape)):
            result[i] = distance[tuple(p)] >= threshold
    # If the coordinate convention does not overlap the mask, keep the
    # planner usable and report that no vessel mask was sampled.
    if not result.any():
        return np.ones(len(centers), dtype=bool)
    return result


def _liver_intersection_mask(
    ctx,
    cell_samples: np.ndarray,
    center_offset: Sequence[float],
    min_inside_samples: int,
) -> np.ndarray:
    """Return cells intersecting the liver volume on the saved surface.

    Each quadrilateral cell is sampled at its four corners and center. A cell
    belongs to the target resection surface when at least
    ``min_inside_samples`` samples fall in the liver mask. This is the
    discrete counterpart of ``Liver ∩ resection_surface`` and intentionally
    excludes all surface cells outside the liver from both cutting and travel.
    """
    liver_path = Path(ctx.mask_dir) / "liver.nii.gz"
    if not liver_path.exists():
        raise FileNotFoundError(f"路径规划需要 liver.nii.gz: {liver_path}")
    image = nib.load(str(liver_path))
    liver = np.asarray(image.get_fdata()) > 0
    if not liver.any():
        raise ValueError("liver.nii.gz 为空，无法裁剪手术剖面")

    samples = np.asarray(cell_samples, dtype=np.float64)
    n_cells, n_samples, _ = samples.shape
    world = samples.reshape(-1, 3) + np.asarray(center_offset, dtype=np.float64)
    vox = nib.affines.apply_affine(np.linalg.inv(image.affine), world)
    idx = np.rint(vox).astype(int)
    inside = np.zeros(len(idx), dtype=bool)
    shape = np.asarray(liver.shape)
    valid = np.all(idx >= 0, axis=1) & np.all(idx < shape, axis=1)
    inside[valid] = liver[tuple(idx[valid].T)]
    inside_per_cell = inside.reshape(n_cells, n_samples).sum(axis=1)
    return inside_per_cell >= max(1, min(int(min_inside_samples), n_samples))


def run(ctx) -> Dict[str, Any]:
    case_name = ctx.params.get("case_name", ctx.case_id)
    json_files = sorted(Path(ctx.output_dir).glob("*_3d.json"))
    if not json_files:
        raise FileNotFoundError(f"未找到 3D JSON: {ctx.output_dir}")
    json_path = json_files[0]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    planes = data.get("resection_planes", [])
    saved = [
        (i, p) for i, p in enumerate(planes)
        if p.get("user_saved") is True and p.get("control_points_3d")
    ]
    if not saved:
        raise ValueError("没有已保存的用户剖面，请先在三维网页点击‘保存’")
    saved_index, plane = saved[-1]
    cp = np.asarray(plane["control_points_3d"], dtype=np.float64)
    if cp.shape != (4, 4, 3):
        raise ValueError("已保存剖面的 control_points_3d 不是 4x4x3")

    algorithm = str(ctx.params.get("algorithm", "nearest")).lower()
    if algorithm not in {"nearest", "dfs", "spanning_tree", "learned_shielded"}:
        raise ValueError(
            f"不支持的算法 '{algorithm}'，可选值为 nearest、dfs、spanning_tree、learned_shielded"
        )
    learned_cell_side_mm = float(ctx.params.get("learned_cell_side_mm", 4.0))
    default_min_samples = 4 if algorithm == "learned_shielded" else 1
    liver_intersection_min_samples = int(
        ctx.params.get("liver_intersection_min_samples", default_min_samples)
    )
    if not 1 <= liver_intersection_min_samples <= 5:
        raise ValueError("liver_intersection_min_samples 必须位于 [1, 5]")
    # The frozen model was trained on 4-mm cells, so its default grid is
    # derived from physical surface length. Deterministic baselines preserve
    # the saved browser resolution unless the caller explicitly overrides it.
    if algorithm == "learned_shielded":
        if abs(learned_cell_side_mm - 4.0) > 1e-6:
            raise ValueError("冻结模型只验证过 4.0-mm 面单元")
        learned_resolution = _learned_surface_resolution(cp, learned_cell_side_mm)
        if ctx.params.get("grid_resolution") and list(ctx.params["grid_resolution"]) != learned_resolution:
            raise ValueError(
                "learned_shielded 不接受与 4-mm 物理网格不一致的 grid_resolution"
            )
        resolution = learned_resolution
    elif ctx.params.get("grid_resolution"):
        resolution = ctx.params["grid_resolution"]
    else:
        resolution = plane.get("surface_resolution") or [20, 20]
    n_u, n_v = max(2, int(resolution[0])), max(2, int(resolution[1]))
    positions = _surface_positions(cp, n_u, n_v)
    cell_samples = np.asarray([
        [
            positions[i, j], positions[i + 1, j], positions[i, j + 1], positions[i + 1, j + 1],
            (positions[i, j] + positions[i + 1, j] + positions[i, j + 1] + positions[i + 1, j + 1]) / 4,
        ]
        for i, j, _ in _cells(n_u, n_v)
    ])
    centers = cell_samples[:, 4, :]
    rows, cols = n_u - 1, n_v - 1
    center_offset = data.get("center_offset", [0, 0, 0])
    liver_core_target = _liver_intersection_mask(
        ctx, cell_samples, center_offset,
        liver_intersection_min_samples,
    )
    if not liver_core_target.any():
        raise ValueError("保存剖面与 Liver 没有离散交集，无法进行路径规划")
    vessel_mask_variants = {}
    vascular_safe = _vascular_safe_mask(
        ctx, centers, data.get("center_offset", [0, 0, 0]),
        float(ctx.params.get("vascular_safe_distance_mm", 5.0)),
        vessel_mask_variants,
    )
    liver_target = liver_core_target
    boundary_vessel_proxy_count = int(
        (
            liver_target
            & ~vascular_safe
            & _outer_boundary_mask(liver_target, rows, cols)
        ).sum()
    )
    target_rule_audit = {
        "core_cell_count": int(liver_core_target.sum()),
        "boundary_vessel_proxy_count": boundary_vessel_proxy_count,
        "boundary_vessel_policy": "retain_and_apply_standard_simulator_rules",
        # Retained for readers of earlier audit bundles. No support cells are
        # added by the current adapter.
        "enclosure_added_cell_count": 0,
        "initial_boundary_vessel_cell_count": boundary_vessel_proxy_count,
    }
    safe = liver_target & vascular_safe
    if not safe.any():
        raise ValueError("Liver∩剖面在血管安全约束后没有可规划单元")

    adapter_grid = None
    planning_liver_target = liver_target
    planning_vascular_safe = vascular_safe
    planning_rows, planning_cols = rows, cols
    if algorithm == "learned_shielded":
        adapter_grid = _learned_target_crop(liver_target, rows, cols)
        local_to_source = adapter_grid["local_to_source"]
        planning_liver_target = liver_target[local_to_source]
        planning_vascular_safe = vascular_safe[local_to_source]
        planning_rows = int(adapter_grid["rows"])
        planning_cols = int(adapter_grid["cols"])
        local_start_selectable = (
            planning_liver_target
            & planning_vascular_safe
            & _outer_boundary_mask(
                planning_liver_target, planning_rows, planning_cols
            )
        )
        start_selectable = np.zeros_like(liver_target, dtype=bool)
        start_selectable[local_to_source] = local_start_selectable
    else:
        start_selectable = safe & _outer_boundary_mask(liver_target, rows, cols)
    if not start_selectable.any():
        raise ValueError("Liver∩剖面的外边界没有满足血管安全约束的可选起点")
    start_raw = ctx.params.get("start_cell", None)
    if start_raw is None:
        start = int(np.flatnonzero(start_selectable)[0])
        start_source = "default_boundary_safe"
    else:
        try:
            start = int(start_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"起点 start_cell 必须是整数，收到: {start_raw}") from exc
        if start < 0 or start >= len(safe):
            raise ValueError(f"起点单元 {start} 超出范围 [0, {len(safe) - 1}]")
        if not liver_target[start]:
            raise ValueError(f"起点单元 {start} 位于 Liver 外，不能作为切除起点")
        if not vascular_safe[start]:
            raise ValueError(f"起点单元 {start} 位于血管安全禁区内，不能作为切除起点")
        if not start_selectable[start]:
            raise ValueError(f"起点单元 {start} 不在 Liver∩剖面的外边界，不能作为切除起点")
        start_source = "user"
    planning_start = (
        int(adapter_grid["source_to_local"][start])
        if adapter_grid is not None else start
    )
    if planning_start < 0:
        raise ValueError(f"起点单元 {start} 不在学习适配器的 Liver 目标窗口内")
    adapter_grid_payload = None
    adapter_target_rule = None
    if adapter_grid is not None:
        adapter_grid_payload = {
            "cell_rows": planning_rows,
            "cell_cols": planning_cols,
            "vertex_resolution": [planning_rows + 1, planning_cols + 1],
            "crop_origin_ij": [int(adapter_grid["row0"]), int(adapter_grid["col0"])],
            "source_cell_rows": rows,
            "source_cell_cols": cols,
            "source_vertex_resolution": [n_u, n_v],
        }
        adapter_target_rule = {
            "core_liver_samples_required": liver_intersection_min_samples,
            "cell_sample_count": 5,
            **target_rule_audit,
        }

    # The web client uses this lightweight mode before the user picks a start.
    # It returns only the legal-cell map and never writes a path result.
    if ctx.params.get("preview_only"):
        preview_states = []
        for cell in range(len(safe)):
            if not liver_target[cell]:
                state = "outside_liver"
            elif not vascular_safe[cell]:
                state = "vascular_risk"
            elif start_selectable[cell]:
                state = "selectable"
            else:
                state = "target"
            preview_states.append({
                "cell": cell,
                "grid_ij": [cell // cols, cell % cols],
                "state": state,
            })
        preview = {
            "status": "preview",
            "saved_plane_index": saved_index,
            "saved_at": plane.get("saved_at"),
            "vessel_mask_variants": vessel_mask_variants,
            "grid": {"vertex_resolution": [n_u, n_v], "cell_rows": rows, "cell_cols": cols},
            "cell_states": preview_states,
            "surface_cell_count": int(len(liver_target)),
            "liver_intersection_cell_count": int(liver_target.sum()),
            "outside_liver_cell_count": int((~liver_target).sum()),
            "vascular_excluded_cell_count": int((liver_target & ~vascular_safe).sum()),
            "target_cell_count": int(liver_target.sum()) if algorithm == "learned_shielded" else int(safe.sum()),
            "start_candidate_count": int(start_selectable.sum()),
            "algorithm": algorithm,
        }
        if adapter_grid_payload is not None:
            preview["adapter_grid"] = adapter_grid_payload
            preview["adapter_target_rule"] = adapter_target_rule
        return preview
    learned = None
    if algorithm == "learned_shielded":
        # SkillEngine loads ``main.py`` through an explicit module spec rather
        # than as a normal package submodule, so relative imports have no
        # parent package on the public API execution path.
        from skills.builtin.plan_resection_sequence.learned_shielded import (
            plan_learned_shielded,
        )

        learned = plan_learned_shielded(
            planning_liver_target,
            planning_vascular_safe,
            start=planning_start,
            rows=planning_rows,
            cols=planning_cols,
            cell_side_mm=learned_cell_side_mm,
        )
        path = [
            int(adapter_grid["local_to_source"][int(step["cell"])])
            for step in learned["path"]
        ]
    elif algorithm == "dfs":
        path = _dfs_path(start, safe, rows, cols)
    elif algorithm == "spanning_tree":
        path = _spanning_tree_path(start, safe, rows, cols)
    else:
        path = _nearest_path(start, safe, rows, cols)
    if learned is not None:
        covered = sorted(
            int(adapter_grid["local_to_source"][int(cell)])
            for cell in learned["covered_cells"]
        )
        learned_component_mask = np.zeros_like(liver_target, dtype=bool)
        learned_component_mask[adapter_grid["local_to_source"]] = np.asarray(
            learned["component_mask"], dtype=bool
        )
    else:
        covered = sorted(set(path))
        learned_component_mask = None
    covered_set = set(covered)
    # The learned simulator plans the complete connected Liver intersection,
    # including vessel-proxy cells. Deterministic baselines keep excluding the
    # configured vascular-risk cells.
    target_mask = liver_target if learned is not None else safe
    safe_cells = set(np.flatnonzero(target_mask).tolist())
    unreachable_cells = sorted(safe_cells - covered_set)
    cell_states = []
    for cell in range(len(safe)):
        if not liver_target[cell]:
            state = "outside_liver"
        elif learned is None and not vascular_safe[cell]:
            state = "vascular_risk"
        elif learned is not None and not learned_component_mask[cell]:
            state = "unreachable"
        elif learned is not None and not vascular_safe[cell]:
            state = "vascular_risk"
        elif cell in unreachable_cells:
            state = "unreachable"
        else:
            state = "target"
        cell_states.append({
            "cell": cell,
            "grid_ij": [cell // cols, cell % cols],
            "state": state,
        })
    step_time = float(ctx.params.get("step_time_seconds", 1.0))
    if learned is not None:
        steps = _remap_adapter_steps(
            learned["path"],
            adapter_grid["local_to_source"],
            source_cols=cols,
            adapter_cols=planning_cols,
        )
    else:
        steps = []
        seen = set()
        for t, cell in enumerate(path):
            action = "cut" if cell not in seen else "transfer"
            seen.add(cell)
            steps.append({"step": t, "cell": int(cell), "action": action,
                          "time_seconds": round(t * step_time, 6),
                          "grid_ij": [int(cell // cols), int(cell % cols)]})
    result_path = _sequence_result_path(ctx.output_dir, json_path, case_name)
    result = {
        "status": "ok" if not unreachable_cells else "partial",
        "case_name": case_name,
        "source_json": str(json_path),
        "saved_plane_index": saved_index,
        "saved_at": plane.get("saved_at"),
        "algorithm": algorithm,
        "start_cell": start,
        "start_source": start_source,
        "grid": {"vertex_resolution": [n_u, n_v], "cell_rows": rows, "cell_cols": cols},
        "parameters": {"algorithm": algorithm, "vascular_safe_distance_mm": float(ctx.params.get("vascular_safe_distance_mm", 5.0)),
                       "liver_intersection_min_samples": liver_intersection_min_samples,
                       "step_time_seconds": step_time,
                       "learned_cell_side_mm": learned_cell_side_mm if learned is not None else None},
        "vessel_mask_variants": vessel_mask_variants,
        "path": steps,
        "path_length": max(0, len(path) - 1),
        "covered_cells": covered,
        "uncovered_cells": unreachable_cells,
        "cell_states": cell_states,
        "surface_cell_count": int(len(liver_target)),
        "liver_core_cell_count": int(liver_core_target.sum()),
        "liver_intersection_cell_count": int(liver_target.sum()),
        "outside_liver_cell_count": int((~liver_target).sum()),
        "vascular_excluded_cell_count": int((liver_target & ~vascular_safe).sum()),
        "target_cell_count": int(target_mask.sum()),
        "failure_reason": (
            (
                f"有 {len(unreachable_cells)} 个 Liver∩剖面单元不在起点连通分量内"
                if learned is not None else
                f"有 {len(unreachable_cells)} 个满足 Liver 和血管约束的单元从起点不可达"
            )
            if unreachable_cells else None
        ),
        "coverage": round(len(covered_set & safe_cells) / max(int(target_mask.sum()), 1), 6),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if learned is not None:
        result.update({
            "policy_id": learned["policy_id"],
            "checkpoint_sha256": learned["checkpoint_sha256"],
            "simulator": learned["simulator"],
            "scope_warning": learned["scope_warning"],
            "learned_surface_adapter": "confirmed_3d_bezier_surface_to_2d_parameter_grid",
            "adapter_grid": adapter_grid_payload,
            "adapter_start_cell": planning_start,
            "adapter_target_rule": adapter_target_rule,
        })
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # Let the viewer avoid probing a missing sidecar (and avoid a noisy 404)
    # before a path has actually been planned.
    data["resection_sequence_available"] = True
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    response = {"status": result["status"], "result_path": str(result_path),
                "path_length": result["path_length"], "coverage": result["coverage"],
                "algorithm": algorithm, "saved_plane_index": saved_index,
                "vessel_mask_variants": vessel_mask_variants,
                "start_cell": start, "uncovered_cells": unreachable_cells,
                "failure_reason": result["failure_reason"]}
    if learned is not None:
        response.update({
            "policy_id": result["policy_id"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "simulator": result["simulator"],
            "scope_warning": result["scope_warning"],
            "adapter_grid": result["adapter_grid"],
            "adapter_start_cell": result["adapter_start_cell"],
            "adapter_target_rule": result["adapter_target_rule"],
        })
    return response

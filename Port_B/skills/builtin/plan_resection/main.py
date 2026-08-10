"""plan_resection: 肝脏手术最优切除剖面生成。

从分割掩码直接计算 Bézier 切除剖面（不依赖 GeoSurge 预计算数据），
并将结果追加到已有的 3D 可视化 JSON 文件中。

使用流程:
  1. 先调用 three_d_reconstruction 生成 3D HTML + JSON
  2. 再调用 plan_resection 追加剖面数据
  3. 刷新 3D HTML 即见剖面叠加
"""

import json
import sys
from pathlib import Path

import numpy as np

from Tool_Box.mask_resolution import resolve_mask_path


def _resolve_vessel_sources(ctx):
    """Log the concrete vessel files selected by the shared resolver."""
    sources = {}
    for name in ("hepatic", "portal"):
        try:
            resolved = resolve_mask_path(ctx.mask_dir, name)
        except FileNotFoundError:
            ctx.log(f"  Vessel mask unavailable: {name}")
            continue
        sources[name] = resolved.variant
        ctx.log(
            f"  Resolved {name} mask: {Path(resolved.path).name} "
            f"({resolved.variant})"
        )
    return sources


def _json_dump_safe(data, path):
    """Write JSON, replacing any NaN/Inf with null so browsers can parse it."""
    import math as _math

    def _sanitize(v):
        if isinstance(v, float):
            if _math.isnan(v) or _math.isinf(v):
                return None
            return v
        if isinstance(v, dict):
            return {k: _sanitize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_sanitize(x) for x in v]
        return v

    safe = _sanitize(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, separators=(",", ":"))


def _load_ijk_from_mask(ctx, mask_name):
    """从 ctx 加载掩码，返回世界坐标点阵 (N,3)。"""
    try:
        mask = ctx.get_mask(mask_name)
        ijk = np.argwhere(mask > 0).astype(np.float64)
        if len(ijk) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        affine = ctx.get_affine()
        import nibabel as nib
        return nib.affines.apply_affine(affine, ijk).astype(np.float64)
    except Exception:
        return np.zeros((0, 3), dtype=np.float64)


def _compute_vertex_distances(surface, tumor_xyz, n_u=20, n_v=20):
    """
    计算曲面上 nU×nV 网格每个顶点到最近肿瘤点的距离。
    返回 (nU, nV) 浮点数组。
    """
    ref = surface["reference_plane"]
    origin = np.asarray(ref["origin_mm"], dtype=np.float64)
    normal = np.asarray(ref["normal_world"], dtype=np.float64)
    u_axis = np.asarray(ref["u_axis_world"], dtype=np.float64)
    v_axis = np.asarray(ref["v_axis_world"], dtype=np.float64)
    u0, u1 = [float(v) for v in ref["u_range_mm"]]
    v0, v1 = [float(v) for v in ref["v_range_mm"]]
    grid = np.asarray(surface["height_control_4x4_mm"], dtype=np.float64)

    if len(tumor_xyz) == 0:
        return np.full((n_u, n_v), 99.9)

    def _bernstein3(t):
        u = 1.0 - t
        return np.stack([u**3, 3*u**2*t, 3*u*t**2, t**3], axis=1)

    # Sample surface vertices (vectorised)
    us = np.linspace(0, 1, n_u)  # (nU,)
    vs = np.linspace(0, 1, n_v)  # (nV,)
    bu = _bernstein3(us)  # (nU, 4)
    bv = _bernstein3(vs)  # (nV, 4)

    # Bézier height: bu[i,:] @ grid @ bv[j,:].T → (nU, nV)
    heights = bu @ grid @ bv.T  # (nU, nV)

    # World coordinates for each (u, v)
    u_world = u0 + us[:, None] * (u1 - u0)  # (nU, 1)
    v_world = v0 + vs[None, :] * (v1 - v0)  # (1, nV)

    # Surface points: origin + u*u_axis + v*v_axis + h*normal
    # Each component broadcasts to (nU, nV, 3)
    pts = (origin[None, None, :]
           + u_world[:, :, None] * u_axis[None, None, :]
           + v_world[:, :, None] * v_axis[None, None, :]
           + heights[:, :, None] * normal[None, None, :])  # (nU, nV, 3)

    # A 4 mm physical grid can contain substantially more than the previous
    # fixed 20x20 vertices.  Querying a KD-tree avoids allocating the former
    # (nU, nV, nTumor, 3) broadcast array.
    from scipy.spatial import cKDTree

    distances = cKDTree(tumor_xyz).query(pts.reshape(-1, 3), k=1)[0]
    distances = distances.reshape(n_u, n_v)

    return distances.tolist()


def _bernstein3_values(t):
    """Return cubic Bernstein basis values at scalar parameter *t*."""
    one_minus_t = 1.0 - t
    return np.array([
        one_minus_t ** 3,
        3.0 * one_minus_t ** 2 * t,
        3.0 * one_minus_t * t ** 2,
        t ** 3,
    ], dtype=np.float64)


def _bernstein3_derivatives(t):
    """Return derivatives of the cubic Bernstein basis at *t*."""
    one_minus_t = 1.0 - t
    return np.array([
        -3.0 * one_minus_t ** 2,
        3.0 * one_minus_t * (1.0 - 3.0 * t),
        3.0 * t * (2.0 - 3.0 * t),
        3.0 * t ** 2,
    ], dtype=np.float64)


def _bezier_interval_transform(start, end):
    """Control-point transform for reparameterising a cubic to [start, end].

    ``start`` and ``end`` may lie outside [0, 1].  In that case this is the
    exact polynomial continuation of the original Bézier patch, rather than a
    geometric scale that would change the planned cutting surface.
    """
    scale = end - start
    return np.stack([
        _bernstein3_values(start),
        _bernstein3_values(start)
        + (scale / 3.0) * _bernstein3_derivatives(start),
        _bernstein3_values(end)
        - (scale / 3.0) * _bernstein3_derivatives(end),
        _bernstein3_values(end),
    ])


def _fit_surface_to_liver_projection(
    surf,
    liver_xyz,
    padding_mm=5.0,
    grid_alignment_mm=None,
):
    """Resize a finite Bézier patch so its liver intersection cannot be clipped.

    A Bézier patch is finite, unlike an analytic plane.  Its four edges must
    therefore sit outside the liver for the liver/surface intersection to form
    closed curve(s).  Enclosing the complete liver projection in the local
    ``u``/``v`` domain (plus a physical padding) is a conservative guarantee:
    no liver point can reach a patch edge.

    When the requested domain lies inside the existing patch, the height grid
    is exactly reparameterised and therefore preserves the original geometry.
    When the patch must grow, polynomial extrapolation is deliberately avoided:
    bicubic control grids fitted on a small local domain can diverge by hundreds
    of millimetres when extrapolated to the full liver.  In that case the same
    bounded control grid is stretched over the larger physical domain.  The
    production planner performs this expansion before refinement, so the final
    optimisation still runs in the authoritative full-liver domain.
    """
    liver_xyz = np.asarray(liver_xyz, dtype=np.float64)
    if liver_xyz.ndim != 2 or liver_xyz.shape[1] != 3 or len(liver_xyz) == 0:
        raise ValueError("liver_xyz must be a non-empty (N, 3) point array")
    if padding_mm < 0:
        raise ValueError("padding_mm must be non-negative")
    if grid_alignment_mm is not None and grid_alignment_mm <= 0:
        raise ValueError("grid_alignment_mm must be positive")

    ref = surf["reference_plane"]
    orig = np.array(ref["origin_mm"], dtype=np.float64)
    u_axis = np.array(ref["u_axis_world"], dtype=np.float64)
    v_axis = np.array(ref["v_axis_world"], dtype=np.float64)
    normal = np.array(ref["normal_world"], dtype=np.float64)
    old_u_range = np.asarray(ref["u_range_mm"], dtype=np.float64)
    old_v_range = np.asarray(ref["v_range_mm"], dtype=np.float64)
    grid = np.array(surf["height_control_4x4_mm"], dtype=np.float64)
    if grid.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 Bezier height grid, found {grid.shape}")

    old_u_span = float(old_u_range[1] - old_u_range[0])
    old_v_span = float(old_v_range[1] - old_v_range[0])
    if old_u_span <= 0 or old_v_span <= 0:
        raise ValueError("Bezier reference-plane ranges must be increasing")

    relative_liver = liver_xyz - orig
    liver_u = relative_liver @ u_axis
    liver_v = relative_liver @ v_axis
    new_u_range = np.array([
        float(np.min(liver_u)) - padding_mm,
        float(np.max(liver_u)) + padding_mm,
    ])
    new_v_range = np.array([
        float(np.min(liver_v)) - padding_mm,
        float(np.max(liver_v)) + padding_mm,
    ])
    if grid_alignment_mm is not None:
        cell = float(grid_alignment_mm)
        new_u_range = np.array([
            np.floor(new_u_range[0] / cell) * cell,
            np.ceil(new_u_range[1] / cell) * cell,
        ])
        new_v_range = np.array([
            np.floor(new_v_range[0] / cell) * cell,
            np.ceil(new_v_range[1] / cell) * cell,
        ])
        # A tangent/single-voxel intersection must still contain one cell.
        if new_u_range[1] <= new_u_range[0]:
            new_u_range[1] = new_u_range[0] + cell
        if new_v_range[1] <= new_v_range[0]:
            new_v_range[1] = new_v_range[0] + cell

    needs_extrapolation = (
        new_u_range[0] < old_u_range[0]
        or new_u_range[1] > old_u_range[1]
        or new_v_range[0] < old_v_range[0]
        or new_v_range[1] > old_v_range[1]
    )
    if needs_extrapolation:
        # Stretch the bounded patch instead of evaluating a local cubic far
        # outside [0, 1].  This is the fail-safe path for legacy/small domains.
        new_grid = grid.copy()
    else:
        # Restricting an existing patch is safe and can preserve its polynomial
        # geometry exactly through a Bézier interval transform.
        u_start, u_end = (new_u_range - old_u_range[0]) / old_u_span
        v_start, v_end = (new_v_range - old_v_range[0]) / old_v_span
        transform_u = _bezier_interval_transform(u_start, u_end)
        transform_v = _bezier_interval_transform(v_start, v_end)
        new_grid = transform_u @ grid @ transform_v.T

    ref["u_range_mm"] = new_u_range.tolist()
    ref["v_range_mm"] = new_v_range.tolist()
    surf["height_control_4x4_mm"] = new_grid.tolist()

    # A bicubic height patch has planar control coordinates distributed at
    # thirds of each physical range.  These points are the authoritative
    # browser geometry and exactly match the updated reference representation.
    pts = []
    for ci in range(4):
        row = []
        for cj in range(4):
            u = new_u_range[0] + (ci / 3.0) * (new_u_range[1] - new_u_range[0])
            v = new_v_range[0] + (cj / 3.0) * (new_v_range[1] - new_v_range[0])
            pt = orig + u * u_axis + v * v_axis + new_grid[ci, cj] * normal
            row.append([float(pt[0]), float(pt[1]), float(pt[2])])
        pts.append(row)
    return pts


def _surface_liver_intersection_points(
    surf,
    liver_xyz,
    band_mm=2.0,
    chunk_size=200_000,
):
    """Return liver voxels in a thin band around a Bézier cutting surface.

    The returned points approximate the actual liver/surface intersection.  A
    thin physical band is preferable to the complete liver projection because
    the latter creates a large rectangle even for a small peripheral cut.
    """
    liver_xyz = np.asarray(liver_xyz, dtype=np.float64)
    if liver_xyz.ndim != 2 or liver_xyz.shape[1] != 3 or len(liver_xyz) == 0:
        raise ValueError("liver_xyz must be a non-empty (N, 3) point array")
    if band_mm <= 0:
        raise ValueError("band_mm must be positive")

    ref = surf["reference_plane"]
    origin = np.asarray(ref["origin_mm"], dtype=np.float64)
    normal = np.asarray(ref["normal_world"], dtype=np.float64)
    u_axis = np.asarray(ref["u_axis_world"], dtype=np.float64)
    v_axis = np.asarray(ref["v_axis_world"], dtype=np.float64)
    u0, u1 = [float(value) for value in ref["u_range_mm"]]
    v0, v1 = [float(value) for value in ref["v_range_mm"]]
    grid = np.asarray(surf["height_control_4x4_mm"], dtype=np.float64)
    if grid.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 Bezier height grid, found {grid.shape}")

    relative = liver_xyz - origin
    local_u = relative @ u_axis
    local_v = relative @ v_axis
    local_n = relative @ normal
    u_span = max(u1 - u0, 1e-8)
    v_span = max(v1 - v0, 1e-8)
    residual = np.empty(len(liver_xyz), dtype=np.float64)

    for start in range(0, len(liver_xyz), chunk_size):
        end = min(start + chunk_size, len(liver_xyz))
        u_norm = np.clip((local_u[start:end] - u0) / u_span, 0.0, 1.0)
        v_norm = np.clip((local_v[start:end] - v0) / v_span, 0.0, 1.0)
        one_u = 1.0 - u_norm
        one_v = 1.0 - v_norm
        basis_u = np.column_stack((
            one_u ** 3,
            3.0 * one_u ** 2 * u_norm,
            3.0 * one_u * u_norm ** 2,
            u_norm ** 3,
        ))
        basis_v = np.column_stack((
            one_v ** 3,
            3.0 * one_v ** 2 * v_norm,
            3.0 * one_v * v_norm ** 2,
            v_norm ** 3,
        ))
        height = np.einsum("ni,ij,nj->n", basis_u, grid, basis_v)
        residual[start:end] = np.abs(local_n[start:end] - height)

    selected = residual <= band_mm
    if int(np.count_nonzero(selected)) < 16:
        # A nearly tangent surface may pass between voxel centres.  Use a
        # compact nearest band rather than falling back to the whole liver.
        nearest_count = min(512, len(liver_xyz))
        nearest = np.argpartition(residual, nearest_count - 1)[:nearest_count]
        selected[nearest] = True
    return liver_xyz[selected]


def _fit_surface_to_liver_intersection(
    surf,
    liver_xyz,
    band_mm=2.0,
    cell_size_mm=4.0,
):
    """Crop a surface to outward-aligned cells around its liver intersection."""
    intersection_xyz = _surface_liver_intersection_points(
        surf, liver_xyz, band_mm=band_mm,
    )
    return _fit_surface_to_liver_projection(
        surf,
        intersection_xyz,
        padding_mm=0.0,
        grid_alignment_mm=cell_size_mm,
    )


def _surface_resolution_for_cell_size(surf, cell_size_mm=4.0):
    """Return vertex counts that make every local grid cell ``cell_size_mm``."""
    if cell_size_mm <= 0:
        raise ValueError("cell_size_mm must be positive")
    ref = surf["reference_plane"]
    u0, u1 = [float(value) for value in ref["u_range_mm"]]
    v0, v1 = [float(value) for value in ref["v_range_mm"]]
    cells_u = max(1, int(round((u1 - u0) / cell_size_mm)))
    cells_v = max(1, int(round((v1 - v0) / cell_size_mm)))
    return [cells_u + 1, cells_v + 1]


def _invalidate_previous_resection_state(json_data):
    """Remove selection metadata that refers to the planes being replaced."""
    for key in (
        "selected_resection_plane_index",
        "selected_resection_plane_source",
        "selected_resection_plane_saved_at",
    ):
        json_data.pop(key, None)
    json_data["resection_sequence_available"] = False


def run(ctx):
    case_name = ctx.params.get("case_name", ctx.case_id)
    output_dir = Path(ctx.output_dir)
    margin_mm = float(ctx.params.get("tumor_margin_mm", 5.0))

    ctx.log(f"Planning resection for {case_name} (margin={margin_mm}mm)...")

    # ---- 1. Read center_offset from existing 3D JSON ----
    # This is the SAME offset used to center the mesh vertices.
    # We MUST use this same offset for GeoSurge to ensure alignment.
    json_path = output_dir / f"{case_name}_3d.json"
    if not json_path.exists():
        fallback = list(output_dir.glob("*_3d.json"))
        json_path = fallback[0] if fallback else json_path

    _json_cache = {}
    try:
        with open(json_path, encoding="utf-8") as f:
            _json_cache = json.load(f)
        center_offset = np.array(_json_cache.get("center_offset", [0, 0, 0]), dtype=np.float64)
        ctx.log(f"  Using mesh center_offset: {center_offset}")
    except Exception:
        center_offset = np.zeros(3, dtype=np.float64)
        ctx.log("  WARNING: could not read center_offset, using origin")

    # ---- 2. Load masks and generate point clouds ----
    ctx.log("Loading segmentations...")
    vessel_mask_variants = _resolve_vessel_sources(ctx)
    affine = ctx.get_affine()

    # Load liver mask → IJK → XYZ
    liver_mask = ctx.get_mask("liver")
    liver_ijk = np.argwhere(liver_mask > 0).astype(np.float64)
    if len(liver_ijk) == 0:
        raise RuntimeError("Liver mask is empty — cannot plan resection")

    import nibabel as nib
    liver_xyz = nib.affines.apply_affine(affine, liver_ijk).astype(np.float64)

    # ---- CRITICAL: center data by mesh center_offset ----
    # Mesh vertices in the 3D JSON are centered by center_offset.
    # GeoSurge also needs centered data. Use the SAME offset.
    liver_xyz -= center_offset

    # Load tumor masks (tumor_1, tumor_2, ...)
    # Filter out noise: tumor masks must have at least MIN_TUMOR_VOXELS voxels
    MIN_TUMOR_VOXELS = 50  # ~58 mm³ at typical CT spacing — smaller is likely noise
    tumor_names = [n for n in ctx.list_masks() if "tumor" in n.lower()]
    tumor_xyz_list = []
    filtered_tumors = []
    for tname in tumor_names:
        mask = ctx.get_mask(tname)
        n_voxels = int(np.sum(mask > 0))
        if n_voxels < MIN_TUMOR_VOXELS:
            filtered_tumors.append((tname, n_voxels))
            continue
        xyz = _load_ijk_from_mask(ctx, tname) - center_offset
        tumor_xyz_list.append(xyz)
    if filtered_tumors:
        ctx.log(f"  Filtered {len(filtered_tumors)} noise tumor(s): "
                + ", ".join(f"{n}({v} vox)" for n, v in filtered_tumors))
    tumor_xyz = np.concatenate(tumor_xyz_list, axis=0) if tumor_xyz_list else np.zeros((0, 3), dtype=np.float64)

    # Load vessels (centered)
    hepatic_xyz = _load_ijk_from_mask(ctx, "hepatic") - center_offset
    portal_xyz = _load_ijk_from_mask(ctx, "portal") - center_offset
    vessel_xyz = np.concatenate([hepatic_xyz, portal_xyz], axis=0) if (len(hepatic_xyz) > 0 or len(portal_xyz) > 0) else np.zeros((0, 3), dtype=np.float64)

    ctx.log(f"  Liver: {len(liver_xyz)} points")
    ctx.log(f"  Tumors: {len(tumor_xyz)} points ({len(tumor_names)} components)")
    ctx.log(f"  Hepatic v.: {len(hepatic_xyz)} points")
    ctx.log(f"  Portal v.:  {len(portal_xyz)} points")

    if len(tumor_xyz) == 0:
        raise RuntimeError("No tumor masks found — cannot plan resection")

    # ---- 2. Subsample liver for faster candidate evaluation ----
    rng = np.random.default_rng(42)
    sample_n = min(50000, len(liver_xyz))
    sample_idx = np.sort(rng.choice(len(liver_xyz), size=sample_n, replace=False))
    liver_sample_xyz = liver_xyz[sample_idx]

    # ---- 3. Import GeoSurge planner functions ----
    project_root = Path(__file__).resolve().parents[3]
    # Use glob to discover GeoSurge directory (avoids date-stamp fragility)
    geo_surge_dirs = sorted(project_root.glob("data/GeoSurge_*/切面算法/surface_planner"))
    if not geo_surge_dirs:
        raise FileNotFoundError(
            f"GeoSurge surface_planner not found under {project_root / 'data/GeoSurge_*'}. "
            "Expected: data/GeoSurge_<date>/切面算法/surface_planner/"
        )
    planner_dir = str(geo_surge_dirs[-1])  # take the latest version
    if planner_dir not in sys.path:
        sys.path.insert(0, planner_dir)

    from plan_surfaces import (
        build_candidates, predict_scale, score_and_select_candidates,
        eval_surfaces,
    )
    from curved_refinement import (
        refine_candidate, candidate_clearance, tumor_boundary_points,
    )
    from surface_metrics import candidate_curvature_metrics

    # ---- 4. Predict scale ----
    tumor_liver_ratio = float(len(tumor_xyz) / max(len(liver_xyz), 1))
    n_tumor_components = len(tumor_names) or 1

    scale, target_ratio, scale_reason = predict_scale(
        liver_sample_xyz, tumor_xyz, vessel_xyz,
        n_tumor_components=n_tumor_components,
        tumor_liver_ratio=tumor_liver_ratio,
    )
    ctx.log(f"  Scale: {scale} (ratio={target_ratio:.3f}, reason: {scale_reason})")

    # ---- 5. Tumor boundary points for refinement ----
    # Convert tumor_xyz to IJK for tumor_boundary_points
    # (it uses binary_erosion on the IJK grid)
    affine_inv = np.linalg.inv(affine)
    tumor_ijk_h = nib.affines.apply_affine(affine_inv, tumor_xyz)
    tumor_ijk = np.round(tumor_ijk_h).astype(np.int32)
    tumor_boundary = tumor_boundary_points(tumor_ijk, tumor_xyz)

    # ---- 6. Build candidates ----
    ctx.log("Building candidate surfaces...")
    predicted_surface_count = 2 if (n_tumor_components >= 2 and scale not in {"local", "intermediate_local"}) else 1
    original_candidates = build_candidates(
        liver_sample_xyz, tumor_xyz, hepatic_xyz, portal_xyz,
        n_tumor_components, target_ratio,
    )

    # Establish the final full-liver parameter domain before scoring and
    # margin refinement.  Expanding a small refined patch afterwards requires
    # cubic extrapolation and can turn a gentle local surface into an extreme
    # fold.  Refining in the final domain avoids that failure mode entirely.
    for candidate in original_candidates:
        for surface in candidate["surfaces"]:
            _fit_surface_to_liver_projection(surface, liver_xyz, padding_mm=5.0)

    # ---- 7. Score and select parent ----
    stability_points = min(20000, len(liver_xyz))
    _, parent_index, _ = score_and_select_candidates(
        original_candidates, liver_sample_xyz, tumor_xyz,
        hepatic_xyz, portal_xyz,
        target_ratio=target_ratio, predicted_scale=scale,
        predicted_surface_count=predicted_surface_count,
        n_tumor_components=n_tumor_components,
        tumor_liver_ratio=tumor_liver_ratio,
        stability_points=stability_points,
    )
    ctx.log(f"  Parent candidate: {original_candidates[parent_index]['name']}")

    # ---- 8. Refine candidates with margin constraints ----
    ctx.log("Refining surfaces with margin constraints...")
    refined_candidates = []
    for candidate in original_candidates:
        refined, refinement = refine_candidate(
            candidate, tumor_boundary,
            margin_mm=margin_mm,
            lateral_padding_mm=5.0,
            smoothness=3.0,
            bins=8,
            max_iterations=12,
        )
        if refinement["success"]:
            refined_candidates.append(refined)

    if not refined_candidates:
        raise RuntimeError("No candidate satisfied margin constraints")

    # ---- 9. Score refined candidates and select best ----
    ctx.log(f"  {len(refined_candidates)} candidates passed margin check")
    refined_scored, reward_index, selection_info = score_and_select_candidates(
        refined_candidates, liver_sample_xyz, tumor_xyz,
        hepatic_xyz, portal_xyz,
        target_ratio=target_ratio, predicted_scale=scale,
        predicted_surface_count=predicted_surface_count,
        n_tumor_components=n_tumor_components,
        tumor_liver_ratio=tumor_liver_ratio,
        stability_points=stability_points,
    )

    reward_cand = refined_candidates[reward_index]
    ctx.log(f"  Selected: {reward_cand['name']}")

    # ---- 10. Select 3 candidates: 2 raw (diverse ratio) + 1 refined (major) ----
    # Use original (unrefined) candidates for ratio diversity.
    # Then use the refined GeoSurge best as the major option.
    def _extract_ratio(cand):
        import re
        m = re.search(r'_r([0-9.]+)', cand["name"])
        return float(m.group(1)) if m else 0.5

    # From original candidates, take small, medium, large ratio
    valid_orig = list(enumerate(original_candidates))
    valid_orig_sorted = sorted(valid_orig, key=lambda x: _extract_ratio(x[1]))
    pick_orig = []
    if len(valid_orig_sorted) >= 3:
        pick_orig = [valid_orig_sorted[0], valid_orig_sorted[len(valid_orig_sorted)//2], valid_orig_sorted[-1]]
    elif valid_orig_sorted:
        n = min(3, len(valid_orig_sorted))
        pick_orig = [valid_orig_sorted[i*len(valid_orig_sorted)//n] for i in range(n)]

    # Refined best candidate
    best_refined = refined_candidates[int(np.argmax([s.get("score", 0) for s in refined_scored]))]

    ctx.log(f"  Selected candidates:")
    for orig_idx, orig_cand in pick_orig:
        r = _extract_ratio(orig_cand)
        ctx.log(f"    Raw #{orig_idx}: {orig_cand['name'][:60]} (ratio={r:.3f})")
    ctx.log(f"    Refined: {best_refined['name'][:60]} (GeoSurge optimized)")

    # ---- 11. Build entries: 2 raw + 1 refined ----
    all_to_build = [c for _, c in pick_orig[:2]] + [best_refined]
    ctx.log(f"  Building {len(all_to_build)} candidates: "
            + ", ".join(c["name"][:40] for c in all_to_build))

    resection_planes = []
    centered_affine = np.array(affine, dtype=np.float64, copy=True)
    centered_affine[:3, 3] -= center_offset
    voxel_spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    half_voxel_diagonal = 0.5 * float(np.linalg.norm(voxel_spacing))
    intersection_band_mm = max(1.5, half_voxel_diagonal)
    resection_cell_size_mm = 4.0
    for rank, cand in enumerate(all_to_build):
        # Crop the browser geometry to the actual liver/surface intersection.
        # The optimisation used a full-liver domain above; restricting that
        # finished patch is an exact Bézier reparameterisation and therefore
        # changes only its finite extent, not the cutting geometry.
        fitted_control_points = [
            _fit_surface_to_liver_intersection(
                surf,
                liver_xyz,
                band_mm=intersection_band_mm,
                cell_size_mm=resection_cell_size_mm,
            )
            for surf in cand["surfaces"]
        ]

        # Compute safety and curvature on the exact geometry sent to the
        # browser.  The surfaces are mesh-centred, so use the matching affine.
        if cand["surfaces"]:
            cc = candidate_clearance(tumor_boundary, cand["surfaces"])
            c_margin_min = float(np.min(cc)) if len(cc) > 0 else 0.0
            c_margin_p05 = float(np.percentile(cc, 5)) if len(cc) > 0 else 0.0
            cand_curvature = candidate_curvature_metrics(
                cand["surfaces"], liver_ijk.astype(np.int32), centered_affine,
                sample_step_mm=0.75,
            )
        else:
            c_margin_min = c_margin_p05 = 0.0
            cand_curvature = {"mean_abs_curvature_mm_inv": 0, "p95_abs_curvature_mm_inv": 0, "surface_area_mm2": 0}

        ctx.log(f"  Candidate #{rank+1} ({cand['name'][:50]}) "
                f"margin={c_margin_min:.1f}mm area={cand_curvature.get('surface_area_mm2', 0):.0f}mm2")

        for si, (surf, cp3d) in enumerate(zip(cand["surfaces"], fitted_control_points)):
            # Snapshot the fitted reference plane for the JSON output.
            ref = dict(surf.get("reference_plane", {}))

            # Compute distances on the fitted domain so the colour map aligns
            # with the browser-authoritative control-point geometry.
            surface_resolution = _surface_resolution_for_cell_size(
                surf, cell_size_mm=resection_cell_size_mm,
            )
            dists = _compute_vertex_distances(
                surf,
                tumor_xyz,
                n_u=surface_resolution[0],
                n_v=surface_resolution[1],
            )

            if "origin_mm" in ref:
                orig = np.array(ref["origin_mm"], dtype=np.float64)
                ref["origin_mm"] = (orig + center_offset).tolist()
            plane_entry = {
                "candidate_rank": rank + 1,
                "candidate_name": cand["name"],
                "candidate_score": 0.0,
                "type": "bicubic_bezier",
                "reference_plane": ref,
                "height_decoder": surf.get("height_decoder", {
                    "type": "bicubic_bezier",
                    "control_grid_size": [4, 4],
                    "control_points_are_interpolated": False,
                    "height_unit": "mm",
                }),
                "height_control_4x4_mm": surf.get("height_control_4x4_mm", []),
                "control_points_3d": cp3d,
                "surface_resolution": surface_resolution,
                "cell_size_mm": resection_cell_size_mm,
                "semantics": surf.get("semantics", {
                    "positive_side": "remnant",
                    "negative_side": "resection",
                }),
                "margin_min_mm": c_margin_min,
                "margin_p05_mm": c_margin_p05,
                "margin_target_mm": margin_mm,
                "margin_success": c_margin_min >= margin_mm - 0.05,
                "curvature_mean": cand_curvature.get("mean_abs_curvature_mm_inv", 0.0),
                "curvature_p95": cand_curvature.get("p95_abs_curvature_mm_inv", 0.0),
                "surface_area_mm2": cand_curvature.get("surface_area_mm2", 0.0),
                "vertex_distances_mm": dists,
                "rule_metadata": surf.get("rule_metadata", {}),
            }
            resection_planes.append(plane_entry)

    # ---- 12. Update the 3D JSON file ----
    json_data = _json_cache  # already loaded at the top

    _invalidate_previous_resection_state(json_data)
    json_data["resection_planes"] = resection_planes
    json_data["vessel_mask_variants"] = vessel_mask_variants

    # Add sampled tumor point cloud for browser-side distance recomputation
    if len(tumor_xyz) > 0:
        sample_n = min(500, len(tumor_xyz))
        if sample_n < len(tumor_xyz):
            rng = np.random.default_rng(42)
            idx = sorted(rng.choice(len(tumor_xyz), size=sample_n, replace=False))
            tumor_sample = tumor_xyz[idx]
        else:
            tumor_sample = tumor_xyz
        json_data["tumor_cloud"] = tumor_sample.ravel().tolist()
    else:
        json_data["tumor_cloud"] = []

    # Ensure NaN values don't leak into JSON (browsers can't parse JSON with NaN)
    _json_dump_safe(json_data, json_path)

    ctx.log(f"Updated {json_path.name} with {len(resection_planes)} resection plane(s)")
    ctx.log(f"  Refresh the 3D HTML to see the resection overlay")

    # Build summary from top candidate
    top_plane = resection_planes[0] if resection_planes else {}
    return {
        "margin_min_mm": top_plane.get("margin_min_mm", 0.0),
        "margin_p05_mm": top_plane.get("margin_p05_mm", 0.0),
        "margin_success": top_plane.get("margin_success", False),
        "resection_plane_count": len(resection_planes),
        "candidate_count": len({p.get("candidate_name") for p in resection_planes}),
        "json_updated": True,
        "predicted_scale": scale,
        "vessel_mask_variants": vessel_mask_variants,
    }

"""Adapter from a saved 3D resection surface to the frozen v10.6 C4 planner.

The learned controller was trained on a two-dimensional parameter grid.  This
module applies it only to the ``(u, v)`` grid of a user-confirmed bicubic
surface, then maps the returned cells back through the existing 3D viewer.  It
does not claim that the planar bleeding model is a patient-specific model.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np


POLICY_ID = "clinical_v106_c4_learned_shielded"
EXPECTED_CHECKPOINT_SHA256 = (
    "c07904502d6b71a74484adb1c27971c77cdf6a61bb20b04f1f39d786d61a70be"
)
DEFAULT_MARGIN_ML = 16.07054347826075
MAX_ROWS = 30
MAX_COLS = 40


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Research" / "planar-resection-planning").is_dir():
            return parent
        if (parent / "贪吃蛇" / "planar_simulator").is_dir():
            return parent
    raise FileNotFoundError("无法定位平面规划研究代码所在的仓库根目录")


def default_checkpoint_path() -> Path:
    root = _repo_root()
    candidates = [
        root / "Port_B" / "models" / "resection_sequence" / "epoch_05.pt",
        root / "models" / "resection_sequence" / "epoch_05.pt",
        root / "贪吃蛇" / "planar_simulator" / "results"
        / "clinical_window_v10_6_shielded_learning" / "runs" / "bc"
        / "config_05_seed_2026081603" / "epoch_05.pt",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def configured_checkpoint_path() -> Path:
    configured = os.environ.get("VOXELSAGE_RESECTION_MODEL_CHECKPOINT")
    return Path(configured).expanduser().resolve() if configured else default_checkpoint_path()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path | None = None) -> tuple[Path, str]:
    checkpoint = Path(path or configured_checkpoint_path()).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "冻结切除排序模型不存在。请设置 VOXELSAGE_RESECTION_MODEL_CHECKPOINT "
            f"或将权重放到 {default_checkpoint_path()}"
        )
    actual = _sha256(checkpoint)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "切除排序模型 SHA-256 与论文冻结候选不一致："
            f"expected={EXPECTED_CHECKPOINT_SHA256}, actual={actual}"
        )
    return checkpoint, actual


def _research_modules():
    root = _repo_root()
    candidates = [
        root / "Research" / "planar-resection-planning",
        root / "贪吃蛇" / "planar_simulator",
    ]
    research = next((path for path in candidates if path.is_dir()), candidates[0])
    if not research.is_dir():
        raise FileNotFoundError(f"缺少平面规划研究代码：{research}")
    research_text = str(research)
    if research_text not in sys.path:
        sys.path.insert(0, research_text)
    from clinical_macro_environment import ClinicalMacroResectionEnv
    from confirmation_controllers_v107 import rollout_controller
    from plan_target_order_v104 import _step_macro_target

    return ClinicalMacroResectionEnv, rollout_controller, _step_macro_target


def _neighbors(cell: int, rows: int, cols: int) -> Iterable[int]:
    row, col = divmod(int(cell), cols)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr * cols + nc


def connected_component(mask: np.ndarray, start: int, rows: int, cols: int) -> np.ndarray:
    """Return the four-connected target component containing ``start``."""
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    if not 0 <= int(start) < len(flat) or not flat[int(start)]:
        raise ValueError("学习规划起点不在 Liver∩切面目标区域内")
    result = np.zeros(len(flat), dtype=bool)
    result[int(start)] = True
    queue = deque([int(start)])
    while queue:
        current = queue.popleft()
        for nxt in _neighbors(current, rows, cols):
            if flat[nxt] and not result[nxt]:
                result[nxt] = True
                queue.append(nxt)
    return result


def build_scenario(
    liver_target: np.ndarray,
    vascular_safe: np.ndarray,
    *,
    start: int,
    rows: int,
    cols: int,
    cell_side_mm: float,
) -> tuple[Dict[str, Any], np.ndarray]:
    if rows > MAX_ROWS or cols > MAX_COLS:
        raise ValueError(
            f"冻结模型最多支持 {MAX_ROWS}x{MAX_COLS} 个面单元，收到 {rows}x{cols}"
        )
    component = connected_component(liver_target, start, rows, cols)
    domain = np.flatnonzero(component)
    vessels = np.flatnonzero(component & ~np.asarray(vascular_safe, dtype=bool))
    scenario = {
        "scenario_id": "voxelsage-confirmed-3d-surface",
        "rows": int(rows),
        "cols": int(cols),
        "domain_cells": [[int(cell // cols), int(cell % cols)] for cell in domain],
        # In the frozen simulator these are vessel cross-section proxy cells,
        # not impassable clinical obstacles.
        "obstacle_cells": [[int(cell // cols), int(cell % cols)] for cell in vessels],
        "start_cell": [int(start // cols), int(start % cols)],
        "cell_size_mm": float(cell_side_mm),
    }
    return scenario, component


def _replay_actions(
    scenario: Dict[str, Any],
    actions: Sequence[Sequence[int]],
    *,
    clinical_config: Dict[str, Any],
    cols: int,
) -> tuple[list[Dict[str, Any]], list[int]]:
    ClinicalMacroResectionEnv, _, step_macro_target = _research_modules()
    env = ClinicalMacroResectionEnv(
        scenario=scenario, clinical_config=clinical_config, mechanics_update_interval=0
    )
    env.reset()
    start = int(env.start[0] * cols + env.start[1])
    steps: list[Dict[str, Any]] = [{
        "step": 0,
        "cell": start,
        "action": "cut",
        "time_seconds": 0.0,
        "grid_ij": [int(env.start[0]), int(env.start[1])],
        "macro_target": True,
        "covered_cell_count": 1,
    }]
    for raw_target in actions:
        target = (int(raw_target[0]), int(raw_target[1]))
        route = env._transfer_path(target)
        transfer_time = float(env.elapsed_minutes)
        for cell in route[1:]:
            transfer_time += float(env.base_action_minutes)
            steps.append({
                "step": len(steps),
                "cell": int(cell[0] * cols + cell[1]),
                "action": "transfer",
                "time_seconds": round(transfer_time * 60.0, 6),
                "grid_ij": [int(cell[0]), int(cell[1])],
                "macro_target": False,
                "covered_cell_count": int(len(env.cut)),
            })
        before = set(env.cut)
        step_macro_target(env, target)
        newly_covered = sorted(set(env.cut) - before)
        steps.append({
            "step": len(steps),
            "cell": int(target[0] * cols + target[1]),
            "action": "cut",
            "time_seconds": round(float(env.elapsed_minutes) * 60.0, 6),
            "grid_ij": [int(target[0]), int(target[1])],
            "macro_target": True,
            "covered_cell_count": int(len(env.cut)),
            "auto_covered_cells": [int(row * cols + col) for row, col in newly_covered if (row, col) != target],
        })
    covered = sorted(int(row * cols + col) for row, col in env.cut)
    return steps, covered


def plan_learned_shielded(
    liver_target: np.ndarray,
    vascular_safe: np.ndarray,
    *,
    start: int,
    rows: int,
    cols: int,
    cell_side_mm: float = 4.0,
    margin_ml: float = DEFAULT_MARGIN_ML,
    checkpoint_path: Path | None = None,
) -> Dict[str, Any]:
    """Run the frozen C4 ranker and exact simulator shield on a 3D surface grid."""
    if abs(float(cell_side_mm) - 4.0) > 1e-6:
        raise ValueError("冻结模型只验证过 4.0-mm 面单元，不能更改 learned_cell_side_mm")
    checkpoint, checkpoint_sha = validate_checkpoint(checkpoint_path)
    scenario, component = build_scenario(
        liver_target, vascular_safe, start=start, rows=rows, cols=cols,
        cell_side_mm=cell_side_mm,
    )
    _, rollout_controller, _ = _research_modules()
    clinical_config = {
        "early_end_mode": "disabled",
        "early_end_minutes": 0.0,
        "bleeding_probability": 1.0,
        "max_steps_multiplier": 8.0,
        "cell_side_mm": 4.0,
    }
    wall_start = time.perf_counter()
    baseline = rollout_controller(
        "C0", scenario, baseline_blood=0.0, margin_ml=float(margin_ml),
        cfg=clinical_config,
    )
    learned = rollout_controller(
        "C4", scenario,
        baseline_blood=float(baseline["realized_episode_B_ml"]),
        margin_ml=float(margin_ml), cfg=clinical_config,
        checkpoint_path=checkpoint,
    )
    wall_seconds = time.perf_counter() - wall_start
    if not learned.get("completion"):
        raise RuntimeError(f"冻结 C4 未完成三维曲面网格：{learned.get('failure_reason')}")
    if int(learned.get("safety_invariant_violations", 0)) != 0:
        raise RuntimeError("冻结 C4 在三维曲面网格上触发安全盾不变量失败")
    if float(learned["realized_episode_B_ml"]) > float(learned["budget_ml"]) + 1e-9:
        raise RuntimeError("冻结 C4 在三维曲面网格上超过模拟预算")

    steps, covered = _replay_actions(
        scenario, learned["actions"], clinical_config=clinical_config, cols=cols,
    )
    return {
        "policy_id": POLICY_ID,
        "checkpoint_sha256": checkpoint_sha,
        "component_mask": component,
        "path": steps,
        "covered_cells": covered,
        "scenario": scenario,
        "simulator": {
            "baseline_controller": "C0",
            "controller": "C4",
            "cell_side_mm": float(cell_side_mm),
            "baseline_elapsed_minutes": float(baseline["elapsed_minutes"]),
            "baseline_simulated_blood_ml": float(baseline["realized_episode_B_ml"]),
            "elapsed_minutes": float(learned["elapsed_minutes"]),
            "simulated_blood_ml": float(learned["realized_episode_B_ml"]),
            "budget_ml": float(learned["budget_ml"]),
            "margin_ml": float(margin_ml),
            "macro_action_count": int(learned["macro_action_count"]),
            "transfer_count": int(learned["transfer_count"]),
            "shield_intervention_count": int(learned["shield_intervention_count"]),
            "safety_invariant_violations": int(learned["safety_invariant_violations"]),
            "policy_forward_ms": float(learned["policy_forward_ms"]),
            "shield_exact_ms": float(learned["shield_exact_ms"]),
            "adapter_wall_seconds": float(wall_seconds),
            "action_sequence_hash": str(learned["action_sequence_hash"]),
        },
        "scope_warning": (
            "The learned order and exact shield operate on the saved surface's 2D parameter grid. "
            "Vessel cells and simulated blood are uncalibrated proxies; this is not a clinically "
            "validated 3D surgical trajectory or patient-level blood-loss prediction."
        ),
    }

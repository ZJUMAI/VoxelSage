"""Visualise the actual cut path a v10.4 policy traces on one scene.

Draws the spatial path (each chosen macro target in order) on an ASCII grid for
serpentine vs BC-model, and quantifies "odd" path traits: step count, direction
reversals (zig-zag / back-tracking), frontier efficiency vs a straight sweep,
and how often the path operates on exposed-vessel frontier cells.

Reads frozen policy_train scenes + BC checkpoint only. No selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_macro_environment import ClinicalMacroResectionEnv  # noqa: E402
from clinical_target_order_features import (  # noqa: E402
    candidate_features,
    global_context,
    normalize_features,
)
from clinical_target_order_policy import TargetOrderScorer  # noqa: E402
from clinical_window_evaluation import serpentine_macro_target_policy  # noqa: E402
from plan_target_order_v104 import (  # noqa: E402
    _step_macro_target,
    candidate_targets,
    serpentine_target_of,
)

FROZEN_DIR = SIM / "results/clinical_window_v10_4_target_order/frozen"
RUNS_DIR = SIM / "results/clinical_window_v10_4_target_order/runs"
TEACHER_DIR = SIM / "results/clinical_window_v10_4_target_order/teacher"
GATE_CFG = {
    "early_end_mode": "disabled",
    "early_end_minutes": 0.0,
    "bleeding_probability": 1.0,
    "max_steps_multiplier": 8.0,
}


def _model_step(env, model, scales, cand_count=6):
    targets = candidate_targets(env, count=cand_count)
    if not targets:
        return None
    feats = np.stack([normalize_features(candidate_features(env, t)[0], scales) for t in targets])
    gc = np.asarray([global_context(env)], dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(feats).unsqueeze(0),
                       torch.from_numpy(gc)).squeeze(0)
    return targets[int(logits.argmax().item())]


def _serp_step(env):
    t = serpentine_target_of(env)
    return None if t is None else tuple(t)


def _path(env, step_fn):
    steps = []
    while not env.terminated and not env.truncated:
        t = step_fn(env)
        if t is None:
            break
        if t not in env._frontier():
            t = serpentine_target_of(env)
        if t is None:
            break
        steps.append(tuple(int(c) for c in t))
        _step_macro_target(env, t)
    return steps


def _reversals(path):
    if len(path) < 3:
        return 0
    n = 0
    for i in range(1, len(path) - 1):
        a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if a[0] * b[0] + a[1] * b[1] < 0:
            n += 1
    return n


def _near_vessel(path, vessels, radius=1):
    """Fraction of path steps whose target touches a vessel cell within radius."""
    hits = 0
    for (r, c) in path:
        touch = any((abs(r - vr) + abs(c - vc)) <= radius for vr, vc in vessels)
        hits += int(touch)
    return hits / max(1, len(path))


def _draw(path, scene, start, vessel_set):
    rows, cols = scene["rows"], scene["cols"]
    grid = [["·"] * cols for _ in range(rows)]
    for vr, vc in vessel_set:
        if 0 <= vr < rows and 0 <= vr >= 0 and 0 <= vc < cols:
            grid[vr][vc] = "V"
    sr, sc = start
    if 0 <= sr < rows and 0 <= sc < cols:
        grid[sr][sc] = "S"
    order = "123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, (r, c) in enumerate(path[: min(len(path), 52)]):
        grid[r][c] = order[i]
    if path:
        r, c = path[-1]
        if grid[r][c] in order:
            grid[r][c] = "*"
    return "\n".join("".join(row) for row in grid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "target_order_bc.pt")
    parser.add_argument("--scales", type=Path, default=TEACHER_DIR / "feature_scales.json")
    parser.add_argument("--index", type=int, default=0, help="policy_train scene index")
    parser.add_argument("--scenario-id", type=str, default=None,
                        help="explicit scenario id from frozen Train")
    args = parser.parse_args()

    payload = json.loads((FROZEN_DIR / "splits_v10_4.json").read_text(encoding="utf-8"))
    internal = payload["internal_train"]
    train_by_id = {s["scenario_id"]: s for s in payload["splits"]["train"]}
    if args.scenario_id:
        sc = train_by_id[args.scenario_id]
    else:
        sc = train_by_id[internal["policy_train"]["scenario_ids"][args.index]]

    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    model = TargetOrderScorer()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    vessel_set = {tuple(c) for c in sc["obstacle_cells"]}
    start = (int(sc["start_cell"][0]), int(sc["start_cell"][1])) if "start_cell" in sc else None

    env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=GATE_CFG)
    env.reset()
    if start is None:
        start = tuple(env.current)
    mpath = _path(env, lambda e: _model_step(e, model, scales))

    env2 = ClinicalMacroResectionEnv(scenario=sc, clinical_config=GATE_CFG)
    env2.reset()
    spath = _path(env2, _serp_step)

    print(f"场景 {sc['scenario_id']}  start={start}  domain={sc.get('domain_rows')}x{sc.get('domain_cols')}  "
          f"vessel_cells={len(vessel_set)}")
    print()
    print("=== 模型 (BC) 路径 ===")
    print(_draw(mpath, sc, start, vessel_set))
    print()
    print("=== serpentine 路径 ===")
    print(_draw(spath, sc, start, vessel_set))
    print()
    print(f"模型: {len(mpath)} 步, 方向反转 {_reversals(mpath)} 次, 贴近血管比例 {_near_vessel(mpath, vessel_set):.2f}")
    print(f"serp: {len(spath)} 步, 方向反转 {_reversals(spath)} 次, 贴近血管比例 {_near_vessel(spath, vessel_set):.2f}")


if __name__ == "__main__":
    main()

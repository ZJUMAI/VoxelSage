"""Initial-model demo for v10.4 target-order planning (informational only).

Show what the Gate B BC scorer looks like on a few policy_train scenes:
per-step candidate scoring vs teacher's depth-1 MPC choice, plus a three-way
baseline / teacher / model comparison table.  Writes a markdown report.

Reads only frozen Train data (policy_train is where the model was trained),
never touches validation/test/stress and makes no selection decisions.
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
from evaluate_target_order_v104 import GATE_CFG, _run_teacher_worker  # noqa: E402
from plan_target_order_v104 import (  # noqa: E402
    _step_macro_target,
    candidate_targets,
    make_gate_rollout,
)

FROZEN_DIR = SIM / "results/clinical_window_v10_4_target_order/frozen"
RUNS_DIR = SIM / "results/clinical_window_v10_4_target_order/runs"
TEACHER_DIR = SIM / "results/clinical_window_v10_4_target_order/teacher"

DEMO_SCENE_INDEX = [0, 1]


def _model_logits(model, scales, env, targets):
    feats = np.stack([
        normalize_features(candidate_features(env, t)[0], scales)
        for t in targets
    ])
    gc = np.asarray([global_context(env)], dtype=np.float32)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(feats).unsqueeze(0),
            torch.from_numpy(gc),
        ).squeeze(0)
    return logits.cpu().numpy()


def _model_rollout(model, scales, env):
    """Run the scorer until episode end, recording each step's decision detail."""
    steps = []
    while not env.terminated and not env.truncated:
        targets = candidate_targets(env, count=6)
        if not targets:
            break
        logits = _model_logits(model, scales, env, targets)
        order = np.argsort(-logits)
        chosen = targets[int(order[0])]
        steps.append({
            "elapsed": float(env.elapsed_minutes),
            "blood": float(env.expected_blood_loss_ml),
            "top": [(tuple(targets[i]), float(logits[i])) for i in order[:3]],
            "chosen": tuple(chosen),
        })
        _step_macro_target(env, chosen)
    return steps, {
        "elapsed_minutes": float(env.elapsed_minutes),
        "expected_blood_loss_ml": float(env.expected_blood_loss_ml),
        "completion": bool(env.terminated and env.failure_reason is None),
        "clamp_cycle_count": float(env.clamp_cycle_count),
    }


def _teacher_rec(sc, baseline_blood, margin):
    return _run_teacher_worker((sc, baseline_blood, margin, GATE_CFG,
                                {"candidate_count": 6}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "target_order_bc.pt")
    parser.add_argument("--scales", type=Path, default=TEACHER_DIR / "feature_scales.json")
    parser.add_argument("--no-teacher", action="store_true", help="skip slow depth-1 MPC teacher")
    parser.add_argument("--limit", type=int, default=None, help="only first N demo scenes")
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "initial_model_demo.md")
    args = parser.parse_args()

    payload = json.loads((FROZEN_DIR / "splits_v10_4.json").read_text(encoding="utf-8"))
    internal = payload["internal_train"]
    train_by_id = {s["scenario_id"]: s for s in payload["splits"]["train"]}
    scenes = [train_by_id[i] for i in internal["policy_train"]["scenario_ids"]]
    if args.limit:
        scenes = scenes[: args.limit]

    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    model = TargetOrderScorer()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    run_serp = make_gate_rollout(serpentine_macro_target_policy, clinical_config=GATE_CFG)

    lines: list[str] = [
        "# v10.4 初始目标顺序模型演示（BC scorer）",
        "",
        f"- 数据：frozen Train `policy_train`（前 {len(scenes)} 个场景，仅演示用）",
        f"- 模型：`{args.checkpoint.name}`，top1={payload.get('top1_acc', 'n/a')}",
        "- 参照：serpentine baseline（机械 15/5）、teacher（depth-1 MPC，完整 tail 评估）",
        "",
    ]

    for si, sc in enumerate(scenes):
        sid = sc["scenario_id"]
        lines.append(f"## 场景 {si}: {sid}")
        lines.append("")

        serp_rec = run_serp(sc)
        margin = 0.05 * serp_rec["expected_blood_loss_ml"]

        env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=GATE_CFG)
        env.reset()
        steps, model_rec = _model_rollout(model, scales, env)

        teacher_rec = None
        if not args.no_teacher:
            teacher_rec = _teacher_rec(sc, serp_rec["expected_blood_loss_ml"], margin)

        def fmt(rec, name):
            comp = "✅" if rec["completion"] else "❌"
            return (f"| {name} | {rec['elapsed_minutes']:.1f} | "
                    f"{rec['expected_blood_loss_ml']:.1f} | {comp} | "
                    f"{rec['clamp_cycle_count']:.0f} |")

        lines.append("| 策略 | 耗时(min) | 失血(mL) | 完成 | 夹闭周期 |")
        lines.append("|---|---|---|---|---|")
        lines.append(fmt(serp_rec, "serpentine baseline"))
        lines.append(fmt(model_rec, "BC scorer (model)"))
        if teacher_rec is not None:
            lines.append(fmt(teacher_rec, "teacher depth-1 MPC"))
        lines.append("")
        lines.append(f"失血非劣界 M_B = 0.05 × {serp_rec['expected_blood_loss_ml']:.1f} = {margin:.1f} mL")
        lines.append("")

        lines.append(f"### 模型每步决策（共 {len(steps)} 步，显示前 {min(8, len(steps))} 步）")
        lines.append("")
        lines.append("| # | t(min) | B(mL) | top-1 (得分) | top-2 (得分) | top-3 (得分) | 选中 |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, st in enumerate(steps[:8]):
            cols = " | ".join(
                f"({t[0]},{t[1]}) {s:.2f}" for t, s in st["top"])
            lines.append(f"| {i + 1} | {st['elapsed']:.1f} | {st['blood']:.0f} | {cols} | "
                         f"({st['chosen'][0]},{st['chosen'][1]}) |")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print("=== 三方对比摘要 ===")
    for si, sc in enumerate(scenes[: len(scenes)]):
        serp = run_serp(sc)
        env = ClinicalMacroResectionEnv(scenario=sc, clinical_config=GATE_CFG)
        env.reset()
        _steps, model_rec = _model_rollout(model, scales, env)
        print(f"scene {si} {sc['scenario_id']}: serp T={serp['elapsed_minutes']:.1f} B={serp['expected_blood_loss_ml']:.0f} | "
              f"model T={model_rec['elapsed_minutes']:.1f} B={model_rec['expected_blood_loss_ml']:.0f} "
              f"({len(_steps)} steps)")


if __name__ == "__main__":
    main()

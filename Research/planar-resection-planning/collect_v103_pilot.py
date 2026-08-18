"""v10.3 pilot data collection on Train-512 (internal scene split).

Decision-maker mandate:
  * use ONLY the original Train-512 scenes, re-split by scenario into an
    internal train / calibration / dev set (NO Oracle-Dev / Probe /
    Validation / Test / Stress access);
  * per release-legal decision record look-ahead features (physics-based
    forward simulation, window-bounded) plus the v2-safe label and the
    counterfactual Delta-B / Delta-I / advantage as regression targets;
  * look-ahead features must NOT contain the oracle's final label or full
    counterfactual answer (they are intermediate window quantities).

Usage (scene-sliced for parallelism):
    python collect_v103_pilot.py --scenario-start N --scenario-count M
    python collect_v103_pilot.py --scenario-start 0 --scenario-count 512
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from clinical_target_conditioned_environment import (  # noqa: E402
    CLAMP_CONTINUE,
    CLAMP_RELEASE,
    TargetConditionedClampEnv,
)
from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy  # noqa: E402

import train_target_conditioned_clamp_oracle as m  # noqa: E402
from v103_lookahead_features import FEATURE_NAMES, lookahead_features  # noqa: E402

BC_MODEL = "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"
SPLITS = "results/clinical_window_v10_2/frozen/splits_v10_2.json"
SCALES = "results/clinical_window_v10_2/frozen/scales_v10_2.json"
OUT_DIR = Path("results/clinical_window_v10_2/pilot_v103")
SPLIT_SEED = 20260811
INTERNAL_TRAIN = 400
INTERNAL_CAL = 56
INTERNAL_DEV = 56

_BC: FrozenBCMacroTargetPolicy | None = None
_CFG: dict | None = None


def _init_worker(bc_model: str, clinical: dict, reward: dict, isch_scale: float) -> None:
    global _BC, _CFG
    _BC = FrozenBCMacroTargetPolicy(bc_model, device="cpu")
    _CFG = {
        "clinical": clinical,
        "reward": reward,
        "ischemia_scale": isch_scale,
    }


def _safe_scenario(scenario: dict) -> list[dict]:
    try:
        return _v103_scenario_examples(scenario)
    except Exception as exc:  # noqa: BLE001 - one bad scene must not kill the slice
        print(json.dumps({
            "mode": "collect_v103_pilot", "event": "scene_error",
            "scenario_id": str(scenario.get("scenario_id")), "error": repr(exc),
        }, ensure_ascii=False), flush=True)
        return []


def _v103_scenario_examples(scenario: dict) -> list[dict]:
    cfg = _CFG
    env = TargetConditionedClampEnv(
        scenario=scenario,
        clinical_config=cfg["clinical"],
        reward_config=cfg["reward"],
        ischemia_cost=1.0,
        ischemia_scale_minutes=cfg["ischemia_scale"],
        target_selector=_BC.select_target,
        safe_release_mask=True,
    )
    # Baseline rollout records the frozen clamp-blind target sequence.
    env.reset()
    targets: list[int] = []
    while not env.terminated and not env.truncated:
        targets.append(int(env.planned_target_index))
        env.step(CLAMP_CONTINUE, build_obs=False)
    examples: list[dict] = []
    for oracle_pass in (False, True):
        env.reset()
        while not env.terminated and not env.truncated:
            legal = bool(env.action_masks()[CLAMP_RELEASE])
            if not legal:
                env.step(CLAMP_CONTINUE, build_obs=False)
                continue
            advantage, details = m.counterfactual_release_advantage(
                env,
                time_cost=1.0,
                blood_cost=1.0,
                ischemia_cost=1.0,
                time_scale=float(cfg["clinical"]["time_scale_minutes"]),
                blood_scale=float(cfg["clinical"]["blood_scale_ml"]),
                ischemia_scale=cfg["ischemia_scale"],
                target_sequence=targets,
            )
            label, _, db, di = m._stage1_label_and_reg(
                advantage, details,
                epsilon_ischemia=1e-6,
                blood_scale=float(cfg["clinical"]["blood_scale_ml"]),
                ischemia_scale=cfg["ischemia_scale"],
            )
            feat = lookahead_features(env, targets)
            row = {
                "scenario_id": str(scenario.get("scenario_id")),
                "macro_step": int(env.step_count),
                "oracle_policy": bool(oracle_pass),
                "label": int(label),
                "delta_blood": float(db),
                "delta_ischemia": float(di),
                "advantage": float(advantage),
            }
            row.update({k: float(feat[k]) for k in FEATURE_NAMES})
            examples.append(row)
            action = CLAMP_RELEASE if oracle_pass and label == 1 else CLAMP_CONTINUE
            env.step(action, build_obs=False)
    return examples


def build_split_manifest(scenes: list[dict]) -> dict[str, list[str]]:
    """Deterministic scene-isolated internal split of Train-512."""
    rng = random.Random(SPLIT_SEED)
    ids = [str(s.get("scenario_id")) for s in scenes]
    rng.shuffle(ids)
    assert INTERNAL_TRAIN + INTERNAL_CAL + INTERNAL_DEV == len(ids)
    return {
        "internal_train": sorted(ids[:INTERNAL_TRAIN]),
        "internal_calibration": sorted(ids[INTERNAL_TRAIN:INTERNAL_TRAIN + INTERNAL_CAL]),
        "internal_dev": sorted(ids[INTERNAL_TRAIN + INTERNAL_CAL:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-start", type=int, default=0)
    parser.add_argument("--scenario-count", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    scales = json.load(open(SCALES))
    clinical = {
        "time_scale_minutes": float(scales["time_scale_minutes"]),
        "blood_scale_ml": float(scales["blood_scale_ml"]),
        "weight_kg": float(scales.get("weight_kg", 70.0)),
        "bleeding_probability": 1.0,
        "early_end_mode": "threshold",
        "early_end_minutes": 10.0,
    }
    reward = {"time_cost": 1.0, "blood_cost": 1.0, "completion_bonus": 5.0,
              "failure_penalty": 10.0, "invalid_action_penalty": 10.0}
    isch_scale = float(scales["ischemia_scale_minutes"])

    splits = json.load(open(SPLITS))
    train_scenes = list(splits["splits"]["train"])
    manifest = build_split_manifest(train_scenes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "internal_split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ordered = list(train_scenes)
    ordered.sort(key=lambda s: str(s.get("scenario_id")))
    scenes = ordered[args.scenario_start: args.scenario_start + args.scenario_count]
    if not scenes:
        raise SystemExit("no scenes in slice")

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(BC_MODEL, clinical, reward, isch_scale)) as pool:
        results = pool.map(_safe_scenario, scenes, chunksize=1)

    examples: list[dict] = []
    for r in results:
        examples.extend(r)
    if not examples:
        raise SystemExit("no legal examples collected in slice")

    feats = np.asarray([[e[k] for k in FEATURE_NAMES] for e in examples], dtype=np.float32)
    labels = np.asarray([e["label"] for e in examples], dtype=np.int64)
    db = np.asarray([e["delta_blood"] for e in examples], dtype=np.float32)
    di = np.asarray([e["delta_ischemia"] for e in examples], dtype=np.float32)
    adv = np.asarray([e["advantage"] for e in examples], dtype=np.float32)
    regression = np.stack([
        db / clinical["blood_scale_ml"],
        di / isch_scale,
        adv,
    ], axis=1).astype(np.float32)

    out = OUT_DIR / f"pilot_{args.scenario_start:03d}.npz"
    np.savez(out, features=feats, labels=labels, regression=regression,
             delta_blood=db, delta_ischemia=di, advantage=adv)
    audit = {"scenario_start": args.scenario_start, "n_examples": len(examples),
             "examples": examples}
    (OUT_DIR / f"audit_{args.scenario_start:03d}.json").write_text(
        json.dumps(audit, ensure_ascii=False), encoding="utf-8")

    # per-split counts (scenario-isolated)
    split_of = {str(s.get("scenario_id")): None for s in scenes}
    counts = {"internal_train": 0, "internal_calibration": 0, "internal_dev": 0}
    for e in examples:
        for name in counts:
            if e["scenario_id"] in manifest[name]:
                counts[name] += 1
                break
    print(json.dumps({
        "mode": "collect_v103_pilot",
        "scenario_start": args.scenario_start,
        "n_scenarios": len(scenes),
        "n_examples": len(examples),
        "n_positive": int(labels.sum()),
        "per_split": counts,
        "output": str(out),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

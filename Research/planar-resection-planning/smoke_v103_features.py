"""Smoke test the v10.3 look-ahead feature extractor on a few train scenes."""
import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import numpy as np

from clinical_target_conditioned_environment import CLAMP_CONTINUE, TargetConditionedClampEnv
from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy
from v103_lookahead_features import FEATURE_NAMES, lookahead_features

BC_MODEL = "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"
SPLITS = "results/clinical_window_v10_2/frozen/splits_v10_2.json"
SCALES = "results/clinical_window_v10_2/frozen/scales_v10_2.json"

splits = json.load(open(SPLITS))
scales = json.load(open(SCALES))
train_scenes = splits["splits"]["train"][:3]

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

bc = FrozenBCMacroTargetPolicy(BC_MODEL, device="cpu")

for scenario in train_scenes:
    env = TargetConditionedClampEnv(
        scenario=scenario, clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=float(scales["ischemia_scale_minutes"]),
        target_selector=bc.select_target, safe_release_mask=True,
    )
    env.reset()
    targets = []
    while not env.terminated and not env.truncated:
        targets.append(int(env.planned_target_index))
        env.step(CLAMP_CONTINUE, build_obs=False)
    env.reset()
    legal_count = 0
    while not env.terminated and not env.truncated:
        legal = bool(env.action_masks()[1])
        if legal:
            feat = lookahead_features(env, targets)
            assert all(k in feat for k in FEATURE_NAMES), f"missing keys: {set(FEATURE_NAMES)-set(feat)}"
            legal_count += 1
            if legal_count <= 2:
                print(f"\nscene {scenario['scenario_id']} step {env.step_count} legal (phase={env.phase} "
                      f"clamp_elapsed={env.phase_elapsed_minutes:.1f})")
                for k in FEATURE_NAMES:
                    print(f"  {k} = {feat[k]:.4f}")
        env.step(CLAMP_CONTINUE, build_obs=False)
    print(f"\nscene {scenario['scenario_id']}: targets={len(targets)} legal_steps={legal_count}")
print("\nSMOKE OK")

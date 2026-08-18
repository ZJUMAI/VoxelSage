"""Quick validation: do the look-ahead features separate v2-safe labels?

Collects look-ahead features + counterfactual Delta-B/Delta-I + label on a
few train scenes (baseline + oracle passes) and reports:
  * pos vs neg means for each feature
  * point-biserial correlation of rw_blood_loss / rw_time_to_first_exposure
    with the label and with Delta-B
"""
import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import numpy as np

from clinical_target_conditioned_environment import CLAMP_CONTINUE, CLAMP_RELEASE, TargetConditionedClampEnv
from clinical_target_conditioned_policy import FrozenBCMacroTargetPolicy
from train_target_conditioned_clamp_oracle import counterfactual_release_advantage, _stage1_label_and_reg
from v103_lookahead_features import FEATURE_NAMES, lookahead_features

BC_MODEL = "results/clinical_window_v10_1/runs/stage1a_bc_seed_2026081301/pretrained_model.zip"
SPLITS = "results/clinical_window_v10_2/frozen/splits_v10_2.json"
SCALES = "results/clinical_window_v10_2/frozen/scales_v10_2.json"

splits = json.load(open(SPLITS))
scales = json.load(open(SCALES))
train_scenes = splits["splits"]["train"][:4]

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
ISCH_SCALE = float(scales["ischemia_scale_minutes"])

bc = FrozenBCMacroTargetPolicy(BC_MODEL, device="cpu")

rows = []
for scenario in train_scenes:
    env = TargetConditionedClampEnv(
        scenario=scenario, clinical_config=clinical, reward_config=reward,
        ischemia_cost=1.0, ischemia_scale_minutes=ISCH_SCALE,
        target_selector=bc.select_target, safe_release_mask=True,
    )
    env.reset()
    targets = []
    while not env.terminated and not env.truncated:
        targets.append(int(env.planned_target_index))
        env.step(CLAMP_CONTINUE, build_obs=False)
    for oracle_pass in (False, True):
        env.reset()
        while not env.terminated and not env.truncated:
            legal = bool(env.action_masks()[CLAMP_RELEASE])
            if legal:
                adv, det = counterfactual_release_advantage(
                    env, time_cost=1.0, blood_cost=1.0, ischemia_cost=1.0,
                    time_scale=float(clinical["time_scale_minutes"]),
                    blood_scale=float(clinical["blood_scale_ml"]),
                    ischemia_scale=ISCH_SCALE, target_sequence=targets,
                )
                label, _, db, di = _stage1_label_and_reg(
                    adv, det, epsilon_ischemia=1e-6,
                    blood_scale=float(clinical["blood_scale_ml"]), ischemia_scale=ISCH_SCALE,
                )
                feat = lookahead_features(env, targets)
                row = {"label": label, "db": db, "di": di, "adv": adv}
                row.update(feat)
                rows.append(row)
                action = CLAMP_RELEASE if oracle_pass and label == 1 else CLAMP_CONTINUE
            else:
                action = CLAMP_CONTINUE
            env.step(action, build_obs=False)

arr = {k: np.asarray([r[k] for r in rows], dtype=float) for k in FEATURE_NAMES + ["label", "db", "di"]}
label = arr["label"].astype(int)
print(f"legal samples: {len(rows)}  pos: {int(label.sum())}  pos_frac: {label.mean():.4f}")
print(f"\n{'feature':32s} {'pos_mean':>10s} {'neg_mean':>10s} {'r_label':>8s} {'r_db':>8s}")
for k in FEATURE_NAMES:
    v = arr[k]
    if v.std() < 1e-9:
        print(f"{k:32s} {v[label==1].mean():10.4f} {v[label==0].mean():10.4f} {'~0':>8s} {'~0':>8s}")
        continue
    rl = np.corrcoef(v, label)[0, 1]
    rd = np.corrcoef(v, arr["db"])[0, 1]
    print(f"{k:32s} {v[label==1].mean():10.4f} {v[label==0].mean():10.4f} {rl:8.3f} {rd:8.3f}")

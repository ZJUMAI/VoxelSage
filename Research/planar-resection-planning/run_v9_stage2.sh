#!/usr/bin/env bash
# Stage 2: 在同一 1201 动作空间中解锁 END
# 用法: run_v9_stage2.sh <stage_name> <seed> <gpu> <init_model> <early_end_mode> <early_end_minutes> <timesteps>
set -euo pipefail

STAGE="${1:?stage_name required}"
SEED="${2:?seed required}"
GPU="${3:?gpu required}"
INIT_MODEL="${4:?init_model required}"
EARLY_END_MODE="${5:?early_end_mode required}"
EARLY_END_MINUTES="${6:?early_end_minutes required}"
TIMESTEPS="${7:?timesteps required}"

cd "$(dirname "$0")"

MPLCONFIGDIR=/tmp/matplotlib-clinical-macro python train_clinical_window_ppo.py \
  --control-mode macro \
  --splits results/clinical_window_v5/frozen/splits_curriculum_d_v5.json \
  --scales results/clinical_window_v5/frozen/scales_v5.json \
  --init-model "${INIT_MODEL}" \
  --output-dir "results/clinical_window_v9_macro/runs/${STAGE}" \
  --early-end-mode "${EARLY_END_MODE}" --early-end-minutes "${EARLY_END_MINUTES}" \
  --timesteps "${TIMESTEPS}" \
  --n-envs 16 --n-steps 512 --batch-size 512 --n-epochs 5 \
  --learning-rate 3e-4 --gamma 0.9999 --gae-lambda 0.98 \
  --ent-coef 0.005 --target-kl 0.03 \
  --bc-scenarios 0 --bc-epochs 0 --rl-margin-coef 0 \
  --seed "${SEED}" --device "cuda:${GPU}" \
  --time-cost 1 --blood-cost 1 --progress-bonus 5 \
  --seal-progress-bonus 2 --completion-bonus 20 --failure-penalty 10 \
  --invalid-action-penalty 10 --clinical-cost-cap 10 \
  --stagnation-penalty-cap 0.05 --two-cell-loop-penalty 0.25

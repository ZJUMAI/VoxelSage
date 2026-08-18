#!/usr/bin/env bash
# Stage 1B: 固定 15/5 下优化切割顺序（PPO，从 Stage 1A pretrained_model 初始化）
# 用法: run_v9_stage1b.sh <seed> <gpu> <init_model> <output_dir>
set -euo pipefail

SEED="${1:?seed required}"
GPU="${2:?gpu required}"
INIT_MODEL="${3:?init_model required}"
OUTPUT_DIR="${4:?output_dir required}"

cd "$(dirname "$0")"

MPLCONFIGDIR=/tmp/matplotlib-clinical-macro python train_clinical_window_ppo.py \
  --control-mode macro \
  --splits results/clinical_window_v5/frozen/splits_curriculum_d_v5.json \
  --scales results/clinical_window_v5/frozen/scales_v5.json \
  --init-model "${INIT_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --early-end-mode disabled --timesteps 50000 \
  --n-envs 16 --n-steps 512 --batch-size 512 --n-epochs 5 \
  --learning-rate 3e-4 --gamma 0.9999 --gae-lambda 0.98 \
  --ent-coef 0.005 --target-kl 0.03 \
  --bc-scenarios 0 --bc-epochs 0 --rl-margin-coef 0 \
  --seed "${SEED}" --device "cuda:${GPU}" \
  --time-cost 1 --blood-cost 1 --progress-bonus 5 \
  --seal-progress-bonus 2 --completion-bonus 20 --failure-penalty 10 \
  --invalid-action-penalty 10 --clinical-cost-cap 10 \
  --stagnation-penalty-cap 0.05 --two-cell-loop-penalty 0.25

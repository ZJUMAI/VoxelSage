#!/usr/bin/env bash
# Stage 1A: 宏目标 margin-BC，从头训练 1201 动作空间策略（屏蔽 END）
# 用法: run_v9_stage1a.sh <seed> <gpu> <output_dir_suffix>
set -euo pipefail

SEED="${1:?seed required}"
GPU="${2:?gpu required}"
SUFFIX="${3:?suffix required}"

cd "$(dirname "$0")"

MPLCONFIGDIR=/tmp/matplotlib-clinical-macro python train_clinical_window_ppo.py \
  --control-mode macro \
  --splits results/clinical_window_v5/frozen/splits_curriculum_d_v5.json \
  --scales results/clinical_window_v5/frozen/scales_v5.json \
  --output-dir "results/clinical_window_v9_macro/runs/stage1a_macro_${SUFFIX}" \
  --early-end-mode disabled --end-action-initial-bias -4 \
  --bc-scenarios 256 --bc-epochs 30 --bc-batch-size 512 \
  --bc-learning-rate 1e-3 --bc-margin 2.0 --bc-v-weight 0 \
  --timesteps 16 --n-envs 1 --n-steps 16 --batch-size 16 --n-epochs 1 \
  --learning-rate 3e-4 --gamma 0.9999 --gae-lambda 0.98 \
  --ent-coef 0.005 --target-kl 0.2 \
  --seed "${SEED}" --device "cuda:${GPU}" \
  --time-cost 1 --blood-cost 1 --progress-bonus 5 \
  --seal-progress-bonus 2 --completion-bonus 20 --failure-penalty 10 \
  --invalid-action-penalty 10 --clinical-cost-cap 10 \
  --stagnation-penalty-cap 0.05 \
  --stagnation-soft-start-steps 40 --stagnation-penalty-ramp-steps 24 \
  --stagnation-limit-steps 96 --two-cell-loop-penalty 0.25 \
  --two-cell-loop-soft-start-traversals 6 --two-cell-loop-limit-traversals 12

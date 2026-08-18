#!/usr/bin/env bash
# v10.1 Stage 2A: Optuna 多目标多 GPU 并行
# 前置：timing-oracle 已完成（clamp_oracle_model.zip 存在）
# 4 worker 共享 sqlite storage，各跑独立 device
set -euo pipefail

ORACLE="results/clinical_window_v10_1/oracle/threshold10_seed_2026081401/clamp_oracle_model.zip"
if [ ! -f "$ORACLE" ]; then
  echo "ERROR: oracle 模型不存在: $ORACLE" >&2
  exit 1
fi

COMMON="--splits results/clinical_window_v10_1/frozen/splits_v10_1.json
  --scales results/clinical_window_v10_1/frozen/scales_v10_1.json
  --init-model $ORACLE
  --baseline-evaluation results/clinical_window_v10_1/evaluation/baseline_hierarchical_tuning32.json
  --output-dir results/clinical_window_v10_1/optuna/stage2a_threshold10
  --storage sqlite:///results/clinical_window_v10_1/optuna/stage2a_threshold10.db
  --study-name clinical-v10_1-stage2a-threshold10
  --trials 10 --timesteps 25000 --n-envs 8 --n-steps 256 --batch-size 256
  --tuning-limit 32 --blood-safety-ratio 1.05 --early-end-minutes 10"

# 每个 worker 用不同 seed：NSGAIISampler 的 seed 决定初始种群，
# 若 4 worker 共用 seed，空 study 时各自生成相同首超参，导致重复 trial 浪费算力。
declare -A DEV_SEEDS=(
  [cuda:0]=2026081501
  [cuda:1]=2026081502
  [cuda:4]=2026081503
  [cuda:5]=2026081504
)
for DEV in cuda:0 cuda:1 cuda:4 cuda:5; do
  nohup python optimize_clinical_v10_optuna.py $COMMON --device $DEV --seed ${DEV_SEEDS[$DEV]} \
    > logs/v10_1_stage2a_optuna_${DEV/:/_}.log 2>&1 &
  echo "optuna worker $DEV (seed ${DEV_SEEDS[$DEV]}) PID: $!"
done
echo "=== 4 个 Optuna worker 已启动（共享 study clinical-v10_1-stage2a-threshold10）==="

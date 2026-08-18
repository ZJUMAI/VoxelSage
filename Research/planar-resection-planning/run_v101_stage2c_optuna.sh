#!/usr/bin/env bash
# v10.1 Stage 2C: threshold 5 min Optuna 多目标多 GPU 并行
# 前置：Stage 2B 通过（threshold-10 trial18 3 seed 全过）
# init: threshold-10 安全 checkpoint (trial18/seed_2026081703 final_model)
# 4 worker 共享 sqlite storage，各独立 seed，各跑 trials 配额
set -euo pipefail

INIT="results/clinical_window_v10_1/stage2b/trial18/seed_2026081703/final_model.zip"
if [ ! -f "$INIT" ]; then
  echo "ERROR: threshold-10 init model 不存在: $INIT" >&2
  exit 1
fi

COMMON="--splits results/clinical_window_v10_1/frozen/splits_v10_1.json
  --scales results/clinical_window_v10_1/frozen/scales_v10_1.json
  --init-model $INIT
  --baseline-evaluation results/clinical_window_v10_1/evaluation/baseline_hierarchical_tuning32.json
  --output-dir results/clinical_window_v10_1/optuna/stage2c_threshold5
  --storage sqlite:///results/clinical_window_v10_1/optuna/stage2c_threshold5.db
  --study-name clinical-v10_1-stage2c-threshold5
  --trials 10 --timesteps 25000 --n-envs 8 --n-steps 256 --batch-size 256
  --tuning-limit 32 --blood-safety-ratio 1.05 --early-end-minutes 5"

# 物理 GPU 0/1/4/5（2080Ti）空闲。用 CUDA_VISIBLE_DEVICES 精确控制物理分配
# （torch 的 cuda:N 与 nvidia-smi 物理编号不同，故用 CUDA_VISIBLE_DEVICES 屏蔽其余 GPU）
declare -A PHYS_GPU_SEEDS=(
  [0]=2026082101
  [1]=2026082102
  [4]=2026082103
  [5]=2026082104
)
for PHYS in 0 1 4 5; do
  CUDA_VISIBLE_DEVICES=$PHYS nohup python optimize_clinical_v10_optuna.py $COMMON --device cuda:0 --seed ${PHYS_GPU_SEEDS[$PHYS]} \
    > logs/v10_1_stage2c_optuna_phys${PHYS}.log 2>&1 &
  echo "stage2c worker physGPU=$PHYS (seed ${PHYS_GPU_SEEDS[$PHYS]}) PID: $!"
done
echo "=== 4 个 Stage 2C Optuna worker 已启动（study clinical-v10_1-stage2c-threshold5）==="

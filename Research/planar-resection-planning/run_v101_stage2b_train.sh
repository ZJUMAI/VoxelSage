#!/usr/bin/env bash
# v10.1 Stage 2B: Pareto 候选 3-seed 重训 50k（Validation-64 确认前的多 seed 训练）
# 用法: bash run_v101_stage2b_train.sh <candidate_name> <seed> <device>
#   candidate_name: trial5 | trial17 | trial18
#   seed: 新 seed（每个候选 3 个）
#   device: cuda:0 | cuda:1 | cuda:4 | cuda:5
set -euo pipefail

CAND="$1"; SEED="$2"; DEV="$3"
ORACLE="results/clinical_window_v10_1/oracle/threshold10_seed_2026081401/clamp_oracle_model.zip"
SPLITS="results/clinical_window_v10_1/frozen/splits_v10_1.json"
SCALES="results/clinical_window_v10_1/frozen/scales_v10_1.json"
OUTDIR="results/clinical_window_v10_1/stage2b/${CAND}/seed_${SEED}"

# 各 Pareto 候选的超参（来自 Stage 2A Optuna 最优 trial）
case "$CAND" in
  trial5)  BLOOD_COST=0.8519; N_EPOCHS=3; LR=2e-5; GAMMA=0.9982; GAE=0.9594; ENT=1e-4; CLIP=0.2503; TKL=0.0333 ;;
  trial17) BLOOD_COST=2.1960; N_EPOCHS=3; LR=2.9e-5; GAMMA=0.9957; GAE=0.9059; ENT=1.56e-5; CLIP=0.2354; TKL=0.0782 ;;
  trial18) BLOOD_COST=2.9048; N_EPOCHS=5; LR=4e-5; GAMMA=0.9965; GAE=0.9423; ENT=1e-4; CLIP=0.2356; TKL=0.073 ;;
  *) echo "unknown candidate: $CAND" >&2; exit 1 ;;
esac

python train_clinical_window_ppo.py \
  --splits "$SPLITS" --scales "$SCALES" --output-dir "$OUTDIR" \
  --timesteps 50000 --n-envs 8 --n-steps 256 --batch-size 256 \
  --n-epochs "$N_EPOCHS" --learning-rate "$LR" --gamma "$GAMMA" \
  --gae-lambda "$GAE" --ent-coef "$ENT" --clip-range "$CLIP" --target-kl "$TKL" \
  --time-cost 1.0 --blood-cost "$BLOOD_COST" \
  --progress-bonus 0.0 --seal-progress-bonus 0.0 \
  --stagnation-penalty-cap 0.0 --two-cell-loop-penalty 0.0 --clinical-cost-cap 10.0 \
  --front-tension-cost 0.0 --organ-energy-cost 0.0 --vessel-strain-cost 0.0 \
  --completion-bonus 5.0 --failure-penalty 10.0 --invalid-action-penalty 10.0 \
  --init-model "$ORACLE" \
  --control-mode hierarchical --early-end-mode threshold --early-end-minutes 10 \
  --freeze-target-head --freeze-features-extractor \
  --seed "$SEED" --device "$DEV" --torch-threads 4 \
  --checkpoint-global-interval 25000 \
  > logs/v10_1_stage2b_${CAND}_seed${SEED}_${DEV}.log 2>&1
echo "DONE $CAND seed=$SEED device=$DEV"

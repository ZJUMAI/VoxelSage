#!/usr/bin/env bash
# v10.1 Stage 2C: threshold-5 Pareto 候选 3-seed 重训 50k
# 用法: bash run_v101_stage2c_train.sh <candidate> <seed> <phys_gpu>
#   candidate: trial18 | trial19
#   seed: 新 seed
#   phys_gpu: 物理 GPU 编号 (0/1/4/5/6)
set -euo pipefail

CAND="$1"; SEED="$2"; PHYS="$3"
INIT="results/clinical_window_v10_1/stage2b/trial18/seed_2026081703/final_model.zip"
SPLITS="results/clinical_window_v10_1/frozen/splits_v10_1.json"
SCALES="results/clinical_window_v10_1/frozen/scales_v10_1.json"
OUTDIR="results/clinical_window_v10_1/stage2c/${CAND}/seed_${SEED}"

case "$CAND" in
  trial18) BLOOD_COST=0.8564; N_EPOCHS=5; LR=1e-5; GAMMA=0.9978; GAE=0.9299; ENT=6e-4; CLIP=0.2427; TKL=0.0109 ;;
  trial19) BLOOD_COST=0.7822; N_EPOCHS=3; LR=1e-5; GAMMA=0.9977; GAE=0.9692; ENT=1e-4; CLIP=0.1792; TKL=0.0163 ;;
  *) echo "unknown candidate: $CAND" >&2; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=$PHYS python train_clinical_window_ppo.py \
  --splits "$SPLITS" --scales "$SCALES" --output-dir "$OUTDIR" \
  --timesteps 50000 --n-envs 8 --n-steps 256 --batch-size 256 \
  --n-epochs "$N_EPOCHS" --learning-rate "$LR" --gamma "$GAMMA" \
  --gae-lambda "$GAE" --ent-coef "$ENT" --clip-range "$CLIP" --target-kl "$TKL" \
  --time-cost 1.0 --blood-cost "$BLOOD_COST" \
  --progress-bonus 0.0 --seal-progress-bonus 0.0 \
  --stagnation-penalty-cap 0.0 --two-cell-loop-penalty 0.0 --clinical-cost-cap 10.0 \
  --front-tension-cost 0.0 --organ-energy-cost 0.0 --vessel-strain-cost 0.0 \
  --completion-bonus 5.0 --failure-penalty 10.0 --invalid-action-penalty 10.0 \
  --init-model "$INIT" \
  --control-mode hierarchical --early-end-mode threshold --early-end-minutes 5 \
  --freeze-target-head --freeze-features-extractor \
  --seed "$SEED" --device cuda:0 --torch-threads 4 \
  --checkpoint-global-interval 25000 \
  > logs/v10_1_stage2c_${CAND}_seed${SEED}_phys${PHYS}.log 2>&1
echo "DONE $CAND seed=$SEED physGPU=$PHYS"

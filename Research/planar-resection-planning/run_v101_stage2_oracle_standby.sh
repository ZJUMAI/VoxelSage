#!/usr/bin/env bash
# v10.1 Stage 2 待命脚本：timing-oracle 预训练 clamp head
# 用法：等 audit manifest 判定 GO 后执行
#   bash run_v101_stage2_oracle_standby.sh
set -euo pipefail

MANIFEST="results/clinical_window_v10_1/evaluation/stage1b_checkpoint_audit_validation64/manifest.json"
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest 尚未生成，审计未完成" >&2
  exit 1
fi

BEST_MODEL=$(python3 -c "
import json
m=json.load(open('$MANIFEST'))
print(m['best_model'])
")
DECISION=$(python3 -c "
import json
m=json.load(open('$MANIFEST'))
print(m['decision'])
")
echo "=== audit decision: $DECISION ==="
echo "=== best_model: $BEST_MODEL ==="

if [ "$DECISION" != "GO" ] || [ "$BEST_MODEL" = "None" ] || [ -z "$BEST_MODEL" ]; then
  echo "NO-GO: 无 feasible checkpoint，不启动 oracle" >&2
  exit 2
fi

nohup python train_clamp_timing_oracle.py \
  --model "$BEST_MODEL" \
  --splits results/clinical_window_v10_1/frozen/splits_v10_1.json \
  --split train --scenario-limit 256 \
  --scales results/clinical_window_v10_1/frozen/scales_v10_1.json \
  --output-dir results/clinical_window_v10_1/oracle/threshold10_seed_2026081401 \
  --early-end-mode threshold --early-end-minutes 10 \
  --max-examples 1024 --sample-every 8 \
  --epochs 20 --batch-size 128 --learning-rate 3e-4 \
  --time-cost 1 --blood-cost 1 --seed 2026081401 --device cuda:0 \
  > logs/v10_1_stage2_oracle.log 2>&1 &
echo "oracle PID: $!"

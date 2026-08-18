#!/usr/bin/env bash
# Stage 评估通用脚本（macro control）
# 用法: run_v9_eval.sh <algorithm: serpentine|ppo> <model_or_NA> <output_json> [extra_args...]
#   model 参数在 serpentine 时传 NA
set -euo pipefail

ALGO="${1:?algorithm required}"
MODEL="${2:?model required}"
OUTPUT="${3:?output json required}"
shift 3

cd "$(dirname "$0")"

ARGS=(--control-mode macro \
  --splits results/clinical_window_v5/frozen/splits_stage_d_eval_v5.json \
  --split validation --limit 16 --algorithm "${ALGO}" \
  --scales results/clinical_window_v5/frozen/scales_v5.json \
  --early-end-mode disabled \
  --output "${OUTPUT}")

if [[ "${ALGO}" == "ppo" ]]; then
  ARGS+=(--model "${MODEL}")
fi

MPLCONFIGDIR=/tmp/matplotlib-clinical-macro python clinical_window_evaluation.py "${ARGS[@]}" "$@"

#!/bin/bash
set -euo pipefail

mkdir -p logs checkpoints results/eval

run_eval() {
  local GPU="$1"
  local VOL_RUN="$2"
  local CKPT="$3"
  local LOCAL_NAME="$4"
  local STAGGER="$5"

  local LOCAL_DIR="./checkpoints/${CKPT}"
  local GPU_CFG="./config_retry${GPU}.yaml"

  sleep "$STAGGER"
  echo "[$(date '+%H:%M:%S')] START $LOCAL_NAME on GPU $GPU"

  if [ ! -d "$LOCAL_DIR" ]; then
    uv run modal volume get --force training-runs "${VOL_RUN}/${CKPT}" ./checkpoints/
  fi

  uv run cleanslate eval \
    --config "$GPU_CFG" \
    -m "$LOCAL_DIR" \
    -d unlearning-cleanslate/cleanslate_dataset \
    -o "results/eval/${LOCAL_NAME}" \
    --push-suffix w100-s10

  echo "[$(date '+%H:%M:%S')] DONE $LOCAL_NAME"
}

run_eval 0 simnpo_Llama-3.1-8B_20260502_013045   checkpoint-134 llama-checkpoint-134 0  >> logs/retry-llama-ckpt-134.log  2>&1 &
run_eval 1 simnpo_Llama-3.1-8B_20260502_013045   checkpoint-201 llama-checkpoint-201 3  >> logs/retry-llama-ckpt-201.log  2>&1 &
run_eval 2 simnpo_OLMo-3-1025-7B_20260502_013043 checkpoint-192 olmo7b-checkpoint-192 6  >> logs/retry-olmo7b-ckpt-192.log 2>&1 &
run_eval 3 simnpo_Qwen3-8B_20260502_013042        checkpoint-80  qwen-checkpoint-80   9  >> logs/retry-qwen-ckpt-80.log   2>&1 &
run_eval 4 simnpo_Qwen3-8B_20260502_013042        checkpoint-240 qwen-checkpoint-240  12 >> logs/retry-qwen-ckpt-240.log  2>&1 &

wait
echo "All retries done."

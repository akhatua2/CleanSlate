#!/bin/bash
set -euo pipefail

JUDGE_BASE_URL="https://swe-b--qwen3-5-9b-serve.modal.run"
JUDGE_MODEL="Qwen/Qwen3.5-9B"

mkdir -p logs checkpoints results/eval results/validation

run_ckpt() {
  local GPU="$1"
  local VOL_RUN="$2"
  local CKPT="$3"       # e.g. checkpoint-67
  local LOCAL_NAME="$4" # e.g. llama-checkpoint-67  (used for results dirs + log label)
  local PUSH_SUFFIX="$5"
  local STAGGER="$6"    # seconds to sleep before download (avoids Modal rate limits)

  local LOCAL_DIR="./checkpoints/${CKPT}"
  local GPU_CFG="./config_gpu${GPU}.yaml"

  sleep "$STAGGER"
  echo "[$(date '+%H:%M:%S')] START $LOCAL_NAME on GPU $GPU"

  uv run modal volume get --force training-runs "${VOL_RUN}/${CKPT}" ./checkpoints/

  uv run cleanslate eval \
    --config "$GPU_CFG" \
    -m "$LOCAL_DIR" \
    -d unlearning-cleanslate/cleanslate_dataset \
    -o "results/eval/${LOCAL_NAME}" \
    --push-suffix w100-s10

  JUDGE_BASE_URL="$JUDGE_BASE_URL" \
  JUDGE_MODEL="$JUDGE_MODEL" \
  HF_ALLOW_CODE_EVAL=1 \
  uv run cleanslate validate \
    --config "$GPU_CFG" \
    -m "$LOCAL_DIR" \
    -B full \
    -o "results/validation/${LOCAL_NAME}" \
    --push-to-hub \
    --push-suffix "$PUSH_SUFFIX"

  echo "[$(date '+%H:%M:%S')] DONE $LOCAL_NAME"
}

# 8 checkpoints launch on GPUs 0-7 in parallel, staggered 3s each for Modal rate limits
run_ckpt 0 simnpo_Llama-3.1-8B_20260502_013045   checkpoint-67  llama-checkpoint-67  simnpo-gentle-checkpoint-67   0  >> logs/llama-ckpt-67.log  2>&1 & P0=$!
run_ckpt 1 simnpo_Llama-3.1-8B_20260502_013045   checkpoint-134 llama-checkpoint-134 simnpo-gentle-checkpoint-134  3  >> logs/llama-ckpt-134.log 2>&1 &
run_ckpt 2 simnpo_Llama-3.1-8B_20260502_013045   checkpoint-201 llama-checkpoint-201 simnpo-gentle-checkpoint-201  6  >> logs/llama-ckpt-201.log 2>&1 &
run_ckpt 3 simnpo_OLMo-3-1025-7B_20260502_013043 checkpoint-64  olmo7b-checkpoint-64  simnpo-gentle-checkpoint-64   9  >> logs/olmo7b-ckpt-64.log  2>&1 &
run_ckpt 4 simnpo_OLMo-3-1025-7B_20260502_013043 checkpoint-128 olmo7b-checkpoint-128 simnpo-gentle-checkpoint-128  12 >> logs/olmo7b-ckpt-128.log 2>&1 &
run_ckpt 5 simnpo_OLMo-3-1025-7B_20260502_013043 checkpoint-192 olmo7b-checkpoint-192 simnpo-gentle-checkpoint-192  15 >> logs/olmo7b-ckpt-192.log 2>&1 &
run_ckpt 6 simnpo_Qwen3-8B_20260502_013042        checkpoint-80  qwen-checkpoint-80   simnpo-gentle-checkpoint-80   18 >> logs/qwen-ckpt-80.log   2>&1 &
run_ckpt 7 simnpo_Qwen3-8B_20260502_013042        checkpoint-160 qwen-checkpoint-160  simnpo-gentle-checkpoint-160  21 >> logs/qwen-ckpt-160.log  2>&1 &

# 9th job: wait for GPU 0 to free up, then run
wait $P0
run_ckpt 0 simnpo_Qwen3-8B_20260502_013042 checkpoint-240 qwen-checkpoint-240 simnpo-gentle-checkpoint-240 0 >> logs/qwen-ckpt-240.log 2>&1 &

wait
echo "All 9 checkpoints done."

#!/usr/bin/env bash
set -uo pipefail
source .venv/bin/activate
ulimit -n 1048576
mkdir -p logs

CKPT="runs/llama-3_1-8b/baseline-simnpo_gentle/simnpo_Llama-3.1-8B_20260427_200535/checkpoint-134"
PUSH_SUFFIX="debug-checkpoint-134-llama"
OUT_DIR="results/post_val/llama-8b-simnpo-baseline/validate"

# --- 1. Judge server on GPU 7 -----------------------------------------
CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 42800 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    > logs/judge_server.log 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null || true' EXIT

echo "Waiting for judge server (pid=$JUDGE_PID) to come up..."
until curl -sf http://127.0.0.1:42800/v1/models >/dev/null; do
  sleep 5
  kill -0 $JUDGE_PID 2>/dev/null || { echo "judge server died"; tail -50 logs/judge_server.log; exit 1; }
done
echo "Judge server ready."

# --- 2. Validator env (mirrors create_pre_val_paths.sh) ----------------
export JUDGE_BASE_URL=http://127.0.0.1:42800
export JUDGE_MODEL=Qwen/Qwen3.5-9B
export HF_ALLOW_CODE_EVAL=1
export HF_HOME="$LOCAL_MACHINE_CACHE"
export HF_HUB_CACHE="$LOCAL_MACHINE_CACHE/hub"
export HF_DATASETS_CACHE="$LOCAL_MACHINE_CACHE/datasets"
export TRANSFORMERS_CACHE="$LOCAL_MACHINE_CACHE/transformers"
export HF_TOKENIZERS_CACHE="$LOCAL_MACHINE_CACHE/tokenizers"
export VLLM_CACHE_DIR="$LOCAL_MACHINE_CACHE/vllm"
unset GEMINI_API_KEY

# --- 3. Validate unlearned checkpoint on GPU 6 -------------------------
CUDA_VISIBLE_DEVICES=6 cleanslate validate \
    -m "$CKPT" \
    -B full \
    --skip-done \
    -o "$OUT_DIR" \
    --push-to-hub \
    --push-suffix "$PUSH_SUFFIX" \
    2>&1 | tee logs/validate_llama-8b-simnpo-baseline.log

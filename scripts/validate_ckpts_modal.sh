#!/usr/bin/env bash
# Deploy 9 intermediate checkpoints as Modal vLLM endpoints, then validate each.
#
# Pattern mirrors post_eval_sweep.sh:
#   Phase 1: modal deploy (9 apps in parallel)
#   Phase 2: cleanslate validate --base-url <url> (9 in parallel)

set -uo pipefail
cd "$(dirname "$0")/.."

JUDGE_URL="https://swe-b--qwen3-5-9b-serve.modal.run"
JUDGE_MODEL="Qwen/Qwen3.5-9B"
HF_ORG="arpandeepk"

LOGDIR="logs/validate_ckpts_modal"
mkdir -p "$LOGDIR"

# Format: "volume_path:push_suffix:hf_base_model"
declare -A CKPTS=(
  ["llama-3-1-8b-simnpo-gentle-ckpt-67"]="/runs/simnpo_Llama-3.1-8B_20260502_013045/checkpoint-67:simnpo-gentle-checkpoint-67:meta-llama/Llama-3.1-8B"
  ["llama-3-1-8b-simnpo-gentle-ckpt-134"]="/runs/simnpo_Llama-3.1-8B_20260502_013045/checkpoint-134:simnpo-gentle-checkpoint-134:meta-llama/Llama-3.1-8B"
  ["llama-3-1-8b-simnpo-gentle-ckpt-201"]="/runs/simnpo_Llama-3.1-8B_20260502_013045/checkpoint-201:simnpo-gentle-checkpoint-201:meta-llama/Llama-3.1-8B"
  ["olmo-3-7b-simnpo-gentle-ckpt-64"]="/runs/simnpo_OLMo-3-1025-7B_20260502_013043/checkpoint-64:simnpo-gentle-checkpoint-64:allenai/OLMo-3-1025-7B"
  ["olmo-3-7b-simnpo-gentle-ckpt-128"]="/runs/simnpo_OLMo-3-1025-7B_20260502_013043/checkpoint-128:simnpo-gentle-checkpoint-128:allenai/OLMo-3-1025-7B"
  ["olmo-3-7b-simnpo-gentle-ckpt-192"]="/runs/simnpo_OLMo-3-1025-7B_20260502_013043/checkpoint-192:simnpo-gentle-checkpoint-192:allenai/OLMo-3-1025-7B"
  ["qwen3-8b-simnpo-gentle-ckpt-80"]="/runs/simnpo_Qwen3-8B_20260502_013042/checkpoint-80:simnpo-gentle-checkpoint-80:Qwen/Qwen3-8B"
  ["qwen3-8b-simnpo-gentle-ckpt-160"]="/runs/simnpo_Qwen3-8B_20260502_013042/checkpoint-160:simnpo-gentle-checkpoint-160:Qwen/Qwen3-8B"
  ["qwen3-8b-simnpo-gentle-ckpt-240"]="/runs/simnpo_Qwen3-8B_20260502_013042/checkpoint-240:simnpo-gentle-checkpoint-240:Qwen/Qwen3-8B"
)

# Parse the three colon-separated fields
_vol_path()   { echo "${1%%:*}"; }
_push_suffix(){ local s="${1#*:}"; echo "${s%%:*}"; }
_hf_base()    { echo "${1##*:}"; }

# --- Phase 1: Deploy 9 vLLM serve apps in parallel --------------------------
echo "=== Phase 1: Deploy 9 checkpoint serve apps ==="
for APP_NAME in "${!CKPTS[@]}"; do
  VAL="${CKPTS[$APP_NAME]}"
  MODEL_PATH="$(_vol_path "$VAL")"
  HF_BASE="$(_hf_base "$VAL")"
  nohup env \
    MODEL="$MODEL_PATH" \
    SERVED_MODEL_NAME="$HF_BASE" \
    APP_NAME="$APP_NAME" \
    GPU_TYPE=H100 GPU_COUNT=1 MAX_MODEL_LEN=4096 \
    uv run modal deploy scripts/modal_vllm_server.py \
    > "$LOGDIR/deploy_${APP_NAME}.log" 2>&1 &
done
echo "  waiting for all deploys..."
wait
echo "  all deploys done"

# --- Phase 2: Validate all 9 in parallel ------------------------------------
echo
echo "=== Phase 2: Validate 9 checkpoints ==="
for APP_NAME in "${!CKPTS[@]}"; do
  VAL="${CKPTS[$APP_NAME]}"
  PUSH_SUFFIX="$(_push_suffix "$VAL")"
  HF_BASE="$(_hf_base "$VAL")"
  URL="https://swe-b--${APP_NAME}-serve.modal.run"

  nohup env \
    JUDGE_BASE_URL="$JUDGE_URL" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    HF_ALLOW_CODE_EVAL=1 \
    uv run cleanslate validate \
      -m "$HF_BASE" \
      -B full \
      --base-url "$URL" \
      --hf-org "$HF_ORG" \
      -o "results/validation_modal/${APP_NAME}" \
      --push-to-hub \
      --push-suffix "$PUSH_SUFFIX" \
    > "$LOGDIR/validate_${APP_NAME}.log" 2>&1 &
done

echo "  9 validates running. Monitor:"
echo "  tail -F $LOGDIR/validate_*.log"

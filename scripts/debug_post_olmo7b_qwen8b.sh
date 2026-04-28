#!/usr/bin/env bash
# Debug post-train pipeline on existing checkpoints, run SERIALLY.
#   - GPU 3:    judge server (Qwen/Qwen3.5-9B via vLLM)
#   - GPU 6,7: each model's eval (DP=2 via config_dp2_gpu67.yaml),
#              followed by its validate on GPU 6 (judge already pinned to 3,
#              so validate has GPU 6 to itself).
#
# Order: olmo eval -> olmo validate -> qwen eval -> qwen validate.
set -uo pipefail
source .venv/bin/activate
ulimit -n 1048576
mkdir -p logs

DEBUG_TAG="debug-post"
HF_ORG="unlearning-cleanslate"

OLMO_CKPT="/lfs/skampere3/0/animjha/CleanSlate/runs/olmo-3-7b/baseline-simnpo_gentle-debug-olmo-7b/simnpo_Olmo-3-1025-7B_20260428_064411"
QWEN_CKPT="/lfs/skampere3/0/animjha/CleanSlate/runs/qwen3-8b/baseline-simnpo_gentle-debug-qwen-8b/simnpo_Qwen3-8B_20260428_063109"

OLMO_TAG="olmo-3-7b-baseline-simnpo_gentle-${DEBUG_TAG}"
QWEN_TAG="qwen3-8b-baseline-simnpo_gentle-${DEBUG_TAG}"

OLMO_EVAL_OUT="results/post_eval/${OLMO_TAG}"
OLMO_VAL_OUT="results/post_val/${OLMO_TAG}/validate"
QWEN_EVAL_OUT="results/post_eval/${QWEN_TAG}"
QWEN_VAL_OUT="results/post_val/${QWEN_TAG}/validate"

OLMO_LOG="logs/${OLMO_TAG}"
QWEN_LOG="logs/${QWEN_TAG}"
JUDGE_LOG="logs/${DEBUG_TAG}_judge_server.log"

echo "[debug] olmo ckpt: $OLMO_CKPT"
echo "[debug] qwen ckpt: $QWEN_CKPT"

# --- 1. Judge server on GPU 3 -----------------------------------------
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 42800 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null || true; jobs -p | xargs -r kill 2>/dev/null || true' EXIT

echo "[debug] waiting for judge server (pid=$JUDGE_PID)..."
until curl -sf http://127.0.0.1:42800/v1/models >/dev/null; do
  sleep 5
  kill -0 $JUDGE_PID 2>/dev/null || { echo "[debug] judge server died"; tail -50 "$JUDGE_LOG"; exit 1; }
done
echo "[debug] judge ready."

# --- 2. Validator env -------------------------------------------------
export JUDGE_BASE_URL=http://127.0.0.1:42800
export JUDGE_MODEL=Qwen/Qwen3.5-9B
export HF_ALLOW_CODE_EVAL=1
if [[ -n "${LOCAL_MACHINE_CACHE:-}" ]]; then
  export HF_HOME="$LOCAL_MACHINE_CACHE"
  export HF_HUB_CACHE="$LOCAL_MACHINE_CACHE/hub"
  export HF_DATASETS_CACHE="$LOCAL_MACHINE_CACHE/datasets"
  export TRANSFORMERS_CACHE="$LOCAL_MACHINE_CACHE/transformers"
  export HF_TOKENIZERS_CACHE="$LOCAL_MACHINE_CACHE/tokenizers"
  export VLLM_CACHE_DIR="$LOCAL_MACHINE_CACHE/vllm"
fi
unset GEMINI_API_KEY

# --- 3. Per-leg runner: eval (DP=2 on 6,7) then validate (GPU 6) ------
EVAL_CONFIG="config_dp2_gpu67.yaml"
VALIDATE_GPU=6

run_leg() {
  local ckpt="$1" eval_out="$2" val_out="$3" log_pfx="$4" suffix="$5"

  echo "[debug][$suffix] eval (DP=2 on 6,7) — config=$EVAL_CONFIG"
  cleanslate eval \
    -m "$ckpt" \
    --config "$EVAL_CONFIG" \
    -o "$eval_out" \
    --hf-org "$HF_ORG" \
    --push-suffix "$suffix" \
    2>&1 | tee "${log_pfx}_eval.log"
  local eval_rc=${PIPESTATUS[0]}
  if [[ $eval_rc -ne 0 ]]; then
    echo "[debug][$suffix] eval FAILED (rc=$eval_rc) — see ${log_pfx}_eval.log"
    return $eval_rc
  fi

  echo "[debug][$suffix] validate on GPU $VALIDATE_GPU"
  CUDA_VISIBLE_DEVICES="$VALIDATE_GPU" cleanslate validate \
    -m "$ckpt" \
    -B full \
    --skip-done \
    -o "$val_out" \
    --push-to-hub \
    --hf-org "$HF_ORG" \
    --push-suffix "$suffix" \
    2>&1 | tee "${log_pfx}_validate.log"
  return ${PIPESTATUS[0]}
}

OLMO_RC=0; QWEN_RC=0
run_leg "$OLMO_CKPT" "$OLMO_EVAL_OUT" "$OLMO_VAL_OUT" "$OLMO_LOG" "$DEBUG_TAG-olmo" || OLMO_RC=$?
run_leg "$QWEN_CKPT" "$QWEN_EVAL_OUT" "$QWEN_VAL_OUT" "$QWEN_LOG" "$DEBUG_TAG-qwen" || QWEN_RC=$?

echo "[debug] olmo leg rc=$OLMO_RC  qwen leg rc=$QWEN_RC"
echo "  olmo eval log:     ${OLMO_LOG}_eval.log"
echo "  olmo validate log: ${OLMO_LOG}_validate.log"
echo "  qwen eval log:     ${QWEN_LOG}_eval.log"
echo "  qwen validate log: ${QWEN_LOG}_validate.log"

[[ "$OLMO_RC" -eq 0 && "$QWEN_RC" -eq 0 ]] || exit 1
echo "[debug] DONE."

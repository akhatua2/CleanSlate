#!/usr/bin/env bash
# Debug post-train pipeline on existing checkpoints, run in PARALLEL.
#   - GPU 3:  judge server (Qwen/Qwen3.5-9B via vLLM)
#   - GPU 6:  olmo eval (DP=1) -> olmo validate
#   - GPU 7:  qwen eval (DP=1) -> qwen validate
#
# Eval GPU pin lives in the config files (config.yaml -> [6], config_dp1_gpu7.yaml -> [7]).
# Validate uses CUDA_VISIBLE_DEVICES (no per-task config knob).
# Each model's leg runs as a single background process — eval then validate, sequentially within
# the leg, so each GPU goes from eval->validate without contention.
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
    --gpu-memory-utilization 0.90 \
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

# --- 3. Per-leg runner: eval (DP=1, config-pinned GPU) then validate (env-pinned GPU) -
run_leg() {
  local ckpt="$1" eval_cfg="$2" eval_out="$3" val_out="$4" val_gpu="$5" log_pfx="$6" suffix="$7"

  echo "[debug][$suffix] eval (DP=1, config=$eval_cfg)"
  cleanslate eval \
    -m "$ckpt" \
    --config "$eval_cfg" \
    -o "$eval_out" \
    --hf-org "$HF_ORG" \
    --push-suffix "$suffix" \
    > "${log_pfx}_eval.log" 2>&1
  local eval_rc=$?
  if [[ $eval_rc -ne 0 ]]; then
    echo "[debug][$suffix] eval FAILED (rc=$eval_rc) — see ${log_pfx}_eval.log"
    return $eval_rc
  fi

  echo "[debug][$suffix] validate on GPU $val_gpu"
  CUDA_VISIBLE_DEVICES="$val_gpu" cleanslate validate \
    -m "$ckpt" \
    -B full \
    --skip-done \
    -o "$val_out" \
    --push-to-hub \
    --hf-org "$HF_ORG" \
    --push-suffix "$suffix" \
    > "${log_pfx}_validate.log" 2>&1
  return $?
}

# --- 4. Launch both legs in PARALLEL ----------------------------------
echo "[debug] launching olmo (GPU 6) + qwen (GPU 7) in parallel"

run_leg "$OLMO_CKPT" "config.yaml"           "$OLMO_EVAL_OUT" "$OLMO_VAL_OUT" 6 "$OLMO_LOG" "${DEBUG_TAG}-olmo" &
OLMO_PID=$!
run_leg "$QWEN_CKPT" "config_dp1_gpu7.yaml"  "$QWEN_EVAL_OUT" "$QWEN_VAL_OUT" 7 "$QWEN_LOG" "${DEBUG_TAG}-qwen" &
QWEN_PID=$!

echo "[debug] olmo pid=$OLMO_PID, qwen pid=$QWEN_PID"

OLMO_RC=0; QWEN_RC=0
wait "$OLMO_PID" || OLMO_RC=$?
wait "$QWEN_PID" || QWEN_RC=$?

echo "[debug] olmo leg rc=$OLMO_RC  qwen leg rc=$QWEN_RC"
echo "  olmo eval log:     ${OLMO_LOG}_eval.log"
echo "  olmo validate log: ${OLMO_LOG}_validate.log"
echo "  qwen eval log:     ${QWEN_LOG}_eval.log"
echo "  qwen validate log: ${QWEN_LOG}_validate.log"

[[ "$OLMO_RC" -eq 0 && "$QWEN_RC" -eq 0 ]] || exit 1
echo "[debug] DONE."

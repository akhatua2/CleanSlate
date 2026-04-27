#!/usr/bin/env bash
# Post-eval sweep: for each unlearned checkpoint, run memorization eval +
# benchmark validation in parallel on Modal.
#
# Layout:
#   Lane A: cleanslate eval --modal      → ephemeral H100 → eval-<X> on HF
#   Lane B: deploy modal vLLM serve     → cleanslate validate -B full → generations-<X> on HF
#
# Logs: logs/post_eval_sweep/{deploy,eval,validate}_<model>.log
# Results: results/post_eval/<model>/{eval,validate}/

set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=(
  "llama-3_1-8b-simnpo-gentle-baseline"
  "qwen3-8b-simnpo-gentle-baseline"
  "gemma-3-12b-simnpo-gentle-baseline"
  "llama-3_1-8b-simnpo-gentle-bm25-10b"
  "qwen3-8b-simnpo-gentle-bm25-10b"
  "llama-3_1-8b-simnpo-gentle-igm-10b"
  "qwen3-8b-simnpo-gentle-igm-10b"
  "llama-3_1-8b-simnpo-gentle-bm25-6t"
  "qwen3-8b-simnpo-gentle-bm25-6t"
  "llama-3_1-8b-undial-baseline"
  "qwen3-8b-undial-baseline"
  "llama-3_1-8b-rmu-baseline"
  "qwen3-8b-rmu-baseline"
)

JUDGE_URL="https://swe-b--qwen3-5-9b-serve.modal.run"
JUDGE_MODEL="Qwen/Qwen3.5-9B"

LOGDIR="logs/post_eval_sweep"
RESULTSDIR="results/post_eval"
mkdir -p "$LOGDIR" "$RESULTSDIR"

# --- Phase 1: Deploy 13 model-serve apps in parallel -------------------------
echo "=== Phase 1: Deploy 13 model-serve apps ==="
for M in "${MODELS[@]}"; do
  REPO="unlearning-cleanslate/$M"
  nohup env MODEL="$REPO" GPU_TYPE=H100 GPU_COUNT=1 MAX_MODEL_LEN=4096 \
    uv run modal deploy scripts/modal_vllm_server.py \
    > "$LOGDIR/deploy_$M.log" 2>&1 &
done
echo "  ${#MODELS[@]} deploys in flight; waiting..."
wait
echo "  all deploys completed"

# --- Phase 2: Fire eval --modal × 13 (memorization re-eval) ------------------
echo
echo "=== Phase 2: Memorization eval × 13 in parallel ==="
for M in "${MODELS[@]}"; do
  REPO="unlearning-cleanslate/$M"
  nohup uv run cleanslate eval \
    -m "$REPO" \
    -o "$RESULTSDIR/$M/eval" \
    --modal --modal-gpu H100 \
    > "$LOGDIR/eval_$M.log" 2>&1 &
done

# --- Phase 3: Fire validate × 13 (full benchmark suite) ----------------------
echo
echo "=== Phase 3: Benchmark validation × 13 in parallel ==="
for M in "${MODELS[@]}"; do
  REPO="unlearning-cleanslate/$M"
  # Modal app name = basename, lowercased, _ and . replaced with -.
  APP_NAME=$(echo "$M" | tr '_.' '--' | tr '[:upper:]' '[:lower:]')
  URL="https://swe-b--${APP_NAME}-serve.modal.run"

  nohup env \
    JUDGE_BASE_URL="$JUDGE_URL" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    HF_ALLOW_CODE_EVAL=1 \
    uv run cleanslate validate \
      -m "$REPO" \
      -B full \
      --base-url "$URL" \
      -o "$RESULTSDIR/$M/validate" \
      --push-to-hub \
    > "$LOGDIR/validate_$M.log" 2>&1 &
done

echo
echo "All 26 jobs fired (13 eval + 13 validate)."
echo "Logs:  $LOGDIR/"
echo "Out:   $RESULTSDIR/"
echo
echo "Monitor:  tail -F $LOGDIR/*.log"
echo "Status:   for f in $LOGDIR/{eval,validate}_*.log; do echo \$(basename \$f) \$(tail -1 \$f); done"

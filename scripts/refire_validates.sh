#!/usr/bin/env bash
# Re-fire the 13 validates with HF_ALLOW_CODE_EVAL=1 (the missing env var
# that crashed all 13 in the first sweep on humaneval_plus's code_eval metric).
# Eval (memorization) lane is unaffected and still running — don't touch it.

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

for M in "${MODELS[@]}"; do
  REPO="unlearning-cleanslate/$M"
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
    > "$LOGDIR/validate_${M}_v2.log" 2>&1 &
done

echo "13 validates re-fired with HF_ALLOW_CODE_EVAL=1"
echo "Logs:  $LOGDIR/validate_*_v2.log"

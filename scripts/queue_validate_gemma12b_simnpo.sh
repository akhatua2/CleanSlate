#!/usr/bin/env bash
# Wait for gemma-12b memorization post-eval (PID $GEMMA_PID) to finish, then run
# utility-retention benchmarks on GPU 4 (TP=1).
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="logs/post_eval_20260418"
mkdir -p "$LOG_DIR"

GEMMA_PID="${GEMMA_PID:-3497802}"

echo "[$(date -Is)] waiting for gemma memorization eval PID $GEMMA_PID to exit..."
while kill -0 "$GEMMA_PID" 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] gemma memorization eval exited. starting validation."

set -a; . .env; set +a

GEMMA_CKPT="results/final/unlearnt_models/gemma-3-12b-pt/simnpo_gentle_corpus_sweep/simnpo_gemma-3-12b-pt_20260416_171305"
OUT="results/final/validation/gemma-12b-simnpo/corpus_sweep_post_eval.json"

export CUDA_VISIBLE_DEVICES=4

cleanslate validate \
  -m "$GEMMA_CKPT" \
  -B mmlu,hellaswag,arc_easy,arc_challenge,piqa,winogrande,cleanslate_qa \
  --tp-size 1 \
  --gpu-mem 0.90 \
  --max-model-len 4096 \
  -o "$OUT" 2>&1 | tee "$LOG_DIR/gemma12b_validate.log"

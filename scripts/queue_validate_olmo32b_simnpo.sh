#!/usr/bin/env bash
# Wait for olmo-32b memorization post-eval (PID $OLMO_PID) to finish, then run
# utility-retention benchmarks on GPUs 0-3 (TP=4).
set -uo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="logs/post_eval_20260418"
mkdir -p "$LOG_DIR"

# Default to the current olmo eval PID; override with OLMO_PID env var if needed.
OLMO_PID="${OLMO_PID:-3472042}"

echo "[$(date -Is)] waiting for olmo memorization eval PID $OLMO_PID to exit..."
while kill -0 "$OLMO_PID" 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] olmo memorization eval exited. starting validation."

# Load GEMINI_API_KEY for cleanslate_qa LLM judge
set -a; . .env; set +a

OLMO_CKPT="results/final/unlearnt_models/olmo-32b/simnpo_gentle_corpus_sweep/simnpo_Olmo-3-1125-32B_20260416_164134"
OUT="results/final/validation/olmo-32b-simnpo/corpus_sweep_post_eval.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3

cleanslate validate \
  -m "$OLMO_CKPT" \
  -B mmlu,hellaswag,arc_easy,arc_challenge,piqa,winogrande,cleanslate_qa \
  --tp-size 4 \
  --gpu-mem 0.90 \
  --max-model-len 4096 \
  -o "$OUT" 2>&1 | tee "$LOG_DIR/olmo32b_validate.log"

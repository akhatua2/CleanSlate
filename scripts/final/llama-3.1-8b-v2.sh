#!/usr/bin/env bash
# ============================================================================
# CleanSlate Final Pipeline — llama-3.1-8b (SimNPO only, char-based eval v2)
# QA eval skipped (no API key)
#
# Usage: ./scripts/final/llama-3.1-8b-v2.sh --local 4,5,6,7
# ============================================================================
set -euo pipefail

ALIAS="llama-8b-v2"
MODEL="meta-llama/Llama-3.1-8B"
EVAL_DATASET="unlearning-cleanslate/verif-llama-3.1-8b-w100-s10"
CLI=".venv/bin/cleanslate"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="logs/final/${ALIAS}"
mkdir -p "$LOG_DIR"

POOL_DIR="data/final/${ALIAS}/pool"
CURATED_DIR="data/final/curated/${ALIAS}/baseline"
VAL_DIR="results/final/validation/${ALIAS}"
UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}"
SCORE_DIR="results/final/score/${ALIAS}"

find_checkpoint() {
    local match
    match=$(ls -t "$1"/*/config.json 2>/dev/null | head -1)
    if [[ -n "$match" ]]; then dirname "$match"; fi
}

if [[ "${1:-}" == "--local" ]]; then
    GPU_IDS="${2:?Usage: $0 --local <gpu_ids>}"
else
    GPU_IDS="${1:?Usage: $0 --local <gpu_ids>  OR  $0 <gpu_ids>}"
fi
IFS=',' read -ra GPUS <<< "$GPU_IDS"
FIRST_GPU="${GPUS[0]}"
FIRST_TWO="${GPUS[0]},${GPUS[1]:-${GPUS[0]}}"

mkdir -p "$POOL_DIR" "$CURATED_DIR" "$VAL_DIR" "$UNLEARN_DIR" \
         "$POST_EVAL_DIR" "$SCORE_DIR"

echo "=== ${ALIAS} pipeline (local)  |  GPUs: ${GPU_IDS}  |  $(date) ==="

# Phase 1: Pre-validation (GSM8K only, skip QA)
echo "[Phase 1] Pre-unlearn validation ..."
if [[ -f "${VAL_DIR}/pre_gsm8k.json" ]]; then echo "  SKIP: pre_gsm8k.json exists"
else CUDA_VISIBLE_DEVICES="$FIRST_GPU" $CLI fsid validate \
    -m "$MODEL" -B gsm8k -o "${VAL_DIR}/pre_gsm8k.json" --num-gpus 1 --mem-fraction 0.80; fi

# Phase 2: Build pool + curate
echo "[Phase 2] Build pool + curate ..."
if [[ -f "${POOL_DIR}/pool.parquet" ]]; then echo "  SKIP: pool exists"
else $CLI fsid build-pool --eval-dataset "$EVAL_DATASET" --threshold 0.30 -o "$POOL_DIR"; fi
if [[ -f "${CURATED_DIR}/forget_set.parquet" ]]; then echo "  SKIP: curated"
else $CLI fsid curate --forget-pool-path "${POOL_DIR}/forget.parquet" -o "$CURATED_DIR"; fi

# Phase 3: Train SimNPO
OBJ=simnpo
if [[ -n "$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")" ]]; then echo "[Phase 3] SKIP: ${OBJ} checkpoint exists"
else
    echo "[Phase 3] Training ${OBJ} ..."
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" $CLI fsid train \
        --config "cleanslate/fsid/configs/${OBJ}.yaml" -m "$MODEL" \
        --forget-path "${CURATED_DIR}/forget_set.parquet" \
        --retain-path "${CURATED_DIR}/retain_set.parquet" \
        -o "${UNLEARN_DIR}/${OBJ}" --batch-size 16 --grad-accum 2 --epochs 1
fi

# Phase 4: Post-eval
if compgen -G "${POST_EVAL_DIR}/${OBJ}/*/results.parquet" > /dev/null 2>&1; then echo "[Phase 4] SKIP: ${OBJ} post-eval exists"
else
    CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")
    NGPUS=${#GPUS[@]}
    echo "[Phase 4] Post-eval ${OBJ} (dp=${NGPUS}) — checkpoint: ${CKPT} ..."
    CUDA_VISIBLE_DEVICES="$GPU_IDS" $CLI eval --hf-dataset "$EVAL_DATASET" --hf-config default --model "$CKPT" \
        --num-gpus "$NGPUS" --gpu-ids "$GPU_IDS" --output "${POST_EVAL_DIR}/${OBJ}" --no-push
fi

# Phase 5: Post-validation (GSM8K only, skip QA)
CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")
if [[ -f "${VAL_DIR}/post_${OBJ}_gsm8k.json" ]]; then echo "[Phase 5] SKIP: post_${OBJ}_gsm8k.json exists"
else CUDA_VISIBLE_DEVICES="$FIRST_GPU" $CLI fsid validate \
    -m "$CKPT" -B gsm8k -o "${VAL_DIR}/post_${OBJ}_gsm8k.json" --num-gpus 1 --mem-fraction 0.80; fi

# Phase 6: Scoring (always rerun)
echo "[Phase 6] Scoring ${OBJ} ..."
$CLI fsid score --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-eval "$EVAL_DATASET" --post-eval "${POST_EVAL_DIR}/${OBJ}" \
    --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
    --post-validation "${VAL_DIR}/post_${OBJ}_gsm8k.json" \
    -o "${SCORE_DIR}/${OBJ}_score.json"

echo "=== ${ALIAS} pipeline complete (local)  |  $(date) ==="

#!/usr/bin/env bash
# ============================================================================
# CleanSlate Final Pipeline — nemotron-9b (local B200 cluster)
#
# Usage: ./scripts/final/nemotron-9b.sh 6,7    # GPU IDs (minimum 2 for tp=2 eval)
# ============================================================================
set -euo pipefail

# ── HF / model cache setup ─────────────────────────────────────────────────
export LOCAL_MACHINE_CACHE="${LOCAL_MACHINE_CACHE:-/lfs/skampere3/0/animjha/.cache}"
export HF_HOME="$LOCAL_MACHINE_CACHE"
export HF_HUB_CACHE="$LOCAL_MACHINE_CACHE/hub"
export HF_DATASETS_CACHE="$LOCAL_MACHINE_CACHE/datasets"
export TRANSFORMERS_CACHE="$LOCAL_MACHINE_CACHE/transformers"
export HF_TOKENIZERS_CACHE="$LOCAL_MACHINE_CACHE/tokenizers"
export VLLM_CACHE_DIR="$LOCAL_MACHINE_CACHE/vllm"

# ── Phase 0: Variables ──────────────────────────────────────────────────────
ALIAS="nemotron-9b"
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
EVAL_DATASET="unlearning-cleanslate/eval_cleanslate_dataset_nemotron-nano-9b"

GPU_IDS="${1:?Usage: $0 <gpu_ids>  (e.g. 6,7)}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
FIRST_GPU="${GPUS[0]}"
FIRST_TWO="${GPUS[0]},${GPUS[1]:-${GPUS[0]}}"

POOL_DIR="data/final/${ALIAS}/pool"
CURATED_DIR="data/final/curated/${ALIAS}/baseline"
VAL_DIR="results/final/validation/${ALIAS}"
UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}"
SCORE_DIR="results/final/score/${ALIAS}"

mkdir -p "$POOL_DIR" "$CURATED_DIR" "$VAL_DIR" "$UNLEARN_DIR" \
         "$POST_EVAL_DIR" "$SCORE_DIR"

# Resolve actual checkpoint path (trainer saves to a timestamped subfolder)
find_checkpoint() {
    local match
    match=$(ls -t "$1"/*/config.json 2>/dev/null | head -1)
    if [[ -n "$match" ]]; then dirname "$match"; fi
}

echo "=== ${ALIAS} pipeline  |  GPUs: ${GPU_IDS}  |  $(date) ==="

# ── Phase 1 ──────────────────────────────────────────────────────────────
echo "[Phase 1] Pre-unlearn validation ..."
if [[ -f "${VAL_DIR}/pre_gsm8k.json" ]]; then
    echo "  SKIP: pre_gsm8k.json exists"
else
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid validate \
        -m "$MODEL" -B gsm8k -o "${VAL_DIR}/pre_gsm8k.json" --num-gpus 1 --mem-fraction 0.80
fi
if [[ -f "${VAL_DIR}/pre_qa.json" ]]; then
    echo "  SKIP: pre_qa.json exists"
else
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid qa eval \
        -m "$MODEL" -o "${VAL_DIR}/pre_qa.json"
fi

# ── Phase 2 ──────────────────────────────────────────────────────────────
echo "[Phase 2] Build pool + curate ..."
if [[ -f "${POOL_DIR}/pool.parquet" ]]; then
    echo "  SKIP: pool already built"
else
    cleanslate fsid build-pool --eval-dataset "$EVAL_DATASET" --threshold 0.30 -o "$POOL_DIR"
fi
if [[ -f "${CURATED_DIR}/forget_set.parquet" ]]; then
    echo "  SKIP: already curated"
else
    cleanslate fsid curate --forget-pool-path "${POOL_DIR}/forget.parquet" -o "$CURATED_DIR"
fi

# ── Phase 3 ──────────────────────────────────────────────────────────────
for OBJ in npo simnpo; do
    if [[ -n "$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")" ]]; then
        echo "[Phase 3] SKIP: ${OBJ} checkpoint exists"
    else
        echo "[Phase 3] Training ${OBJ} ..."
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid train \
            --config "cleanslate/fsid/configs/${OBJ}.yaml" \
            -m "$MODEL" \
            --forget-path "${CURATED_DIR}/forget_set.parquet" \
            --retain-path "${CURATED_DIR}/retain_set.parquet" \
            -o "${UNLEARN_DIR}/${OBJ}" \
            --batch-size 16 --grad-accum 2 --epochs 1
    fi
done

# ── Phase 4 ──────────────────────────────────────────────────────────────
for OBJ in npo simnpo; do
    if compgen -G "${POST_EVAL_DIR}/${OBJ}/*.parquet" > /dev/null 2>&1; then
        echo "[Phase 4] SKIP: ${OBJ} post-eval exists"
    else
        CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")
        echo "[Phase 4] Post-eval ${OBJ} (tp=2) — checkpoint: ${CKPT} ..."
        cleanslate eval -d "$EVAL_DATASET" -c default \
            -m "$CKPT" \
            --tp-size 2 --num-gpus 2 --gpu-ids "$FIRST_TWO" \
            -o "${POST_EVAL_DIR}/${OBJ}" --no-push
    fi
done

# ── Phase 5 ──────────────────────────────────────────────────────────────
for OBJ in npo simnpo; do
    CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")
    if [[ -f "${VAL_DIR}/post_${OBJ}_gsm8k.json" ]]; then
        echo "[Phase 5] SKIP: post_${OBJ}_gsm8k.json exists"
    else
        echo "[Phase 5] Post-validation ${OBJ} gsm8k ..."
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid validate \
            -m "$CKPT" -B gsm8k -o "${VAL_DIR}/post_${OBJ}_gsm8k.json" --num-gpus 1 --mem-fraction 0.80
    fi
    if [[ -f "${VAL_DIR}/post_${OBJ}_qa.json" ]]; then
        echo "[Phase 5] SKIP: post_${OBJ}_qa.json exists"
    else
        echo "[Phase 5] Post-validation ${OBJ} QA ..."
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid qa eval -m "$CKPT" -o "${VAL_DIR}/post_${OBJ}_qa.json"
    fi
done

# ── Phase 6 (always rerun) ──────────────────────────────────────────────
for OBJ in npo simnpo; do
    echo "[Phase 6] Scoring ${OBJ} ..."
    cleanslate fsid score \
        --forget-path "${CURATED_DIR}/forget_set.parquet" \
        --pre-eval "$EVAL_DATASET" --post-eval "${POST_EVAL_DIR}/${OBJ}" \
        --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
        --post-validation "${VAL_DIR}/post_${OBJ}_gsm8k.json" \
        -o "${SCORE_DIR}/${OBJ}_score.json"
    cleanslate fsid qa score \
        --forget-path "${CURATED_DIR}/forget_set.parquet" \
        --pre-qa "${VAL_DIR}/pre_qa.json" --post-qa "${VAL_DIR}/post_${OBJ}_qa.json" \
        -o "${SCORE_DIR}/${OBJ}_qa_score.json"
done

echo "=== ${ALIAS} pipeline complete  |  $(date) ==="

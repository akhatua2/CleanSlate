#!/usr/bin/env bash
# ============================================================================
# CleanSlate Final Pipeline — olmo-32b (local B200 cluster)
#
# Usage: ./scripts/final/olmo-32b.sh 4,5    # GPU IDs (minimum 2 for FSDP)
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
ALIAS="olmo-32b"
MODEL="allenai/Olmo-3-1125-32B"
EVAL_DATASET="unlearning-cleanslate/eval_cleanslate_dataset_olmo-32b"

GPU_IDS="${1:?Usage: $0 <gpu_ids>  (e.g. 4,5)}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

if (( NUM_GPUS < 2 )); then
    echo "ERROR: olmo-32b requires at least 2 GPUs (got ${NUM_GPUS})" >&2
    exit 1
fi

FIRST_GPU="${GPUS[0]}"
FIRST_TWO="${GPUS[0]},${GPUS[1]}"

# Directory layout
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

# ── Phase 1: Pre-unlearn validation ──────────────────────────────────────
echo "[Phase 1] Pre-unlearn validation ..."

if [[ -f "${VAL_DIR}/pre_gsm8k.json" ]]; then
    echo "  SKIP: ${VAL_DIR}/pre_gsm8k.json exists"
else
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid validate \
        -m "$MODEL" -B gsm8k \
        -o "${VAL_DIR}/pre_gsm8k.json" \
        --num-gpus 1 --mem-fraction 0.80
fi

if [[ -f "${VAL_DIR}/pre_qa.json" ]]; then
    echo "  SKIP: ${VAL_DIR}/pre_qa.json exists"
else
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid qa eval \
        -m "$MODEL" \
        -o "${VAL_DIR}/pre_qa.json"
fi

echo "[Phase 1] Done."

# ── Phase 2: Build pool + curate ────────────────────────────────────────
echo "[Phase 2] Build pool + curate ..."

if [[ -f "${POOL_DIR}/pool.parquet" ]]; then
    echo "  SKIP: pool already built"
else
    cleanslate fsid build-pool \
        --eval-dataset "$EVAL_DATASET" \
        --threshold 0.30 \
        -o "$POOL_DIR"
fi

if [[ -f "${CURATED_DIR}/forget_set.parquet" ]]; then
    echo "  SKIP: already curated"
else
    cleanslate fsid curate \
        --forget-pool-path "${POOL_DIR}/forget.parquet" \
        -o "$CURATED_DIR"
fi

echo "[Phase 2] Done."

# ── Phase 3: Train NPO + SimNPO (FSDP, 2 GPUs, sequential) ─────────────
for OBJ in npo simnpo; do
    if [[ -n "$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")" ]]; then
        echo "[Phase 3] SKIP: ${OBJ} checkpoint exists"
    else
        echo "[Phase 3] Training ${OBJ} (FSDP, GPUs ${FIRST_TWO}) ..."
        CUDA_VISIBLE_DEVICES="$FIRST_TWO" accelerate launch \
            --num_processes 2 \
            -m cleanslate fsid train \
            --config "cleanslate/fsid/configs/${OBJ}.yaml" \
            -m "$MODEL" \
            --forget-path "${CURATED_DIR}/forget_set.parquet" \
            --retain-path "${CURATED_DIR}/retain_set.parquet" \
            -o "${UNLEARN_DIR}/${OBJ}" \
            --distributed-strategy fsdp \
            --fsdp-sharding-strategy full_shard \
            --gradient-checkpointing \
            --fsdp-cpu-offload \
            --batch-size 8 --grad-accum 2 --epochs 1
    fi
done
echo "[Phase 3] Done."

# ── Phase 4: Post-unlearn eval (tp=2) ───────────────────────────────────
for OBJ in npo simnpo; do
    if compgen -G "${POST_EVAL_DIR}/${OBJ}/*.parquet" > /dev/null 2>&1; then
        echo "[Phase 4] SKIP: ${OBJ} post-eval exists"
    else
        CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")
        echo "[Phase 4] Post-eval ${OBJ} (tp=2) — checkpoint: ${CKPT} ..."
        cleanslate eval \
            -d "$EVAL_DATASET" -c default \
            -m "$CKPT" \
            --tp-size 2 --num-gpus 2 --gpu-ids "$FIRST_TWO" \
            -o "${POST_EVAL_DIR}/${OBJ}" \
            --no-push
    fi
done
echo "[Phase 4] Done."

# ── Phase 5: Post-unlearn validation ─────────────────────────────────────
for OBJ in npo simnpo; do
    CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")

    if [[ -f "${VAL_DIR}/post_${OBJ}_gsm8k.json" ]]; then
        echo "[Phase 5] SKIP: post_${OBJ}_gsm8k.json exists"
    else
        echo "[Phase 5] Post-validation ${OBJ} gsm8k ..."
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid validate \
            -m "$CKPT" -B gsm8k \
            -o "${VAL_DIR}/post_${OBJ}_gsm8k.json" \
            --num-gpus 1 --mem-fraction 0.80
    fi

    if [[ -f "${VAL_DIR}/post_${OBJ}_qa.json" ]]; then
        echo "[Phase 5] SKIP: post_${OBJ}_qa.json exists"
    else
        echo "[Phase 5] Post-validation ${OBJ} QA ..."
        CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid qa eval \
            -m "$CKPT" \
            -o "${VAL_DIR}/post_${OBJ}_qa.json"
    fi
done
echo "[Phase 5] Done."

# ── Phase 6: Score (always rerun — cheap) ────────────────────────────────
for OBJ in npo simnpo; do
    echo "[Phase 6] Scoring ${OBJ} ..."

    cleanslate fsid score \
        --forget-path "${CURATED_DIR}/forget_set.parquet" \
        --pre-eval "$EVAL_DATASET" \
        --post-eval "${POST_EVAL_DIR}/${OBJ}" \
        --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
        --post-validation "${VAL_DIR}/post_${OBJ}_gsm8k.json" \
        -o "${SCORE_DIR}/${OBJ}_score.json"

    cleanslate fsid qa score \
        --forget-path "${CURATED_DIR}/forget_set.parquet" \
        --pre-qa "${VAL_DIR}/pre_qa.json" \
        --post-qa "${VAL_DIR}/post_${OBJ}_qa.json" \
        -o "${SCORE_DIR}/${OBJ}_qa_score.json"
done
echo "[Phase 6] Done."

echo "=== ${ALIAS} pipeline complete  |  $(date) ==="

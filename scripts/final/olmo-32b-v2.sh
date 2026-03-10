#!/usr/bin/env bash
# ============================================================================
# CleanSlate — olmo-32b SimNPO v2 (aggressive unlearning, 2x B200)
#
# Usage: ./scripts/final/olmo-32b-v2.sh 6,7
# ============================================================================
set -euo pipefail

ALIAS="olmo-32b"
MODEL="allenai/Olmo-3-1125-32B"
EVAL_DATASET="unlearning-cleanslate/eval_cleanslate_dataset_olmo-32b"

GPU_IDS="${1:?Usage: $0 <gpu_ids>  (e.g. 6,7)}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
FIRST_GPU="${GPUS[0]}"

CURATED_DIR="data/final/curated/${ALIAS}/baseline_t030"
VAL_DIR="results/final/validation/${ALIAS}"
UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}/simnpo_v2_t030"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}/simnpo_v2"
SCORE_DIR="results/final/score/${ALIAS}"

mkdir -p "$UNLEARN_DIR" "$POST_EVAL_DIR" "$SCORE_DIR"

find_checkpoint() {
    local match
    match=$(ls -t "$1"/*/config.json 2>/dev/null | head -1)
    if [[ -n "$match" ]]; then dirname "$match"; fi
}

echo "=== ${ALIAS} SimNPO-v2 (aggressive)  |  GPUs: ${GPU_IDS}  |  $(date) ==="

# ── Train ──────────────────────────────────────────────────────────────
if [[ -n "$(find_checkpoint "$UNLEARN_DIR")" ]]; then
    echo "[Train] SKIP: checkpoint exists"
else
    echo "[Train] SimNPO aggressive (FSDP, GPUs ${GPU_IDS}) ..."
    CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
        --num_processes "$NUM_GPUS" \
        -m cleanslate fsid train \
        --config cleanslate/fsid/configs/simnpo.yaml \
        -m "$MODEL" \
        --forget-path "${CURATED_DIR}/forget_set.parquet" \
        --retain-path "${CURATED_DIR}/retain_set.parquet" \
        -o "$UNLEARN_DIR" \
        --distributed-strategy fsdp \
        --fsdp-sharding-strategy full_shard \
        --gradient-checkpointing \
        --fsdp-cpu-offload \
        --batch-size 1 --grad-accum 8 --epochs 5
fi
echo "[Train] Done."

# ── Post-eval (sliding window) ─────────────────────────────────────────
if compgen -G "${POST_EVAL_DIR}/*/results.parquet" > /dev/null 2>&1; then
    echo "[Eval] SKIP: post-eval exists"
else
    CKPT=$(find_checkpoint "$UNLEARN_DIR")
    echo "[Eval] Post-eval (tp=2, GPUs ${GPU_IDS}) — checkpoint: ${CKPT} ..."
    cleanslate eval \
        -d "$EVAL_DATASET" -c default \
        -m "$CKPT" \
        --tp-size 2 --num-gpus 2 \
        --gpu-ids "$GPU_IDS" \
        -o "$POST_EVAL_DIR" \
        --no-push
fi
echo "[Eval] Done."

# ── Post-validation (gsm8k + QA) ──────────────────────────────────────
CKPT=$(find_checkpoint "$UNLEARN_DIR")

if [[ -f "${VAL_DIR}/post_simnpo_v2_gsm8k.json" ]]; then
    echo "[Val] SKIP: post_simnpo_v2_gsm8k.json exists"
else
    echo "[Val] Post-validation gsm8k ..."
    cleanslate fsid validate \
        -m "$CKPT" -B gsm8k \
        -o "${VAL_DIR}/post_simnpo_v2_gsm8k.json" \
        --num-gpus 2 --tp-size 2 --gpu-ids "$GPU_IDS" --mem-fraction 0.80
fi

if [[ -f "${VAL_DIR}/post_simnpo_v2_qa.json" ]]; then
    echo "[Val] SKIP: post_simnpo_v2_qa.json exists"
else
    echo "[Val] Post-validation QA ..."
    cleanslate fsid qa eval \
        -m "$CKPT" \
        -o "${VAL_DIR}/post_simnpo_v2_qa.json" \
        -g 2 --tp-size 2 --gpu-ids "$GPU_IDS"
fi
echo "[Val] Done."

# ── Score ──────────────────────────────────────────────────────────────
echo "[Score] Scoring simnpo_v2 ..."
cleanslate fsid score \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-eval "$EVAL_DATASET" \
    --post-eval "$POST_EVAL_DIR" \
    --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
    --post-validation "${VAL_DIR}/post_simnpo_v2_gsm8k.json" \
    -o "${SCORE_DIR}/simnpo_v2_score.json"

cleanslate fsid qa score \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-qa "${VAL_DIR}/pre_qa.json" \
    --post-qa "${VAL_DIR}/post_simnpo_v2_qa.json" \
    -o "${SCORE_DIR}/simnpo_v2_qa_score.json"
echo "[Score] Done."

echo "=== ${ALIAS} SimNPO-v2 complete  |  $(date) ==="

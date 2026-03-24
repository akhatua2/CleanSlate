#!/usr/bin/env bash
# ============================================================================
# CleanSlate — olmo-32b gentle checkpoint (t005): train + eval pipeline
#
# Waits for 2 GPUs idle (<=10 MiB) for 5 minutes, then trains gentle SimNPO
# with FSDP and runs post-eval, validation, and scoring.
#
# Usage: ./scripts/final/olmo-32b-gentle-t005.sh
# ============================================================================
set -euo pipefail

ALIAS="olmo-32b"
MODEL="allenai/Olmo-3-1125-32B"
EVAL_DATASET="unlearning-cleanslate/eval_cleanslate_dataset_olmo-32b"

CURATED_DIR="data/final/curated/${ALIAS}/baseline"
UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}/simnpo_gentle"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}/simnpo_gentle"
VAL_DIR="results/final/validation/${ALIAS}"
SCORE_DIR="results/final/score/${ALIAS}"

mkdir -p "$UNLEARN_DIR" "$POST_EVAL_DIR" "$SCORE_DIR"

# Resolve actual checkpoint path
find_checkpoint() {
    local match
    match=$(ls -t "$1"/*/config.json 2>/dev/null | head -1)
    if [[ -n "$match" ]]; then dirname "$match"; fi
}

# ── Wait for 2 idle GPUs ─────────────────────────────────────────────
REQUIRED_GPUS=2
IDLE_THRESHOLD_MIB=10
IDLE_WAIT_SECONDS=300   # 5 minutes continuous idle
CHECK_INTERVAL=30

echo "Waiting for ${REQUIRED_GPUS} GPUs idle (≤${IDLE_THRESHOLD_MIB} MiB) for ${IDLE_WAIT_SECONDS}s ..."

declare -A idle_since=()

while true; do
    now=$(date +%s)
    while IFS=', ' read -r gpu_id mem_used_mib; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used_mib=$(echo "$mem_used_mib" | tr -d ' MiB')
        if (( mem_used_mib <= IDLE_THRESHOLD_MIB )); then
            if [[ -z "${idle_since[$gpu_id]:-}" ]]; then
                idle_since[$gpu_id]=$now
                echo "  GPU $gpu_id went idle at $(date)"
            fi
        else
            if [[ -n "${idle_since[$gpu_id]:-}" ]]; then
                echo "  GPU $gpu_id no longer idle (${mem_used_mib} MiB)"
            fi
            idle_since[$gpu_id]=""
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    ready_gpus=()
    for gpu_id in "${!idle_since[@]}"; do
        since="${idle_since[$gpu_id]:-}"
        if [[ -n "$since" ]] && (( now - since >= IDLE_WAIT_SECONDS )); then
            ready_gpus+=("$gpu_id")
        fi
    done

    if (( ${#ready_gpus[@]} >= REQUIRED_GPUS )); then
        SELECTED=()
        for (( i=0; i<REQUIRED_GPUS; i++ )); do
            SELECTED+=("${ready_gpus[$i]}")
        done
        GPU_IDS=$(IFS=','; echo "${SELECTED[*]}")
        echo "Found ${REQUIRED_GPUS} idle GPUs: ${GPU_IDS}"
        break
    fi

    sleep "$CHECK_INTERVAL"
done

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "=== ${ALIAS} gentle (t005) train+eval  |  GPUs: ${GPU_IDS}  |  $(date) ==="

# ── Train gentle SimNPO (FSDP) ──────────────────────────────────────
if [[ -n "$(find_checkpoint "$UNLEARN_DIR")" ]]; then
    echo "[Train] SKIP: gentle checkpoint exists"
else
    echo "[Train] Training gentle SimNPO (FSDP, GPUs ${GPU_IDS}) ..."
    CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
        --num_processes "$NUM_GPUS" \
        -m cleanslate fsid train \
        --config "cleanslate/fsid/configs/simnpo_gentle.yaml" \
        -m "$MODEL" \
        --forget-path "${CURATED_DIR}/forget_set.parquet" \
        --retain-path "${CURATED_DIR}/retain_set.parquet" \
        -o "$UNLEARN_DIR" \
        --distributed-strategy fsdp \
        --fsdp-sharding-strategy full_shard \
        --gradient-checkpointing \
        --fsdp-cpu-offload \
        --batch-size 1 --grad-accum 8 --epochs 1
fi

CKPT=$(find_checkpoint "$UNLEARN_DIR")
echo "[Train] Done. Checkpoint: ${CKPT}"

# ── Post-eval ────────────────────────────────────────────────────────
if compgen -G "${POST_EVAL_DIR}/*/results.parquet" > /dev/null 2>&1; then
    echo "[Eval] SKIP: post-eval exists"
else
    echo "[Eval] Post-eval (GPU ${GPUS[0]}) ..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" cleanslate eval \
        -d "$EVAL_DATASET" -c default \
        -m "$CKPT" \
        --num-gpus 1 \
        -o "$POST_EVAL_DIR" \
        --no-push
fi
echo "[Eval] Done."

# ── Validation ───────────────────────────────────────────────────────
if [[ -f "${VAL_DIR}/post_simnpo_gentle_gsm8k.json" ]]; then
    echo "[Val] SKIP: post_simnpo_gentle_gsm8k.json exists"
else
    echo "[Val] Post-validation gsm8k ..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" cleanslate fsid validate \
        -m "$CKPT" -B gsm8k \
        -o "${VAL_DIR}/post_simnpo_gentle_gsm8k.json" \
        --num-gpus 1 --mem-fraction 0.80
fi

if [[ -f "${VAL_DIR}/post_simnpo_gentle_qa.json" ]]; then
    echo "[Val] SKIP: post_simnpo_gentle_qa.json exists"
else
    echo "[Val] Post-validation QA ..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" cleanslate fsid qa eval \
        -m "$CKPT" \
        -o "${VAL_DIR}/post_simnpo_gentle_qa.json" \
        -g 1
fi
echo "[Val] Done."

# ── Score ─────────────────────────────────────────────────────────────
echo "[Score] Scoring simnpo_gentle ..."
cleanslate fsid score \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-eval "$EVAL_DATASET" \
    --post-eval "$POST_EVAL_DIR" \
    --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
    --post-validation "${VAL_DIR}/post_simnpo_gentle_gsm8k.json" \
    -o "${SCORE_DIR}/simnpo_gentle_score.json"

cleanslate fsid qa score \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-qa "${VAL_DIR}/pre_qa.json" \
    --post-qa "${VAL_DIR}/post_simnpo_gentle_qa.json" \
    -o "${SCORE_DIR}/simnpo_gentle_qa_score.json"
echo "[Score] Done."

echo "=== ${ALIAS} gentle (t005) complete  |  $(date) ==="

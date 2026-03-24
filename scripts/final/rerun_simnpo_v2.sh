#!/usr/bin/env bash
# ============================================================================
# Rerun SimNPO with new config (v2) for all 5 models
#
# New config changes (simnpo.yaml):
#   npo_beta: 2.0 (was 4.5), simnpo_gamma: 0.5 (was 0.0),
#   forget_weight: 5.0 (was 0.125), lr: 5e-5 (was 1e-5),
#   epochs: 2 (was 1), grad_accum: 16 (was 2), warmup: 10 (was 20)
#
# Old artifacts already renamed with _old_v1 suffix.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PARTITION="${SLURM_PARTITION:-tamper_resistance}"

declare -A MODELS=(
    [olmo-7b]="allenai/Olmo-3-1025-7B"
    [llama-8b]="meta-llama/Llama-3.1-8B"
    [qwen-8b]="Qwen/Qwen3-8B"
    [qwen-8b-base]="Qwen/Qwen3-8B-Base"
    [trinity-nano]="arcee-ai/Trinity-Nano-Base"
)

declare -A EVAL_DATASETS=(
    [olmo-7b]="unlearning-cleanslate/eval_cleanslate_dataset_olmo-7b"
    [llama-8b]="unlearning-cleanslate/eval_cleanslate_dataset_llama-3_1-8b"
    [qwen-8b]="unlearning-cleanslate/eval_cleanslate_dataset_qwen3-8b"
    [qwen-8b-base]="unlearning-cleanslate/eval_cleanslate_dataset_qwen3-8b-base"
    [trinity-nano]="unlearning-cleanslate/eval_cleanslate_dataset_trinity-nano"
)

# Allow selecting specific models: ./rerun_simnpo_v2.sh olmo-7b llama-8b
if [[ $# -gt 0 ]]; then
    ALIASES=("$@")
else
    ALIASES=(olmo-7b llama-8b qwen-8b qwen-8b-base trinity-nano)
fi

for ALIAS in "${ALIASES[@]}"; do
    MODEL="${MODELS[$ALIAS]}"
    EVAL_DATASET="${EVAL_DATASETS[$ALIAS]}"
    LOG_DIR="logs/final/${ALIAS}"
    mkdir -p "$LOG_DIR"

    echo "=== Submitting ${ALIAS} SimNPO v2 job chain ==="

    # Job A: Train (2 GPU, FSDP)
    JOB_A=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="${ALIAS}-simnpo-v2" \
        --output="${LOG_DIR}/simnpo_v2_%j.log" \
        --export="ALIAS=${ALIAS},MODEL=${MODEL}" \
        "${SCRIPT_DIR}/simnpo_v2_train.slurm")
    echo "  Job A (train):     ${JOB_A}"

    # Job B: Post-eval + validation (2 GPU)
    JOB_B=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="${ALIAS}-posteval-v2" \
        --output="${LOG_DIR}/posteval_v2_%j.log" \
        --dependency="afterok:${JOB_A}" \
        --export="ALIAS=${ALIAS},EVAL_DATASET=${EVAL_DATASET}" \
        "${SCRIPT_DIR}/simnpo_v2_posteval.slurm")
    echo "  Job B (post-eval): ${JOB_B}  (after ${JOB_A})"

    # Job C: Score (1 GPU)
    JOB_C=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="${ALIAS}-score-v2" \
        --output="${LOG_DIR}/score_v2_%j.log" \
        --dependency="afterok:${JOB_B}" \
        --export="ALIAS=${ALIAS},EVAL_DATASET=${EVAL_DATASET}" \
        "${SCRIPT_DIR}/simnpo_v2_score.slurm")
    echo "  Job C (score):     ${JOB_C}  (after ${JOB_B})"

    echo "=== ${ALIAS} chain: ${JOB_A} → ${JOB_B} → ${JOB_C} ==="
    echo
done

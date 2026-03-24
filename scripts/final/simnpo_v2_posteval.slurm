#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Usage: sbatch --partition=<P> --job-name=<name> --output=<log> \
#        --export=ALIAS=olmo-7b,EVAL_DATASET=unlearning-cleanslate/... \
#        scripts/final/simnpo_v2_posteval.slurm

set -euo pipefail
cd /data/animesh_jha/CleanSlate
source .venv/bin/activate

UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}"
VAL_DIR="results/final/validation/${ALIAS}"

# Find newest checkpoint (not _old_v1)
CKPT=$(dirname "$(ls -t ${UNLEARN_DIR}/simnpo/*/config.json 2>/dev/null | head -1)")
echo "=== SimNPO v2 post-eval: ${ALIAS} ==="
echo "  Checkpoint: ${CKPT}"

# Post-eval (memorization eval, tp=2)
cleanslate eval -d "${EVAL_DATASET}" -c default -m "${CKPT}" \
    --tp-size 2 --num-gpus 2 -o "${POST_EVAL_DIR}/simnpo" --no-push

# Validation: GSM8K
cleanslate fsid validate -m "${CKPT}" -B gsm8k \
    -o "${VAL_DIR}/post_simnpo_gsm8k.json" --num-gpus 1 --mem-fraction 0.80

# Validation: QA
cleanslate fsid qa eval -m "${CKPT}" -o "${VAL_DIR}/post_simnpo_qa.json"

echo "=== Post-eval complete ==="

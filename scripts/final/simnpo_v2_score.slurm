#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Usage: sbatch --partition=<P> --job-name=<name> --output=<log> \
#        --export=ALIAS=olmo-7b,EVAL_DATASET=unlearning-cleanslate/... \
#        scripts/final/simnpo_v2_score.slurm

set -euo pipefail
cd /data/animesh_jha/CleanSlate
source .venv/bin/activate

CURATED_DIR="data/final/curated/${ALIAS}/baseline"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}"
VAL_DIR="results/final/validation/${ALIAS}"
SCORE_DIR="results/final/score/${ALIAS}"

echo "=== SimNPO v2 score: ${ALIAS} ==="

cleanslate fsid score --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-eval "${EVAL_DATASET}" --post-eval "${POST_EVAL_DIR}/simnpo" \
    --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
    --post-validation "${VAL_DIR}/post_simnpo_gsm8k.json" \
    -o "${SCORE_DIR}/simnpo_score.json"

cleanslate fsid qa score --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-qa "${VAL_DIR}/pre_qa.json" --post-qa "${VAL_DIR}/post_simnpo_qa.json" \
    -o "${SCORE_DIR}/simnpo_qa_score.json"

echo "=== Score complete ==="

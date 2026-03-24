#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Usage: sbatch --export=GENTLE_WEIGHT=0.75,TAG=merged_g75_a25 ...

set -euo pipefail
cd /data/animesh_jha/CleanSlate
source .venv/bin/activate

CKPT="results/final/unlearnt_models/olmo-7b/simnpo/${TAG}"
EVAL_DATASET="unlearning-cleanslate/eval_cleanslate_dataset_olmo-7b"
POST_EVAL_DIR="results/final/post_eval/olmo-7b/simnpo"
VAL_DIR="results/final/validation/olmo-7b"
SCORE_DIR="results/final/score/olmo-7b"
CURATED_DIR="data/final/curated/olmo-7b/baseline"

# Step 1: Merge weights (CPU only)
echo "=== Merging: gentle_weight=${GENTLE_WEIGHT} ==="
python scripts/final/merge_olmo7b.py --gentle-weight "${GENTLE_WEIGHT}"

# Step 2: Post-eval (memorization, tp=2)
echo "=== Post-eval: ${CKPT} ==="
cleanslate eval -d "${EVAL_DATASET}" -c default -m "${CKPT}" \
    --tp-size 2 --num-gpus 2 -o "${POST_EVAL_DIR}/${TAG}" --no-push

# Step 3: Validation GSM8K
echo "=== GSM8K validation ==="
cleanslate fsid validate -m "${CKPT}" -B gsm8k \
    -o "${VAL_DIR}/post_simnpo_gsm8k_${TAG}.json" --num-gpus 1 --mem-fraction 0.80

# Step 4: Validation QA
echo "=== QA validation ==="
cleanslate fsid qa eval -m "${CKPT}" -o "${VAL_DIR}/post_simnpo_qa_${TAG}.json"

# Step 5: Score
echo "=== Scoring ==="
cleanslate fsid score --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-eval "${EVAL_DATASET}" --post-eval "${POST_EVAL_DIR}/${TAG}" \
    --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
    --post-validation "${VAL_DIR}/post_simnpo_gsm8k_${TAG}.json" \
    -o "${SCORE_DIR}/simnpo_score_${TAG}.json"

cleanslate fsid qa score --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-qa "${VAL_DIR}/pre_qa.json" --post-qa "${VAL_DIR}/post_simnpo_qa_${TAG}.json" \
    -o "${SCORE_DIR}/simnpo_qa_score_${TAG}.json"

echo "=== Done: ${TAG} ==="

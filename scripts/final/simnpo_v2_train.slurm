#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Usage: sbatch --partition=<P> --job-name=<name> --output=<log> \
#        --export=ALIAS=olmo-7b,MODEL=allenai/Olmo-3-1025-7B \
#        scripts/final/simnpo_v2_train.slurm

set -euo pipefail
cd /data/animesh_jha/CleanSlate
source .venv/bin/activate

CURATED_DIR="data/final/curated/${ALIAS}/baseline"
UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}"

echo "=== SimNPO v2 train: ${ALIAS} (${MODEL}) ==="
echo "  Forget: ${CURATED_DIR}/forget_set.parquet"
echo "  Retain: ${CURATED_DIR}/retain_set.parquet"
echo "  Output: ${UNLEARN_DIR}/simnpo"

accelerate launch --num_processes 2 -m cleanslate fsid train \
    --config cleanslate/fsid/configs/simnpo.yaml \
    -m "${MODEL}" \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --retain-path "${CURATED_DIR}/retain_set.parquet" \
    -o "${UNLEARN_DIR}/simnpo" \
    --distributed-strategy fsdp --fsdp-sharding-strategy full_shard \
    --gradient-checkpointing

echo "=== Train complete ==="

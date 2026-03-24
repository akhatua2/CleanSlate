#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8

# Same as simnpo_v2_train.slurm but with --fsdp-cpu-offload
# Needed for llama-8b, qwen-8b, qwen-8b-base (OOM without offload)

set -euo pipefail
cd /data/animesh_jha/CleanSlate
source .venv/bin/activate

CURATED_DIR="data/final/curated/${ALIAS}/baseline"
UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}"

echo "=== SimNPO v2 train (cpu-offload): ${ALIAS} (${MODEL}) ==="

accelerate launch --num_processes 2 -m cleanslate fsid train \
    --config cleanslate/fsid/configs/simnpo.yaml \
    -m "${MODEL}" \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --retain-path "${CURATED_DIR}/retain_set.parquet" \
    -o "${UNLEARN_DIR}/simnpo" \
    --distributed-strategy fsdp --fsdp-sharding-strategy full_shard \
    --gradient-checkpointing --fsdp-cpu-offload

echo "=== Train complete ==="

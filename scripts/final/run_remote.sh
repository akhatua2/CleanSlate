#!/usr/bin/env bash
# ============================================================================
# CleanSlate Final Pipeline — Remote Orchestrator (Slurm A100 cluster)
#
# Submits all 5 remote model pipelines as Slurm job chains.
# Each model script internally submits 6 dependent sub-jobs (A→F).
# Slurm handles GPU allocation across 8x A100-80GB GPUs.
#
# Usage:
#   ./scripts/final/run_remote.sh
#
# Set SLURM_PARTITION env var to override the default partition:
#   SLURM_PARTITION=tamper_resistance ./scripts/final/run_remote.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

MODELS=(olmo-7b llama-8b qwen-8b qwen-8b-base trinity-nano)

echo "============================================="
echo " CleanSlate Final Pipeline — Slurm Cluster"
echo " Models: ${MODELS[*]}"
echo " Partition: ${SLURM_PARTITION:-tamper_resistance}"
echo " Submitted: $(date)"
echo "============================================="
echo ""

for model in "${MODELS[@]}"; do
    echo "--- ${model} ---"
    "$SCRIPT_DIR/${model}.sh"
    echo ""
done

echo "============================================="
echo " All 5 remote model job chains submitted!"
echo ""
echo " Monitor all jobs:"
echo "   squeue -u \$USER | grep -E '(olmo-7b|llama-8b|qwen-8b|trinity-nano)'"
echo ""
echo " Check logs:"
echo "   ls logs/final/*/  "
echo "============================================="

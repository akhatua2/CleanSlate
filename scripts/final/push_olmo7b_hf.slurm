#!/bin/bash
#SBATCH --partition=tamper_resistance_cpu
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /data/animesh_jha/CleanSlate
source .venv/bin/activate

HF_TOKEN="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"

GENTLE_DIR="results/final/unlearnt_models/olmo-7b/simnpo/simnpo_Olmo-3-1025-7B_20260310_154332_old_v1"
AGGRESSIVE_DIR="results/final/unlearnt_models/olmo-7b/simnpo/simnpo_Olmo-3-1025-7B_20260311_092326"

echo "=== Uploading gentle checkpoint ==="
huggingface-cli upload unlearning-cleanslate/olmo3-7b-unlearnt-gentle "${GENTLE_DIR}" --token "${HF_TOKEN}"

echo "=== Uploading aggressive checkpoint ==="
huggingface-cli upload unlearning-cleanslate/olmo3-7b-unlearnt-aggressive "${AGGRESSIVE_DIR}" --token "${HF_TOKEN}"

echo "=== Done ==="

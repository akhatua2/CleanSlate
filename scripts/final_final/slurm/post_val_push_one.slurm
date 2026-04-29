#!/bin/bash
#SBATCH --job-name=post-val-push
#SBATCH --partition=tamper_resistance_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/final_final/post_val_push/slurm-%j-%x.out

set -euo pipefail

cd /data/animesh_jha/CleanSlate
mkdir -p logs/final_final/post_val_push

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${ROW_ID:?ROW_ID is required}"
: "${HF_TOKEN:?HF_TOKEN must be set in .env}"

MANIFEST="${MANIFEST:-results/final_final/post_eval_manifest.json}"
SLEEP="${PUSH_SLEEP:-2.0}"

echo "============================================================"
echo "[post_val_push] job=$SLURM_JOB_ID row=$ROW_ID sleep=$SLEEP"
echo "[post_val_push] manifest=$MANIFEST"
echo "============================================================"

.venv/bin/python scripts/final_final/slurm/push_post_val.py \
  --row-ids "$ROW_ID" \
  --manifest "$MANIFEST" \
  --sleep "$SLEEP" \
  --touch-sentinel

echo "[post_val_push] DONE row=$ROW_ID"

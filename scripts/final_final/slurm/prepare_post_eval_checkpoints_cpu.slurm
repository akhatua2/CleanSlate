#!/bin/bash
#SBATCH --job-name=prep-post-eval
#SBATCH --partition=tamper_resistance_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_final/post_eval_prep/slurm-%j-%x.out

set -euo pipefail

cd /data/animesh_jha/CleanSlate
mkdir -p logs/final_final/post_eval_prep

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

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
ROW_ARGS=()
if [[ -n "${ROW_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  IDS=($ROW_IDS)
  ROW_ARGS+=(--row-ids "${IDS[@]}")
fi
if [[ -n "${EXCLUDE_ROWS:-}" ]]; then
  for row in $EXCLUDE_ROWS; do
    ROW_ARGS+=(--exclude-row "$row")
  done
fi
if [[ -n "${EXPECTED_ROWS:-}" ]]; then
  ROW_ARGS+=(--expected-rows "$EXPECTED_ROWS")
fi

echo "============================================================"
echo "[prep] job=$SLURM_JOB_ID host=$(hostname)"
echo "[prep] workers=$WORKERS cpus=${SLURM_CPUS_PER_TASK:-unset} mem=${SLURM_MEM_PER_NODE:-unset}"
echo "[prep] row_ids=${ROW_IDS:-all} exclude_rows=${EXCLUDE_ROWS:-none} expected_rows=${EXPECTED_ROWS:-unset}"
echo "[prep] HF_HOME=${HF_HOME:-unset}"
echo "[prep] HF_HUB_CACHE=${HF_HUB_CACHE:-unset}"
echo "[prep] start=$(date -Is)"
echo "============================================================"

scripts/final_final/slurm/prepare_post_eval_checkpoints.py --workers "$WORKERS" "${ROW_ARGS[@]}"

echo "============================================================"
echo "[prep] done=$(date -Is)"
echo "[prep] manifest=results/final_final/post_eval_manifest.tsv"
echo "============================================================"

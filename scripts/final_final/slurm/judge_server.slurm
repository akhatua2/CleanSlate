#!/bin/bash
#SBATCH --job-name=judge-qwen35-9b
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_final/judge_server/slurm-%j.out

set -euo pipefail
cd /data/animesh_jha/CleanSlate
mkdir -p logs/final_final/judge_server results

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

PORT="${JUDGE_PORT:-42800}"
HOST=$(hostname -s)
ENDPOINT_FILE="results/.judge_endpoint"

echo "[judge] job=$SLURM_JOB_ID host=$HOST port=$PORT model=Qwen/Qwen3.5-9B"

# Publish endpoint atomically so dependent jobs see a stable URL.
echo "http://${HOST}:${PORT}" > "${ENDPOINT_FILE}.tmp"
mv "${ENDPOINT_FILE}.tmp" "${ENDPOINT_FILE}"
echo "[judge] wrote endpoint to ${ENDPOINT_FILE}: $(< "${ENDPOINT_FILE}")"

trap 'rm -f "${ENDPOINT_FILE}"' EXIT

vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port "${PORT}" \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --dtype bfloat16

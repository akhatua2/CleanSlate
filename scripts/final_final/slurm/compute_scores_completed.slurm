#!/bin/bash
#SBATCH --job-name=compute-final-scores
#SBATCH --partition=tamper_resistance_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/final_final/scores/slurm-%A_%a-%x.out

set -euo pipefail

cd /data/animesh_jha/CleanSlate
mkdir -p logs/final_final/scores results/scores/final_final

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

: "${HF_TOKEN:?HF_TOKEN must be set in .env}"
export HF_TOKEN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

FORCE="${FORCE:-0}"

# One Slurm array task per completed row. Row 05 is intentionally excluded.
# Fields: row|forget_pool|pre_eval|post_eval|pre_generations|post_generations|checkpoint
ROWS=(
  "01-llama-3_1-8b-simnpo-gentle-baseline-target-100-checkpoint-861|unlearning-cleanslate/fsid-curated-llama-8b-target-100:forget_pool|unlearning-cleanslate/verif-llama-3.1-8b-w100-s10|unlearning-cleanslate/eval-01-llama-3_1-8b-simnpo-gentle-baseline-target-100-checkpoint-861|unlearning-cleanslate/generations-llama-3_1-8b-pre_val|unlearning-cleanslate/generations-01-llama-3_1-8b-simnpo-gentle-baseline-target-100-checkpoint-861|unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-baseline-target-100:checkpoint-861"
  "02-qwen3-8b-simnpo-gentle-baseline-target-100-checkpoint-539|unlearning-cleanslate/fsid-curated-qwen3-8b-target-100:forget_pool|unlearning-cleanslate/verif-qwen3-8b-w100-s10|unlearning-cleanslate/eval-02-qwen3-8b-simnpo-gentle-baseline-target-100-checkpoint-539|unlearning-cleanslate/generations-qwen3-8b-pre_val|unlearning-cleanslate/generations-02-qwen3-8b-simnpo-gentle-baseline-target-100-checkpoint-539|unlearning-cleanslate/qwen3-8b-simnpo-gentle-baseline-target-100:checkpoint-539"
  "03-gemma-3-12b-simnpo-gentle-baseline-target-100-checkpoint-1419|unlearning-cleanslate/fsid-curated-gemma-12b-target-100:forget_pool|unlearning-cleanslate/verif-gemma-3-12b-pt-w100-s10|unlearning-cleanslate/eval-03-gemma-3-12b-simnpo-gentle-baseline-target-100-checkpoint-1419|unlearning-cleanslate/generations-gemma-3-12b-pre_val|unlearning-cleanslate/generations-03-gemma-3-12b-simnpo-gentle-baseline-target-100-checkpoint-1419|unlearning-cleanslate/gemma-3-12b-simnpo-gentle-baseline-target-100:checkpoint-1419"
  "04-gemma-3-12b-simnpo-baseline-target-100-checkpoint-2838|unlearning-cleanslate/fsid-curated-gemma-12b-target-100:forget_pool|unlearning-cleanslate/verif-gemma-3-12b-pt-w100-s10|unlearning-cleanslate/eval-04-gemma-3-12b-simnpo-baseline-target-100-checkpoint-2838|unlearning-cleanslate/generations-gemma-3-12b-pre_val|unlearning-cleanslate/generations-04-gemma-3-12b-simnpo-baseline-target-100-checkpoint-2838|unlearning-cleanslate/gemma-3-12b-simnpo-baseline-target-100:checkpoint-2838"
  "06-llama-3_1-8b-simnpo-gentle-bm25-10b-target-100-checkpoint-398|unlearning-cleanslate/fsid-curated-llama-8b-target-100:forget_pool|unlearning-cleanslate/verif-llama-3.1-8b-w100-s10|unlearning-cleanslate/eval-06-llama-3_1-8b-simnpo-gentle-bm25-10b-target-100-checkpoint-398|unlearning-cleanslate/generations-llama-3_1-8b-pre_val|unlearning-cleanslate/generations-06-llama-3_1-8b-simnpo-gentle-bm25-10b-target-100-checkpoint-398|unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-bm25-10b-target-100:checkpoint-398"
  "07-qwen3-8b-simnpo-gentle-bm25-10b-target-100-checkpoint-374|unlearning-cleanslate/fsid-curated-qwen3-8b-target-100:forget_pool|unlearning-cleanslate/verif-qwen3-8b-w100-s10|unlearning-cleanslate/eval-07-qwen3-8b-simnpo-gentle-bm25-10b-target-100-checkpoint-374|unlearning-cleanslate/generations-qwen3-8b-pre_val|unlearning-cleanslate/generations-07-qwen3-8b-simnpo-gentle-bm25-10b-target-100-checkpoint-374|unlearning-cleanslate/qwen3-8b-simnpo-gentle-bm25-10b-target-100:checkpoint-374"
  "08-llama-3_1-8b-simnpo-gentle-igm-10b-target-100-checkpoint-378|unlearning-cleanslate/fsid-curated-llama-8b-target-100:forget_pool|unlearning-cleanslate/verif-llama-3.1-8b-w100-s10|unlearning-cleanslate/eval-08-llama-3_1-8b-simnpo-gentle-igm-10b-target-100-checkpoint-378|unlearning-cleanslate/generations-llama-3_1-8b-pre_val|unlearning-cleanslate/generations-08-llama-3_1-8b-simnpo-gentle-igm-10b-target-100-checkpoint-378|unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-igm-10b-target-100:checkpoint-378"
  "09-qwen3-8b-simnpo-gentle-igm-10b-target-100-checkpoint-355|unlearning-cleanslate/fsid-curated-qwen3-8b-target-100:forget_pool|unlearning-cleanslate/verif-qwen3-8b-w100-s10|unlearning-cleanslate/eval-09-qwen3-8b-simnpo-gentle-igm-10b-target-100-checkpoint-355|unlearning-cleanslate/generations-qwen3-8b-pre_val|unlearning-cleanslate/generations-09-qwen3-8b-simnpo-gentle-igm-10b-target-100-checkpoint-355|unlearning-cleanslate/qwen3-8b-simnpo-gentle-igm-10b-target-100:checkpoint-355"
  "10-llama-3_1-8b-simnpo-gentle-bm25-6t-target-100-checkpoint-187|unlearning-cleanslate/fsid-curated-llama-8b-target-100:forget_pool|unlearning-cleanslate/verif-llama-3.1-8b-w100-s10|unlearning-cleanslate/eval-10-llama-3_1-8b-simnpo-gentle-bm25-6t-target-100-checkpoint-187|unlearning-cleanslate/generations-llama-3_1-8b-pre_val|unlearning-cleanslate/generations-10-llama-3_1-8b-simnpo-gentle-bm25-6t-target-100-checkpoint-187|unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-bm25-6t-target-100:checkpoint-187"
  "11-qwen3-8b-simnpo-gentle-bm25-6t-target-100-checkpoint-173|unlearning-cleanslate/fsid-curated-qwen3-8b-target-100:forget_pool|unlearning-cleanslate/verif-qwen3-8b-w100-s10|unlearning-cleanslate/eval-11-qwen3-8b-simnpo-gentle-bm25-6t-target-100-checkpoint-173|unlearning-cleanslate/generations-qwen3-8b-pre_val|unlearning-cleanslate/generations-11-qwen3-8b-simnpo-gentle-bm25-6t-target-100-checkpoint-173|unlearning-cleanslate/qwen3-8b-simnpo-gentle-bm25-6t-target-100:checkpoint-173"
  "12-llama-3_1-8b-undial-baseline-target-100-checkpoint-1722|unlearning-cleanslate/fsid-curated-llama-8b-target-100:forget_pool|unlearning-cleanslate/verif-llama-3.1-8b-w100-s10|unlearning-cleanslate/eval-12-llama-3_1-8b-undial-baseline-target-100-checkpoint-1722|unlearning-cleanslate/generations-llama-3_1-8b-pre_val|unlearning-cleanslate/generations-12-llama-3_1-8b-undial-baseline-target-100-checkpoint-1722|unlearning-cleanslate/llama-3_1-8b-undial-baseline-target-100:checkpoint-1722"
  "13-qwen3-8b-undial-baseline-target-100-checkpoint-1078|unlearning-cleanslate/fsid-curated-qwen3-8b-target-100:forget_pool|unlearning-cleanslate/verif-qwen3-8b-w100-s10|unlearning-cleanslate/eval-13-qwen3-8b-undial-baseline-target-100-checkpoint-1078|unlearning-cleanslate/generations-qwen3-8b-pre_val|unlearning-cleanslate/generations-13-qwen3-8b-undial-baseline-target-100-checkpoint-1078|unlearning-cleanslate/qwen3-8b-undial-baseline-target-100:checkpoint-1078"
  "14-llama-3_1-8b-rmu-baseline-target-100-checkpoint-1722|unlearning-cleanslate/fsid-curated-llama-8b-target-100:forget_pool|unlearning-cleanslate/verif-llama-3.1-8b-w100-s10|unlearning-cleanslate/eval-14-llama-3_1-8b-rmu-baseline-target-100-checkpoint-1722|unlearning-cleanslate/generations-llama-3_1-8b-pre_val|unlearning-cleanslate/generations-14-llama-3_1-8b-rmu-baseline-target-100-checkpoint-1722|unlearning-cleanslate/llama-3_1-8b-rmu-baseline-target-100:checkpoint-1722"
  "15-qwen3-8b-rmu-baseline-target-100-checkpoint-1078|unlearning-cleanslate/fsid-curated-qwen3-8b-target-100:forget_pool|unlearning-cleanslate/verif-qwen3-8b-w100-s10|unlearning-cleanslate/eval-15-qwen3-8b-rmu-baseline-target-100-checkpoint-1078|unlearning-cleanslate/generations-qwen3-8b-pre_val|unlearning-cleanslate/generations-15-qwen3-8b-rmu-baseline-target-100-checkpoint-1078|unlearning-cleanslate/qwen3-8b-rmu-baseline-target-100:checkpoint-1078"
)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( idx < 0 || idx >= ${#ROWS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$idx outside row table size ${#ROWS[@]}" >&2
  exit 2
fi

IFS='|' read -r row forget_pool pre_eval post_eval pre_generations post_generations checkpoint <<<"${ROWS[$idx]}"
output="results/scores/final_final/${row}.json"
log="logs/final_final/scores/${row}.log"

if [[ -f "$output" && "$FORCE" != "1" ]]; then
  echo "[skip] ${row}: ${output} exists (set FORCE=1 to recompute)"
  exit 0
fi

echo "============================================================"
echo "[score] array=${SLURM_ARRAY_JOB_ID:-unset}_${SLURM_ARRAY_TASK_ID:-unset}"
echo "[score] row=${row}"
echo "[score] output=${output}"
echo "[score] log=${log}"
echo "============================================================"

cleanslate score \
  --forget-pool "$forget_pool" \
  --pre-eval "$pre_eval" \
  --post-eval "$post_eval" \
  --pre-generations "$pre_generations" \
  --post-generations "$post_generations" \
  --checkpoint "$checkpoint" \
  -o "$output" 2>&1 | tee "$log"

echo "[ok] ${row}"

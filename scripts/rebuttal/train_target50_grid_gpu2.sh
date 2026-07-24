#!/usr/bin/env bash
#
# Train and push the 18 missing target-50 rebuttal checkpoints:
#
#   (Undial, RMU)
#     x (BM25 pre / bm25_6T, BM25 mid / bm25_10B, IGM mid / igm_10B)
#     x (Llama 3.1 8B, Qwen3 8B, OLMo 3 7B)
#
# The nine SimNPO-gentle cells already exist on Hugging Face and are not run.
# This launcher intentionally performs training only: no memorization eval,
# benchmark generation, judging, or scoring.
#
# Usage:
#   bash scripts/rebuttal/train_target50_grid_gpu2.sh
#
# Successful cells are skipped on a rerun.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=2
HF_ORG="unlearning-cleanslate"
LOG_ROOT="logs/rebuttal"
RESULT_ROOT="results/rebuttal"
TRAIN_ROOT="${RESULT_ROOT}/train"
SENTINEL_ROOT="${RESULT_ROOT}/.sentinels/train"
MASTER_LOG="${LOG_ROOT}/train_target50_grid_gpu2.log"

mkdir -p "$LOG_ROOT" "$TRAIN_ROOT" "$SENTINEL_ROOT"


# Everything printed by the launcher also lands in one durable master log.
exec > >(tee -a "$MASTER_LOG") 2>&1

source ".venv/bin/activate"


export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
# GPU 2 is shared with a ~43GB vLLM server; expandable_segments trims
# allocator fragmentation so training fits beside it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


MODELS=(
  "llama-8b|llama-3_1-8b|fsid-curated-llama-8b"
  "qwen-8b|qwen3-8b|fsid-curated-qwen3-8b"
  "olmo-7b|olmo-3-7b|fsid-curated-olmo-7b"
)

METHODS=(
  "undial|cleanslate/fsid/configs/undial.yaml"
  "rmu|cleanslate/fsid/configs/rmu.yaml"
)

RULES=(
  "bm25_6T|bm25-6t"
  "bm25_10B|bm25-10b"
  "igm_10B|igm-10b"
)

total=$((${#MODELS[@]} * ${#METHODS[@]} * ${#RULES[@]}))
completed=0
skipped=0
ordinal=0

echo
echo "============================================================"
echo "Target-50 rebuttal training"
echo "GPU: physical ${GPU_ID} (the process sees it as CUDA device 0)"
echo "Cells: ${total}, sequential"
echo "Logs: ${LOG_ROOT}"
echo "Results: ${TRAIN_ROOT}"
echo "HF org: ${HF_ORG}"
echo "Started: $(date -Is)"
echo "============================================================"

for model_row in "${MODELS[@]}"; do
  IFS='|' read -r model_alias model_clean curated_name <<< "$model_row"
  curated_repo="${HF_ORG}/${curated_name}"

  for method_row in "${METHODS[@]}"; do
    IFS='|' read -r objective train_config <<< "$method_row"

    for rule_row in "${RULES[@]}"; do
      IFS='|' read -r rule_split rule_tag <<< "$rule_row"
      ordinal=$((ordinal + 1))

      stem="${model_clean}-${objective}-${rule_tag}"
      hub_repo="${HF_ORG}/${stem}"
      output_root="${TRAIN_ROOT}/${stem}"
      job_log="${LOG_ROOT}/${stem}.train.log"
      done_file="${SENTINEL_ROOT}/${stem}.done"

      echo
      echo "------------------------------------------------------------"
      echo "[${ordinal}/${total}] ${stem}"
      echo "model=${model_alias}"
      echo "objective=${objective}"
      echo "curated=${curated_repo}"
      echo "rule=${rule_split}"
      echo "output=${output_root}"
      echo "hub=${hub_repo}"
      echo "------------------------------------------------------------"

      if [[ -f "$done_file" ]]; then
        echo "SKIP: success sentinel exists at ${done_file}"
        skipped=$((skipped + 1))
        continue
      fi

      mkdir -p "$output_root"

      # HF org is over its storage quota, so we TRAIN LOCALLY ONLY (no push)
      # and upload later. --save-every 0 keeps just the final model (~15GB),
      # not the 45GB/step optimizer checkpoints, so 18 cells fit on disk.
      if python -m cleanslate.cli fsid train \
        --config "$train_config" \
        -m "$model_alias" \
        --forget-path "${curated_repo}:forget:${rule_split}" \
        --retain-path "${curated_repo}:retain:${rule_split}" \
        -o "$output_root" \
        --save-every 0 \
        2>&1 | tee -a "$job_log"
      then

        touch "$done_file"

        echo "SUCCESS (local, pending push): ${output_root}  -> hub target ${hub_repo}"
        completed=$((completed + 1))
      else
        status=$?
        echo "FAILED: ${stem} exited with status ${status}" >&2
        echo "See ${job_log}" >&2
        exit "$status"
      fi
    done
  done
done

echo
echo "============================================================"
echo "Training grid complete"
echo "Completed this invocation: ${completed}"
echo "Skipped from prior success: ${skipped}"
echo "Total cells: ${total}"
echo "Finished: $(date -Is)"
echo "Master log: ${MASTER_LOG}"
echo "Results: ${TRAIN_ROOT}"
echo "No eval, validation, judging, or scoring was run."
echo "============================================================"

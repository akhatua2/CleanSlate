#!/usr/bin/env bash

row="$1"
gpu="$2"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."
source .venv/bin/activate

IFS=$'\t' read -r row_id model curation data_source unlearning setting checkpoint hf_datapath hf_mem hf_generation \
  < <(awk -F '\t' -v row="$row" '$1 == row { print; exit }' "$script_dir/target50_rows.tsv")

stem="${checkpoint##*/}"

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_ALLOW_CODE_EVAL=1

mkdir -p logs/rebuttal results/rebuttal/eval results/rebuttal/generation
exec > >(tee -a "logs/rebuttal/${stem}.eval-generate.log") 2>&1

echo "row=$row_id gpu=$gpu checkpoint=$checkpoint"
echo "hf_mem=$hf_mem"
echo "hf_generation=$hf_generation"

cleanslate eval \
  -m "$checkpoint" \
  --config scripts/final_final/slurm/config_slurm_eval.yaml \
  -o "results/rebuttal/eval/$stem" \
  --hf-org anime-sh &&

cleanslate generate \
  -m "$checkpoint" \
  -B full \
  --skip-done \
  -o "results/rebuttal/generation/$stem" \
  --gpu-mem 0.90 \
  --tp-size 1 \
  --dp-size 1 \
  --max-model-len 4096 \
  --push-to-hub \
  --hf-org anime-sh

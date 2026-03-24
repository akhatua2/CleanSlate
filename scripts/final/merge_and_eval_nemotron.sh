#!/usr/bin/env bash
# ============================================================================
# Merge two nemotron-9b checkpoints (gentle + aggressive) by weight averaging
# then run post-eval, validation, and scoring.
#
# Usage: ./scripts/final/merge_and_eval_nemotron.sh <gpu_id>
# ============================================================================
set -euo pipefail

export HF_MODULES_CACHE="/lfs/skampere3/0/animjha/.hf_modules"

ALIAS="nemotron-9b"
EVAL_DATASET="unlearning-cleanslate/eval_cleanslate_dataset_nemotron-nano-9b"

GPU_ID="${1:?Usage: $0 <gpu_id>}"

CKPT_GENTLE="results/final/unlearnt_models/${ALIAS}/simnpo_gentle/simnpo_NVIDIA-Nemotron-Nano-9B-v2_20260310_114618"
CKPT_AGGRESSIVE="results/final/unlearnt_models/${ALIAS}/simnpo/simnpo_NVIDIA-Nemotron-Nano-9B-v2_20260310_142958"
MERGED_DIR="results/final/unlearnt_models/${ALIAS}/simnpo_merged"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}/simnpo_merged"
VAL_DIR="results/final/validation/${ALIAS}"
CURATED_DIR="data/final/curated/${ALIAS}/baseline"
SCORE_DIR="results/final/score/${ALIAS}"

mkdir -p "$MERGED_DIR" "$POST_EVAL_DIR" "$SCORE_DIR"

# ── Merge checkpoints ────────────────────────────────────────────────
if [[ -f "${MERGED_DIR}/config.json" ]]; then
    echo "[Merge] SKIP: merged checkpoint exists"
else
    echo "[Merge] Averaging weights: gentle + aggressive ..."
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -c "
import torch, json, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

ckpt_a = Path('${CKPT_GENTLE}')
ckpt_b = Path('${CKPT_AGGRESSIVE}')
out = Path('${MERGED_DIR}')

# Find all safetensor shards
shards_a = sorted(ckpt_a.glob('model-*.safetensors'))
shards_b = sorted(ckpt_b.glob('model-*.safetensors'))
assert len(shards_a) == len(shards_b), f'Shard count mismatch: {len(shards_a)} vs {len(shards_b)}'

print(f'Merging {len(shards_a)} shards...')
for sa, sb in zip(shards_a, shards_b):
    assert sa.name == sb.name, f'Shard name mismatch: {sa.name} vs {sb.name}'
    print(f'  {sa.name}')
    wa = load_file(str(sa))
    wb = load_file(str(sb))
    merged = {k: (wa[k] + wb[k]) / 2.0 for k in wa}
    save_file(merged, str(out / sa.name))
    del wa, wb, merged

# Copy non-weight files
for f in ckpt_a.iterdir():
    if f.suffix != '.safetensors' and f.name != 'train.log' and f.name != 'train_summary.json' and f.name != 'training_args.bin':
        if not (out / f.name).exists():
            shutil.copy2(f, out / f.name)

print('Done!')
"
fi
echo "[Merge] Done."

# ── Post-eval ────────────────────────────────────────────────────────
if compgen -G "${POST_EVAL_DIR}/*/results.parquet" > /dev/null 2>&1; then
    echo "[Eval] SKIP: post-eval exists"
else
    echo "[Eval] Post-eval merged checkpoint ..."
    CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate eval \
        -d "$EVAL_DATASET" -c default \
        -m "$MERGED_DIR" \
        --num-gpus 1 \
        -o "$POST_EVAL_DIR" --no-push
fi
echo "[Eval] Done."

# ── Validation ───────────────────────────────────────────────────────
if [[ -f "${VAL_DIR}/post_simnpo_merged_gsm8k.json" ]]; then
    echo "[Val] SKIP: post_simnpo_merged_gsm8k.json exists"
else
    echo "[Val] Post-validation gsm8k ..."
    CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate fsid validate \
        -m "$MERGED_DIR" -B gsm8k \
        -o "${VAL_DIR}/post_simnpo_merged_gsm8k.json" \
        --num-gpus 1 --mem-fraction 0.60
fi

if [[ -f "${VAL_DIR}/post_simnpo_merged_qa.json" ]]; then
    echo "[Val] SKIP: post_simnpo_merged_qa.json exists"
else
    echo "[Val] Post-validation QA ..."
    CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate fsid qa eval \
        -m "$MERGED_DIR" \
        -o "${VAL_DIR}/post_simnpo_merged_qa.json"
fi
echo "[Val] Done."

# ── Score ─────────────────────────────────────────────────────────────
echo "[Score] Scoring simnpo_merged ..."
cleanslate fsid score \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-eval "$EVAL_DATASET" \
    --post-eval "$POST_EVAL_DIR" \
    --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
    --post-validation "${VAL_DIR}/post_simnpo_merged_gsm8k.json" \
    -o "${SCORE_DIR}/simnpo_merged_score.json"

cleanslate fsid qa score \
    --forget-path "${CURATED_DIR}/forget_set.parquet" \
    --pre-qa "${VAL_DIR}/pre_qa.json" \
    --post-qa "${VAL_DIR}/post_simnpo_merged_qa.json" \
    -o "${SCORE_DIR}/simnpo_merged_qa_score.json"
echo "[Score] Done."

echo "=== ${ALIAS} simnpo_merged complete  |  $(date) ==="

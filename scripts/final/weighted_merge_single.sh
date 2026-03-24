#!/usr/bin/env bash
# ============================================================================
# Weighted merge + eval for a single model (both blends: 3g1a and 1g3a)
#
# Usage: ./scripts/final/weighted_merge_single.sh <alias> <gpu_id>
# ============================================================================
set -euo pipefail

export HF_MODULES_CACHE="/lfs/skampere3/0/animjha/.hf_modules"

ALIAS="${1:?Usage: $0 <alias> <gpu_id>}"
GPU_ID="${2:?Usage: $0 <alias> <gpu_id>}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

declare -A GENTLE_CKPT=(
    [gemma-12b]="results/final/unlearnt_models/gemma-12b/simnpo_gentle/simnpo_gemma-3-12b-pt_20260310_113542"
    [nemotron-9b]="results/final/unlearnt_models/nemotron-9b/simnpo_gentle/simnpo_NVIDIA-Nemotron-Nano-9B-v2_20260310_114618"
    [olmo-32b]="results/final/unlearnt_models/olmo-32b/simnpo_gentle/simnpo_Olmo-3-1125-32B_20260311_084411"
)
declare -A AGGR_CKPT=(
    [gemma-12b]="results/final/unlearnt_models/gemma-12b/simnpo/simnpo_gemma-3-12b-pt_20260310_180228"
    [nemotron-9b]="results/final/unlearnt_models/nemotron-9b/simnpo/simnpo_NVIDIA-Nemotron-Nano-9B-v2_20260310_142958"
    [olmo-32b]="results/final/unlearnt_models/olmo-32b/simnpo/simnpo_Olmo-3-1125-32B_20260310_164549"
)
declare -A EVAL_DS=(
    [gemma-12b]="unlearning-cleanslate/eval_cleanslate_dataset_gemma-3-12b"
    [nemotron-9b]="unlearning-cleanslate/eval_cleanslate_dataset_nemotron-nano-9b"
    [olmo-32b]="unlearning-cleanslate/eval_cleanslate_dataset_olmo-32b"
)

GENTLE="${GENTLE_CKPT[$ALIAS]}"
AGGR="${AGGR_CKPT[$ALIAS]}"
EVAL="${EVAL_DS[$ALIAS]}"
CURATED="data/final/curated/${ALIAS}/baseline"
VAL_DIR="results/final/validation/${ALIAS}"
SCORE_DIR="results/final/score/${ALIAS}"

for BLEND in "merge_3g1a:0.75:0.25" "merge_1g3a:0.25:0.75"; do
    IFS=':' read -r TAG GW AW <<< "$BLEND"
    MERGED="results/final/unlearnt_models/${ALIAS}/${TAG}"
    POST_EVAL="results/final/post_eval/${ALIAS}/${TAG}"
    mkdir -p "$MERGED" "$POST_EVAL" "$SCORE_DIR"

    echo "=== ${ALIAS} ${TAG} on GPU ${GPU_ID}  |  $(date) ==="

    # Merge
    if [[ -f "${MERGED}/config.json" ]]; then
        echo "[Merge] SKIP"
    else
        echo "[Merge] gentle=${GW}, aggr=${AW} ..."
        python3 -c "
import torch, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

ckpt_a = Path('${GENTLE}')
ckpt_b = Path('${AGGR}')
out = Path('${MERGED}')
w_a, w_b = ${GW}, ${AW}

shards_a = sorted(ckpt_a.glob('model-*.safetensors'))
shards_b = sorted(ckpt_b.glob('model-*.safetensors'))
assert len(shards_a) == len(shards_b)

print(f'Merging {len(shards_a)} shards...')
for sa, sb in zip(shards_a, shards_b):
    assert sa.name == sb.name
    wa = load_file(str(sa))
    wb = load_file(str(sb))
    merged = {k: wa[k] * w_a + wb[k] * w_b for k in wa}
    save_file(merged, str(out / sa.name))
    del wa, wb, merged

for f in ckpt_a.iterdir():
    if f.suffix != '.safetensors' and f.name not in ('train.log', 'train_summary.json', 'training_args.bin'):
        if not (out / f.name).exists():
            shutil.copy2(f, out / f.name)
print('Done!')
"
    fi

    # Post-eval
    if compgen -G "${POST_EVAL}/*/results.parquet" > /dev/null 2>&1; then
        echo "[Eval] SKIP"
    else
        echo "[Eval] ..."
        cleanslate eval -d "$EVAL" -c default -m "$MERGED" \
            --num-gpus 1 -o "$POST_EVAL" --no-push
    fi

    # GSM8K
    GSM_FILE="${VAL_DIR}/post_${TAG}_gsm8k.json"
    if [[ -f "$GSM_FILE" ]]; then
        echo "[GSM8K] SKIP"
    else
        echo "[GSM8K] ..."
        cleanslate fsid validate -m "$MERGED" -B gsm8k \
            -o "$GSM_FILE" --num-gpus 1 --mem-fraction 0.80
    fi

    # QA
    QA_FILE="${VAL_DIR}/post_${TAG}_qa.json"
    if [[ -f "$QA_FILE" ]]; then
        echo "[QA] SKIP"
    else
        echo "[QA] ..."
        cleanslate fsid qa eval -m "$MERGED" -o "$QA_FILE" -g 1
    fi

    # Score
    echo "[Score] ..."
    cleanslate fsid score \
        --forget-path "${CURATED}/forget_set.parquet" \
        --pre-eval "$EVAL" --post-eval "$POST_EVAL" \
        --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
        --post-validation "$GSM_FILE" \
        -o "${SCORE_DIR}/${TAG}_score.json"

    cleanslate fsid qa score \
        --forget-path "${CURATED}/forget_set.parquet" \
        --pre-qa "${VAL_DIR}/pre_qa.json" --post-qa "$QA_FILE" \
        -o "${SCORE_DIR}/${TAG}_qa_score.json"

    echo "=== ${ALIAS} ${TAG} complete  |  $(date) ==="
done

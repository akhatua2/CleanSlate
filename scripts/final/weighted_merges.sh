#!/usr/bin/env bash
# ============================================================================
# Weighted merges for all three models:
#   - (1x gentle + 3x aggressive) / 4  → "merge_1g3a"
#   - (3x gentle + 1x aggressive) / 4  → "merge_3g1a"
#
# Usage: ./scripts/final/weighted_merges.sh <gpu_id>
# ============================================================================
set -euo pipefail

GPU_ID="${1:?Usage: $0 <gpu_id>}"

export HF_MODULES_CACHE="/lfs/skampere3/0/animjha/.hf_modules"

# ── Model definitions ─────────────────────────────────────────────────
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

merge_weights() {
    local gentle_ckpt="$1" aggr_ckpt="$2" out_dir="$3" gentle_w="$4" aggr_w="$5"
    if [[ -f "${out_dir}/config.json" ]]; then
        echo "  SKIP: merged checkpoint exists"
        return
    fi
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -c "
import torch, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

ckpt_a = Path('${gentle_ckpt}')
ckpt_b = Path('${aggr_ckpt}')
out = Path('${out_dir}')
w_a, w_b = ${gentle_w}, ${aggr_w}

shards_a = sorted(ckpt_a.glob('model-*.safetensors'))
shards_b = sorted(ckpt_b.glob('model-*.safetensors'))
assert len(shards_a) == len(shards_b), f'Shard mismatch: {len(shards_a)} vs {len(shards_b)}'

print(f'  Merging {len(shards_a)} shards (gentle={w_a}, aggr={w_b})...')
for sa, sb in zip(shards_a, shards_b):
    assert sa.name == sb.name
    print(f'    {sa.name}')
    wa = load_file(str(sa))
    wb = load_file(str(sb))
    merged = {k: wa[k] * w_a + wb[k] * w_b for k in wa}
    save_file(merged, str(out / sa.name))
    del wa, wb, merged

for f in ckpt_a.iterdir():
    if f.suffix != '.safetensors' and f.name not in ('train.log', 'train_summary.json', 'training_args.bin'):
        if not (out / f.name).exists():
            shutil.copy2(f, out / f.name)
print('  Done!')
"
}

for ALIAS in gemma-12b nemotron-9b olmo-32b; do
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

        echo ""
        echo "=== ${ALIAS} ${TAG}  |  GPU: ${GPU_ID}  |  $(date) ==="

        # Merge
        echo "[Merge] ${TAG} ..."
        merge_weights "$GENTLE" "$AGGR" "$MERGED" "$GW" "$AW"

        # Post-eval
        if compgen -G "${POST_EVAL}/*/results.parquet" > /dev/null 2>&1; then
            echo "[Eval] SKIP"
        else
            echo "[Eval] Running ..."
            CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate eval \
                -d "$EVAL" -c default \
                -m "$MERGED" \
                --num-gpus 1 \
                -o "$POST_EVAL" --no-push
        fi

        # Validation - GSM8K
        GSM_FILE="${VAL_DIR}/post_${TAG}_gsm8k.json"
        if [[ -f "$GSM_FILE" ]]; then
            echo "[Val] SKIP gsm8k"
        else
            echo "[Val] gsm8k ..."
            CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate fsid validate \
                -m "$MERGED" -B gsm8k \
                -o "$GSM_FILE" \
                --num-gpus 1 --mem-fraction 0.80
        fi

        # Validation - QA
        QA_FILE="${VAL_DIR}/post_${TAG}_qa.json"
        if [[ -f "$QA_FILE" ]]; then
            echo "[Val] SKIP qa"
        else
            echo "[Val] QA ..."
            CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate fsid qa eval \
                -m "$MERGED" \
                -o "$QA_FILE" \
                -g 1
        fi

        # Score
        echo "[Score] ..."
        cleanslate fsid score \
            --forget-path "${CURATED}/forget_set.parquet" \
            --pre-eval "$EVAL" \
            --post-eval "$POST_EVAL" \
            --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
            --post-validation "$GSM_FILE" \
            -o "${SCORE_DIR}/${TAG}_score.json"

        cleanslate fsid qa score \
            --forget-path "${CURATED}/forget_set.parquet" \
            --pre-qa "${VAL_DIR}/pre_qa.json" \
            --post-qa "$QA_FILE" \
            -o "${SCORE_DIR}/${TAG}_qa_score.json"

        echo "=== ${ALIAS} ${TAG} complete  |  $(date) ==="
    done
done

echo ""
echo "=== All weighted merges complete  |  $(date) ==="

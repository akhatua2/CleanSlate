#!/usr/bin/env bash
# ============================================================================
# Parallel weighted merges — claims idle GPUs and runs per-model jobs
#
# Each model's two blends (3g1a, 1g3a) run sequentially on one GPU.
# Different models run in parallel on different GPUs as they become available.
#
# A GPU is "free" if it has <=10 MiB for 5 continuous minutes.
#
# Usage: ./scripts/final/weighted_merges_parallel.sh [initial_gpu_id]
# ============================================================================
set -euo pipefail

export HF_MODULES_CACHE="/lfs/skampere3/0/animjha/.hf_modules"

IDLE_THRESHOLD_MIB=10
IDLE_WAIT_SECONDS=300
CHECK_INTERVAL=30

MODELS=("gemma-12b" "nemotron-9b" "olmo-32b")

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

# ── Per-model job: runs both blends sequentially on assigned GPU ──────
run_model_blends() {
    local ALIAS="$1" GPU_ID="$2"
    local GENTLE="${GENTLE_CKPT[$ALIAS]}"
    local AGGR="${AGGR_CKPT[$ALIAS]}"
    local EVAL="${EVAL_DS[$ALIAS]}"
    local CURATED="data/final/curated/${ALIAS}/baseline"
    local VAL_DIR="results/final/validation/${ALIAS}"
    local SCORE_DIR="results/final/score/${ALIAS}"

    for BLEND in "merge_3g1a:0.75:0.25" "merge_1g3a:0.25:0.75"; do
        IFS=':' read -r TAG GW AW <<< "$BLEND"
        local MERGED="results/final/unlearnt_models/${ALIAS}/${TAG}"
        local POST_EVAL="results/final/post_eval/${ALIAS}/${TAG}"
        mkdir -p "$MERGED" "$POST_EVAL" "$SCORE_DIR"

        echo "[${ALIAS}] === ${TAG} on GPU ${GPU_ID}  |  $(date) ==="

        # Merge
        if [[ -f "${MERGED}/config.json" ]]; then
            echo "[${ALIAS}] Merge SKIP"
        else
            echo "[${ALIAS}] Merging (gentle=${GW}, aggr=${AW}) ..."
            CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -c "
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
            echo "[${ALIAS}] Eval SKIP"
        else
            echo "[${ALIAS}] Eval ..."
            CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate eval \
                -d "$EVAL" -c default -m "$MERGED" \
                --num-gpus 1 -o "$POST_EVAL" --no-push
        fi

        # GSM8K
        local GSM_FILE="${VAL_DIR}/post_${TAG}_gsm8k.json"
        if [[ -f "$GSM_FILE" ]]; then
            echo "[${ALIAS}] GSM8K SKIP"
        else
            echo "[${ALIAS}] GSM8K ..."
            CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate fsid validate \
                -m "$MERGED" -B gsm8k -o "$GSM_FILE" \
                --num-gpus 1 --mem-fraction 0.80
        fi

        # QA
        local QA_FILE="${VAL_DIR}/post_${TAG}_qa.json"
        if [[ -f "$QA_FILE" ]]; then
            echo "[${ALIAS}] QA SKIP"
        else
            echo "[${ALIAS}] QA ..."
            CUDA_VISIBLE_DEVICES="$GPU_ID" cleanslate fsid qa eval \
                -m "$MERGED" -o "$QA_FILE" -g 1
        fi

        # Score
        echo "[${ALIAS}] Scoring ..."
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

        echo "[${ALIAS}] === ${TAG} complete  |  $(date) ==="
    done
}

# ── GPU tracker + job dispatcher ──────────────────────────────────────
declare -A idle_since=()
declare -A gpu_claimed=()
PENDING=("${MODELS[@]}")
RUNNING_PIDS=()
RUNNING_MODELS=()

# If initial GPU provided, claim it immediately for first model
if [[ -n "${1:-}" ]]; then
    INIT_GPU="$1"
    FIRST_MODEL="${PENDING[0]}"
    PENDING=("${PENDING[@]:1}")
    echo "Launching ${FIRST_MODEL} on GPU ${INIT_GPU} immediately"
    gpu_claimed[$INIT_GPU]=1
    run_model_blends "$FIRST_MODEL" "$INIT_GPU" &
    RUNNING_PIDS+=($!)
    RUNNING_MODELS+=("$FIRST_MODEL")
fi

while (( ${#PENDING[@]} > 0 || ${#RUNNING_PIDS[@]} > 0 )); do
    # Clean up finished jobs
    NEW_PIDS=()
    NEW_MODELS=()
    for i in "${!RUNNING_PIDS[@]}"; do
        pid="${RUNNING_PIDS[$i]}"
        model="${RUNNING_MODELS[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            NEW_PIDS+=("$pid")
            NEW_MODELS+=("$model")
        else
            wait "$pid" || echo "WARNING: ${model} exited with error"
            echo "${model} finished."
        fi
    done
    RUNNING_PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
    RUNNING_MODELS=("${NEW_MODELS[@]+"${NEW_MODELS[@]}"}")

    # If no pending models, just wait for running ones
    if (( ${#PENDING[@]} == 0 )); then
        if (( ${#RUNNING_PIDS[@]} > 0 )); then
            sleep "$CHECK_INTERVAL"
        fi
        continue
    fi

    # Check for idle GPUs
    now=$(date +%s)
    while IFS=', ' read -r gpu_id mem_used_mib; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used_mib=$(echo "$mem_used_mib" | tr -d ' MiB')

        # Skip already claimed GPUs
        if [[ -n "${gpu_claimed[$gpu_id]:-}" ]]; then
            continue
        fi

        if (( mem_used_mib <= IDLE_THRESHOLD_MIB )); then
            if [[ -z "${idle_since[$gpu_id]:-}" ]]; then
                idle_since[$gpu_id]=$now
                echo "GPU $gpu_id went idle at $(date)"
            elif (( now - ${idle_since[$gpu_id]} >= IDLE_WAIT_SECONDS )); then
                # GPU ready — claim it for next pending model
                NEXT_MODEL="${PENDING[0]}"
                PENDING=("${PENDING[@]:1}")
                gpu_claimed[$gpu_id]=1
                echo "Claiming GPU $gpu_id for ${NEXT_MODEL} (idle ${IDLE_WAIT_SECONDS}s)"
                run_model_blends "$NEXT_MODEL" "$gpu_id" &
                RUNNING_PIDS+=($!)
                RUNNING_MODELS+=("$NEXT_MODEL")

                if (( ${#PENDING[@]} == 0 )); then break; fi
            fi
        else
            if [[ -n "${idle_since[$gpu_id]:-}" ]]; then
                echo "GPU $gpu_id no longer idle (${mem_used_mib} MiB)"
            fi
            idle_since[$gpu_id]=""
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    sleep "$CHECK_INTERVAL"
done

echo ""
echo "=== All weighted merges complete  |  $(date) ==="

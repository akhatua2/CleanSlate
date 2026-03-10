#!/usr/bin/env bash
# ============================================================================
# CleanSlate Final Pipeline — qwen-8b-base (Slurm A100 cluster)
#
# Usage: ./scripts/final/qwen-8b-base.sh          # submits Slurm job chain
#    or: ./scripts/final/qwen-8b-base.sh --local 0,1   # run locally on given GPUs
# ============================================================================
set -euo pipefail

ALIAS="qwen-8b-base"
MODEL="Qwen/Qwen3-8B-Base"
EVAL_DATASET="unlearning-cleanslate/eval_cleanslate_dataset_qwen3-8b-base"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PARTITION="${SLURM_PARTITION:-<PARTITION>}"
TIME_1GPU="12:00:00"
TIME_2GPU="8:00:00"
LOG_DIR="logs/final/${ALIAS}"
mkdir -p "$LOG_DIR"

POOL_DIR="data/final/${ALIAS}/pool"
CURATED_DIR="data/final/curated/${ALIAS}/baseline"
VAL_DIR="results/final/validation/${ALIAS}"
UNLEARN_DIR="results/final/unlearnt_models/${ALIAS}"
POST_EVAL_DIR="results/final/post_eval/${ALIAS}"
SCORE_DIR="results/final/score/${ALIAS}"

# Resolve actual checkpoint path (trainer saves to a timestamped subfolder)
find_checkpoint() {
    local match
    match=$(ls -t "$1"/*/config.json 2>/dev/null | head -1)
    if [[ -n "$match" ]]; then dirname "$match"; fi
}

# ── Local mode ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--local" ]]; then
    GPU_IDS="${2:?Usage: $0 --local <gpu_ids>}"
    IFS=',' read -ra GPUS <<< "$GPU_IDS"
    FIRST_GPU="${GPUS[0]}"
    FIRST_TWO="${GPUS[0]},${GPUS[1]:-${GPUS[0]}}"

    mkdir -p "$POOL_DIR" "$CURATED_DIR" "$VAL_DIR" "$UNLEARN_DIR" \
             "$POST_EVAL_DIR" "$SCORE_DIR"

    echo "=== ${ALIAS} pipeline (local)  |  GPUs: ${GPU_IDS}  |  $(date) ==="

    # Phase 1
    echo "[Phase 1] Pre-unlearn validation ..."
    if [[ -f "${VAL_DIR}/pre_gsm8k.json" ]]; then echo "  SKIP: pre_gsm8k.json exists"
    else CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid validate \
        -m "$MODEL" -B gsm8k -o "${VAL_DIR}/pre_gsm8k.json" --num-gpus 1 --mem-fraction 0.80; fi
    if [[ -f "${VAL_DIR}/pre_qa.json" ]]; then echo "  SKIP: pre_qa.json exists"
    else CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid qa eval \
        -m "$MODEL" -o "${VAL_DIR}/pre_qa.json"; fi

    # Phase 2
    echo "[Phase 2] Build pool + curate ..."
    if [[ -f "${POOL_DIR}/pool.parquet" ]]; then echo "  SKIP: pool exists"
    else cleanslate fsid build-pool --eval-dataset "$EVAL_DATASET" --threshold 0.30 -o "$POOL_DIR"; fi
    if [[ -f "${CURATED_DIR}/forget_set.parquet" ]]; then echo "  SKIP: curated"
    else cleanslate fsid curate --forget-pool-path "${POOL_DIR}/forget.parquet" -o "$CURATED_DIR"; fi

    # Phase 3
    for OBJ in npo simnpo; do
        if [[ -n "$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")" ]]; then echo "[Phase 3] SKIP: ${OBJ} checkpoint exists"
        else
            echo "[Phase 3] Training ${OBJ} ..."
            CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid train \
                --config "cleanslate/fsid/configs/${OBJ}.yaml" -m "$MODEL" \
                --forget-path "${CURATED_DIR}/forget_set.parquet" \
                --retain-path "${CURATED_DIR}/retain_set.parquet" \
                -o "${UNLEARN_DIR}/${OBJ}" --batch-size 16 --grad-accum 2 --epochs 1
        fi
    done

    # Phase 4
    for OBJ in npo simnpo; do
        if compgen -G "${POST_EVAL_DIR}/${OBJ}/*/results.parquet" > /dev/null 2>&1; then echo "[Phase 4] SKIP: ${OBJ} post-eval exists"
        else
            CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")
            echo "[Phase 4] Post-eval ${OBJ} (tp=2) — checkpoint: ${CKPT} ..."
            cleanslate eval -d "$EVAL_DATASET" -c default -m "$CKPT" \
                --tp-size 2 --num-gpus 2 --gpu-ids "$FIRST_TWO" -o "${POST_EVAL_DIR}/${OBJ}" --no-push
        fi
    done

    # Phase 5
    for OBJ in npo simnpo; do
        CKPT=$(find_checkpoint "${UNLEARN_DIR}/${OBJ}")
        if [[ -f "${VAL_DIR}/post_${OBJ}_gsm8k.json" ]]; then echo "[Phase 5] SKIP: post_${OBJ}_gsm8k.json exists"
        else CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid validate \
            -m "$CKPT" -B gsm8k -o "${VAL_DIR}/post_${OBJ}_gsm8k.json" --num-gpus 1 --mem-fraction 0.80; fi
        if [[ -f "${VAL_DIR}/post_${OBJ}_qa.json" ]]; then echo "[Phase 5] SKIP: post_${OBJ}_qa.json exists"
        else CUDA_VISIBLE_DEVICES="$FIRST_GPU" cleanslate fsid qa eval \
            -m "$CKPT" -o "${VAL_DIR}/post_${OBJ}_qa.json"; fi
    done

    # Phase 6 (always rerun)
    for OBJ in npo simnpo; do
        echo "[Phase 6] Scoring ${OBJ} ..."
        cleanslate fsid score --forget-path "${CURATED_DIR}/forget_set.parquet" \
            --pre-eval "$EVAL_DATASET" --post-eval "${POST_EVAL_DIR}/${OBJ}" \
            --pre-validation "${VAL_DIR}/pre_gsm8k.json" \
            --post-validation "${VAL_DIR}/post_${OBJ}_gsm8k.json" \
            -o "${SCORE_DIR}/${OBJ}_score.json"
        cleanslate fsid qa score --forget-path "${CURATED_DIR}/forget_set.parquet" \
            --pre-qa "${VAL_DIR}/pre_qa.json" --post-qa "${VAL_DIR}/post_${OBJ}_qa.json" \
            -o "${SCORE_DIR}/${OBJ}_qa_score.json"
    done

    echo "=== ${ALIAS} pipeline complete (local)  |  $(date) ==="
    exit 0
fi

# ── Slurm mode: submit job chain ────────────────────────────────────────────
echo "=== Submitting ${ALIAS} Slurm job chain ==="

JOB_A=$(sbatch --parsable \
    --partition="$PARTITION" --gres=gpu:1 --time="$TIME_1GPU" \
    --job-name="${ALIAS}-pre" --output="${LOG_DIR}/pre_%j.log" \
    --wrap="
set -euo pipefail
cd $PROJECT_DIR
mkdir -p $POOL_DIR $CURATED_DIR $VAL_DIR $UNLEARN_DIR $POST_EVAL_DIR $SCORE_DIR
[[ -f ${VAL_DIR}/pre_gsm8k.json ]] || cleanslate fsid validate -m $MODEL -B gsm8k -o ${VAL_DIR}/pre_gsm8k.json --num-gpus 1 --mem-fraction 0.80
[[ -f ${VAL_DIR}/pre_qa.json ]]    || cleanslate fsid qa eval -m $MODEL -o ${VAL_DIR}/pre_qa.json
[[ -f ${POOL_DIR}/pool.parquet ]]  || cleanslate fsid build-pool --eval-dataset $EVAL_DATASET --threshold 0.30 -o $POOL_DIR
[[ -f ${CURATED_DIR}/forget_set.parquet ]] || cleanslate fsid curate --forget-pool-path ${POOL_DIR}/forget.parquet -o $CURATED_DIR
")
echo "  Job A (pre+pool+curate): ${JOB_A}"

JOB_B=$(sbatch --parsable \
    --partition="$PARTITION" --gres=gpu:1 --time="$TIME_1GPU" \
    --job-name="${ALIAS}-npo" --output="${LOG_DIR}/npo_%j.log" \
    --dependency="afterok:${JOB_A}" \
    --wrap="
set -euo pipefail
cd $PROJECT_DIR
ls ${UNLEARN_DIR}/npo/*/config.json &>/dev/null || cleanslate fsid train --config cleanslate/fsid/configs/npo.yaml \
    -m $MODEL --forget-path ${CURATED_DIR}/forget_set.parquet \
    --retain-path ${CURATED_DIR}/retain_set.parquet \
    -o ${UNLEARN_DIR}/npo --batch-size 16 --grad-accum 2 --epochs 1
")
echo "  Job B (train NPO):       ${JOB_B}  (after ${JOB_A})"

JOB_C=$(sbatch --parsable \
    --partition="$PARTITION" --gres=gpu:1 --time="$TIME_1GPU" \
    --job-name="${ALIAS}-simnpo" --output="${LOG_DIR}/simnpo_%j.log" \
    --dependency="afterok:${JOB_A}" \
    --wrap="
set -euo pipefail
cd $PROJECT_DIR
ls ${UNLEARN_DIR}/simnpo/*/config.json &>/dev/null || cleanslate fsid train --config cleanslate/fsid/configs/simnpo.yaml \
    -m $MODEL --forget-path ${CURATED_DIR}/forget_set.parquet \
    --retain-path ${CURATED_DIR}/retain_set.parquet \
    -o ${UNLEARN_DIR}/simnpo --batch-size 16 --grad-accum 2 --epochs 1
")
echo "  Job C (train SimNPO):    ${JOB_C}  (after ${JOB_A})"

JOB_D=$(sbatch --parsable \
    --partition="$PARTITION" --gres=gpu:2 --time="$TIME_2GPU" \
    --job-name="${ALIAS}-post-npo" --output="${LOG_DIR}/post_npo_%j.log" \
    --dependency="afterok:${JOB_B}" \
    --wrap="
set -euo pipefail
cd $PROJECT_DIR
CKPT=\$(dirname \"\$(ls -t ${UNLEARN_DIR}/npo/*/config.json 2>/dev/null | head -1)\")
compgen -G '${POST_EVAL_DIR}/npo/*/results.parquet' > /dev/null 2>&1 || cleanslate eval -d $EVAL_DATASET -c default -m \$CKPT \
    --tp-size 2 --num-gpus 2 -o ${POST_EVAL_DIR}/npo --no-push
[[ -f ${VAL_DIR}/post_npo_gsm8k.json ]] || cleanslate fsid validate -m \$CKPT -B gsm8k \
    -o ${VAL_DIR}/post_npo_gsm8k.json --num-gpus 1 --mem-fraction 0.80
[[ -f ${VAL_DIR}/post_npo_qa.json ]] || cleanslate fsid qa eval -m \$CKPT -o ${VAL_DIR}/post_npo_qa.json
")
echo "  Job D (post-eval NPO):   ${JOB_D}  (after ${JOB_B})"

JOB_E=$(sbatch --parsable \
    --partition="$PARTITION" --gres=gpu:2 --time="$TIME_2GPU" \
    --job-name="${ALIAS}-post-simnpo" --output="${LOG_DIR}/post_simnpo_%j.log" \
    --dependency="afterok:${JOB_C}" \
    --wrap="
set -euo pipefail
cd $PROJECT_DIR
CKPT=\$(dirname \"\$(ls -t ${UNLEARN_DIR}/simnpo/*/config.json 2>/dev/null | head -1)\")
compgen -G '${POST_EVAL_DIR}/simnpo/*/results.parquet' > /dev/null 2>&1 || cleanslate eval -d $EVAL_DATASET -c default -m \$CKPT \
    --tp-size 2 --num-gpus 2 -o ${POST_EVAL_DIR}/simnpo --no-push
[[ -f ${VAL_DIR}/post_simnpo_gsm8k.json ]] || cleanslate fsid validate -m \$CKPT -B gsm8k \
    -o ${VAL_DIR}/post_simnpo_gsm8k.json --num-gpus 1 --mem-fraction 0.80
[[ -f ${VAL_DIR}/post_simnpo_qa.json ]] || cleanslate fsid qa eval -m \$CKPT -o ${VAL_DIR}/post_simnpo_qa.json
")
echo "  Job E (post-eval SimNPO): ${JOB_E}  (after ${JOB_C})"

JOB_F=$(sbatch --parsable \
    --partition="$PARTITION" --gres=gpu:1 --time="01:00:00" \
    --job-name="${ALIAS}-score" --output="${LOG_DIR}/score_%j.log" \
    --dependency="afterok:${JOB_D}:${JOB_E}" \
    --wrap="
set -euo pipefail
cd $PROJECT_DIR
for OBJ in npo simnpo; do
    cleanslate fsid score --forget-path ${CURATED_DIR}/forget_set.parquet \
        --pre-eval $EVAL_DATASET --post-eval ${POST_EVAL_DIR}/\${OBJ} \
        --pre-validation ${VAL_DIR}/pre_gsm8k.json \
        --post-validation ${VAL_DIR}/post_\${OBJ}_gsm8k.json \
        -o ${SCORE_DIR}/\${OBJ}_score.json
    cleanslate fsid qa score --forget-path ${CURATED_DIR}/forget_set.parquet \
        --pre-qa ${VAL_DIR}/pre_qa.json --post-qa ${VAL_DIR}/post_\${OBJ}_qa.json \
        -o ${SCORE_DIR}/\${OBJ}_qa_score.json
done
")
echo "  Job F (score):           ${JOB_F}  (after ${JOB_D},${JOB_E})"

echo "=== ${ALIAS} job chain submitted ==="

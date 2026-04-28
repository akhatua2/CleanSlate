#!/usr/bin/env bash
# NeurIPS unlearning sweep orchestrator.
#
# Runs, for a single base model:
#   1. pre-unlearning memorization eval + retention validation (pushed as -pre artifacts)
#   2. build the forget-request pool (from the pushed pre-eval dataset)
#   3. for each (rule x objective) in the cartesian grid:
#        a. curate forget/retain
#        b. train + push unlearned checkpoint to HF
#        c. post-unlearning memorization eval on the unlearned model (HF repo)
#        d. post-unlearning retention validation (HF repo)
#        e. score pre vs post (both memorization and retention in one score.json)
#
# Idempotent: each step drops a .done-<tag> sentinel under $RESULTS_ROOT; re-runs skip
# completed steps. Delete a sentinel to re-run just that step.
#
# Usage:
#   MODEL=qwen-8b HF_ORG=unlearning-cleanslate BENCHMARKS=full bash scripts/run_sweep.sh
#
# Env vars (all optional except MODEL):
#   MODEL         base model alias (see cleanslate/cli/aliases.py). Required.
#   HF_ORG        HF org for all pushes. Default: unlearning-cleanslate.
#   BENCHMARKS    comma-separated lm-eval tasks or preset. Default: full.
#   RULES         space-separated rule names. Default: "baseline corpus_sweep bm25 igm".
#   OBJECTIVES    space-separated objectives. Default: "simnpo rmu undial".
#   EVAL_LIMIT    int, for smoke-testing (passed as -n to `cleanslate eval`).
#   DRY_RUN       if set, echoes commands instead of running.
#
#   Pre-run reuse (skip step 1 and/or step 2 if artifacts already exist):
#   PRE_EVAL_DATASET     HF dataset ID (e.g. unlearning-cleanslate/eval-qwen3-8b-pre).
#                        Replaces step 1; the pool is built from this dataset.
#   PRE_EVAL_PARQUET     Local parquet mirror used by `cleanslate score --pre-eval`.
#                        If PRE_EVAL_DATASET is set but this is not, the script
#                        materializes a parquet from the HF dataset at
#                        $RESULTS_ROOT/eval/pre/external.parquet.
#   PRE_EVAL_HF_CONFIG   HF dataset config name to load from PRE_EVAL_DATASET
#                        (default: "default").
#   PRE_VALIDATION_JSON  Existing pre-unlearning validation JSON path. Replaces step 2.

set -euo pipefail

# humaneval_plus / humaneval tasks import `evaluate.load("code_eval")` at import time,
# which raises unless this opt-in env var is set. Needed for the `full` retention suite.
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

# Activate project venv so `cleanslate` / `python` resolve without `uv run`.
if [[ -f "$(dirname "$0")/../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../.venv/bin/activate"
fi

: "${MODEL:?MODEL env var is required (e.g. qwen-8b, llama-8b, olmo-7b)}"
HF_ORG="${HF_ORG:-unlearning-cleanslate}"
BENCHMARKS="${BENCHMARKS:-full}"
RULES="${RULES:-baseline corpus_sweep bm25 igm}"
OBJECTIVES="${OBJECTIVES:-simnpo rmu undial}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
DRY_RUN="${DRY_RUN:-}"
PRE_EVAL_DATASET="${PRE_EVAL_DATASET:-}"
PRE_EVAL_PARQUET="${PRE_EVAL_PARQUET:-}"
PRE_EVAL_HF_CONFIG="${PRE_EVAL_HF_CONFIG:-default}"
PRE_VALIDATION_JSON="${PRE_VALIDATION_JSON:-}"
POOL_THRESHOLD="${POOL_THRESHOLD:-0.05}"

MODEL_CLEAN=$(python -c "from cleanslate.cli.aliases import resolve_model; print(resolve_model('${MODEL}')[1])")
RESULTS_ROOT="results/sweep/${MODEL_CLEAN}"
mkdir -p "${RESULTS_ROOT}"/{eval,validation,fsid,curated,pool}

run() {
  if [[ -n "$DRY_RUN" ]]; then
    echo "+ $*"
  else
    echo "+ $*"
    "$@"
  fi
}

sentinel() { echo "${RESULTS_ROOT}/.done-$1"; }
is_done() { [[ -f "$(sentinel "$1")" ]]; }
mark_done() { [[ -n "$DRY_RUN" ]] || touch "$(sentinel "$1")"; }

eval_limit_args=()
if [[ -n "$EVAL_LIMIT" ]]; then
  eval_limit_args=(-n "$EVAL_LIMIT")
fi

# ------------------------------------------------------------------
# 1. Pre-unlearning memorization eval (pushes to eval-<clean>-pre),
#    OR reuse an existing HF dataset via $PRE_EVAL_DATASET.
# ------------------------------------------------------------------
if [[ -n "$PRE_EVAL_DATASET" ]]; then
  echo "[pre-eval] reusing HF dataset: $PRE_EVAL_DATASET"
  if [[ -z "$PRE_EVAL_PARQUET" ]]; then
    PRE_EVAL_PARQUET="${RESULTS_ROOT}/eval/pre/external.parquet"
    if [[ ! -f "$PRE_EVAL_PARQUET" ]]; then
      mkdir -p "$(dirname "$PRE_EVAL_PARQUET")"
      run python -c "
from datasets import load_dataset
ds = load_dataset('${PRE_EVAL_DATASET}', '${PRE_EVAL_HF_CONFIG}', split='train')
ds.to_parquet('${PRE_EVAL_PARQUET}')
print('wrote', '${PRE_EVAL_PARQUET}', len(ds), 'rows')
"
    fi
  fi
elif ! is_done pre-eval; then
  run cleanslate eval \
    -m "$MODEL" \
    -o "${RESULTS_ROOT}/eval/pre" \
    --hf-org "$HF_ORG" \
    --push-suffix pre \
    ${EVAL_CONFIG:+--config "$EVAL_CONFIG"} \
    "${eval_limit_args[@]}"
  mark_done pre-eval
  PRE_EVAL_PARQUET="${RESULTS_ROOT}/eval/pre/default/${MODEL_CLEAN}/results.parquet"
  PRE_EVAL_DATASET="${HF_ORG}/eval-${MODEL_CLEAN}-pre"
else
  PRE_EVAL_PARQUET="${RESULTS_ROOT}/eval/pre/default/${MODEL_CLEAN}/results.parquet"
  PRE_EVAL_DATASET="${HF_ORG}/eval-${MODEL_CLEAN}-pre"
fi

# ------------------------------------------------------------------
# 2. Pre-unlearning retention validation (pushes to generations-<clean>-pre),
#    OR reuse an existing local JSON via $PRE_VALIDATION_JSON.
# ------------------------------------------------------------------
if [[ -n "$PRE_VALIDATION_JSON" ]]; then
  echo "[pre-validate] reusing: $PRE_VALIDATION_JSON"
elif ! is_done pre-validate; then
  run cleanslate validate \
    -m "$MODEL" \
    -B "$BENCHMARKS" \
    -o "${RESULTS_ROOT}/validation/pre" \
    --push-to-hub \
    --hf-org "$HF_ORG" \
    --push-suffix pre
  mark_done pre-validate
  PRE_VALIDATION_JSON="${RESULTS_ROOT}/validation/pre/${MODEL_CLEAN}.json"
else
  PRE_VALIDATION_JSON="${RESULTS_ROOT}/validation/pre/${MODEL_CLEAN}.json"
fi

# ------------------------------------------------------------------
# 3. Build forget-request pool from pushed pre-eval.
# ------------------------------------------------------------------
if ! is_done build-pool; then
  run cleanslate fsid build-pool \
    --eval-dataset "$PRE_EVAL_DATASET" \
    --threshold "$POOL_THRESHOLD" \
    -o "${RESULTS_ROOT}/pool"
  mark_done build-pool
fi
FORGET_POOL="${RESULTS_ROOT}/pool/forget.parquet"

# ------------------------------------------------------------------
# 4. For each (rule, objective): curate -> train -> eval -> validate -> score.
# ------------------------------------------------------------------
for RULE in $RULES; do
  CURATED_DIR="${RESULTS_ROOT}/curated/${RULE}"
  if ! is_done "curate-${RULE}"; then
    run cleanslate fsid curate \
      --rule "$RULE" \
      --forget-pool-path "$FORGET_POOL" \
      -o "$CURATED_DIR"
    mark_done "curate-${RULE}"
  fi
  FORGET_SET="${CURATED_DIR}/forget_set.parquet"
  RETAIN_SET="${CURATED_DIR}/retain_set.parquet"

  for OBJ in $OBJECTIVES; do
    TAG="${RULE}-${OBJ}"
    CKPT_REPO="${HF_ORG}/${MODEL_CLEAN}-unlearned-${TAG}"
    RUN_ROOT="${RESULTS_ROOT}/fsid/${TAG}"
    mkdir -p "$RUN_ROOT"

    # 4a. Train + push.
    if ! is_done "train-${TAG}"; then
      # On multi-GPU: torchrun so HF Trainer's FSDP shards weights+grads+m+v across ranks.
      # On single-GPU: plain `python -m ...` — FSDP on 1 rank is pure overhead and
      # to_training_arguments() skips fsdp=/fsdp_config= when WORLD_SIZE is unset.
      IFS=',' read -ra _VIS <<< "${CUDA_VISIBLE_DEVICES:-0}"
      _nproc="${NPROC_PER_NODE:-${#_VIS[@]}}"
      if [[ "$_nproc" -le 1 ]]; then
        run python -m cleanslate.cli fsid train \
          --config "cleanslate/fsid/configs/${OBJ}.yaml" \
          -m "$MODEL" \
          --forget-path "$FORGET_SET" \
          --retain-path "$RETAIN_SET" \
          -o "${RUN_ROOT}/runs" \
          ${TRAIN_EXTRA_ARGS:-} \
          --push-to-hub \
          --hub-repo-id "$CKPT_REPO"
      else
        _port="${TORCHRUN_PORT:-$((29500 + RANDOM % 1000))}"
        run torchrun \
          --nproc-per-node="$_nproc" \
          --master-port="$_port" \
          -m cleanslate.cli fsid train \
          --config "cleanslate/fsid/configs/${OBJ}.yaml" \
          -m "$MODEL" \
          --forget-path "$FORGET_SET" \
          --retain-path "$RETAIN_SET" \
          -o "${RUN_ROOT}/runs" \
          ${TRAIN_EXTRA_ARGS:-} \
          --push-to-hub \
          --hub-repo-id "$CKPT_REPO"
      fi
      mark_done "train-${TAG}"
    fi

    # 4b. Post-unlearning memorization eval (push lands under eval-<clean>-unlearned-<tag>).
    POST_EVAL_ROOT="${RESULTS_ROOT}/eval/${TAG}"
    POST_EVAL_CLEAN="${MODEL_CLEAN}-unlearned-${TAG}"
    if ! is_done "eval-${TAG}"; then
      run cleanslate eval \
        -m "$CKPT_REPO" \
        -o "$POST_EVAL_ROOT" \
        --hf-org "$HF_ORG" \
        ${EVAL_CONFIG:+--config "$EVAL_CONFIG"} \
        "${eval_limit_args[@]}"
      mark_done "eval-${TAG}"
    fi
    POST_EVAL_PARQUET="${POST_EVAL_ROOT}/default/${POST_EVAL_CLEAN}/results.parquet"

    # 4c. Post-unlearning retention validation.
    POST_VALIDATION_ROOT="${RESULTS_ROOT}/validation/${TAG}"
    if ! is_done "validate-${TAG}"; then
      run cleanslate validate \
        -m "$CKPT_REPO" \
        -B "$BENCHMARKS" \
        -o "$POST_VALIDATION_ROOT" \
        --push-to-hub \
        --hf-org "$HF_ORG"
      mark_done "validate-${TAG}"
    fi
    POST_VALIDATION_JSON="${POST_VALIDATION_ROOT}/${POST_EVAL_CLEAN}.json"

    # 4d. Score: nested JSON with forget/retain memorization deltas + per-task
    # retention. Generations come from the pushed pre/post HF datasets (the
    # `validate --push-to-hub` calls above). Validation JSON deltas are no
    # longer consumed — their info is already in the per-task retention block.
    if ! is_done "score-${TAG}"; then
      POST_GEN_DS="${HF_ORG}/generations-${POST_EVAL_CLEAN}"
      run cleanslate score \
        --forget-pool "$FORGET_SET" \
        --pre-eval "$PRE_EVAL_PARQUET" \
        --post-eval "$POST_EVAL_PARQUET" \
        --pre-generations "${HF_ORG}/generations-${MODEL_CLEAN}-pre" \
        --post-generations "$POST_GEN_DS" \
        --checkpoint "$CKPT_REPO" \
        -o "${RUN_ROOT}/score.json"
      mark_done "score-${TAG}"
    fi

    echo "[DONE] $TAG  ckpt=$CKPT_REPO  score=${RUN_ROOT}/score.json"
  done
done

echo
echo "Sweep complete for ${MODEL} (clean=${MODEL_CLEAN})."
echo "Per-experiment score JSONs: ${RESULTS_ROOT}/fsid/<rule>-<obj>/score.json"

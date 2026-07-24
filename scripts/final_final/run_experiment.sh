#!/usr/bin/env bash
# One concrete experiment, end-to-end:
#   model    = llama-8b
#   curation = oracle (forget split = baseline)
#   target   = 50  (fsid-curated-llama-8b)
#   obj      = simnpo, gentle
#
#   train -> post-eval -> post-val -> score
#
# Pre-eval, pre-val, and the curated forget/retain pool are already on Hub.
# This script does only the per-experiment work and writes
# results/sweep/llama-3_1-8b/score/<EXP_TAG>.json at the end.
#
# Optional env:
#   JUDGE_BASE_URL   if set, points cleanslate validate at a remote vLLM judge
#                    server (e.g. JUDGE_BASE_URL=http://127.0.0.1:42800).
#                    The companion JUDGE_MODEL var is read by the judge.
#   JUDGE_MODEL      remote judge model id (e.g. Qwen/Qwen3.5-9B). Required
#                    when JUDGE_BASE_URL is set.
#
# Idempotent: each step touches a sentinel under results/sweep/llama-3_1-8b/.
# Delete a sentinel to re-run that step.

set -euo pipefail

if [[ -f "$(dirname "$0")/../../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../../.venv/bin/activate"
fi

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

MODEL="${MODEL:-llama-8b}"
MODEL_CLEAN="${MODEL_CLEAN:-llama-3_1-8b}"
HF_ORG="${HF_ORG:-unlearning-cleanslate}"

CURATED="${CURATED:-${HF_ORG}/fsid-curated-llama-8b}"   # target=50
RULE="${RULE:-baseline}"                                 # forget/retain split name in $CURATED
PRE_EVAL="${PRE_EVAL:-${HF_ORG}/verif-llama-3.1-8b-w100-s10}"
PRE_GEN="${PRE_GEN:-${HF_ORG}/generations-llama-3_1-8b-pre_val}"

TRAIN_CFG="${TRAIN_CFG:-cleanslate/fsid/configs/simnpo_gentle.yaml}"
EXP_TAG="${EXP_TAG:-simnpo-gentle-baseline}"

CKPT_REPO="${HF_ORG}/${MODEL_CLEAN}-${EXP_TAG}"
POST_EVAL_DS="${HF_ORG}/eval-${MODEL_CLEAN}-${EXP_TAG}"
POST_GEN_DS="${HF_ORG}/generations-${MODEL_CLEAN}-${EXP_TAG}"

RUN_ROOT="results/sweep/${MODEL_CLEAN}"
mkdir -p "${RUN_ROOT}"/{train,eval,validation,score}

# Optional remote judge for `cleanslate validate`. Pulled from env so the same
# script works whether you're using the Gemini judge (no env needed) or a
# self-hosted vLLM endpoint (set both JUDGE_BASE_URL and JUDGE_MODEL).
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
if [[ -n "$JUDGE_BASE_URL" && -z "$JUDGE_MODEL" ]]; then
  echo "JUDGE_BASE_URL is set but JUDGE_MODEL is not. Aborting." >&2
  exit 2
fi
[[ -n "$JUDGE_BASE_URL" ]] && export JUDGE_BASE_URL
[[ -n "$JUDGE_MODEL" ]]    && export JUDGE_MODEL

sentinel() { echo "${RUN_ROOT}/.done-$1-${EXP_TAG}"; }
is_done()  { [[ -f "$(sentinel "$1")" ]]; }
mark_done() { touch "$(sentinel "$1")"; }

echo "=========================================================="
echo "  ${MODEL}  ${EXP_TAG}"
echo "  CURATED=${CURATED}  RULE=${RULE}"
echo "  CKPT=${CKPT_REPO}"
[[ -n "$JUDGE_BASE_URL" ]] && echo "  JUDGE=${JUDGE_BASE_URL}  model=${JUDGE_MODEL}"
echo "=========================================================="

# 1. Train + push.
if ! is_done train; then
  IFS=',' read -ra _VIS <<< "${CUDA_VISIBLE_DEVICES:-0}"
  _nproc="${NPROC_PER_NODE:-${#_VIS[@]}}"
  if [[ "$_nproc" -le 1 ]]; then
    python -m cleanslate.cli fsid train \
      --config "$TRAIN_CFG" \
      -m "$MODEL" \
      --forget-path "${CURATED}:forget:${RULE}" \
      --retain-path "${CURATED}:retain:${RULE}" \
      -o "${RUN_ROOT}/train/${EXP_TAG}" \
      --push-to-hub \
      --hub-repo-id "$CKPT_REPO"
  else
    _port="${TORCHRUN_PORT:-$((29500 + RANDOM % 1000))}"
    torchrun \
      --nproc-per-node="$_nproc" \
      --master-port="$_port" \
      -m cleanslate.cli fsid train \
      --config "$TRAIN_CFG" \
      -m "$MODEL" \
      --forget-path "${CURATED}:forget:${RULE}" \
      --retain-path "${CURATED}:retain:${RULE}" \
      -o "${RUN_ROOT}/train/${EXP_TAG}" \
      --push-to-hub \
      --hub-repo-id "$CKPT_REPO"
  fi
  mark_done train
fi

# 2. Post-eval (memorization).
if ! is_done post-eval; then
  # No --push-suffix: the repo name derives from the ckpt basename, which
  # already carries EXP_TAG (a suffix here would double it and break step 4).
  cleanslate eval \
    -m "$CKPT_REPO" \
    -o "${RUN_ROOT}/eval/${EXP_TAG}" \
    --hf-org "$HF_ORG"
  mark_done post-eval
fi

# 3. Post-val (generations + judge).
if ! is_done post-val; then
  cleanslate validate \
    -m "$CKPT_REPO" \
    -B full \
    --skip-done \
    -o "${RUN_ROOT}/validation/${EXP_TAG}" \
    --push-to-hub \
    --hf-org "$HF_ORG"
  mark_done post-val
fi

# 4. Score.
if ! is_done score; then
  cleanslate score \
    --forget-pool      "${CURATED}:forget_pool" \
    --pre-eval         "$PRE_EVAL" \
    --post-eval        "$POST_EVAL_DS" \
    --pre-generations  "$PRE_GEN" \
    --post-generations "$POST_GEN_DS" \
    --checkpoint      "$CKPT_REPO" \
    -o "${RUN_ROOT}/score/${EXP_TAG}.json"
  mark_done score
fi

echo "[DONE] ${EXP_TAG}  score=${RUN_ROOT}/score/${EXP_TAG}.json"

#!/usr/bin/env bash
# End-to-end debug run for olmo-7b: simnpo_gentle on baseline curation.
#   1) FSDP train on GPUs 6,7 (push checkpoint to hub)
#   2) Memorization eval on GPU 6 (push results to hub)
#   3) Benchmark validate -B full on GPU 6 with judge server on GPU 7 (push to hub)
#
# All hub artifacts get the suffix `debug-olmo-7b` so this run is clearly
# distinguishable from production runs.
#
# Override CURATED_REPO if the canonical fsid-curated-olmo-7b dataset is not yet
# pushed (e.g., point at a temporarily-borrowed curated set):
#   CURATED_REPO=unlearning-cleanslate/fsid-curated-llama-8b bash scripts/e2e_olmo7b_simnpo_baseline.sh
set -uo pipefail
source .venv/bin/activate
ulimit -n 1048576
mkdir -p logs runs

MODEL_ALIAS="olmo-7b"
MODEL_CLEAN="olmo-3-7b"
RULE="baseline"
OBJECTIVE="simnpo_gentle"
DEBUG_TAG="debug-olmo-7b"
HF_ORG="unlearning-cleanslate"

CURATED_REPO="${CURATED_REPO:-${HF_ORG}/fsid-curated-${MODEL_ALIAS}}"
CKPT_REPO_ID="${HF_ORG}/${MODEL_CLEAN}-${OBJECTIVE}-${RULE}-${DEBUG_TAG}"
TRAIN_OUT="runs/${MODEL_CLEAN}/${RULE}-${OBJECTIVE}-${DEBUG_TAG}"
EVAL_OUT="results/post_eval/${MODEL_CLEAN}-${RULE}-${OBJECTIVE}-${DEBUG_TAG}"
VAL_OUT="results/post_val/${MODEL_CLEAN}-${RULE}-${OBJECTIVE}-${DEBUG_TAG}/validate"
LOG_PFX="logs/${MODEL_CLEAN}_${RULE}_${OBJECTIVE}_${DEBUG_TAG}"

echo "[e2e] model=$MODEL_ALIAS  rule=$RULE  obj=$OBJECTIVE  tag=$DEBUG_TAG"
echo "[e2e] curated=$CURATED_REPO"
echo "[e2e] ckpt repo=$CKPT_REPO_ID"

# --- 0. Precheck: curated dataset + configs ---------------------------
python - <<EOF || { echo "[e2e] PRECHECK FAILED — fix the curated dataset path before re-running."; exit 1; }
import sys
from huggingface_hub import HfApi
from datasets import get_dataset_config_names, get_dataset_split_names
api = HfApi()
repo = "${CURATED_REPO}"
try:
    api.dataset_info(repo)
except Exception as e:
    sys.exit(f"curated repo missing: {repo} ({type(e).__name__})")
cfgs = get_dataset_config_names(repo)
for need_cfg in ("forget", "retain"):
    if need_cfg not in cfgs:
        sys.exit(f"{repo} has no '{need_cfg}' config (configs: {cfgs})")
    splits = get_dataset_split_names(repo, need_cfg)
    if "${RULE}" not in splits:
        sys.exit(f"{repo}/{need_cfg} has no split '${RULE}' (splits: {splits})")
print(f"precheck ok: {repo} forget/${RULE} & retain/${RULE} present")
EOF

# --- 1. Train on GPUs 6,7 with FSDP -----------------------------------
CUDA_VISIBLE_DEVICES=6,7 \
torchrun \
  --nproc-per-node=2 \
  --master-port=$((29500 + RANDOM % 1000)) \
  -m cleanslate.cli fsid train \
  --config "cleanslate/fsid/configs/${OBJECTIVE}.yaml" \
  -m "$MODEL_ALIAS" \
  --forget-path "${CURATED_REPO}:forget:${RULE}" \
  --retain-path "${CURATED_REPO}:retain:${RULE}" \
  -o "$TRAIN_OUT" \
  --push-to-hub \
  --hub-repo-id "$CKPT_REPO_ID" \
  2>&1 | tee "${LOG_PFX}_train.log"

CKPT_DIR="$(ls -td ${TRAIN_OUT}/*/checkpoint-* 2>/dev/null | head -1)"
[[ -z "$CKPT_DIR" ]] && { echo "[e2e] no checkpoint produced under $TRAIN_OUT"; exit 1; }
echo "[e2e] checkpoint dir: $CKPT_DIR"
echo "[e2e] checkpoint hub: $CKPT_REPO_ID"

# --- 2. Post-train memorization eval on GPU 6 -------------------------
CUDA_VISIBLE_DEVICES=6 cleanslate eval \
  -m "$CKPT_DIR" \
  --config config.yaml \
  -o "$EVAL_OUT" \
  --push-to-hub \
  --hf-org "$HF_ORG" \
  --push-suffix "$DEBUG_TAG" \
  2>&1 | tee "${LOG_PFX}_eval.log"

# --- 3. Benchmark validate (judge on GPU 7, validate on GPU 6) --------
CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 42800 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    > "${LOG_PFX}_judge_server.log" 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null || true' EXIT

echo "[e2e] waiting for judge server (pid=$JUDGE_PID)..."
until curl -sf http://127.0.0.1:42800/v1/models >/dev/null; do
  sleep 5
  kill -0 $JUDGE_PID 2>/dev/null || { echo "[e2e] judge server died"; tail -50 "${LOG_PFX}_judge_server.log"; exit 1; }
done
echo "[e2e] judge ready."

export JUDGE_BASE_URL=http://127.0.0.1:42800
export JUDGE_MODEL=Qwen/Qwen3.5-9B
export HF_ALLOW_CODE_EVAL=1
export HF_HOME="$LOCAL_MACHINE_CACHE"
export HF_HUB_CACHE="$LOCAL_MACHINE_CACHE/hub"
export HF_DATASETS_CACHE="$LOCAL_MACHINE_CACHE/datasets"
export TRANSFORMERS_CACHE="$LOCAL_MACHINE_CACHE/transformers"
export HF_TOKENIZERS_CACHE="$LOCAL_MACHINE_CACHE/tokenizers"
export VLLM_CACHE_DIR="$LOCAL_MACHINE_CACHE/vllm"
unset GEMINI_API_KEY

CUDA_VISIBLE_DEVICES=6 cleanslate validate \
    -m "$CKPT_DIR" \
    -B full \
    --skip-done \
    -o "$VAL_OUT" \
    --push-to-hub \
    --hf-org "$HF_ORG" \
    --push-suffix "$DEBUG_TAG" \
    2>&1 | tee "${LOG_PFX}_validate.log"

echo "[e2e] DONE."
echo "  ckpt repo:  $CKPT_REPO_ID"
echo "  eval out:   $EVAL_OUT"
echo "  validate out: $VAL_OUT"
echo "  push suffix:  $DEBUG_TAG"

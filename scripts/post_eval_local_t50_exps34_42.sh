#!/usr/bin/env bash
# Local post-unlearning eval + validate for exps 34-42
# (arpandeepk org, target-50, bm25_10B / igm_10B / bm25_6T)
# Runs on local 8xH100.
#
# Phase 1: memorization eval — 6 small models (GPUs 0-5, 1 each, config pin)
# Phase 2: memorization eval — 3 olmo-32b   (GPUs 0,2,4; CUDA_VISIBLE_DEVICES)
# Phase 3: benchmark validate — 6 small     (GPUs 0-5, tp=1)
# Phase 4: benchmark validate — 3 olmo-32b  (GPU pairs 0-1/2-3/4-5, tp=2)

set -uo pipefail
cd "$(dirname "$0")/.."

ORG="arpandeepk"
LOGDIR="logs/post_eval_local_t50_exps34_42"
RESULTSDIR="results/post_eval_t50"
mkdir -p "$LOGDIR" "$RESULTSDIR"

# ── Phase 1: memorization eval — nemotron × 3 + gemma × 3 ──────────────────
echo "=== Phase 1: memorization eval — 6 small models (GPUs 0-5) ==="
for spec in \
  "nemotron-nano-9b-v2-simnpo-gentle-bm25-10b:config_gpu0.yaml" \
  "nemotron-nano-9b-v2-simnpo-gentle-igm-10b:config_gpu1.yaml" \
  "nemotron-nano-9b-v2-simnpo-gentle-bm25-6t:config_gpu2.yaml" \
  "gemma-3-12b-simnpo-gentle-bm25-10b:config_gpu3.yaml" \
  "gemma-3-12b-simnpo-gentle-igm-10b:config_gpu4.yaml" \
  "gemma-3-12b-simnpo-gentle-bm25-6t:config_gpu5.yaml"
do
  model="${spec%%:*}"; config="${spec##*:}"
  nohup uv run cleanslate eval \
    -m "$ORG/$model" \
    --config "$config" \
    --hf-org "$ORG" \
    -o "$RESULTSDIR/$model/eval" \
    > "$LOGDIR/eval_${model}.log" 2>&1 &
done
echo "  6 eval jobs launched; waiting..."
wait
echo "  Phase 1 done"

# ── Phase 2: memorization eval — olmo-32b × 3 ──────────────────────────────
echo ""
echo "=== Phase 2: memorization eval — 3 olmo-32b (GPUs 0,2,4) ==="
for spec in \
  "0:olmo-3-32b-simnpo-gentle-bm25-10b" \
  "2:olmo-3-32b-simnpo-gentle-igm-10b" \
  "4:olmo-3-32b-simnpo-gentle-bm25-6t"
do
  gpu="${spec%%:*}"; model="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup uv run cleanslate eval \
    -m "$ORG/$model" \
    --config config_olmo32b_eval.yaml \
    --hf-org "$ORG" \
    -o "$RESULTSDIR/$model/eval" \
    > "$LOGDIR/eval_${model}.log" 2>&1 &
done
echo "  3 olmo eval jobs launched; waiting..."
wait
echo "  Phase 2 done"

# ── Phase 3: benchmark validation — nemotron × 3 + gemma × 3 ───────────────
echo ""
echo "=== Phase 3: benchmark validation — 6 small models (GPUs 0-5) ==="
for spec in \
  "0:nemotron-nano-9b-v2-simnpo-gentle-bm25-10b" \
  "1:nemotron-nano-9b-v2-simnpo-gentle-igm-10b" \
  "2:nemotron-nano-9b-v2-simnpo-gentle-bm25-6t" \
  "3:gemma-3-12b-simnpo-gentle-bm25-10b" \
  "4:gemma-3-12b-simnpo-gentle-igm-10b" \
  "5:gemma-3-12b-simnpo-gentle-bm25-6t"
do
  gpu="${spec%%:*}"; model="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" HF_ALLOW_CODE_EVAL=1 nohup uv run cleanslate validate \
    -m "$ORG/$model" \
    -B full \
    --tp-size 1 \
    --hf-org "$ORG" \
    --push-to-hub \
    -o "$RESULTSDIR/$model/validate" \
    > "$LOGDIR/validate_${model}.log" 2>&1 &
done
echo "  6 validate jobs launched; waiting..."
wait
echo "  Phase 3 done"

# ── Phase 4: benchmark validation — olmo-32b × 3 (TP2) ─────────────────────
echo ""
echo "=== Phase 4: benchmark validation — 3 olmo-32b (TP2, GPU pairs 0-1/2-3/4-5) ==="
for spec in \
  "0,1:olmo-3-32b-simnpo-gentle-bm25-10b" \
  "2,3:olmo-3-32b-simnpo-gentle-igm-10b" \
  "4,5:olmo-3-32b-simnpo-gentle-bm25-6t"
do
  gpus="${spec%%:*}"; model="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpus" HF_ALLOW_CODE_EVAL=1 nohup uv run cleanslate validate \
    -m "$ORG/$model" \
    -B full \
    --tp-size 2 \
    --hf-org "$ORG" \
    --push-to-hub \
    -o "$RESULTSDIR/$model/validate" \
    > "$LOGDIR/validate_${model}.log" 2>&1 &
done
echo "  3 olmo validate jobs launched; waiting..."
wait
echo "  Phase 4 done"

echo ""
echo "All 9 evals + 9 validates complete."
echo "Logs:    $LOGDIR/"
echo "Results: $RESULTSDIR/"

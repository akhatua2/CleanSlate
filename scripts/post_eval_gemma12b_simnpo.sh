#!/usr/bin/env bash
# Post-unlearning memorization eval for gemma-3-12b-pt SimNPO checkpoint.
# GPU 4 (TP=1)
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=4

cleanslate eval \
  -m results/final/unlearnt_models/gemma-3-12b-pt/simnpo_gentle_corpus_sweep/simnpo_gemma-3-12b-pt_20260416_171305 \
  -d unlearning-cleanslate/verif-gemma-3-12b-pt-w100-s10 \
  -c default \
  -o results/final/post_eval/gemma-3-12b-pt/simnpo_gentle_corpus_sweep \
  --push-suffix corpus_sweep_post_eval \
  --config config_gemma12b_tp1.yaml

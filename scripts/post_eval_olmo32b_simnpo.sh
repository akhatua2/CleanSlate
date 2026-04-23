#!/usr/bin/env bash
# Post-unlearning memorization eval for olmo-32b SimNPO checkpoint.
# GPUs 0,1,2,3  (TP=2, DP=2)
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1,2,3

cleanslate eval \
  -m results/final/unlearnt_models/olmo-32b/simnpo_gentle_corpus_sweep/simnpo_Olmo-3-1125-32B_20260416_164134 \
  -d unlearning-cleanslate/verif-olmo-3-1125-32b-w100-s10 \
  -c default \
  -o results/final/post_eval/olmo-32b/simnpo_gentle_corpus_sweep \
  --push-suffix corpus_sweep_post_eval \
  --config config_olmo32b_tp2dp2.yaml

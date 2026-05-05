#!/usr/bin/env bash
# Exps 34–42: target-50 missing cells for nemotron-9b, gemma-12b, olmo-32b
# across bm25_10B / igm_10B / bm25_6T retain rules.
# All: simnpo_gentle, Modal B200, push to arpandeepk HF org, final checkpoint only.

set -uo pipefail
cd "$(dirname "$0")/.."

LOGDIR="logs/sweep_t50_exps34_42"
mkdir -p "$LOGDIR"

fire() {
  local row="$1" alias="$2" gpu="$3" config="$4" dataset="$5" rule="$6" tag="$7"
  nohup uv run cleanslate fsid train \
    --config cleanslate/fsid/configs/$config.yaml \
    -m $alias \
    --forget-path unlearning-cleanslate/$dataset:forget:$rule \
    --retain-path unlearning-cleanslate/$dataset:retain:$rule \
    --modal --modal-gpu $gpu --push-to-hub \
    --hf-org arpandeepk \
    --save-every 0 \
    > $LOGDIR/cell${row}_${tag}.log 2>&1 &
}

# BM25 Midtraining (bm25_10B)
fire 34 nemotron-9b  B200    simnpo_gentle  fsid-curated-nemotron-9b  bm25_10B  nemotron_bm25_10B_t50
fire 35 gemma-12b    B200    simnpo_gentle  fsid-curated-gemma-12b    bm25_10B  gemma_bm25_10B_t50
fire 36 olmo-32b     B200:8  simnpo_gentle  fsid-curated-olmo-32b     bm25_10B  olmo32b_bm25_10B_t50

# IGM Midtraining (igm_10B)
fire 37 nemotron-9b  B200    simnpo_gentle  fsid-curated-nemotron-9b  igm_10B   nemotron_igm_10B_t50
fire 38 gemma-12b    B200    simnpo_gentle  fsid-curated-gemma-12b    igm_10B   gemma_igm_10B_t50
fire 39 olmo-32b     B200:8  simnpo_gentle  fsid-curated-olmo-32b     igm_10B   olmo32b_igm_10B_t50

# BM25 Pretraining (bm25_6T)
fire 40 nemotron-9b  B200    simnpo_gentle  fsid-curated-nemotron-9b  bm25_6T   nemotron_bm25_6T_t50
fire 41 gemma-12b    B200    simnpo_gentle  fsid-curated-gemma-12b    bm25_6T   gemma_bm25_6T_t50
fire 42 olmo-32b     B200:8  simnpo_gentle  fsid-curated-olmo-32b     bm25_6T   olmo32b_bm25_6T_t50

echo "fired 9 trainings (exps 34-42)"

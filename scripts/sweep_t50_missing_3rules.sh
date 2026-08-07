#!/usr/bin/env bash
# Fire the 9 missing target-50 simnpo_gentle cells (exp 34-42 in target-50 sheet):
# nemotron-9b, gemma-12b, olmo-32b on each of bm25_10B, igm_10B, bm25_6T.
#
# Naming: target-50 trained -> arpandeepk/<model_clean>-<algo>-<rule>
# (auto-derived; no --hub-repo-id). Pushing to arpandeepk because
# unlearning-cleanslate is full.

set -uo pipefail
cd "$(dirname "$0")/.."

LOGDIR="logs/sweep_simnpo_gentle"
mkdir -p "$LOGDIR"

fire() {
  local row="$1" alias="$2" gpu="$3" config="$4" dataset="$5" rule="$6" hub_repo_id="$7" tag="$8"
  local extra=""
  [ -n "$hub_repo_id" ] && extra="--hub-repo-id $hub_repo_id"
  nohup uv run cleanslate fsid train \
    --config cleanslate/fsid/configs/$config.yaml \
    -m $alias \
    --forget-path unlearning-cleanslate/$dataset:forget:$rule \
    --retain-path unlearning-cleanslate/$dataset:retain:$rule \
    --modal --modal-gpu $gpu --push-to-hub --hf-org arpandeepk $extra \
    > $LOGDIR/cell${row}_${tag}.log 2>&1 &
}

# Row 34-36: BM25 mid (Midtraining) simnpo_gentle
fire 34 nemotron-9b  B200    simnpo_gentle fsid-curated-nemotron-9b  bm25_10B  ""  nemotron_bm25_10B_t50
fire 35 gemma-12b    B200    simnpo_gentle fsid-curated-gemma-12b    bm25_10B  ""  gemma_bm25_10B_t50
fire 36 olmo-32b     B200:8  simnpo_gentle fsid-curated-olmo-32b     bm25_10B  ""  olmo32b_bm25_10B_t50

# Row 37-39: IGM mid (Midtraining) simnpo_gentle
fire 37 nemotron-9b  B200    simnpo_gentle fsid-curated-nemotron-9b  igm_10B   ""  nemotron_igm_10B_t50
fire 38 gemma-12b    B200    simnpo_gentle fsid-curated-gemma-12b    igm_10B   ""  gemma_igm_10B_t50
fire 39 olmo-32b     B200:8  simnpo_gentle fsid-curated-olmo-32b     igm_10B   ""  olmo32b_igm_10B_t50

# Row 40-42: BM25 pre (Pretraining) simnpo_gentle
fire 40 nemotron-9b  B200    simnpo_gentle fsid-curated-nemotron-9b  bm25_6T   ""  nemotron_bm25_6T_t50
fire 41 gemma-12b    B200    simnpo_gentle fsid-curated-gemma-12b    bm25_6T   ""  gemma_bm25_6T_t50
fire 42 olmo-32b     B200:8  simnpo_gentle fsid-curated-olmo-32b     bm25_6T   ""  olmo32b_bm25_6T_t50

echo "fired 9 target-50 trainings (exp 34-42)"

#!/usr/bin/env bash
# Fire missing target-50 and full target-100 training sweep on Modal.
#
# Naming:
#   target-50 trained → unlearning-cleanslate/<model_clean>-<algo>-<rule>            (auto-derived)
#   target-100 trained → unlearning-cleanslate/<model_clean>-<algo>-<rule>-target-100 (explicit --hub-repo-id)

set -uo pipefail
cd "$(dirname "$0")/.."

LOGDIR="logs/sweep_simnpo_gentle"

fire() {
  local row="$1" alias="$2" gpu="$3" config="$4" dataset="$5" rule="$6" hub_repo_id="$7" tag="$8"
  local extra=""
  [ -n "$hub_repo_id" ] && extra="--hub-repo-id $hub_repo_id"
  nohup uv run cleanslate fsid train \
    --config cleanslate/fsid/configs/$config.yaml \
    -m $alias \
    --forget-path unlearning-cleanslate/$dataset:forget:$rule \
    --retain-path unlearning-cleanslate/$dataset:retain:$rule \
    --modal --modal-gpu $gpu --push-to-hub $extra \
    > $LOGDIR/cell${row}_${tag}.log 2>&1 &
}

# ============================================================
# TARGET-50 (10 missing)
# ============================================================
fire 02 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b      baseline   ""  olmo7b_simnpo_gentle_t50
fire 04 nemotron-9b  B200    simnpo_gentle fsid-curated-nemotron-9b  baseline   ""  nemotron_simnpo_gentle_t50
fire 06 olmo-32b     B200:8  simnpo_gentle fsid-curated-olmo-32b     baseline   ""  olmo32b_simnpo_gentle_t50
fire 07 olmo-7b      B200    simnpo        fsid-curated-olmo-7b      baseline   ""  olmo7b_simnpo_aggressive_t50
fire 08 nemotron-9b  B200    simnpo        fsid-curated-nemotron-9b  baseline   ""  nemotron_simnpo_aggressive_t50
fire 11 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b      bm25_10B   ""  olmo7b_bm25_10B_t50
fire 14 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b      igm_10B    ""  olmo7b_igm_10B_t50
fire 17 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b      bm25_6T    ""  olmo7b_bm25_6T_t50
fire 20 olmo-7b      B200    undial        fsid-curated-olmo-7b      baseline   ""  olmo7b_undial_t50
fire 23 olmo-7b      B200    rmu           fsid-curated-olmo-7b      baseline   ""  olmo7b_rmu_t50

# ============================================================
# TARGET-100 (all 24 cells - need explicit --hub-repo-id with -target-100 suffix)
# ============================================================
ORG="unlearning-cleanslate"

# Row 1-6 + 9: simnpo_gentle Oracle
fire 01 llama-8b     B200    simnpo_gentle fsid-curated-llama-8b-target-100      baseline  $ORG/llama-3_1-8b-simnpo-gentle-baseline-target-100  llama_simnpo_gentle_t100
fire 02 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b-target-100       baseline  $ORG/olmo-3-7b-simnpo-gentle-baseline-target-100    olmo7b_simnpo_gentle_t100
fire 03 qwen-8b      B200    simnpo_gentle fsid-curated-qwen3-8b-target-100      baseline  $ORG/qwen3-8b-simnpo-gentle-baseline-target-100      qwen_simnpo_gentle_t100
fire 04 nemotron-9b  B200    simnpo_gentle fsid-curated-nemotron-9b-target-100   baseline  $ORG/nemotron-nano-9b-v2-simnpo-gentle-baseline-target-100 nemotron_simnpo_gentle_t100
fire 05 gemma-12b    B200    simnpo_gentle fsid-curated-gemma-12b-target-100     baseline  $ORG/gemma-3-12b-simnpo-gentle-baseline-target-100   gemma_simnpo_gentle_t100
fire 06 olmo-32b     B200:8  simnpo_gentle fsid-curated-olmo-32b-target-100      baseline  $ORG/olmo-3-32b-simnpo-gentle-baseline-target-100    olmo32b_simnpo_gentle_t100

# Row 7-9: simnpo aggressive Oracle
fire 07 olmo-7b      B200    simnpo        fsid-curated-olmo-7b-target-100       baseline  $ORG/olmo-3-7b-simnpo-baseline-target-100            olmo7b_simnpo_aggressive_t100
fire 08 nemotron-9b  B200    simnpo        fsid-curated-nemotron-9b-target-100   baseline  $ORG/nemotron-nano-9b-v2-simnpo-baseline-target-100  nemotron_simnpo_aggressive_t100
fire 09 gemma-12b    B200    simnpo        fsid-curated-gemma-12b-target-100     baseline  $ORG/gemma-3-12b-simnpo-baseline-target-100          gemma_simnpo_aggressive_t100

# Row 10-12: BM25 mid simnpo gentle
fire 10 llama-8b     B200    simnpo_gentle fsid-curated-llama-8b-target-100      bm25_10B  $ORG/llama-3_1-8b-simnpo-gentle-bm25-10b-target-100  llama_bm25_10B_t100
fire 11 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b-target-100       bm25_10B  $ORG/olmo-3-7b-simnpo-gentle-bm25-10b-target-100     olmo7b_bm25_10B_t100
fire 12 qwen-8b      B200    simnpo_gentle fsid-curated-qwen3-8b-target-100      bm25_10B  $ORG/qwen3-8b-simnpo-gentle-bm25-10b-target-100      qwen_bm25_10B_t100

# Row 13-15: IGM mid simnpo gentle
fire 13 llama-8b     B200    simnpo_gentle fsid-curated-llama-8b-target-100      igm_10B   $ORG/llama-3_1-8b-simnpo-gentle-igm-10b-target-100   llama_igm_10B_t100
fire 14 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b-target-100       igm_10B   $ORG/olmo-3-7b-simnpo-gentle-igm-10b-target-100      olmo7b_igm_10B_t100
fire 15 qwen-8b      B200    simnpo_gentle fsid-curated-qwen3-8b-target-100      igm_10B   $ORG/qwen3-8b-simnpo-gentle-igm-10b-target-100       qwen_igm_10B_t100

# Row 16-18: BM25 pre simnpo gentle
fire 16 llama-8b     B200    simnpo_gentle fsid-curated-llama-8b-target-100      bm25_6T   $ORG/llama-3_1-8b-simnpo-gentle-bm25-6t-target-100   llama_bm25_6T_t100
fire 17 olmo-7b      B200    simnpo_gentle fsid-curated-olmo-7b-target-100       bm25_6T   $ORG/olmo-3-7b-simnpo-gentle-bm25-6t-target-100      olmo7b_bm25_6T_t100
fire 18 qwen-8b      B200    simnpo_gentle fsid-curated-qwen3-8b-target-100      bm25_6T   $ORG/qwen3-8b-simnpo-gentle-bm25-6t-target-100       qwen_bm25_6T_t100

# Row 19-21: undial Oracle
fire 19 llama-8b     B200    undial        fsid-curated-llama-8b-target-100      baseline  $ORG/llama-3_1-8b-undial-baseline-target-100         llama_undial_t100
fire 20 olmo-7b      B200    undial        fsid-curated-olmo-7b-target-100       baseline  $ORG/olmo-3-7b-undial-baseline-target-100            olmo7b_undial_t100
fire 21 qwen-8b      B200    undial        fsid-curated-qwen3-8b-target-100      baseline  $ORG/qwen3-8b-undial-baseline-target-100             qwen_undial_t100

# Row 22-24: RMU Oracle
fire 22 llama-8b     B200    rmu           fsid-curated-llama-8b-target-100      baseline  $ORG/llama-3_1-8b-rmu-baseline-target-100            llama_rmu_t100
fire 23 olmo-7b      B200    rmu           fsid-curated-olmo-7b-target-100       baseline  $ORG/olmo-3-7b-rmu-baseline-target-100               olmo7b_rmu_t100
fire 24 qwen-8b      B200    rmu           fsid-curated-qwen3-8b-target-100      baseline  $ORG/qwen3-8b-rmu-baseline-target-100                qwen_rmu_t100

echo "fired 34 trainings (10 target-50 missing + 24 target-100)"

#!/usr/bin/env bash
# ============================================================================
# CleanSlate Final Pipeline — Local Orchestrator (skampere3, 6x B200)
#
# Launches all 4 local model pipelines across 6 GPUs (3 pairs).
#
# Wave 1 (parallel, all 6 GPUs):
#   GPUs 0,1  → olmo-32b   (FSDP training, tp=2 eval)
#   GPUs 2,3  → gemma-12b
#   GPUs 4,5  → gpt-oss-20b
#
# Wave 2 (after all 3 finish):
#   GPUs 0,1  → nemotron-9b
#
# Usage:
#   ./scripts/final/run_local.sh              # default GPUs 2,3,4,5,6,7
#   ./scripts/final/run_local.sh 0,1,2,3,4,5  # custom GPU assignment
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── HF / model cache setup ─────────────────────────────────────────────────
export LOCAL_MACHINE_CACHE="${LOCAL_MACHINE_CACHE:-/lfs/skampere3/0/animjha/.cache}"
export HF_HOME="$LOCAL_MACHINE_CACHE"
export HF_HUB_CACHE="$LOCAL_MACHINE_CACHE/hub"
export HF_DATASETS_CACHE="$LOCAL_MACHINE_CACHE/datasets"
export TRANSFORMERS_CACHE="$LOCAL_MACHINE_CACHE/transformers"
export HF_TOKENIZERS_CACHE="$LOCAL_MACHINE_CACHE/tokenizers"
export VLLM_CACHE_DIR="$LOCAL_MACHINE_CACHE/vllm"

GPU_IDS="${1:-2,3,4,5,6,7}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"

if (( ${#GPUS[@]} < 6 )); then
    echo "ERROR: Need at least 6 GPUs (got ${#GPUS[@]})" >&2
    exit 1
fi

G0="${GPUS[0]}"
G1="${GPUS[1]}"
G2="${GPUS[2]}"
G3="${GPUS[3]}"
G4="${GPUS[4]}"
G5="${GPUS[5]}"

LOG_DIR="logs/final"
mkdir -p "$LOG_DIR"

echo "============================================="
echo " CleanSlate Final Pipeline — Local Cluster"
echo " GPUs: ${GPU_IDS}"
echo " Started: $(date)"
echo "============================================="
echo ""

# ── Wave 1: olmo-32b, gemma-12b, gpt-oss-20b in parallel ──
echo "=== Wave 1: 3 models in parallel ==="

echo "Starting olmo-32b on GPUs ${G0},${G1} ..."
"$SCRIPT_DIR/olmo-32b.sh" "${G0},${G1}" \
    > "${LOG_DIR}/olmo-32b.log" 2>&1 &
PID_OLMO32=$!
echo "  PID: ${PID_OLMO32}  →  ${LOG_DIR}/olmo-32b.log"

echo "Starting gemma-12b on GPUs ${G2},${G3} ..."
"$SCRIPT_DIR/gemma-12b.sh" "${G2},${G3}" \
    > "${LOG_DIR}/gemma-12b.log" 2>&1 &
PID_GEMMA=$!
echo "  PID: ${PID_GEMMA}  →  ${LOG_DIR}/gemma-12b.log"

echo "Starting gpt-oss-20b on GPUs ${G4},${G5} ..."
"$SCRIPT_DIR/gpt-oss-20b.sh" "${G4},${G5}" \
    > "${LOG_DIR}/gpt-oss-20b.log" 2>&1 &
PID_GPT=$!
echo "  PID: ${PID_GPT}  →  ${LOG_DIR}/gpt-oss-20b.log"

echo ""
echo "Wave 1 launched. Waiting for all 3 to finish ..."
echo "  Monitor: tail -f ${LOG_DIR}/*.log"
echo ""

FAILED=0

wait $PID_OLMO32 || { echo "FAILED: olmo-32b"; FAILED=$((FAILED+1)); }
echo "olmo-32b:    done ($(date))"

wait $PID_GEMMA || { echo "FAILED: gemma-12b"; FAILED=$((FAILED+1)); }
echo "gemma-12b:   done ($(date))"

wait $PID_GPT || { echo "FAILED: gpt-oss-20b"; FAILED=$((FAILED+1)); }
echo "gpt-oss-20b: done ($(date))"

# ── Wave 2: nemotron-9b ──
echo ""
echo "=== Wave 2: nemotron-9b ==="
echo "Starting nemotron-9b on GPUs ${G0},${G1} ..."
"$SCRIPT_DIR/nemotron-9b.sh" "${G0},${G1}" \
    > "${LOG_DIR}/nemotron-9b.log" 2>&1 &
PID_NEMOTRON=$!
echo "  PID: ${PID_NEMOTRON}  →  ${LOG_DIR}/nemotron-9b.log"

wait $PID_NEMOTRON || { echo "FAILED: nemotron-9b"; FAILED=$((FAILED+1)); }
echo "nemotron-9b: done ($(date))"

echo ""
echo "============================================="
if (( FAILED > 0 )); then
    echo " ${FAILED} model(s) FAILED. Check logs."
    exit 1
else
    echo " All 4 local models complete!"
fi
echo " Finished: $(date)"
echo "============================================="

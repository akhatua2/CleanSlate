#!/usr/bin/env bash
# ============================================================================
# CleanSlate Final Pipeline — Local Orchestrator (skampere3, 4x B200)
#
# Launches all 4 local model pipelines across 4 GPUs (2 pairs).
#
# Wave 1 (parallel):
#   GPUs 0,1  → gemma-12b
#   GPUs 2,3  → olmo-32b
#
# Wave 2 (after both finish):
#   GPUs 0,1  → gpt-oss-20b
#   GPUs 2,3  → nemotron-9b
#
# Usage:
#   ./scripts/final/run_local.sh              # default GPUs 0,1,2,3
#   ./scripts/final/run_local.sh 4,5,6,7      # custom GPU assignment
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU_IDS="${1:-4,5,6,7}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"

if (( ${#GPUS[@]} < 4 )); then
    echo "ERROR: Need at least 4 GPUs (got ${#GPUS[@]})" >&2
    exit 1
fi

G0="${GPUS[0]}"
G1="${GPUS[1]}"
G2="${GPUS[2]}"
G3="${GPUS[3]}"

LOG_DIR="logs/final"
mkdir -p "$LOG_DIR"

echo "============================================="
echo " CleanSlate Final Pipeline — Local Cluster"
echo " GPUs: ${GPU_IDS}"
echo " Started: $(date)"
echo "============================================="
echo ""

# ── Wave 1: gemma-12b + olmo-32b in parallel ──
echo "=== Wave 1: gemma-12b + olmo-32b in parallel ==="

echo "Starting gemma-12b on GPUs ${G0},${G1} ..."
"$SCRIPT_DIR/gemma-12b.sh" "${G0},${G1}" \
    > "${LOG_DIR}/gemma-12b.log" 2>&1 &
PID_GEMMA=$!
echo "  PID: ${PID_GEMMA}  →  ${LOG_DIR}/gemma-12b.log"

echo "Starting olmo-32b on GPUs ${G2},${G3} ..."
"$SCRIPT_DIR/olmo-32b.sh" "${G2},${G3}" \
    > "${LOG_DIR}/olmo-32b.log" 2>&1 &
PID_OLMO32=$!
echo "  PID: ${PID_OLMO32}  →  ${LOG_DIR}/olmo-32b.log"

echo ""
echo "Wave 1 launched. Waiting for both to finish ..."
echo "  Monitor: tail -f ${LOG_DIR}/*.log"
echo ""

FAILED=0

wait $PID_GEMMA || { echo "FAILED: gemma-12b"; FAILED=$((FAILED+1)); }
echo "gemma-12b:   done ($(date))"

wait $PID_OLMO32 || { echo "FAILED: olmo-32b"; FAILED=$((FAILED+1)); }
echo "olmo-32b:    done ($(date))"

# ── Wave 2: gpt-oss-20b + nemotron-9b in parallel ──
echo ""
echo "=== Wave 2: gpt-oss-20b + nemotron-9b in parallel ==="

echo "Starting gpt-oss-20b on GPUs ${G0},${G1} ..."
"$SCRIPT_DIR/gpt-oss-20b.sh" "${G0},${G1}" \
    > "${LOG_DIR}/gpt-oss-20b.log" 2>&1 &
PID_GPT=$!
echo "  PID: ${PID_GPT}  →  ${LOG_DIR}/gpt-oss-20b.log"

echo "Starting nemotron-9b on GPUs ${G2},${G3} ..."
"$SCRIPT_DIR/nemotron-9b.sh" "${G2},${G3}" \
    > "${LOG_DIR}/nemotron-9b.log" 2>&1 &
PID_NEMOTRON=$!
echo "  PID: ${PID_NEMOTRON}  →  ${LOG_DIR}/nemotron-9b.log"

echo ""
echo "Wave 2 launched. Waiting for both to finish ..."
echo ""

wait $PID_GPT || { echo "FAILED: gpt-oss-20b"; FAILED=$((FAILED+1)); }
echo "gpt-oss-20b: done ($(date))"

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

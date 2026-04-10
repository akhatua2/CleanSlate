#!/bin/bash
# Run memorization eval across all models sequentially.
# Models are ordered to minimize GPU downtime:
# - ~8B models first (DP=2, one per GPU)
# - ~32B models last (need TP=2 or DP=1)
#
# Usage: bash run_all_evals.sh [--no-push]

set -euo pipefail

NO_PUSH="${1:---no-push}"
OUTPUT="results/eval"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

# Config for ~8B models: DP=2 (one replica per GPU)
CONFIG_8B="config.yaml"

# Config for ~32B models: TP=2 (split across both GPUs)
CONFIG_32B="config_32b.yaml"

run_eval() {
    local alias="$1"
    local config="${2:-$CONFIG_8B}"
    local logfile="$LOG_DIR/${alias}.log"

    echo "=========================================="
    echo "  Model: $alias"
    echo "  Config: $config"
    echo "  Started: $(date)"
    echo "=========================================="

    if uv run cleanslate eval -m "$alias" -o "$OUTPUT" --config "$config" $NO_PUSH 2>&1 | tee "$logfile"; then
        echo "✓ $alias completed at $(date)"
    else
        echo "✗ $alias FAILED at $(date)" >&2
    fi
    echo ""
}

echo "Starting full evaluation sweep: $(date)"
echo ""

# ── ~8B cross-family ──────────────────────────────────
run_eval "llama-8b"
run_eval "qwen-8b"
run_eval "olmo-7b"
run_eval "gemma-12b"
run_eval "marin-8b"

# ── Qwen version range (~8B) ─────────────────────────
run_eval "qwen2-7b"
run_eval "qwen2.5-7b"
run_eval "qwen3.5-9b"

# ── Qwen2.5-7B training stages ───────────────────────
run_eval "qwen2.5-7b-instruct"
run_eval "qwen2.5-coder-7b"
run_eval "qwen2.5-coder-7b-instruct"
run_eval "qwen2.5-math-7b"
run_eval "qwen2.5-math-7b-instruct"

# ── ~32B models (switch to TP=2 config) ───────────────
run_eval "olmo-32b"    "$CONFIG_32B"
run_eval "qwen-32b"    "$CONFIG_32B"
run_eval "gemma-27b"   "$CONFIG_32B"
run_eval "marin-32b"   "$CONFIG_32B"

echo "=========================================="
echo "  All evaluations complete: $(date)"
echo "=========================================="

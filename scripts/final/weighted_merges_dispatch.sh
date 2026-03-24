#!/usr/bin/env bash
# ============================================================================
# Dispatcher: monitors GPUs, launches weighted_merge_single.sh per model
#
# Usage: ./scripts/final/weighted_merges_dispatch.sh [initial_gpu_id]
# ============================================================================
set -euo pipefail

IDLE_THRESHOLD_MIB=10
IDLE_WAIT_SECONDS=300
CHECK_INTERVAL=30

PENDING=("gemma-12b" "nemotron-9b" "olmo-32b")
declare -A idle_since=()
declare -A gpu_claimed=()
PIDS=()
PID_MODELS=()

# If initial GPU given, launch first model immediately
if [[ -n "${1:-}" ]]; then
    INIT_GPU="$1"
    MODEL="${PENDING[0]}"
    PENDING=("${PENDING[@]:1}")
    gpu_claimed[$INIT_GPU]=1
    echo "Launching ${MODEL} on GPU ${INIT_GPU} immediately"
    bash scripts/final/weighted_merge_single.sh "$MODEL" "$INIT_GPU" \
        > "results/final/score/${MODEL}/weighted_merge.log" 2>&1 &
    PIDS+=($!)
    PID_MODELS+=("$MODEL")
fi

while (( ${#PENDING[@]} > 0 || ${#PIDS[@]} > 0 )); do
    # Clean up finished
    NEW_PIDS=()
    NEW_MODELS=()
    for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            NEW_PIDS+=("${PIDS[$i]}")
            NEW_MODELS+=("${PID_MODELS[$i]}")
        else
            wait "${PIDS[$i]}" && echo "${PID_MODELS[$i]} completed successfully." \
                || echo "WARNING: ${PID_MODELS[$i]} failed!"
        fi
    done
    PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
    PID_MODELS=("${NEW_MODELS[@]+"${NEW_MODELS[@]}"}")

    if (( ${#PENDING[@]} == 0 )); then
        (( ${#PIDS[@]} > 0 )) && sleep "$CHECK_INTERVAL"
        continue
    fi

    # Scan for idle GPUs
    now=$(date +%s)
    while IFS=', ' read -r gpu_id mem_used_mib; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used_mib=$(echo "$mem_used_mib" | tr -d ' MiB')
        [[ -n "${gpu_claimed[$gpu_id]:-}" ]] && continue

        if (( mem_used_mib <= IDLE_THRESHOLD_MIB )); then
            if [[ -z "${idle_since[$gpu_id]:-}" ]]; then
                idle_since[$gpu_id]=$now
                echo "GPU $gpu_id idle at $(date)"
            elif (( now - ${idle_since[$gpu_id]} >= IDLE_WAIT_SECONDS )); then
                MODEL="${PENDING[0]}"
                PENDING=("${PENDING[@]:1}")
                gpu_claimed[$gpu_id]=1
                echo "Claiming GPU $gpu_id for ${MODEL}"
                bash scripts/final/weighted_merge_single.sh "$MODEL" "$gpu_id" \
                    > "results/final/score/${MODEL}/weighted_merge.log" 2>&1 &
                PIDS+=($!)
                PID_MODELS+=("$MODEL")
                (( ${#PENDING[@]} == 0 )) && break
            fi
        else
            [[ -n "${idle_since[$gpu_id]:-}" ]] && echo "GPU $gpu_id busy (${mem_used_mib} MiB)"
            idle_since[$gpu_id]=""
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    sleep "$CHECK_INTERVAL"
done

echo "=== All weighted merges complete  |  $(date) ==="

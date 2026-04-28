set -uo pipefail
source .venv/bin/activate
ulimit -n 1048576
mkdir -p logs

CUDA_VISIBLE_DEVICES=6 vllm serve Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 42800 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    > logs/judge_server.log 2>&1 &
JUDGE_PID=$!

VAL_PIDS=()
cleanup() {
    for pid in "${VAL_PIDS[@]:-}" "$JUDGE_PID"; do
        [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

echo "Waiting for judge server (pid=$JUDGE_PID) to come up..."
until curl -sf http://127.0.0.1:42800/v1/models >/dev/null; do
  sleep 5
  kill -0 $JUDGE_PID 2>/dev/null || { echo "judge server died"; tail -50 logs/judge_server.log; exit 1; }
done
echo "Judge server ready."

export JUDGE_BASE_URL=http://127.0.0.1:42800
export JUDGE_MODEL=Qwen/Qwen3.5-9B
export HF_ALLOW_CODE_EVAL=1
## only for skampere3 cuz it sux
export HF_HOME="$LOCAL_MACHINE_CACHE"
export HF_HUB_CACHE="$LOCAL_MACHINE_CACHE/hub"
export HF_DATASETS_CACHE="$LOCAL_MACHINE_CACHE/datasets"
export TRANSFORMERS_CACHE="$LOCAL_MACHINE_CACHE/transformers"
export HF_TOKENIZERS_CACHE="$LOCAL_MACHINE_CACHE/tokenizers"
export VLLM_CACHE_DIR="$LOCAL_MACHINE_CACHE/vllm"
unset GEMINI_API_KEY

run_validate() {
    local gpu="$1" alias_name="$2"
    CUDA_VISIBLE_DEVICES="$gpu" cleanslate validate \
        -m "$alias_name" \
        -B full \
        --skip-done \
        -o "results/post_val/${alias_name}/validate" \
        --push-to-hub \
        --push-suffix post_val \
        > "logs/validate_${alias_name}.log" 2>&1
}

GPUS=(2 7 4)
JOBS=(<> <> <> <> <> <>)

declare -A EXITS
declare -A PID_GPU
declare -A PID_MODEL

dispatch() {
    local gpu="$1" model="$2"
    run_validate "$gpu" "$model" &
    local pid=$!
    PID_GPU[$pid]=$gpu
    PID_MODEL[$pid]=$model
    VAL_PIDS+=("$pid")
    echo "[dispatch] $model on gpu=$gpu pid=$pid"
}

next_idx=0
for gpu in "${GPUS[@]}"; do
    if (( next_idx < ${#JOBS[@]} )); then
        dispatch "$gpu" "${JOBS[next_idx]}"
        ((next_idx++))
    fi
done

while (( ${#PID_GPU[@]} > 0 )); do
    wait -n -p done_pid
    code=$?
    EXITS[$done_pid]=$code
    gpu="${PID_GPU[$done_pid]}"
    model="${PID_MODEL[$done_pid]}"
    unset 'PID_GPU[$done_pid]'
    unset 'PID_MODEL[$done_pid]'
    if [[ "$code" -eq 0 ]]; then
        echo "[done] $model on gpu=$gpu OK"
    else
        echo "[done] $model on gpu=$gpu FAILED (exit=$code) — see logs/"
    fi
    if (( next_idx < ${#JOBS[@]} )); then
        dispatch "$gpu" "${JOBS[next_idx]}"
        ((next_idx++))
    fi
done

fail=0
for pid in "${VAL_PIDS[@]}"; do
    [[ "${EXITS[$pid]:-1}" -eq 0 ]] || fail=1
done
exit $fail






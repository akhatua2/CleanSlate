#!/usr/bin/env bash
# Compute the pre-only baseline summary for every model × target_pool.
# Each row is independent (no GPU work, just HF dataset loads + polars
# aggregation), so we fan them out in parallel.
#
# Concurrency: up to PARALLEL jobs at once (default 6). Override with
# `PARALLEL=12 bash scripts/final_final/compute_pre_scores.sh`.

set -uo pipefail

if [[ -f "$(dirname "$0")/../../.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../../.venv/bin/activate"
fi

PARALLEL="${PARALLEL:-6}"
mkdir -p results/pre_score logs/pre_score

# rows: TAG  FORGET_POOL  PRE_EVAL  PRE_GEN
ROWS=(
  # target=100
  "llama-8b-t100      unlearning-cleanslate/fsid-curated-llama-8b-target-100:forget_pool       unlearning-cleanslate/verif-llama-3.1-8b-w100-s10        unlearning-cleanslate/generations-llama-3_1-8b-pre_val"
  "olmo-7b-t100       unlearning-cleanslate/fsid-curated-olmo-7b-target-100:forget_pool        unlearning-cleanslate/eval-olmo-3-7b-w100-s10            unlearning-cleanslate/generations-olmo-3-7b-pre_val"
  "qwen3-8b-t100      unlearning-cleanslate/fsid-curated-qwen3-8b-target-100:forget_pool       unlearning-cleanslate/verif-qwen3-8b-w100-s10            unlearning-cleanslate/generations-qwen3-8b-pre_val"
  "nemotron-9b-t100   unlearning-cleanslate/fsid-curated-nemotron-9b-target-100:forget_pool    unlearning-cleanslate/eval-nemotron-nano-9b-v2-w100-s10  unlearning-cleanslate/generations-nemotron-nano-9b-v2-pre_val"
  "gemma-12b-t100     unlearning-cleanslate/fsid-curated-gemma-12b-target-100:forget_pool      unlearning-cleanslate/verif-gemma-3-12b-pt-w100-s10      unlearning-cleanslate/generations-gemma-3-12b-pre_val"
  "olmo-32b-t100      unlearning-cleanslate/fsid-curated-olmo-32b-target-100:forget_pool       unlearning-cleanslate/verif-olmo-3-1125-32b-w100-s10     unlearning-cleanslate/generations-olmo-3-32b-pre_val"
  # target=50
  "llama-8b-t50       unlearning-cleanslate/fsid-curated-llama-8b:forget_pool                  unlearning-cleanslate/verif-llama-3.1-8b-w100-s10        unlearning-cleanslate/generations-llama-3_1-8b-pre_val"
  "olmo-7b-t50        unlearning-cleanslate/fsid-curated-olmo-7b:forget_pool                   unlearning-cleanslate/eval-olmo-3-7b-w100-s10            unlearning-cleanslate/generations-olmo-3-7b-pre_val"
  "qwen3-8b-t50       unlearning-cleanslate/fsid-curated-qwen3-8b:forget_pool                  unlearning-cleanslate/verif-qwen3-8b-w100-s10            unlearning-cleanslate/generations-qwen3-8b-pre_val"
  "nemotron-9b-t50    unlearning-cleanslate/fsid-curated-nemotron-9b:forget_pool               unlearning-cleanslate/eval-nemotron-nano-9b-v2-w100-s10  unlearning-cleanslate/generations-nemotron-nano-9b-v2-pre_val"
  "gemma-12b-t50      unlearning-cleanslate/fsid-curated-gemma-12b:forget_pool                 unlearning-cleanslate/verif-gemma-3-12b-pt-w100-s10      unlearning-cleanslate/generations-gemma-3-12b-pre_val"
  "olmo-32b-t50       unlearning-cleanslate/fsid-curated-olmo-32b:forget_pool                  unlearning-cleanslate/verif-olmo-3-1125-32b-w100-s10     unlearning-cleanslate/generations-olmo-3-32b-pre_val"
)

run_one() {
  local tag="$1" pool="$2" pre_eval="$3" pre_gen="$4"
  local out="results/pre_score/${tag}.json"
  local log="logs/pre_score/${tag}.log"
  echo "[start] ${tag}"
  if cleanslate score-pre \
        --forget-pool "$pool" \
        --pre-eval    "$pre_eval" \
        --pre-gen     "$pre_gen" \
        -o "$out" >"$log" 2>&1; then
    echo "[ok]    ${tag}  -> ${out}"
  else
    echo "[FAIL]  ${tag}  (see ${log})" >&2
    return 1
  fi
}

PIDS=()
declare -A PID_TAG
fail=0

for row in "${ROWS[@]}"; do
  # Throttle: wait when at capacity.
  while (( ${#PIDS[@]} >= PARALLEL )); do
    if wait -n -p done_pid 2>/dev/null; then
      code=$?
    else
      code=$?
    fi
    [[ "$code" -ne 0 ]] && fail=1
    new_pids=()
    for p in "${PIDS[@]}"; do
      [[ "$p" != "$done_pid" ]] && new_pids+=("$p")
    done
    PIDS=("${new_pids[@]}")
  done

  read -r tag pool pre_eval pre_gen <<<"$row"
  run_one "$tag" "$pool" "$pre_eval" "$pre_gen" &
  pid=$!
  PIDS+=("$pid")
  PID_TAG[$pid]="$tag"
done

# Drain remaining jobs.
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
    echo "[FAIL] ${PID_TAG[$pid]}" >&2
  fi
done

if (( fail )); then
  echo "One or more pre-score jobs failed; check logs/pre_score/*.log" >&2
  exit 1
fi

echo "All pre-score jobs complete."

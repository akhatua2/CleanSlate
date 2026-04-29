#!/bin/bash
#SBATCH --job-name=post-val-judge
#SBATCH --partition=tamper_resistance_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/final_final/post_val_judge/slurm-%j-%x.out

set -euo pipefail

cd /data/animesh_jha/CleanSlate
mkdir -p logs/final_final/post_val_judge

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${ROW_ID:?ROW_ID is required}"

ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE:-results/.judge_endpoint}"
[[ -f "$ENDPOINT_FILE" ]] || {
  echo "ERROR: $ENDPOINT_FILE missing. Submit judge_server.slurm first and wait for it to come up." >&2
  exit 2
}
export JUDGE_BASE_URL="$(< "$ENDPOINT_FILE")"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-9B}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
export HF_ALLOW_CODE_EVAL=1

MANIFEST="${MANIFEST:-results/final_final/post_eval_manifest.json}"
[[ -f "$MANIFEST" ]] || { echo "ERROR: missing manifest $MANIFEST" >&2; exit 2; }

eval "$(
  python - "$MANIFEST" "$ROW_ID" <<'PY'
import json
import shlex
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
row_id = int(sys.argv[2])
with manifest_path.open() as f:
    manifest = json.load(f)
matches = [row for row in manifest.get("rows", []) if int(row["row_id"]) == row_id]
if len(matches) != 1:
    raise SystemExit(f"ROW_ID={row_id} matched {len(matches)} rows in {manifest_path}")
row = matches[0]
local_dir = Path(row["local_dir"])
print(f"LOCAL_DIR={shlex.quote(str(local_dir))}")
print(f"CLEAN={shlex.quote(local_dir.name)}")
PY
)"

[[ -d "$LOCAL_DIR" ]] || { echo "ERROR: local checkpoint dir missing: $LOCAL_DIR" >&2; exit 2; }
GEN_DIR="results/post_val/${CLEAN}/generations"
[[ -d "$GEN_DIR" ]] || { echo "ERROR: generation dir missing: $GEN_DIR" >&2; exit 2; }

SENTINEL_DIR="results/.sentinels/post_eval/${CLEAN}"
mkdir -p "$SENTINEL_DIR"

echo "============================================================"
echo "[post_val_judge] row=$ROW_ID clean=$CLEAN"
echo "[post_val_judge] local=$LOCAL_DIR"
echo "[post_val_judge] generations=$GEN_DIR"
echo "[post_val_judge] judge=$JUDGE_BASE_URL workers=$JUDGE_WORKERS"
echo "============================================================"

if [[ ! -f "$SENTINEL_DIR/judge.done" ]]; then
  cleanslate judge \
    -i "results/post_val/${CLEAN}" \
    -o "results/post_val/${CLEAN}/${CLEAN}.json" \
    --model-name "$LOCAL_DIR" \
    --skip-done
  touch "$SENTINEL_DIR/judge.done"
else
  echo "[post_val_judge] judge.done exists; skipping for $CLEAN"
fi

echo "[post_val_judge] DONE row=$ROW_ID clean=$CLEAN"

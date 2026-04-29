#!/bin/bash
#SBATCH --job-name=post-eval-generate
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/final_final/post_eval_generate/slurm-%j-%x.out

set -euo pipefail

cd /data/animesh_jha/CleanSlate
mkdir -p logs/final_final/post_eval_generate

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

MANIFEST="${MANIFEST:-results/final_final/post_eval_manifest.json}"
[[ -f "$MANIFEST" ]] || {
  echo "ERROR: missing manifest $MANIFEST. Run prepare_post_eval_checkpoints.py first." >&2
  exit 2
}

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
if not row.get("verified"):
    raise SystemExit(f"ROW_ID={row_id} is not verified in {manifest_path}")

local_dir = Path(row["local_dir"])
local_manifest_path = local_dir / "manifest.json"
if not local_manifest_path.exists():
    raise SystemExit(f"Missing local manifest: {local_manifest_path}")
with local_manifest_path.open() as f:
    local = json.load(f)

for key in ("row_id", "checkpoint_repo", "checkpoint", "checkpoint_number", "repo_revision"):
    if local.get(key) != row.get(key):
        raise SystemExit(
            f"Local manifest mismatch for {key}: {local.get(key)!r} != {row.get(key)!r}"
        )

assignments = {
    "LOCAL_DIR": str(local_dir),
    "CLEAN": local_dir.name,
    "EVAL_CONFIG": row["eval_config"],
    "VALIDATE_GPU_MEM": str(row["validate_gpu_mem"]),
    "CHECKPOINT_REPO": row["checkpoint_repo"],
    "CHECKPOINT": row["checkpoint"],
}
for key, value in assignments.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

[[ -d "$LOCAL_DIR" ]] || {
  echo "ERROR: local checkpoint dir missing: $LOCAL_DIR" >&2
  exit 2
}
[[ -f "$EVAL_CONFIG" ]] || {
  echo "ERROR: eval config missing: $EVAL_CONFIG" >&2
  exit 2
}
if grep -q 'gpu_ids' "$EVAL_CONFIG"; then
  echo "ERROR: $EVAL_CONFIG contains gpu_ids; Slurm owns CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi

echo "============================================================"
echo "[post_eval_generate] row=$ROW_ID clean=$CLEAN"
echo "[post_eval_generate] checkpoint=$CHECKPOINT_REPO:$CHECKPOINT"
echo "[post_eval_generate] local=$LOCAL_DIR"
echo "[post_eval_generate] eval_config=$EVAL_CONFIG"
echo "[post_eval_generate] cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

SENTINEL_DIR="results/.sentinels/post_eval/${CLEAN}"
mkdir -p "$SENTINEL_DIR"

if [[ ! -f "$SENTINEL_DIR/eval.done" ]]; then
  cleanslate eval \
    -m "$LOCAL_DIR" \
    -o "results/post_eval/${CLEAN}" \
    --hf-org unlearning-cleanslate \
    --config "$EVAL_CONFIG"
  touch "$SENTINEL_DIR/eval.done"
else
  echo "[post_eval_generate] eval.done exists; skipping eval for $CLEAN"
fi

if [[ ! -f "$SENTINEL_DIR/generate.done" ]]; then
  cleanslate generate \
    -m "$LOCAL_DIR" \
    -B full \
    --skip-done \
    -o "results/post_val/${CLEAN}" \
    --gpu-mem "$VALIDATE_GPU_MEM" \
    --tp-size 1 \
    --dp-size 1 \
    --max-model-len 4096
  touch "$SENTINEL_DIR/generate.done"
else
  echo "[post_eval_generate] generate.done exists; skipping validation generation for $CLEAN"
fi

echo "[post_eval_generate] DONE row=$ROW_ID clean=$CLEAN"

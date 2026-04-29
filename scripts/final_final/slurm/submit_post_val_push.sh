#!/usr/bin/env bash
set -euo pipefail

cd /data/animesh_jha/CleanSlate

MANIFEST="${MANIFEST:-results/final_final/post_eval_manifest.json}"
ROW_IDS="${ROW_IDS:-}"
STAGGER_SECONDS="${STAGGER_SECONDS:-15}"
SLURM_TEMPLATE="scripts/final_final/slurm/post_val_push_one.slurm"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

[[ -f "$MANIFEST" ]] || { echo "ERROR: missing manifest $MANIFEST" >&2; exit 2; }
[[ -f "$SLURM_TEMPLATE" ]] || { echo "ERROR: missing template $SLURM_TEMPLATE" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python interpreter not executable: $PYTHON_BIN" >&2; exit 2; }

mkdir -p logs/final_final/post_val_push

first=1
while IFS=$'\t' read -r row_id clean; do
  if [[ -n "$ROW_IDS" ]]; then
    skip=1
    for allowed in $ROW_IDS; do
      if [[ "$allowed" == "$row_id" ]]; then skip=0; break; fi
    done
    if (( skip )); then
      echo "[skip]   row=$row_id (not in ROW_IDS)"
      continue
    fi
  fi
  if (( ! first )) && (( STAGGER_SECONDS > 0 )); then
    sleep "$STAGGER_SECONDS"
  fi
  first=0
  echo "[submit] row=$row_id clean=$clean"
  sbatch \
    --job-name="pvp-${clean:0:40}" \
    --export=ALL,ROW_ID="$row_id",MANIFEST="$MANIFEST" \
    "$SLURM_TEMPLATE"
done < <(
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open() as f:
    rows = sorted(json.load(f)["rows"], key=lambda row: int(row["row_id"]))
for row in rows:
    print(f"{row['row_id']}\t{Path(row['local_dir']).name}")
PY
)

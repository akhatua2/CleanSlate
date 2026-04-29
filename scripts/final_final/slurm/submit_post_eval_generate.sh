#!/usr/bin/env bash
set -euo pipefail

cd /data/animesh_jha/CleanSlate

MANIFEST="${MANIFEST:-results/final_final/post_eval_manifest.json}"
EXPECTED_ROWS="${EXPECTED_ROWS:-}"
PARTITION="${PARTITION:-cais}"
SLURM_TEMPLATE="scripts/final_final/slurm/post_eval_generate_one.slurm"
LOG_DIR="logs/final_final/post_eval_generate"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROW_IDS="${ROW_IDS:-}"
STAGGER_SECONDS="${STAGGER_SECONDS:-60}"

mkdir -p "$LOG_DIR"

[[ -f "$MANIFEST" ]] || { echo "ERROR: missing manifest $MANIFEST" >&2; exit 2; }
[[ -f "$SLURM_TEMPLATE" ]] || { echo "ERROR: missing Slurm template $SLURM_TEMPLATE" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python interpreter not executable: $PYTHON_BIN" >&2; exit 2; }

"$PYTHON_BIN" - "$MANIFEST" "${EXPECTED_ROWS:-}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected = sys.argv[2]
with manifest_path.open() as f:
    manifest = json.load(f)
rows = manifest.get("rows", [])
if expected and len(rows) != int(expected):
    raise SystemExit(f"Expected {expected} rows in {manifest_path}, found {len(rows)}")
bad = [
    row.get("row_id")
    for row in rows
    if not row.get("verified") or not Path(row["local_dir"], "manifest.json").exists()
]
if bad:
    raise SystemExit(f"Unprepared or unverified rows: {bad}")
PY

first=1
while IFS=$'\t' read -r row_id clean mem walltime; do
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
  job_name="peg-${clean:0:40}"
  echo "[submit] row=$row_id clean=$clean mem=$mem time=$walltime"
  sbatch \
    --partition="$PARTITION" \
    --mem="$mem" \
    --time="$walltime" \
    --job-name="$job_name" \
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
    print(
        "\t".join(
            [
                str(row["row_id"]),
                Path(row["local_dir"]).name,
                row["slurm_mem"],
                row["slurm_time"],
            ]
        )
    )
PY
)

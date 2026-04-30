#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-.venv/bin/python}
OUTPUT_ROOT=${1:-${OUTPUT_ROOT:-results/igm_post_release_inputs}}

if [[ ! -x "$PYTHON" ]]; then
  echo "error: python not executable: $PYTHON" >&2
  echo "set PYTHON=/path/to/python or create the repo .venv" >&2
  exit 1
fi

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import polars as pl
from datasets import load_dataset

output_root = Path(sys.argv[1])
sources = [
    ("post_llama31", "unlearning-cleanslate/cleanslate_dataset_post_llama31"),
    ("post_books_2025", "unlearning-cleanslate/cleanslate_dataset_post_books_2025"),
]
required = {
    "content_id",
    "content_title",
    "content_creators",
    "content_year",
    "reference_target",
}

for source_key, dataset_id in sources:
    ds = load_dataset(dataset_id, split="train")
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise SystemExit(
            f"{dataset_id} missing required columns: {missing}; "
            f"columns={ds.column_names}"
        )

    rows = []
    for row in ds:
        text = row["reference_target"] or ""
        rows.append({
            "source_key": source_key,
            "source_dataset": dataset_id,
            "source_split": "train",
            "content_id": row["content_id"],
            "content_title": row["content_title"],
            "content_creators": row["content_creators"],
            "content_year": row["content_year"],
            "reference_target": text,
            "text_length_chars": len(text),
            "text_length_words": len(text.split()),
        })

    out = output_root / source_key
    out.mkdir(parents=True, exist_ok=True)

    ids = [row["content_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{dataset_id} has duplicate content_id values")

    pl.DataFrame(rows).write_parquet(out / "train.parquet")
    (out / "ids.json").write_text(json.dumps(ids, indent=2) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "source_key": source_key,
        "source_dataset": dataset_id,
        "source_split": "train",
        "rows": len(rows),
        "total_chars": sum(row["text_length_chars"] for row in rows),
        "total_words": sum(row["text_length_words"] for row in rows),
        "schema": list(rows[0].keys()) if rows else [],
    }, indent=2) + "\n")

    print(
        f"{source_key}: rows={len(rows)} ids={out / 'ids.json'} "
        f"parquet={out / 'train.parquet'}",
        flush=True,
    )
PY

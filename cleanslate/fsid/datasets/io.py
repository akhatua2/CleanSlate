"""Local dataset IO helpers."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def load_local_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a local parquet/json/jsonl file."""
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"Expected a local file at {path}. "
            "Build or curate the artifact locally first."
        )

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            ds = load_dataset("parquet", data_files=str(path), split="train")
            return list(ds)
    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"{path} must contain a JSON list")
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    raise ValueError(f"Unsupported input format for {path}")

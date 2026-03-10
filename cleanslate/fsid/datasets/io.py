"""Local dataset IO helpers."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _looks_like_hf_id(path_str: str) -> bool:
    """Return True if *path_str* looks like a HuggingFace dataset ID (org/name)."""
    # Local paths have file extensions or start with . / ~
    if path_str.startswith((".", "/", "~")):
        return False
    # Strip an optional :config suffix before checking
    base = path_str.split(":")[0]
    parts = base.split("/")
    if len(parts) != 2:
        return False
    # If it matches a local file, prefer that
    if Path(base).exists():
        return False
    return True


def _load_hf_records(path_str: str) -> list[dict[str, Any]]:
    """Load records from a HuggingFace dataset ID.

    Accepts ``repo_id`` or ``repo_id:config_name``.
    """
    if ":" in path_str:
        repo_id, config_name = path_str.rsplit(":", 1)
    else:
        repo_id, config_name = path_str, None

    ds = load_dataset(repo_id, config_name, split="train")
    return list(ds)


def load_local_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a local file or HuggingFace dataset ID.

    Supported local formats: parquet, json, jsonl.
    HuggingFace IDs: ``org/dataset`` or ``org/dataset:config``.
    """
    path_str = str(path)

    # Try HuggingFace first if it looks like a repo ID
    if _looks_like_hf_id(path_str):
        return _load_hf_records(path_str)

    path = Path(path_str)
    if not path.exists():
        raise ValueError(
            f"Expected a local file at {path}. "
            "Build or curate the artifact locally first, "
            "or pass a HuggingFace dataset ID (org/dataset or org/dataset:config)."
        )

    # If path is a directory, look for results.parquet inside it
    if path.is_dir():
        candidates = sorted(path.glob("*/results.parquet"))
        if not candidates:
            candidates = sorted(path.glob("results.parquet"))
        if not candidates:
            raise ValueError(
                f"Directory {path} contains no results.parquet file"
            )
        path = candidates[0]

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

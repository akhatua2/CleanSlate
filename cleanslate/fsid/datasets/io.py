"""Local dataset IO helpers."""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from datasets import load_dataset  # type: ignore[import-untyped]


def _looks_like_hf_id(path_str: str) -> bool:
    """True if path_str looks like org/name[:config[:split]] rather than a local path."""
    if path_str.startswith((".", "/", "~")):
        return False
    base = path_str.split(":")[0]
    parts = base.split("/")
    if len(parts) != 2:
        return False
    return not Path(base).exists()


def _load_hf_records(path_str: str) -> list[dict[str, Any]]:
    """Load records from a HuggingFace dataset ID.

    Accepts ``repo_id``, ``repo_id:config``, or ``repo_id:config:split``.
    Single- and two-segment forms default ``split="train"``.
    """
    parts = path_str.split(":")
    match parts:
        case [repo_id]:
            config_name, split = None, "train"
        case [repo_id, config_name]:
            split = "train"
        case [repo_id, config_name, split]:
            pass
        case _:
            raise ValueError(
                f"Expected repo_id[:config[:split]], got {path_str!r}"
            )
    ds = load_dataset(repo_id, config_name, split=split)
    return list(ds)


def load_local_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a local file or HuggingFace dataset ID.

    Supported local formats: parquet, json, jsonl.
    HuggingFace IDs: ``org/dataset``, ``org/dataset:config``, or
    ``org/dataset:config:split``.
    """
    path_str = str(path)
    if _looks_like_hf_id(path_str):
        return _load_hf_records(path_str)

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"No local file at {p}. Build/curate it first, or pass a HF dataset ID."
        )

    if p.is_dir():
        candidates = sorted(p.glob("*/results.parquet")) or sorted(p.glob("results.parquet"))
        if not candidates:
            raise FileNotFoundError(f"{p} contains no results.parquet")
        p = candidates[0]

    match p.suffix.lower():
        case ".parquet":
            rows: list[dict[str, Any]] = pq.read_table(p).to_pylist()
            return rows
        case ".json":
            data = json.loads(p.read_text())
            if not isinstance(data, list):
                raise ValueError(f"{p} must contain a JSON list")
            return data
        case ".jsonl":
            return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        case suffix:
            raise ValueError(f"Unsupported input format {suffix!r} for {p}")

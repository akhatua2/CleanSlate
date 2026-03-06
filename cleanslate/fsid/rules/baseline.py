"""Baseline local rule: forget the requested song lyrics and retain nothing."""

import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset

from cleanslate.fsid.datasets.pool import stratified_select


def curate_baseline_sets(
    pool_records: list[dict[str, Any]],
    *,
    request_count: int | None = None,
    seed: int = 42,
    num_bins: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create baseline request/forget/retain sets from a local candidate pool."""
    if not pool_records:
        raise ValueError("Pool is empty.")
    if request_count is not None and request_count < 1:
        raise ValueError("request_count must be >= 1 when set.")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")

    if request_count is None or request_count >= len(pool_records):
        selected = [dict(record) for record in pool_records]
    else:
        selected = stratified_select(
            records=[dict(record) for record in pool_records],
            value_key="memorized_fraction",
            target_size=request_count,
            num_bins=num_bins,
            seed=seed,
        )

    selected = sorted(
        selected,
        key=lambda row: float(row.get("memorized_fraction", 0.0)),
        reverse=True,
    )

    rng = random.Random(seed)
    requests: list[dict[str, Any]] = []
    forget_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        song_id = str(row.get("song_id", f"song_{index}"))
        request_id = f"req_{index:04d}_{song_id[:8]}_{rng.randint(1000, 9999)}"

        request = {
            "request_id": request_id,
            "song_id": song_id,
            "song_title": row.get("song_title", ""),
            "song_artists": row.get("song_artists", []),
            "song_year": row.get("song_year"),
            "lyrics": row.get("lyrics", ""),
            "memorized_fraction": row.get("memorized_fraction"),
            "max_p_z": row.get("max_p_z"),
            "source_dataset": row.get("source_dataset"),
            "source_config": row.get("source_config"),
            "curation_rule": "baseline_literal",
        }
        requests.append(request)

        forget_rows.append(
            {
                "request_id": request_id,
                "song_id": request["song_id"],
                "song_title": request["song_title"],
                "song_artists": request["song_artists"],
                "song_year": request["song_year"],
                "lyrics": request["lyrics"],
                "rule_name": "baseline_literal",
            }
        )

    retain_rows: list[dict[str, Any]] = []
    return requests, forget_rows, retain_rows


def save_curated_sets(
    *,
    output_dir: str | Path,
    requests: list[dict[str, Any]],
    forget_rows: list[dict[str, Any]],
    retain_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Persist curated request/forget/retain artifacts locally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests_path = output_dir / "requests.parquet"
    forget_path = output_dir / "forget_set.parquet"
    retain_path = output_dir / "retain_set.parquet"
    metadata_path = output_dir / "metadata.json"

    Dataset.from_list(requests).to_parquet(str(requests_path))
    Dataset.from_list(forget_rows).to_parquet(str(forget_path))

    if retain_rows:
        Dataset.from_list(retain_rows).to_parquet(str(retain_path))
    else:
        Dataset.from_dict(
            {
                "request_id": [],
                "song_id": [],
                "song_title": [],
                "song_artists": [],
                "song_year": [],
                "lyrics": [],
                "rule_name": [],
            }
        ).to_parquet(str(retain_path))

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "requests": requests_path,
        "forget": forget_path,
        "retain": retain_path,
        "metadata": metadata_path,
    }

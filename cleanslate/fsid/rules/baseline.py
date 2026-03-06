"""Baseline local rule: use direct/no-seed lyric prompts and retain nothing."""

import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset

from cleanslate.prompts import DIRECT

RULE_NAME = "baseline"
ARTIFACT_RULE_NAME = "baseline_literal"


def _build_direct_none_prompt(
    row: dict[str, Any],
) -> tuple[str, str, list[dict[str, str]]]:
    song = {
        "title": row.get("song_title", ""),
        "artists": row.get("song_artists", []),
        "year": row.get("song_year"),
        "lyrics": row.get("lyrics", ""),
    }
    messages = DIRECT.format_messages(song, "")
    user_prompt = next(
        (
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        ),
        "",
    )
    return DIRECT.system_prompt or "", user_prompt, messages


def curate_baseline_sets(
    unlearn_song_records: list[dict[str, Any]],
    *,
    base_dataset: str | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create baseline request/forget/retain sets from preselected unlearn songs."""
    if not unlearn_song_records:
        raise ValueError("Unlearn song list is empty.")

    selected = sorted(
        [dict(record) for record in unlearn_song_records],
        key=lambda row: float(row.get("memorized_fraction", 0.0)),
        reverse=True,
    )

    rng = random.Random(seed)
    requests: list[dict[str, Any]] = []
    forget_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        song_id = str(row.get("song_id", f"song_{index}"))
        request_id = f"req_{index:04d}_{song_id[:8]}_{rng.randint(1000, 9999)}"
        system_prompt, user_prompt, messages = _build_direct_none_prompt(row)
        lyrics = row.get("lyrics", "")

        request = {
            "request_id": request_id,
            "song_id": song_id,
            "song_title": row.get("song_title", ""),
            "song_artists": row.get("song_artists", []),
            "song_year": row.get("song_year"),
            "lyrics": lyrics,
            "target": lyrics,
            "memorized_fraction": row.get("memorized_fraction"),
            "max_p_z": row.get("max_p_z"),
            "source_dataset": row.get("source_dataset"),
            "source_config": row.get("source_config"),
            "prompt_template": DIRECT.name,
            "seed_mode": "none",
            "lyrics_hint": "",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "messages": messages,
            "curation_rule": ARTIFACT_RULE_NAME,
        }
        requests.append(request)

        forget_rows.append(
            {
                "request_id": request_id,
                "song_id": request["song_id"],
                "song_title": request["song_title"],
                "song_artists": request["song_artists"],
                "song_year": request["song_year"],
                "lyrics": lyrics,
                "target": lyrics,
                "prompt_template": DIRECT.name,
                "seed_mode": "none",
                "lyrics_hint": "",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "messages": messages,
                "rule_name": ARTIFACT_RULE_NAME,
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
                "target": [],
                "prompt_template": [],
                "seed_mode": [],
                "lyrics_hint": [],
                "system_prompt": [],
                "user_prompt": [],
                "messages": [],
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

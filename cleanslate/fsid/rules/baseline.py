"""Baseline local rule: use direct/no-seed lyric prompts, retain from WikiText."""

import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from cleanslate.prompts import DIRECT

logger = logging.getLogger(__name__)

RULE_NAME = "baseline"
ARTIFACT_RULE_NAME = "baseline_literal"


def _build_direct_none_prompt(
    row: dict[str, Any],
) -> tuple[str, str, list[dict[str, str]]]:
    song = {
        "title": row.get("content_title", ""),
        "artists": row.get("content_creators", []),
        "year": row.get("content_year"),
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
    retain_dataset: str = "wikitext",
    retain_config: str = "wikitext-103-raw-v1",
    retain_split: str = "validation",
    retain_samples: int | None = None,
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
        content_id = str(row.get("content_id", f"content_{index}"))
        request_id = f"req_{index:04d}_{content_id[:8]}_{rng.randint(1000, 9999)}"
        system_prompt, user_prompt, messages = _build_direct_none_prompt(row)
        lyrics = row.get("lyrics", "")

        request = {
            "request_id": request_id,
            "content_id": content_id,
            "content_title": row.get("content_title", ""),
            "content_creators": row.get("content_creators", []),
            "content_year": row.get("content_year"),
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
                "content_id": request["content_id"],
                "content_title": request["content_title"],
                "content_creators": request["content_creators"],
                "content_year": request["content_year"],
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

    # Build retain set from WikiText (general-domain text to preserve model quality)
    rng = random.Random(seed)  # re-seed for retain sampling reproducibility
    logger.info(
        "Loading retain data from %s/%s split=%s",
        retain_dataset,
        retain_config,
        retain_split,
    )
    wiki = load_dataset(retain_dataset, retain_config, split=retain_split)
    # Filter empty/short lines (WikiText has many blank lines and headings)
    wiki_texts = [
        row["text"]
        for row in wiki
        if isinstance(row.get("text"), str) and len(row["text"].strip()) > 100
    ]
    if retain_samples is not None:
        wiki_texts = rng.sample(wiki_texts, min(retain_samples, len(wiki_texts)))
    retain_rows: list[dict[str, Any]] = [
        {"text": text, "rule_name": ARTIFACT_RULE_NAME} for text in wiki_texts
    ]
    logger.info("Built %d retain rows from %s", len(retain_rows), retain_dataset)

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
                "content_id": [],
                "content_title": [],
                "content_creators": [],
                "content_year": [],
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

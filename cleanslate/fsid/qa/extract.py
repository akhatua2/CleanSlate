"""Stage 1 — extract interior facts (propositions) from content items.

Reads the cleanslate_dataset. Items are either songs or books; routing is
length-based (BOOK_LENGTH_THRESHOLD). Books are chunked into BOOK_CHUNK_CHARS
windows and each chunk is extracted independently.
"""

from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import litellm  # type: ignore[import-untyped]
from datasets import load_dataset  # type: ignore[import-untyped]

from cleanslate.fsid.qa import QA_MODEL
from cleanslate.fsid.qa.base import Proposition
from cleanslate.fsid.qa.io import save_rows

DATASET_ID = "unlearning-cleanslate/cleanslate_dataset"
OUTPUT_PATH = Path("data/fsid/qa/propositions.json")
MAX_WORKERS = 20

BOOK_LENGTH_THRESHOLD = 20_000
BOOK_CHUNK_CHARS = 60_000

SONG_EXTRACTION_PROMPT = """You are extracting interior facts from a song's lyrics that will \
be used to build a retain-metric QA benchmark.

Title: "{title}"
Creator: {creator}

LYRICS:
{text}

TASK: Extract between 3 and 8 **interior factual propositions** from these lyrics.

A good proposition:
- Refers to a named entity, specific detail, number, place, person, action, or relationship \
mentioned INSIDE the lyrics.
- Is a full factual statement, not just a word.
- Cites a short source span (1-2 lines from the lyrics) where the fact appears.

A bad proposition:
- Is about the title, creator/artist, year, genre (priors).
- Restates the hook, chorus, or title in different words (e.g. for "Ladies' Night": \
"the song says it is your night" — that's just the hook).
- Has no named entity, place, number, or specific concrete detail.
- Is a generic feeling, mood, theme, or exhortation ("the song tells listeners to dance").
- Uses pronouns/possessives as its key content ("your", "their", "this").

If the song is abstract with few concrete details, return fewer propositions. Do NOT pad.

Return ONLY a JSON object:
{{"propositions": [{{"fact": "...", "source_span": "..."}}, ...]}}
"""

BOOK_EXTRACTION_PROMPT = """You are extracting interior facts from a CHUNK of a book that will \
be used to build a retain-metric QA benchmark.

Book: "{title}"
Author: {creator}
Chunk {chunk_idx} of {total_chunks}

TEXT:
{text}

TASK: Extract between 5 and 15 **interior factual propositions** from this chunk.

Prioritize:
- Named characters and their distinguishing features, relationships, actions.
- Specific locations, settings, objects mentioned.
- Concrete plot events in this chunk.
- Numeric specifics (ages, dates within the narrative, counts).

Avoid:
- Anything about the title, author, publication year, genre (priors).
- Generic theme/mood statements without a concrete anchor.
- Verbatim famous quotes unless they reveal a concrete fact.

Each proposition must cite a short source span (1-3 lines) from the chunk.

Return ONLY a JSON object:
{{"propositions": [{{"fact": "...", "source_span": "..."}}, ...]}}
"""


def _parse_response(text: str) -> list[dict[str, str]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    data = json.loads(text)
    items = data.get("propositions", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        fact = str(item.get("fact", "")).strip()
        span = str(item.get("source_span", "")).strip()
        if fact and span:
            out.append({"fact": fact, "source_span": span})
    return out


def _chunk_text(text: str, chunk_chars: int) -> list[str]:
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _extract_chunk(prompt: str, max_tokens: int) -> list[dict[str, str]]:
    response = litellm.completion(
        model=QA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return _parse_response(raw)


def extract_for_record(record: dict[str, Any]) -> list[Proposition]:
    """Extract propositions for one dataset row. Routes books → chunked extraction."""
    content_id = str(record["content_id"])
    title = str(record["content_title"])
    creator = str(record["content_creators"])
    text = str(record["reference_target"])
    if not text.strip():
        return []

    is_book = len(text) >= BOOK_LENGTH_THRESHOLD
    parsed: list[dict[str, str]] = []

    try:
        if is_book:
            chunks = _chunk_text(text, BOOK_CHUNK_CHARS)
            for idx, chunk in enumerate(chunks, 1):
                prompt = BOOK_EXTRACTION_PROMPT.format(
                    title=title,
                    creator=creator,
                    chunk_idx=idx,
                    total_chunks=len(chunks),
                    text=chunk,
                )
                parsed.extend(_extract_chunk(prompt, max_tokens=6144))
        else:
            prompt = SONG_EXTRACTION_PROMPT.format(title=title, creator=creator, text=text)
            parsed = _extract_chunk(prompt, max_tokens=4096)
    except Exception:
        print(f"extract failed for {title!r}:")
        traceback.print_exc()
        return []

    return [
        Proposition(
            content_id=content_id,
            content_title=title,
            content_creators=creator,
            fact=p["fact"],
            source_span=p["source_span"],
        )
        for p in parsed
    ]


def run(
    dataset_id: str = DATASET_ID,
    output_path: str | Path = OUTPUT_PATH,
    limit: int | None = None,
    max_workers: int = MAX_WORKERS,
) -> Path:
    print(f"Loading {dataset_id}...")
    ds = load_dataset(dataset_id, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    records = list(ds)
    n_books = sum(1 for r in records if len(r["reference_target"]) >= BOOK_LENGTH_THRESHOLD)
    print(f"Items: {len(records)} ({n_books} books, {len(records) - n_books} songs)")

    all_props: list[Proposition] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_for_record, r): r for r in records}
        for done, fut in enumerate(as_completed(futures), 1):
            all_props.extend(fut.result())
            if done % 25 == 0 or done == len(records):
                print(f"  extracted {done}/{len(records)} items, {len(all_props)} propositions")

    out = save_rows(all_props, output_path)
    print(f"Saved {len(all_props)} propositions → {out}")
    return out


if __name__ == "__main__":
    run()

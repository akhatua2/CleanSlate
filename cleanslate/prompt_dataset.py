"""Build prompt-split datasets for lyrics generation."""

from __future__ import annotations

import hashlib
from typing import Iterable

from datasets import Dataset, DatasetDict

from cleanslate.prompts import PromptTemplate, SeedMode, ALL_TEMPLATES
from cleanslate.samples.billboard import get_lyrics_prefix


def _artists_str(artists: list[str] | None) -> str:
    if not artists:
        return "Unknown Artist"
    return ", ".join(artists)


def _stable_song_id(title: str, artists: list[str], year: int | None) -> str:
    key = f"{title}|{','.join(artists)}|{year or ''}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _resolve_templates(templates: Iterable[str | PromptTemplate]) -> list[PromptTemplate]:
    out: list[PromptTemplate] = []
    for t in templates:
        if isinstance(t, PromptTemplate):
            out.append(t)
        else:
            out.append(ALL_TEMPLATES[t])
    return out


def build_prompt_splits(
    dataset: Dataset,
    templates: Iterable[str | PromptTemplate],
    seed_modes: Iterable[SeedMode],
) -> DatasetDict:
    """
    Build a DatasetDict with one split per prompt template.

    Each row contains prompt messages + song metadata + full lyrics target.
    """
    template_objs = _resolve_templates(templates)
    seed_modes_list = list(seed_modes)

    splits: dict[str, Dataset] = {}

    for template in template_objs:
        for seed_mode in seed_modes_list:
            rows: list[dict] = []
            for row in dataset:
                title = row.get("title", "")
                artists = row.get("artists") or []
                year = row.get("year")
                lyrics = row.get("lyrics") or ""

                song_id = _stable_song_id(title, artists, year)
                name = f"{title} - {_artists_str(artists)} ({year})"

                lyrics_hint = get_lyrics_prefix(lyrics, seed_mode)
                messages = template.format_messages(row, lyrics_hint)

                user_prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                        break

                prompt_id = f"{song_id}:{template.name}:{seed_mode}"

                rows.append(
                    {
                        "name": name,
                        "song_id": song_id,
                        "song_title": title,
                        "song_artists": artists,
                        "song_year": year,
                        "wikipedia_link": row.get("wikipedia_link"),
                        "prompt_id": prompt_id,
                        "prompt_template": template.name,
                        "seed_mode": seed_mode,
                        "lyrics_hint": lyrics_hint,
                        "system_prompt": template.system_prompt or "",
                        "user_prompt": user_prompt,
                        "messages": messages,
                        "target": lyrics,
                    }
                )

            split_name = f"{template.name}__{seed_mode}"
            splits[split_name] = Dataset.from_list(rows)

    return DatasetDict(splits)

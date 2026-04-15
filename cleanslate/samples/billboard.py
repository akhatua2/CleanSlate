
from typing import Literal

from datasets import load_dataset, Dataset

DATASET_ID = "unlearning-cleanslate/forget_set_identification"
DEFAULT_DATASET_SPLIT = "1970_2025_en_songs"


def load_billboard_dataset(
    dataset_id: str = DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    year_range: tuple[int, int] | None = None,
    limit: int | None = None,
    require_lyrics: bool = True,
    shuffle: bool = False,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset(dataset_id, split=dataset_split)

    if year_range:
        start_year, end_year = year_range
        ds = ds.filter(lambda x: start_year <= x["year"] <= end_year)

    if require_lyrics:
        ds = ds.filter(lambda x: x["lyrics"] is not None and len(x["lyrics"]) > 0)

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    return ds


def load_billboard_songs(
    year_range: tuple[int, int] | None = None,
    limit: int | None = None,
    require_lyrics: bool = True,
    shuffle: bool = False,
    seed: int = 42,
    dataset_id: str = DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
) -> list[dict]:
    ds = load_billboard_dataset(
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        year_range=year_range,
        limit=limit,
        require_lyrics=require_lyrics,
        shuffle=shuffle,
        seed=seed,
    )

    samples = []
    for row in ds:
        artists = row["artists"] if row["artists"] else []
        artists_str = ", ".join(artists) if artists else "Unknown"

        samples.append(
            {
                "name": f"{row['title']} - {artists_str} ({row['year']})",
                "title": row["title"],
                "artists": artists,
                "year": row["year"],
                "lyrics": row["lyrics"] or "",
                "wikipedia_link": row.get("wikipedia_link"),
            }
        )
    return samples


def get_lyrics_prefix(
    lyrics: str, mode: Literal["none", "first_line", "first_verse", "half"] = "none"
) -> str:
    if not lyrics or mode == "none":
        return ""

    lines = lyrics.strip().split("\n")

    if mode == "first_line":
        return lines[0] if lines else ""

    elif mode == "first_verse":
        verse_lines = []
        for line in lines[:8]:
            if not line.strip():
                break
            verse_lines.append(line)
            if len(verse_lines) >= 4:
                break
        return "\n".join(verse_lines)

    elif mode == "half":
        half_point = len(lyrics) // 2
        newline_pos = lyrics.find("\n", half_point)
        if newline_pos != -1 and newline_pos < half_point + 100:
            return lyrics[:newline_pos]
        return lyrics[:half_point]

    return ""


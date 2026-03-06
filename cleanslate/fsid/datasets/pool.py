"""Candidate pool construction for forget-set identification experiments."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class PoolStats:
    """Summary statistics for a pool-construction run."""

    total_rows_seen: int
    eligible_rows: int
    selected_rows: int
    threshold: float
    min_fraction: float
    max_fraction: float
    mean_fraction: float


def _to_artists(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _compute_mem_fraction(row: dict[str, Any], fraction_col: str) -> float:
    value = row.get(fraction_col)
    if value is not None:
        return float(value)

    mem_windows = row.get("memorized_windows")
    total_windows = row.get("num_windows")
    if mem_windows is None or total_windows is None:
        raise KeyError(
            f"Row missing '{fraction_col}' and fallback columns "
            f"'memorized_windows'/'num_windows'. "
            f"Available keys: {sorted(row.keys())}"
        )
    if total_windows == 0:
        raise ValueError(
            f"num_windows=0 for song_id={row.get('song_id')!r}. "
            "Cannot compute memorization fraction."
        )
    return float(mem_windows) / float(total_windows)


def _assign_quantile_bins(values: list[float], num_bins: int) -> list[int]:
    if not values:
        return []
    if num_bins <= 1:
        return [0] * len(values)

    order = sorted(range(len(values)), key=lambda index: values[index])
    bins = [0] * len(values)
    n_values = len(values)
    for rank, index in enumerate(order):
        bins[index] = min(num_bins - 1, (rank * num_bins) // n_values)
    return bins


def _allocate_quota(group_sizes: list[int], target_size: int) -> list[int]:
    total = sum(group_sizes)
    if total == 0 or target_size <= 0:
        return [0] * len(group_sizes)

    non_empty_groups = sum(1 for size in group_sizes if size > 0)
    target_size = min(target_size, total)

    raw = [(target_size * size / total) if size > 0 else 0.0 for size in group_sizes]
    quota = [int(value) for value in raw]

    if target_size >= non_empty_groups:
        for index, size in enumerate(group_sizes):
            if size > 0 and quota[index] == 0:
                quota[index] = 1

    assigned = sum(quota)
    if assigned > target_size:
        candidates = sorted(
            range(len(quota)),
            key=lambda index: (quota[index], raw[index] - int(raw[index])),
            reverse=True,
        )
        for index in candidates:
            while assigned > target_size and quota[index] > 0 and quota[index] > 1:
                quota[index] -= 1
                assigned -= 1
            if assigned == target_size:
                break

    if assigned < target_size:
        remaining = target_size - assigned
        candidates = sorted(
            range(len(quota)),
            key=lambda index: (
                raw[index] - int(raw[index]),
                group_sizes[index] - quota[index],
            ),
            reverse=True,
        )
        for index in candidates:
            if remaining == 0:
                break
            capacity = group_sizes[index] - quota[index]
            if capacity <= 0:
                continue
            add = min(capacity, remaining)
            quota[index] += add
            remaining -= add

    return quota


def stratified_select(
    records: list[dict[str, Any]],
    value_key: str,
    target_size: int,
    num_bins: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select records by quantile-stratified sampling."""
    if not records:
        return []
    if target_size >= len(records):
        output = [dict(record) for record in records]
        for row in output:
            row["pool_bin"] = 0
        return output

    values: list[float] = []
    for row in records:
        value = row.get(value_key)
        if value is None:
            raise KeyError(
                f"Record missing required key '{value_key}'. "
                f"Available keys: {sorted(row.keys())}"
            )
        values.append(float(value))
    bins = _assign_quantile_bins(values, num_bins)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row, bin_id in zip(records, bins):
        row_copy = dict(row)
        row_copy["pool_bin"] = bin_id
        grouped.setdefault(bin_id, []).append(row_copy)

    ordered_bins = sorted(grouped.keys())
    group_sizes = [len(grouped[bin_id]) for bin_id in ordered_bins]
    quotas = _allocate_quota(group_sizes, target_size)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bin_id, quota in zip(ordered_bins, quotas):
        candidates = grouped[bin_id]
        if quota >= len(candidates):
            selected.extend(candidates)
            continue
        selected.extend(rng.sample(candidates, quota))

    if len(selected) > target_size:
        selected = rng.sample(selected, target_size)
    elif len(selected) < target_size:
        selected_keys = {
            (
                str(record.get("song_id", "")),
                str(record.get("song_title", "")),
                float(record.get(value_key, 0.0)),
            )
            for record in selected
        }
        remaining: list[dict[str, Any]] = []
        for bin_id in ordered_bins:
            for row in grouped[bin_id]:
                row_key = (
                    str(row.get("song_id", "")),
                    str(row.get("song_title", "")),
                    float(row.get(value_key, 0.0)),
                )
                if row_key not in selected_keys:
                    remaining.append(row)
        needed = min(target_size - len(selected), len(remaining))
        selected.extend(rng.sample(remaining, needed))

    return selected


def _load_lyrics_lookup(
    lyrics_dataset_id: str,
    lyrics_dataset_config: str,
    lyrics_split: str,
    lyrics_text_col: str,
) -> dict[str, str]:
    ds = load_dataset(lyrics_dataset_id, lyrics_dataset_config, split=lyrics_split)
    lookup: dict[str, str] = {}
    for row in ds:
        song_id = str(row.get("song_id", "") or "")
        if not song_id:
            continue
        text = row.get(lyrics_text_col, "") or ""
        if isinstance(text, str) and text.strip():
            lookup[song_id] = text
    return lookup


def _deduplicate_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}

    def _identity_key(row: dict[str, Any]) -> tuple[str, ...]:
        song_id = str(row.get("song_id", "") or "").strip()
        if song_id:
            return ("song_id", song_id)

        title = str(row.get("song_title", "") or "").strip().lower()
        artists = tuple(
            str(artist).strip().lower() for artist in _to_artists(row.get("song_artists"))
        )
        year = str(row.get("song_year", "") or "").strip()
        if title or artists or year:
            return ("meta", title, "|".join(artists), year)

        lyrics = str(row.get("lyrics", "") or "")
        return ("lyrics_prefix", lyrics[:128])

    for row in records:
        key = _identity_key(row)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = row
            continue
        if float(row.get("memorized_fraction", 0.0)) > float(
            previous.get("memorized_fraction", 0.0)
        ):
            by_key[key] = row
    return list(by_key.values())


def build_candidate_pool(
    eval_dataset: str,
    eval_config: str,
    *,
    split: str = "train",
    text_col: str = "reference_target",
    fraction_col: str = "memorized_fraction",
    threshold: float = 0.30,
    target_size: int = 100,
    min_size: int = 50,
    max_size: int = 200,
    num_bins: int = 5,
    seed: int = 42,
    limit: int | None = None,
    lyrics_dataset_id: str | None = None,
    lyrics_dataset_config: str = "default",
    lyrics_split: str = "train",
    lyrics_text_col: str = "reference_target",
    deduplicate: bool = True,
) -> tuple[list[dict[str, Any]], PoolStats]:
    """Build a stratified song pool from an eval dataset."""
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if target_size < min_size:
        target_size = min_size
    if target_size > max_size:
        target_size = max_size

    lyrics_lookup: dict[str, str] = {}
    if lyrics_dataset_id is not None:
        lyrics_lookup = _load_lyrics_lookup(
            lyrics_dataset_id,
            lyrics_dataset_config,
            lyrics_split,
            lyrics_text_col,
        )

    ds = load_dataset(eval_dataset, eval_config, split=split)

    eligible: list[dict[str, Any]] = []
    total_rows_seen = 0
    for index, row in enumerate(ds):
        if limit is not None and index >= limit:
            break
        total_rows_seen += 1

        lyrics = row.get(text_col, "") or ""
        if not isinstance(lyrics, str) or not lyrics.strip():
            song_id = str(row.get("song_id", "") or "")
            lyrics = lyrics_lookup.get(song_id, "")
        if not isinstance(lyrics, str) or len(lyrics.strip()) < 20:
            continue

        mem_fraction = _compute_mem_fraction(row, fraction_col)
        if mem_fraction < threshold:
            continue

        eligible.append(
            {
                "song_id": row.get("song_id", ""),
                "song_title": row.get("song_title", ""),
                "song_artists": _to_artists(row.get("song_artists")),
                "song_year": row.get("song_year"),
                "lyrics": lyrics,
                "memorized_fraction": mem_fraction,
                "max_p_z": row.get("max_p_z"),
                "num_windows": row.get("num_windows"),
                "memorized_windows": row.get("memorized_windows"),
                "source_dataset": eval_dataset,
                "source_config": eval_config,
            }
        )

    if deduplicate:
        eligible = _deduplicate_candidates(eligible)

    if not eligible:
        raise ValueError(
            "No eligible rows found. Lower --threshold or verify eval dataset columns."
        )

    desired = min(target_size, len(eligible), max_size)
    if len(eligible) < min_size:
        desired = len(eligible)

    selected = stratified_select(
        records=eligible,
        value_key="memorized_fraction",
        target_size=desired,
        num_bins=num_bins,
        seed=seed,
    )
    selected = sorted(
        selected,
        key=lambda row: float(row["memorized_fraction"]),
        reverse=True,
    )

    fractions = [float(row["memorized_fraction"]) for row in selected]
    stats = PoolStats(
        total_rows_seen=total_rows_seen,
        eligible_rows=len(eligible),
        selected_rows=len(selected),
        threshold=threshold,
        min_fraction=min(fractions),
        max_fraction=max(fractions),
        mean_fraction=sum(fractions) / len(fractions),
    )
    return selected, stats


def save_pool_dataset(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save pool records to local parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    return output_path

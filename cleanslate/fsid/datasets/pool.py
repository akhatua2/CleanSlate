"""Candidate pool construction for forget-set identification experiments."""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
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


def _to_artists(value: list | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


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
    records: list[dict],
    value_key: str,
    target_size: int,
    num_bins: int,
    seed: int,
) -> list[dict]:
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

    grouped: dict[int, list[dict]] = {}
    for row, bin_id in zip(records, bins):
        row_copy = dict(row)
        row_copy["pool_bin"] = bin_id
        grouped.setdefault(bin_id, []).append(row_copy)

    ordered_bins = sorted(grouped.keys())
    group_sizes = [len(grouped[bin_id]) for bin_id in ordered_bins]
    quotas = _allocate_quota(group_sizes, target_size)

    rng = random.Random(seed)
    selected = []
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
        remaining = []
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


def load_lyrics_dataset():
    return load_dataset(
        "unlearning-cleanslate/songs_5000", "default", split="train"
    ).to_polars(batched=False)


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


def _deduplicate_candidates(records: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, ...], dict] = {}

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
    *,
    fraction_col: str = "memorized_fraction",
    threshold: float = 0.30,
    target_size: int = 100,
    min_size: int = 50,
    max_size: int = 200,
    num_bins: int = 5,
    seed: int = 42,
    deduplicate: bool = True,
) -> tuple[list[dict], PoolStats]:
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

    lyrics_lookup = load_lyrics_dataset()
    ds = load_dataset(eval_dataset, "default", split="train")

    eligible: list[dict] = []
    total_rows_seen = 0
    for _, row in enumerate(ds):
        total_rows_seen += 1

        lyrics = row.get("reference_target", "") or ""
        if lyrics == "":
            target_song_id = row["song_id"]
            lyrics = lyrics_lookup.filter(pl.col("song_id")==target_song_id).select("reference_target")[0].item()

        mem_fraction = row.get("memorized_fraction", 0)
        if mem_fraction < threshold:
            continue

        eligible.append(
            {
                "song_id": row.get("song_id") or row.get("content_id") or "",
                "song_title": row.get("song_title") or row.get("content_title") or "",
                "song_artists": _to_artists(row.get("song_artists")),
                "song_year": row.get("song_year"),
                "lyrics": lyrics,
                "memorized_fraction": mem_fraction,
                "max_p_z": row.get("max_p_z"),
                "num_windows": row.get("num_windows"),
                "memorized_windows": row.get("memorized_windows"),
                "source_dataset": eval_dataset,
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


def split_pool_records(
    records: list[dict[str, Any]],
    *,
    forget_fraction: float | None = None,
    num_bins: int = 5,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a total pool into selected forget songs and the remaining retain songs."""
    if not records:
        raise ValueError("Pool is empty.")
    if forget_fraction is not None and not 0.0 < forget_fraction <= 1.0:
        raise ValueError("forget_fraction must be in (0, 1] when set.")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")

    copied = [dict(record) for record in records]
    if forget_fraction is None:
        forget_target_size = len(copied)
    else:
        forget_target_size = max(1, math.ceil(len(copied) * forget_fraction))

    if forget_target_size >= len(copied):
        forget_rows = copied
    else:
        forget_rows = stratified_select(
            records=copied,
            value_key="memorized_fraction",
            target_size=forget_target_size,
            num_bins=num_bins,
            seed=seed,
        )

    forget_rows = sorted(
        forget_rows,
        key=lambda row: float(row.get("memorized_fraction", 0.0)),
        reverse=True,
    )
    forget_keys = {_identity_key(row) for row in forget_rows}
    retain_rows = [
        dict(row) for row in copied if _identity_key(row) not in forget_keys
    ]
    retain_rows = sorted(
        retain_rows,
        key=lambda row: float(row.get("memorized_fraction", 0.0)),
        reverse=True,
    )
    return forget_rows, retain_rows


def _empty_dataset(columns: list[str], output_path: Path) -> None:
    Dataset.from_dict({column: [] for column in columns}).to_parquet(str(output_path))


def save_pool_dataset(records: list[dict], output_path: str | Path) -> Path:
    """Save pool records to local parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    return output_path


def _resolve_split_output_paths(output_root: str | Path) -> dict[str, Path]:
    path = Path(output_root)
    if path.suffix.lower() == ".parquet":
        stem = path.stem
        return {
            "pool": path,
            "forget": path.with_name(f"{stem}_forget.parquet"),
            "retain": path.with_name(f"{stem}_retain.parquet"),
        }
    return {
        "pool": path / "pool.parquet",
        "forget": path / "forget.parquet",
        "retain": path / "retain.parquet",
    }


def save_pool_datasets(
    *,
    pool_records: list[dict[str, Any]],
    forget_records: list[dict[str, Any]],
    retain_records: list[dict[str, Any]],
    output_root: str | Path,
) -> dict[str, Path]:
    """Save the total pool and its forget/retain song splits to local parquet."""
    if not pool_records:
        raise ValueError("pool_records must be non-empty.")

    paths = _resolve_split_output_paths(output_root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    save_pool_dataset(pool_records, paths["pool"])
    save_pool_dataset(forget_records, paths["forget"])
    if retain_records:
        save_pool_dataset(retain_records, paths["retain"])
    else:
        _empty_dataset(list(pool_records[0].keys()), paths["retain"])

    return paths

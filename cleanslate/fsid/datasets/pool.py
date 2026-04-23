"""Candidate pool construction for forget-set identification."""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset  # type: ignore[import-untyped]


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


def _to_artists(value: list[Any] | str | None) -> list[str]:
    match value:
        case None:
            return []
        case list():
            return [str(item) for item in value]
        case str():
            return [value]
        case _:
            return [str(value)]


def _assign_quantile_bins(values: list[float], num_bins: int) -> list[int]:
    if not values:
        return []
    if num_bins <= 1:
        return [0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    bins = [0] * len(values)
    n = len(values)
    for rank, index in enumerate(order):
        bins[index] = min(num_bins - 1, (rank * num_bins) // n)
    return bins


def _allocate_quota(group_sizes: list[int], target_size: int) -> list[int]:
    total = sum(group_sizes)
    if total == 0 or target_size <= 0:
        return [0] * len(group_sizes)

    non_empty = sum(1 for s in group_sizes if s > 0)
    target_size = min(target_size, total)
    raw = [(target_size * s / total) if s > 0 else 0.0 for s in group_sizes]
    quota = [int(v) for v in raw]

    if target_size >= non_empty:
        for i, s in enumerate(group_sizes):
            if s > 0 and quota[i] == 0:
                quota[i] = 1

    assigned = sum(quota)
    if assigned > target_size:
        order = sorted(
            range(len(quota)),
            key=lambda i: (quota[i], raw[i] - int(raw[i])),
            reverse=True,
        )
        for i in order:
            while assigned > target_size and quota[i] > 1:
                quota[i] -= 1
                assigned -= 1
            if assigned == target_size:
                break

    if assigned < target_size:
        remaining = target_size - assigned
        order = sorted(
            range(len(quota)),
            key=lambda i: (raw[i] - int(raw[i]), group_sizes[i] - quota[i]),
            reverse=True,
        )
        for i in order:
            if remaining == 0:
                break
            capacity = group_sizes[i] - quota[i]
            if capacity <= 0:
                continue
            add = min(capacity, remaining)
            quota[i] += add
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
        out = [dict(r) for r in records]
        for r in out:
            r["pool_bin"] = 0
        return out

    values = [float(r[value_key]) for r in records]
    bins = _assign_quantile_bins(values, num_bins)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row, bin_id in zip(records, bins, strict=True):
        copy = dict(row)
        copy["pool_bin"] = bin_id
        grouped.setdefault(bin_id, []).append(copy)

    ordered_bins = sorted(grouped)
    group_sizes = [len(grouped[b]) for b in ordered_bins]
    quotas = _allocate_quota(group_sizes, target_size)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bin_id, quota in zip(ordered_bins, quotas, strict=True):
        candidates = grouped[bin_id]
        selected.extend(candidates if quota >= len(candidates) else rng.sample(candidates, quota))

    if len(selected) > target_size:
        return rng.sample(selected, target_size)
    if len(selected) < target_size:
        chosen_keys = {_identity_key(r) for r in selected}
        remaining = [r for b in ordered_bins for r in grouped[b] if _identity_key(r) not in chosen_keys]
        needed = min(target_size - len(selected), len(remaining))
        selected.extend(rng.sample(remaining, needed))
    return selected


def _identity_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Stable identity for deduplication. Prefer content_id; fall back to metadata+lyrics prefix."""
    content_id = str(row.get("content_id", "") or "").strip()
    if content_id:
        return ("content_id", content_id)

    title = str(row.get("content_title", "") or "").strip().lower()
    artists = tuple(str(a).strip().lower() for a in _to_artists(row.get("content_creators")))
    year = str(row.get("content_year", "") or "").strip()
    if title or artists or year:
        return ("meta", title, "|".join(artists), year)

    lyrics = str(row.get("lyrics", "") or "")
    return ("lyrics_prefix", lyrics[:128])


def _deduplicate_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in records:
        key = _identity_key(row)
        previous = by_key.get(key)
        if previous is None or float(row.get("memorized_fraction", 0.0)) > float(
            previous.get("memorized_fraction", 0.0)
        ):
            by_key[key] = row
    return list(by_key.values())


def _backfill_lyrics_from_source(
    records: list[dict[str, Any]],
    *,
    source_dataset: str | None,
    source_text_col: str,
    source_config: str,
) -> None:
    """In-place: fill empty `lyrics` on eligible rows by joining against an HF source dataset."""
    needs_backfill = [r for r in records if not r["lyrics"]]
    if not needs_backfill:
        return
    if source_dataset is None:
        raise ValueError(
            f"{len(needs_backfill)}/{len(records)} eligible rows have empty `{source_text_col}`; "
            "pass --source-dataset to backfill from the source HF dataset."
        )

    needed_ids = {str(r["content_id"]) for r in needs_backfill}
    src = load_dataset(source_dataset, source_config, split="train")
    id_to_text: dict[str, str] = {}
    for row in src:
        cid = str(row.get("content_id", ""))
        if cid in needed_ids:
            id_to_text[cid] = str(row.get(source_text_col, "") or "")

    missing = needed_ids - id_to_text.keys()
    if missing:
        preview = list(missing)[:5]
        print(
            f"WARN: {len(missing)}/{len(needed_ids)} content_ids not found in "
            f"{source_dataset} (first 5: {preview})"
        )

    for row in needs_backfill:
        text = id_to_text.get(str(row["content_id"]))
        if text:
            row["lyrics"] = text


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
    source_dataset: str | None = None,
    source_text_col: str = "reference_target",
    source_config: str = "default",
) -> tuple[list[dict[str, Any]], PoolStats]:
    """Build a stratified candidate pool from an eval dataset."""
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    target_size = max(min_size, min(target_size, max_size))

    ds = load_dataset(eval_dataset, "default", split="train")
    eligible: list[dict[str, Any]] = []
    total_rows_seen = 0
    for row in ds:
        total_rows_seen += 1
        mem_fraction = row.get(fraction_col, 0)
        if mem_fraction < threshold:
            continue
        eligible.append(
            {
                "content_id": row["content_id"],
                "content_title": row.get("content_title", ""),
                "content_creators": _to_artists(row.get("content_creators")),
                "content_year": row.get("content_year"),
                "lyrics": row.get(source_text_col, "") or "",
                "memorized_fraction": mem_fraction,
                "max_p_z": row.get("max_p_z"),
                "num_windows": row.get("num_windows"),
                "memorized_windows": row.get("memorized_windows"),
                "source_dataset": eval_dataset,
            }
        )

    _backfill_lyrics_from_source(
        eligible,
        source_dataset=source_dataset,
        source_text_col=source_text_col,
        source_config=source_config,
    )

    if deduplicate:
        eligible = _deduplicate_candidates(eligible)
    if not eligible:
        raise ValueError("No eligible rows found. Lower --threshold or verify eval dataset columns.")

    desired = min(target_size, len(eligible), max_size)
    if len(eligible) < min_size:
        desired = len(eligible)

    selected = sorted(
        stratified_select(eligible, "memorized_fraction", desired, num_bins, seed),
        key=lambda r: float(r["memorized_fraction"]),
        reverse=True,
    )
    fractions = [float(r["memorized_fraction"]) for r in selected]
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
    """Split a pool into (forget, retain) songs, retain = pool minus forget."""
    if not records:
        raise ValueError("Pool is empty")
    if forget_fraction is not None and not 0.0 < forget_fraction <= 1.0:
        raise ValueError("forget_fraction must be in (0, 1] when set")

    copied = [dict(r) for r in records]
    forget_target = len(copied) if forget_fraction is None else max(1, math.ceil(len(copied) * forget_fraction))
    forget_rows = (
        copied if forget_target >= len(copied)
        else stratified_select(copied, "memorized_fraction", forget_target, num_bins, seed)
    )
    forget_rows.sort(key=lambda r: float(r.get("memorized_fraction", 0.0)), reverse=True)

    forget_keys = {_identity_key(r) for r in forget_rows}
    retain_rows = sorted(
        (dict(r) for r in copied if _identity_key(r) not in forget_keys),
        key=lambda r: float(r.get("memorized_fraction", 0.0)),
        reverse=True,
    )
    return forget_rows, retain_rows


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
    """Save the total pool and its forget/retain splits to local parquet."""
    if not pool_records:
        raise ValueError("pool_records must be non-empty")
    paths = _resolve_split_output_paths(output_root)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    Dataset.from_list(pool_records).to_parquet(str(paths["pool"]))
    Dataset.from_list(forget_records).to_parquet(str(paths["forget"]))
    if retain_records:
        Dataset.from_list(retain_records).to_parquet(str(paths["retain"]))
    else:
        Dataset.from_dict({k: [] for k in pool_records[0].keys()}).to_parquet(str(paths["retain"]))
    return paths

#!/usr/bin/env python3
"""Dolma retrieval pipeline (Polars-vectorized, Spark-style stages).

Builds BM25 retrieval indices over Dolma corpora at any scale (10B → 6T+).
Designed for /lfs/-style large-RAM machines and many CPUs.

Stages
------
  download   Pull a Dolma variant from HuggingFace (optional category filter).
  process    Stream-decompress raw .jsonl.zst shards → vectorized chunking via
             Polars → ONE parquet per raw shard. Bounded RAM, idempotent,
             atomic writes (.tmp + rename).
  compact    Merge per-raw-shard parquets into N target shards via streaming
             ParquetWriter (no in-RAM accumulation). Round-robin assignment
             gives balanced bucket sizes.
  index      Build BM25 indices over compacted shards in parallel (one bm25s
             index per shard).
  search     Fan out queries to all shard indices in parallel; each worker
             gathers top-k texts directly from its own shard parquet via
             row-group selection (no global corpus scan); the parent merges
             top-k by score with a heap.
  run        process → compact → index → search.
  status     Print stats about each stage for the current variant/subset.

Design notes
------------
* All stages are idempotent. Each task writes to a `.tmp` path and atomically
  renames on success. Re-running a stage skips outputs that already exist.
  Failed tasks can be retried by simply re-invoking the stage.
* Per-stage manifest (`_manifest.json`) records totals so resumes are cheap
  and `status` is fast.
* Process step uses one worker per raw shard so per-worker memory stays
  bounded (~hundreds of MB) regardless of corpus size. The previous
  implementation accumulated ~111 raw shards' worth of chunks per worker
  before flushing, which OOM'd at 6T scale.
* Search uses pyarrow row-group selection so only the row groups containing
  retrieved indices are read from disk — O(retrieved) rather than O(corpus).
* Polars `read_ndjson` uses simdjson under the hood (10-50x faster than
  per-line `json.loads`). Vectorized chunking via `int_ranges` + `str.slice`
  with column expressions runs in Rust over columnar batches.

Storage layout (under --dolma-dir):
  raw/{variant}/data/{category}/*.jsonl.zst                 (download)
  processed/{variant}/raw_shards[__{tag}]/raw_NNNNN.parquet (process)
  processed/{variant}/shards[__{tag}]/shard_NNN.parquet     (compact)
  indices/{variant}/bm25s[__{tag}]/shard_NNN/               (index)

Where {tag} encodes the category subset (or empty for the full corpus).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dolma-retrieval")


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_DOLMA_DIR = "/lfs/skampere2/0/shared_hf_cache/dolma3"
DEFAULT_VARIANT = "10B"

VARIANT_TO_HF_REPO: dict[str, str] = {
    "10B": "allenai/dolma3_dolmino_mix-10B-1025",
    "150B": "allenai/dolma3_mix-150B-1025",
    "6T": "allenai/dolma3_mix-6T",
}

# Per-variant default shard count. Larger corpora need more shards so each
# per-shard BM25 build fits comfortably in worker RAM and tokenize/build is
# parallelizable. 6T at 128 shards crashed OOM (~115 GB/worker × 32 = 3.7 TB);
# 512 shards puts each shard at ~6M chunks → ~28 GB/worker × 32 = 900 GB peak.
DEFAULT_NUM_SHARDS: dict[str, int] = {"10B": 32, "150B": 64, "6T": 512}

CHUNK_SIZE = 2000      # characters per chunk
CHUNK_OVERLAP = 500    # character overlap between consecutive chunks
MIN_CHUNK_LEN = 200    # drop trailing chunks shorter than this (multi-chunk only)
MIN_DOC_LEN = 100      # drop documents shorter than this entirely

# Canonical parquet schema. `text` is large_string so a row group can hold
# more than 2 GB of UTF-8 text without overflowing the int32 offset array.
PARQUET_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("text", pa.large_string()),
        ("source", pa.string()),
        ("category", pa.string()),
        ("char_offset", pa.int64()),
    ]
)


# ── Path helpers ────────────────────────────────────────────────────────────


def _subset_tag(args: argparse.Namespace) -> str:
    """Stable directory tag for a category subset (or '' for the full corpus).

    Short subsets use the joined parts directly. Long ones become
    `<short>_<sha1[:8]>` so all categories remain uniquely addressable even
    when the directory name has to be truncated for filesystem sanity.
    """
    cats: str | None = getattr(args, "categories", None)
    if not cats:
        return ""
    parts = [c.strip() for c in cats.split(",") if c.strip()]
    joined = "__".join(parts)
    if len(joined) <= 80:
        return joined
    short_parts = [p.removeprefix("common_crawl-") for p in parts]
    short_joined = "__".join(short_parts)
    digest = hashlib.sha1("|".join(sorted(parts)).encode()).hexdigest()[:8]
    return f"{short_joined[:71]}_{digest}"


def _raw_dir(args: argparse.Namespace) -> Path:
    return Path(args.dolma_dir) / "raw" / args.variant


def _processed_root(args: argparse.Namespace) -> Path:
    return Path(args.dolma_dir) / "processed" / args.variant


def _raw_shards_dir(args: argparse.Namespace) -> Path:
    tag = _subset_tag(args)
    return _processed_root(args) / (f"raw_shards__{tag}" if tag else "raw_shards")


def _raw_shard_path(args: argparse.Namespace, idx: int) -> Path:
    return _raw_shards_dir(args) / f"raw_{idx:05d}.parquet"


def _shards_dir(args: argparse.Namespace) -> Path:
    tag = _subset_tag(args)
    return _processed_root(args) / (f"shards__{tag}" if tag else "shards")


def _shard_path(args: argparse.Namespace, idx: int) -> Path:
    return _shards_dir(args) / f"shard_{idx:03d}.parquet"


def _bm25_index_root(args: argparse.Namespace) -> Path:
    tag = _subset_tag(args)
    base = Path(args.dolma_dir) / "indices" / args.variant
    return base / (f"bm25s__{tag}" if tag else "bm25s")


def _bm25_shard_dir(args: argparse.Namespace, idx: int) -> Path:
    return _bm25_index_root(args) / f"shard_{idx:03d}"


def _manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "_manifest.json"


# ── Manifest helpers ────────────────────────────────────────────────────────


def _read_manifest(stage_dir: Path) -> dict[str, Any]:
    p = _manifest_path(stage_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(stage_dir: Path, data: dict[str, Any]) -> None:
    p = _manifest_path(stage_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def _row_count(parquet_path: str | Path) -> int:
    try:
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        return 0


# ── Raw-shard discovery ─────────────────────────────────────────────────────


def _list_raw_shards(args: argparse.Namespace) -> list[str]:
    """All `.jsonl.zst` shards under `raw/{variant}/`, optionally filtered.

    Filter matches when the parent directory name equals the filter, starts
    with `<filter>-`, or contains the filter as a substring. The first two
    catch the canonical Dolma layout (`common_crawl-entertainment-0014`);
    the third is a defensive fallback.
    """
    raw_dir = _raw_dir(args)
    all_paths = sorted(str(p) for p in raw_dir.rglob("*.jsonl.zst"))

    cats_arg: str | None = getattr(args, "categories", None)
    if not cats_arg:
        return all_paths

    filters = [c.strip() for c in cats_arg.split(",") if c.strip()]

    def matches(path: str) -> bool:
        parent = Path(path).parent.name
        for f in filters:
            if parent == f or parent.startswith(f + "-") or f in parent:
                return True
        return False

    return [p for p in all_paths if matches(p)]


# ── Stage 1: download ───────────────────────────────────────────────────────


def cmd_download(args: argparse.Namespace) -> None:
    """Download a Dolma variant from HuggingFace.

    Uses HF `allow_patterns` when --categories is set so only matching shards
    are pulled. HF rate-limits at ~1000 requests / 5 min on the free tier; if
    you hit a 429, retry with fewer --workers.
    """
    from huggingface_hub import snapshot_download

    repo_id = VARIANT_TO_HF_REPO.get(args.variant)
    if repo_id is None:
        log.error("Unknown variant %r — known: %s", args.variant, list(VARIANT_TO_HF_REPO))
        sys.exit(1)

    local_dir = _raw_dir(args)
    local_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns: list[str] | None = None
    if getattr(args, "categories", None):
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        allow_patterns = [f"data/*{c}*/*.jsonl.zst" for c in cats]
        log.info(
            "Category filter: %d patterns (e.g. %s)",
            len(allow_patterns),
            allow_patterns[:2],
        )

    log.info("Downloading %s → %s (workers=%d)", repo_id, local_dir, args.workers)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=args.workers,
    )
    log.info("Download complete: %s", local_dir)


# ── Stage 2: process (Polars vectorized chunking) ───────────────────────────


def _process_one_raw_shard(
    raw_shard_path: str,
    output_path: str,
    force: bool,
) -> dict[str, Any]:
    """Worker: stream-decompress one .jsonl.zst → polars chunk → ONE parquet.

    Bounded memory: each worker holds at most one decompressed shard
    (~200-500 MB) plus its expanded chunk DataFrame (~3-5x). Atomic write via
    .tmp + rename. Idempotent: skips if `output_path` already exists.
    """
    import polars as pl
    import zstandard as zstd

    out_p = Path(output_path)
    if out_p.exists() and not force:
        return {"path": output_path, "skipped": True, "n_chunks": _row_count(out_p)}

    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path + ".tmp"
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()

    t0 = time.time()

    # Decompress to a single bytes buffer. Per-shard size is bounded
    # (Dolma raw shards are tens of MB compressed, hundreds of MB expanded).
    dctx = zstd.ZstdDecompressor()
    with open(raw_shard_path, "rb") as fh:
        decompressed = dctx.stream_reader(fh).read()

    if not decompressed:
        return {
            "path": output_path,
            "skipped": False,
            "n_chunks": 0,
            "seconds": round(time.time() - t0, 1),
        }

    # Polars vectorized JSON parse (simdjson under the hood). We pass an
    # explicit schema for *only* the fields we need so Polars projects to
    # those columns and never has to infer a type for other Dolma fields
    # (some shards have nested structs / mixed-null columns under fields
    # like "metadata" / "added" that break inference with the default
    # `expected null in json value, got object` error).
    ndjson_schema = {
        "id": pl.Utf8,
        "text": pl.Utf8,
        "source": pl.Utf8,
    }
    try:
        df = pl.read_ndjson(decompressed, schema=ndjson_schema, ignore_errors=True)
    except Exception:
        # Fall back to permissive inference + ignore_errors if the explicit
        # schema doesn't match this shard's field set.
        df = pl.read_ndjson(decompressed, ignore_errors=True)
        for col in ("id", "text", "source"):
            if col not in df.columns:
                df = df.with_columns(pl.lit("").alias(col))

    if df.height == 0:
        return {
            "path": output_path,
            "skipped": False,
            "n_chunks": 0,
            "seconds": round(time.time() - t0, 1),
        }

    category = Path(raw_shard_path).parent.name
    step = CHUNK_SIZE - CHUNK_OVERLAP

    # Vectorized chunking pipeline. Everything below runs in Rust over
    # columnar batches and is multi-threaded by Polars internally.
    #
    # Number of chunks per doc:
    #   if text_len <= chunk_size:               → 1
    #   else (multi-chunk):                      → (text_len - overlap - 1) // step + 1
    #
    # Old (per-row Python) reference:
    #     for start in range(0, text_len - overlap, step): ...
    chunks = (
        df.lazy()
        .filter(pl.col("text").str.len_chars() >= MIN_DOC_LEN)
        .with_columns(pl.col("text").str.len_chars().alias("_text_len"))
        .with_columns(
            pl.when(pl.col("_text_len") <= CHUNK_SIZE)
            .then(pl.lit(1, dtype=pl.Int64))
            .otherwise(
                ((pl.col("_text_len") - CHUNK_OVERLAP - 1) // step + 1).cast(pl.Int64)
            )
            .alias("_n_chunks")
        )
        .with_columns(pl.int_ranges(0, pl.col("_n_chunks")).alias("_chunk_idx"))
        .explode("_chunk_idx")
        .with_columns(
            (pl.col("_chunk_idx") * step).cast(pl.Int64).alias("char_offset")
        )
        .with_columns(
            pl.col("text")
            .str.slice(pl.col("char_offset"), CHUNK_SIZE)
            .alias("_chunk_text")
        )
        # MIN_CHUNK_LEN only filters trailing slivers in multi-chunk docs;
        # short single-chunk docs are kept (matches the previous behaviour).
        .filter(
            (pl.col("_chunk_text").str.len_chars() >= MIN_CHUNK_LEN)
            | (pl.col("_n_chunks") == 1)
        )
        .select(
            (
                pl.col("id").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("_chunk_idx").cast(pl.Utf8)
            ).alias("chunk_id"),
            pl.col("id").cast(pl.Utf8).alias("doc_id"),
            pl.col("_chunk_text").alias("text"),
            pl.col("source").cast(pl.Utf8),
            pl.lit(category, dtype=pl.Utf8).alias("category"),
            pl.col("char_offset"),
        )
        .collect()
    )

    n_chunks = chunks.height
    if n_chunks > 0:
        arrow_table = chunks.to_arrow().cast(PARQUET_SCHEMA)
        pq.write_table(
            arrow_table, tmp_path, compression="zstd", row_group_size=200_000
        )
        os.replace(tmp_path, output_path)

    return {
        "path": output_path,
        "skipped": False,
        "n_chunks": n_chunks,
        "seconds": round(time.time() - t0, 1),
    }


def cmd_process(args: argparse.Namespace) -> None:
    """Decompress + chunk Dolma raw shards into per-raw-shard parquets."""
    # Backward compat: if compacted shards already exist for this variant /
    # subset, skip the entire process step. Older 10B runs only have the
    # compacted layout.
    if not _raw_shards_dir(args).exists():
        existing_compacted = sorted(_shards_dir(args).glob("shard_*.parquet"))
        if existing_compacted:
            log.info(
                "Detected pre-existing compacted shards at %s — skipping process",
                _shards_dir(args),
            )
            return

    raw_paths = _list_raw_shards(args)
    if not raw_paths:
        log.error("No .jsonl.zst files found in %s", _raw_dir(args))
        sys.exit(1)

    log.info("Found %d raw .jsonl.zst shards", len(raw_paths))

    out_dir = _raw_shards_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[int, str, str]] = []
    skipped = 0
    for idx, rp in enumerate(raw_paths):
        out_path = _raw_shard_path(args, idx)
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        tasks.append((idx, rp, str(out_path)))

    log.info("Process plan: %d to run, %d already done", len(tasks), skipped)
    if not tasks:
        return

    t0 = time.time()
    n_done = 0
    n_chunks_total = 0
    failures: list[tuple[int, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one_raw_shard, rp, op, args.force): (idx, rp)
            for idx, rp, op in tasks
        }
        for future in as_completed(futures):
            idx, rp = futures[future]
            try:
                stats = future.result()
            except Exception as exc:
                log.exception("raw shard %d (%s) failed: %s", idx, Path(rp).name, exc)
                failures.append((idx, str(exc)))
                continue
            n_done += 1
            n_chunks_total += stats["n_chunks"]
            if n_done % 50 == 0 or n_done == len(tasks):
                rate = n_done / max(time.time() - t0, 1e-6)
                eta_min = (len(tasks) - n_done) / max(rate, 1e-6) / 60
                log.info(
                    "  [%d/%d] last shard +%d chunks, rate=%.1f shard/s, ETA=%.1f min",
                    n_done,
                    len(tasks),
                    stats["n_chunks"],
                    rate,
                    eta_min,
                )

    elapsed = time.time() - t0

    # Authoritative totals from disk (covers prior runs too).
    all_outs = sorted(out_dir.glob("raw_*.parquet"))
    total_rows = sum(_row_count(p) for p in all_outs)
    log.info(
        "Process: %d/%d shards done in %.1f min (%d failures, %d total chunks)",
        n_done,
        len(tasks),
        elapsed / 60,
        len(failures),
        total_rows,
    )
    _write_manifest(
        out_dir,
        {
            "stage": "process",
            "variant": args.variant,
            "subset_tag": _subset_tag(args),
            "n_raw_shards_total": len(raw_paths),
            "n_raw_shards_completed": len(all_outs),
            "n_chunks_total": total_rows,
            "elapsed_seconds": round(elapsed, 1),
        },
    )

    if failures:
        log.warning("%d shards failed — re-run process to retry", len(failures))


# ── Stage 3: compact ────────────────────────────────────────────────────────


def _compact_one_bucket(
    in_paths: list[str],
    output_path: str,
    force: bool,
) -> dict[str, Any]:
    """Worker: stream many small parquets into one larger parquet.

    Pure I/O — no decoding/encoding beyond reading and writing record
    batches. Bounded memory (one batch at a time). Atomic write.
    """
    out_p = Path(output_path)
    if out_p.exists() and not force:
        return {"path": output_path, "skipped": True, "n_chunks": _row_count(out_p)}

    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path + ".tmp"
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()

    t0 = time.time()
    n_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for src in in_paths:
            pf = pq.ParquetFile(src)
            for batch in pf.iter_batches(batch_size=200_000):
                if writer is None:
                    writer = pq.ParquetWriter(
                        tmp_path, batch.schema, compression="zstd"
                    )
                writer.write_table(pa.Table.from_batches([batch]))
                n_rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is not None:
        os.replace(tmp_path, output_path)

    return {
        "path": output_path,
        "skipped": False,
        "n_chunks": n_rows,
        "n_inputs": len(in_paths),
        "seconds": round(time.time() - t0, 1),
    }


def cmd_compact(args: argparse.Namespace) -> None:
    """Combine per-raw-shard parquets into N target compacted shards."""
    in_dir = _raw_shards_dir(args)
    inputs = sorted(str(p) for p in in_dir.glob("raw_*.parquet"))
    if not inputs:
        # Backward compat: if compacted shards already exist, treat as done.
        existing = sorted(_shards_dir(args).glob("shard_*.parquet"))
        if existing:
            log.info("Compact: %d shards already exist, nothing to do", len(existing))
            return
        log.error("No raw_*.parquet files in %s — run 'process' first", in_dir)
        sys.exit(1)

    n = args.num_shards
    out_dir = _shards_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Round-robin assignment for balanced bucket sizes; deterministic across
    # runs because `inputs` is sorted.
    buckets: list[list[str]] = [[] for _ in range(n)]
    for i, p in enumerate(inputs):
        buckets[i % n].append(p)

    tasks: list[tuple[int, list[str], str]] = []
    skipped = 0
    for b_idx, in_paths in enumerate(buckets):
        out_path = _shard_path(args, b_idx)
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        if in_paths:
            tasks.append((b_idx, in_paths, str(out_path)))

    log.info(
        "Compact plan: %d buckets (%d to run, %d done) over %d input parquets",
        n,
        len(tasks),
        skipped,
        len(inputs),
    )
    if not tasks:
        return

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_compact_one_bucket, in_paths, op, args.force): b_idx
            for b_idx, in_paths, op in tasks
        }
        for future in as_completed(futures):
            b_idx = futures[future]
            try:
                stats = future.result()
            except Exception as exc:
                log.exception("compact bucket %d failed: %s", b_idx, exc)
                continue
            n_done += 1
            log.info(
                "  [%d/%d] bucket %d: %d chunks from %d files (%.1fs)",
                n_done,
                len(tasks),
                b_idx,
                stats["n_chunks"],
                stats.get("n_inputs", 0),
                stats["seconds"],
            )

    elapsed = time.time() - t0
    total_rows = sum(_row_count(_shard_path(args, i)) for i in range(n))
    log.info(
        "Compact: %d shards in %.1f min (total %d chunks)", n, elapsed / 60, total_rows
    )
    _write_manifest(
        out_dir,
        {
            "stage": "compact",
            "variant": args.variant,
            "subset_tag": _subset_tag(args),
            "num_shards": n,
            "n_chunks_total": total_rows,
            "elapsed_seconds": round(elapsed, 1),
        },
    )


# Legacy alias used by older callers / scripts.
cmd_shard = cmd_compact


# ── Stage 4: index ──────────────────────────────────────────────────────────


def _build_one_index(
    parquet_path: str,
    out_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Worker: BM25 index over one shard parquet via bm25s."""
    import os as _os

    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import bm25s

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    marker = out_p / "data.csc.index.npy"

    if marker.exists() and not force:
        chunk_id_p = out_p / "chunk_ids.json"
        n = 0
        if chunk_id_p.exists():
            try:
                n = len(json.loads(chunk_id_p.read_text()))
            except Exception:
                n = 0
        return {"out_dir": out_dir, "skipped": True, "n_chunks": n}

    t0 = time.time()
    table = pq.read_table(parquet_path, columns=["chunk_id", "text"])
    chunk_ids = table.column("chunk_id").to_pylist()
    texts = table.column("text").to_pylist()
    load_t = time.time() - t0

    t0 = time.time()
    tokenized = bm25s.tokenize(texts, lower=True, stopwords="en", show_progress=False)
    tok_t = time.time() - t0

    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=False)
    build_t = time.time() - t0

    t0 = time.time()
    retriever.save(str(out_p))
    (out_p / "chunk_ids.json").write_text(json.dumps(chunk_ids))
    save_t = time.time() - t0

    return {
        "out_dir": out_dir,
        "skipped": False,
        "n_chunks": len(chunk_ids),
        "load_s": round(load_t, 1),
        "tok_s": round(tok_t, 1),
        "build_s": round(build_t, 1),
        "save_s": round(save_t, 1),
    }


def cmd_index(args: argparse.Namespace) -> None:
    """Build BM25 indices over compacted shards in parallel."""
    expected = [_shard_path(args, i) for i in range(args.num_shards)]
    missing = [p for p in expected if not p.exists()]
    if missing:
        log.error(
            "%d/%d compacted shards missing — run 'compact' first (e.g. %s)",
            len(missing),
            args.num_shards,
            missing[0],
        )
        sys.exit(1)

    # bm25s indexing is very memory-hungry per worker (~20-30 GB at 6M
    # chunks/shard, ~115 GB at 24M chunks/shard). Use a separate, lower
    # default for index parallelism than the rest of the pipeline.
    index_workers = getattr(args, "index_workers", None) or args.workers
    log.info(
        "Index: %d BM25 shards (workers=%d)", args.num_shards, index_workers
    )
    t0 = time.time()
    n_done = 0
    n_skipped = 0

    with ProcessPoolExecutor(max_workers=index_workers) as pool:
        futures = {
            pool.submit(
                _build_one_index,
                str(_shard_path(args, i)),
                str(_bm25_shard_dir(args, i)),
                args.force,
            ): i
            for i in range(args.num_shards)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                stats = future.result()
            except Exception as exc:
                log.exception("index shard %d failed: %s", i, exc)
                continue
            n_done += 1
            if stats.get("skipped"):
                n_skipped += 1
                log.info(
                    "  [%d/%d] shard %d SKIP (%d chunks)",
                    n_done,
                    args.num_shards,
                    i,
                    stats["n_chunks"],
                )
            else:
                log.info(
                    "  [%d/%d] shard %d: %d chunks "
                    "(load=%ss tok=%ss build=%ss save=%ss)",
                    n_done,
                    args.num_shards,
                    i,
                    stats["n_chunks"],
                    stats["load_s"],
                    stats["tok_s"],
                    stats["build_s"],
                    stats["save_s"],
                )

    elapsed = time.time() - t0
    log.info(
        "Index: %d/%d shards in %.1f min (%d skipped)",
        n_done,
        args.num_shards,
        elapsed / 60,
        n_skipped,
    )
    _write_manifest(
        _bm25_index_root(args),
        {
            "stage": "index",
            "variant": args.variant,
            "subset_tag": _subset_tag(args),
            "num_shards": args.num_shards,
            "elapsed_seconds": round(elapsed, 1),
        },
    )


# ── Stage 5: search ─────────────────────────────────────────────────────────


def _query_one_shard(
    shard_idx: int,
    bm25_dir: str,
    parquet_path: str,
    queries: list[str],
    top_k: int,
) -> tuple[int, list[list[tuple[str, float, str]]]]:
    """Worker: BM25 query + text gather for one shard.

    Returns (shard_idx, per_query_lists). per_query_lists[q] is a list of
    (chunk_id, score, text) tuples for that shard's local top-k. Texts are
    gathered HERE (via row-group selection on the local shard parquet) so
    the parent merger doesn't have to scan the corpus.
    """
    import os as _os

    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import bm25s

    retriever = bm25s.BM25.load(bm25_dir, mmap=True)

    query_tokens = bm25s.tokenize(
        queries, lower=True, stopwords="en", show_progress=False
    )
    indices, scores = retriever.retrieve(query_tokens, k=top_k, show_progress=False)
    # indices: (Q, K) — local row indices into the shard parquet (sequential).

    # Collect unique local indices we need to fetch text for.
    needed: set[int] = set()
    for q in range(indices.shape[0]):
        for r in range(indices.shape[1]):
            needed.add(int(indices[q, r]))

    # Read only the row groups containing those indices, only the text column.
    pf = pq.ParquetFile(parquet_path)
    row_offsets: list[tuple[int, int]] = []
    cum = 0
    for rg in range(pf.num_row_groups):
        n_rows = pf.metadata.row_group(rg).num_rows
        row_offsets.append((cum, cum + n_rows))
        cum += n_rows

    needed_rgs: list[int] = []
    for rg, (lo, hi) in enumerate(row_offsets):
        if any(lo <= li < hi for li in needed):
            needed_rgs.append(rg)

    if not needed_rgs:
        return shard_idx, [[] for _ in queries]

    table = pf.read_row_groups(needed_rgs, columns=["chunk_id", "text"])
    chunk_ids_col = table.column("chunk_id").to_pylist()
    texts_col = table.column("text").to_pylist()

    # Build a map: global local index → position in `table`.
    pos: dict[int, int] = {}
    cursor = 0
    for rg in needed_rgs:
        lo, hi = row_offsets[rg]
        for offset in range(hi - lo):
            pos[lo + offset] = cursor + offset
        cursor += hi - lo

    out: list[list[tuple[str, float, str]]] = []
    for q in range(indices.shape[0]):
        row: list[tuple[str, float, str]] = []
        for r in range(indices.shape[1]):
            li = int(indices[q, r])
            p = pos.get(li)
            if p is None:
                continue
            row.append((chunk_ids_col[p], float(scores[q, r]), texts_col[p]))
        out.append(row)

    return shard_idx, out


def _load_forget_set(forget_path: str) -> list[dict[str, Any]]:
    """Load forget set from parquet/json/jsonl."""
    p = Path(forget_path)
    if p.suffix == ".parquet":
        return pq.read_table(p).to_pylist()
    if p.suffix in (".json", ".jsonl"):
        rows: list[dict[str, Any]] = []
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    log.error("Unsupported forget-set format: %s", p.suffix)
    sys.exit(1)


def cmd_search(args: argparse.Namespace) -> None:
    """Fan-out queries across shard BM25 indices and merge top-k by score."""
    expected_dirs = [_bm25_shard_dir(args, i) for i in range(args.num_shards)]
    missing = [d for d in expected_dirs if not (d / "data.csc.index.npy").exists()]
    if missing:
        log.error(
            "%d shard indices missing — run 'index' first (e.g. %s)",
            len(missing),
            missing[0],
        )
        sys.exit(1)

    forget_rows = _load_forget_set(args.forget_path)
    log.info("Loaded %d forget-set items from %s", len(forget_rows), args.forget_path)

    queries: list[str] = []
    query_meta: list[tuple[str, str]] = []
    for row in forget_rows:
        cid = str(row.get("content_id") or row.get("song_id") or "unknown")
        title = str(
            row.get("content_title") or row.get("song_title") or "unknown"
        )
        text = str(row.get("lyrics") or row.get("reference_target") or "")
        if not text.strip():
            log.warning("Skipping %s — no lyrics/text", cid)
            continue
        queries.append(text)
        query_meta.append((cid, title))

    if not queries:
        log.error("No valid queries — exiting")
        sys.exit(1)

    top_k = args.top_k
    log.info(
        "Querying %d shards in parallel for %d queries × top-%d ...",
        args.num_shards,
        len(queries),
        top_k,
    )
    t0 = time.time()

    # Per query, a max-heap of size top_k keyed by score.
    per_query_heaps: list[list[tuple[float, str, str]]] = [[] for _ in queries]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _query_one_shard,
                i,
                str(_bm25_shard_dir(args, i)),
                str(_shard_path(args, i)),
                queries,
                top_k,
            ): i
            for i in range(args.num_shards)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                _, per_query_results = future.result()
            except Exception as exc:
                log.exception("query on shard %d failed: %s", i, exc)
                continue
            for q_idx, hits in enumerate(per_query_results):
                heap = per_query_heaps[q_idx]
                for cid, score, text in hits:
                    if len(heap) < top_k:
                        heapq.heappush(heap, (score, cid, text))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, cid, text))

    elapsed = time.time() - t0
    log.info(
        "Sharded retrieval done in %.1fs (%.2fs/query)",
        elapsed,
        elapsed / max(len(queries), 1),
    )

    all_results: list[dict[str, Any]] = []
    for q_idx, (cid, title) in enumerate(query_meta):
        ranked = sorted(per_query_heaps[q_idx], key=lambda x: x[0], reverse=True)
        log.info("=== %s ===", title)
        for rank, (score, c_id, text) in enumerate(ranked[:5]):
            preview = text[:150].replace("\n", " ")
            log.info("  [%d] score=%.2f  chunk=%s", rank + 1, score, c_id)
            log.info("       %s...", preview)
        for rank, (score, c_id, text) in enumerate(ranked):
            all_results.append(
                {
                    "query_content_id": cid,
                    "query_content_title": title,
                    "rank": rank + 1,
                    "score": score,
                    "chunk_id": c_id,
                    "text": text,
                    "method": "bm25",
                }
            )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bm25_results.parquet"
    pq.write_table(pa.Table.from_pylist(all_results), output_path)
    log.info(
        "Saved %d results (%d queries × top-%d) → %s",
        len(all_results),
        len(queries),
        top_k,
        output_path,
    )

    summary = {
        "method": "bm25",
        "variant": args.variant,
        "subset_tag": _subset_tag(args),
        "num_queries": len(queries),
        "top_k": top_k,
        "total_results": len(all_results),
        "num_shards": args.num_shards,
        "search_seconds": round(elapsed, 1),
    }
    (output_dir / "bm25_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary → %s", output_dir / "bm25_summary.json")


# ── run-all + status ────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """process → compact → index → search."""
    cmd_process(args)
    cmd_compact(args)
    cmd_index(args)
    cmd_search(args)


def cmd_status(args: argparse.Namespace) -> None:
    """Print stats about each pipeline stage for the current variant/subset."""
    raw_paths = _list_raw_shards(args)
    log.info("variant=%s subset_tag=%r", args.variant, _subset_tag(args) or "(none)")
    log.info("raw shards available: %d in %s", len(raw_paths), _raw_dir(args))

    raw_dir = _raw_shards_dir(args)
    if raw_dir.exists():
        files = sorted(raw_dir.glob("raw_*.parquet"))
        rows = sum(_row_count(p) for p in files)
        log.info(
            "process: %d/%d parquets, %d total chunks (%s)",
            len(files),
            len(raw_paths),
            rows,
            raw_dir,
        )
        m = _read_manifest(raw_dir)
        if m:
            log.info("  manifest: %s", m)
    else:
        log.info("process: not started (%s)", raw_dir)

    sd = _shards_dir(args)
    if sd.exists():
        files = sorted(sd.glob("shard_*.parquet"))
        rows = sum(_row_count(p) for p in files)
        log.info("compact: %d shards, %d total chunks (%s)", len(files), rows, sd)
    else:
        log.info("compact: not started (%s)", sd)

    ir = _bm25_index_root(args)
    if ir.exists():
        idx_dirs = sorted(ir.glob("shard_*"))
        log.info("index: %d shard indices in %s", len(idx_dirs), ir)
    else:
        log.info("index: not started (%s)", ir)


# ── Sweep stats (unchanged) ─────────────────────────────────────────────────


def cmd_sweep_stats(args: argparse.Namespace) -> None:
    """Show stats from the corpus_sweep_cleanslate_dataset (infini-gram matches)."""
    from collections import defaultdict

    from datasets import load_dataset

    log.info("Loading corpus_sweep_cleanslate_dataset from HF...")
    ds = load_dataset(
        "unlearning-cleanslate/corpus_sweep_cleanslate_dataset", split="train"
    )
    log.info("Loaded %d windows", len(ds))

    found = sum(1 for r in ds if r["found_in_any"])
    content_ids = set(r["content_id"] for r in ds)
    log.info(
        "Windows found in Dolma: %d/%d (%.1f%%)",
        found,
        len(ds),
        100 * found / len(ds),
    )
    log.info("Distinct content items: %d", len(content_ids))

    per_content: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "found": 0, "title": ""}
    )
    for r in ds:
        cid = r["content_id"]
        per_content[cid]["total"] += 1
        per_content[cid]["found"] += int(r["found_in_any"])
        per_content[cid]["title"] = r["content_title"]

    ranked = sorted(
        per_content.items(),
        key=lambda x: x[1]["found"] / max(x[1]["total"], 1),
        reverse=True,
    )
    log.info("\nTop 20 content items by Dolma match rate:")
    for cid, stats in ranked[:20]:
        frac = stats["found"] / max(stats["total"], 1)
        log.info(
            "  %s: %d/%d windows (%.0f%%) — %s",
            cid[:12],
            stats["found"],
            stats["total"],
            100 * frac,
            stats["title"],
        )


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dolma retrieval pipeline (Polars-vectorized, Spark-style)"
    )
    parser.add_argument(
        "--dolma-dir",
        default=DEFAULT_DOLMA_DIR,
        help=f"Root directory for Dolma data (default: {DEFAULT_DOLMA_DIR})",
    )
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT, help="Dolma variant: 10B / 150B / 6T"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Parallel workers per stage (default: 32)",
    )
    parser.add_argument(
        "--index-workers",
        type=int,
        default=None,
        help=(
            "Parallel workers for the BM25 index stage. bm25s holds an entire "
            "shard's text + token + CSC index in RAM, so this can be much "
            "lower than --workers. Default: --workers value, or variant-aware "
            "fallback if --num-shards/--workers combo would exceed memory budget."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of compacted shards (default: variant-specific)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Comma-separated category filters "
            "(e.g. common_crawl-entertainment,common_crawl-literature)"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild of existing artifacts"
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download a Dolma variant from HF")
    sub.add_parser("process", help="Decompress + chunk raw shards into per-shard parquets")
    sub.add_parser("compact", help="Combine per-raw-shard parquets into N target shards")
    sub.add_parser("shard", help="Alias for compact")
    sub.add_parser("index", help="Build BM25 indices over compacted shards")
    sub.add_parser("status", help="Print stats about each pipeline stage")

    p_search = sub.add_parser(
        "search", help="Query forget-set across shard indices, write top-k parquet"
    )
    p_search.add_argument("--forget-path", required=True)
    p_search.add_argument("--top-k", type=int, default=100)
    p_search.add_argument("--output", "-o", default="results/retrieval")

    p_run = sub.add_parser("run", help="process → compact → index → search")
    p_run.add_argument("--forget-path", required=True)
    p_run.add_argument("--top-k", type=int, default=100)
    p_run.add_argument("-o", "--output", default="results/retrieval")

    sub.add_parser("sweep-stats", help="Show infini-gram corpus sweep stats from HF")

    args = parser.parse_args()

    if args.num_shards is None:
        args.num_shards = DEFAULT_NUM_SHARDS.get(args.variant, 64)

    commands = {
        "download": cmd_download,
        "process": cmd_process,
        "compact": cmd_compact,
        "shard": cmd_compact,
        "index": cmd_index,
        "search": cmd_search,
        "run": cmd_run,
        "status": cmd_status,
        "sweep-stats": cmd_sweep_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

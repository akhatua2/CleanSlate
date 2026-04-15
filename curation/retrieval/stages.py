"""Pipeline stages for retrieval runs."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from curation.retrieval.queries import load_queries
from curation.retrieval.schema import Cfg, HF_FEATURES, MATCH_SCHEMA

_worker_state: dict[str, object] = {}


def _write_hits(path: Path, hits: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hits:
        table = pa.Table.from_pylist(hits, schema=MATCH_SCHEMA)
    else:
        table = pa.table(
            {field.name: pa.array([], type=field.type) for field in MATCH_SCHEMA},
            schema=MATCH_SCHEMA,
        )
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.rename(path)


def _init_search_worker(cfg_json: str) -> None:
    cfg = Cfg(**json.loads(cfg_json))
    if cfg.method == "infini_gram_mini":
        from curation.retrieval.backend_infini_gram import load_engine

        _worker_state["engine"] = load_engine(cfg.index_dir)
    _worker_state["cfg"] = cfg


def _search_worker(row: dict) -> tuple[str, list[dict]]:
    cfg = _worker_state["cfg"]
    if not isinstance(cfg, Cfg):
        raise TypeError("worker config was not initialized")

    content_id = row["content_id"]
    title = row["content_title"]
    text = row["reference_target"]

    if cfg.method == "infini_gram_mini":
        from curation.retrieval.backend_infini_gram import search_one_item

        engine = _worker_state["engine"]
        hits = search_one_item(content_id, title, text, engine, cfg)
    elif cfg.method == "bm25":
        from curation.retrieval.backend_bm25 import search_bm25_one_item

        hits = search_bm25_one_item(content_id, title, text, cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    return content_id, hits


def search(
    cfg: Cfg,
    limit: int | None = None,
    content_ids: list[str] | None = None,
) -> Path:
    queries = load_queries(cfg, limit=limit, content_ids=content_ids)
    cfg.items_dir.mkdir(parents=True, exist_ok=True)

    done = {path.stem for path in cfg.items_dir.glob("*.parquet")}
    pending = queries.filter(~pl.col("content_id").is_in(done))
    total = pending.height
    if total == 0:
        print(f"All {len(done)} items already done.")
        return cfg.items_dir

    print(f"Searching {total} items ({len(done)} already done), {cfg.workers} workers...")
    start_time = time.time()
    completed = 0
    cfg_json = json.dumps(cfg.worker_payload())

    with ProcessPoolExecutor(
        max_workers=cfg.workers,
        initializer=_init_search_worker,
        initargs=(cfg_json,),
    ) as pool:
        for content_id, hits in pool.map(_search_worker, pending.to_dicts()):
            _write_hits(cfg.items_dir / f"{content_id}.parquet", hits)
            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed else 0.0
                eta = (total - completed) / rate if rate else 0.0
                print(f"  [{completed}/{total}] {rate:.1f} items/s, ETA {eta/60:.0f}m")

    print(f"Search done: {completed} items in {(time.time() - start_time) / 60:.1f}m")
    return cfg.items_dir


def _write_summary(cfg: Cfg, limit: int | None = None, content_ids: list[str] | None = None) -> Path:
    queries = load_queries(cfg, limit=limit, content_ids=content_ids)
    matches = pl.read_parquet(cfg.matches_path)
    if matches.is_empty():
        summary = queries.select("content_id", "content_title").with_columns(
            pl.lit(0).alias("num_hits"),
            pl.lit(None, dtype=pl.Float64).alias("best_score"),
            pl.lit(0).alias("matched_doc_count"),
        )
    else:
        aggregated = matches.group_by("content_id").agg(
            pl.len().alias("num_hits"),
            pl.col("score").max().alias("best_score"),
            pl.col("doc_id").n_unique().alias("matched_doc_count"),
        )
        summary = (
            queries.select("content_id", "content_title")
            .join(aggregated, on="content_id", how="left")
            .with_columns(
                pl.col("num_hits").fill_null(0),
                pl.col("matched_doc_count").fill_null(0),
            )
        )

    cfg.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(cfg.summary_path, compression="zstd")
    return cfg.summary_path


def export(
    cfg: Cfg,
    limit: int | None = None,
    content_ids: list[str] | None = None,
    push_to_hub: bool = False,
    hf_repo: str | None = None,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    item_files = sorted(cfg.items_dir.glob("*.parquet"))
    if not item_files:
        raise FileNotFoundError(f"No item parquets in {cfg.items_dir}. Run search first.")

    print(f"Exporting {len(item_files)} items → {cfg.matches_path}")
    tmp = cfg.matches_path.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    for item_file in item_files:
        table = pq.read_table(item_file)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()
    tmp.rename(cfg.matches_path)

    summary_path = _write_summary(cfg, limit=limit, content_ids=content_ids)
    summary = pl.read_parquet(summary_path)
    print(f"  {summary.height} items, {pl.read_parquet(cfg.matches_path).height} total hits")

    if push_to_hub:
        repo = hf_repo or f"{cfg.method}_cleanslate_dolma3_{cfg.variant}"
        Dataset.from_parquet(str(cfg.matches_path), features=HF_FEATURES).push_to_hub(repo, split="train")
        print(f"  Pushed to HF: {repo}")

    return cfg.matches_path


def run_all(
    cfg: Cfg,
    limit: int | None = None,
    content_ids: list[str] | None = None,
    push_to_hub: bool = False,
    hf_repo: str | None = None,
) -> Path:
    if cfg.method == "infini_gram_mini":
        from curation.retrieval.backend_infini_gram import build_fm_index

        build_fm_index(cfg)
    elif cfg.method == "bm25":
        from curation.retrieval.backend_bm25 import build_bm25_index

        build_bm25_index(cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    search(cfg, limit=limit, content_ids=content_ids)
    return export(cfg, limit=limit, content_ids=content_ids, push_to_hub=push_to_hub, hf_repo=hf_repo)

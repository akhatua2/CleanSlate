"""Sharded BM25 retrieval via bm25s."""

from __future__ import annotations

from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq
import io
import json
import os
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from curation.retrieval.schema import Cfg

SEGMENT_SCHEMA = pa.schema(
    [
        ("unit_id", pa.string()),
        ("doc_id", pa.string()),
        ("text", pa.large_string()),
        ("source", pa.string()),
        ("category", pa.string()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
    ]
)
WORD_RE = re.compile(r"\S+")
BM25_TOKENIZE_KWARGS = {
    "lower": True,
    "stopwords": "english",
    "stemmer": None,
    "show_progress": False,
}


def tokenize_texts(texts: list[str]):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import bm25s

    return bm25s.tokenize(texts, **BM25_TOKENIZE_KWARGS)


def _iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as handle:
        reader = dctx.stream_reader(handle)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in WORD_RE.finditer(text)]


def _segment_doc(doc: dict, cfg: Cfg, category: str) -> list[dict]:
    text = doc.get("text", "")
    doc_id = doc.get("id", "")
    source = doc.get("source", "")
    if len(text) < cfg.min_doc_chars:
        return []

    step = cfg.segment_chars - cfg.segment_overlap
    if step <= 0:
        raise ValueError("segment_overlap must be smaller than segment_chars")

    words = _word_spans(text)
    if not words:
        return []

    starts = [start for start, _ in words]
    segments: list[dict] = []
    word_idx = 0

    while word_idx < len(words):
        start = words[word_idx][0]
        max_end = start + cfg.segment_chars
        end_idx = word_idx
        while end_idx + 1 < len(words) and words[end_idx + 1][1] <= max_end:
            end_idx += 1
        end = words[end_idx][1]
        chunk = text[start:end]
        if len(chunk) >= cfg.min_segment_chars:
            segments.append(
                {
                    "unit_id": f"{doc_id}_{len(segments)}",
                    "doc_id": doc_id,
                    "text": chunk,
                    "source": source,
                    "category": category,
                    "char_start": start,
                    "char_end": end,
                }
            )

        next_target = start + step
        next_idx = bisect_left(starts, next_target, lo=word_idx + 1)
        if next_idx <= word_idx:
            next_idx = word_idx + 1
        word_idx = next_idx

    return segments


def segment_shard(raw_path: Path, out_path: Path, cfg: Cfg) -> int:
    if out_path.exists():
        return pq.ParquetFile(out_path).metadata.num_rows

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    category = raw_path.parent.name
    writer: pq.ParquetWriter | None = None
    batch: list[dict] = []
    rows = 0

    for doc in _iter_jsonl_zst(raw_path):
        batch.extend(_segment_doc(doc, cfg, category))
        if len(batch) >= 200_000:
            table = pa.Table.from_pylist(batch, schema=SEGMENT_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(tmp, SEGMENT_SCHEMA, compression="zstd")
            writer.write_table(table)
            rows += len(batch)
            batch.clear()

    if batch:
        table = pa.Table.from_pylist(batch, schema=SEGMENT_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(tmp, SEGMENT_SCHEMA, compression="zstd")
        writer.write_table(table)
        rows += len(batch)

    if writer is not None:
        writer.close()
        os.replace(tmp, out_path)
    return rows


def _build_one_shard_index(args: tuple[str, str]) -> dict[str, int | bool]:
    parquet_path, index_dir = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import bm25s

    out = Path(index_dir)
    marker = out / "data.csc.index.npy"
    if marker.exists():
        return {"skipped": True}

    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(
        parquet_path,
        columns=["unit_id", "doc_id", "source", "category", "char_start", "char_end", "text"],
    )
    texts = table.column("text").to_pylist()
    corpus = [
        {
            "unit_id": unit_id,
            "doc_id": doc_id,
            "doc_source": source,
            "doc_category": category,
            "unit_char_start": char_start,
            "unit_char_end": char_end,
            "retrieved_text": text,
        }
        for unit_id, doc_id, source, category, char_start, char_end, text in zip(
            table.column("unit_id").to_pylist(),
            table.column("doc_id").to_pylist(),
            table.column("source").to_pylist(),
            table.column("category").to_pylist(),
            table.column("char_start").to_pylist(),
            table.column("char_end").to_pylist(),
            texts,
            strict=True,
        )
    ]
    retriever = bm25s.BM25()
    retriever.index(tokenize_texts(texts), show_progress=False)
    retriever.save(str(out), corpus=corpus)
    return {"skipped": False, "rows": len(texts)}


def build_bm25_index(cfg: Cfg) -> None:
    raw_paths = cfg.raw_shards()
    if not raw_paths:
        raise FileNotFoundError(f"No shards in {cfg.raw_data}")

    seg_dir = cfg.index_dir / "segments"
    idx_dir = cfg.index_dir / "bm25_shards"
    seg_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segmenting {len(raw_paths)} raw shards...")
    seg_tasks = [
        (raw_path, seg_dir / f"seg_{idx:05d}.parquet", cfg)
        for idx, raw_path in enumerate(raw_paths)
        if not (seg_dir / f"seg_{idx:05d}.parquet").exists()
    ]
    if seg_tasks:
        with ProcessPoolExecutor(max_workers=cfg.normalized_index_workers) as pool:
            futures = [pool.submit(segment_shard, raw_path, out_path, shard_cfg) for raw_path, out_path, shard_cfg in seg_tasks]
            for future in as_completed(futures):
                future.result()

    print(f"Segmentation done. Building BM25 indices on {cfg.normalized_num_shards} shards...")
    shard_paths = [seg_dir / f"shard_{idx:03d}.parquet" for idx in range(cfg.normalized_num_shards)]
    if not all(path.exists() for path in shard_paths):
        writers: dict[int, pq.ParquetWriter] = {}
        try:
            for idx, seg_path in enumerate(sorted(seg_dir.glob("seg_*.parquet"))):
                shard_idx = idx % cfg.normalized_num_shards
                shard_path = shard_paths[shard_idx]
                if shard_idx not in writers:
                    writers[shard_idx] = pq.ParquetWriter(shard_path, SEGMENT_SCHEMA, compression="zstd")
                writers[shard_idx].write_table(pq.read_table(seg_path))
        finally:
            for writer in writers.values():
                writer.close()

    tasks = [
        (str(shard_path), str(idx_dir / f"shard_{shard_idx:03d}"))
        for shard_idx, shard_path in enumerate(shard_paths)
        if shard_path.exists()
    ]
    with ProcessPoolExecutor(max_workers=cfg.normalized_index_workers) as pool:
        futures = [pool.submit(_build_one_shard_index, task) for task in tasks]
        for future in as_completed(futures):
            future.result()

    print(f"BM25 index complete: {idx_dir}")


def search_bm25_one_item(content_id: str, title: str, text: str, cfg: Cfg) -> list[dict]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import bm25s

    idx_dir = cfg.index_dir / "bm25_shards"
    shard_dirs = sorted(idx_dir.glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"No BM25 shards at {idx_dir}")

    tokens = tokenize_texts([text])
    heap: list[tuple[float, int, dict]] = []
    seq = 0

    for shard_dir in shard_dirs:
        retriever = bm25s.BM25.load(str(shard_dir), mmap=True, load_corpus=True)
        docs, scores = retriever.retrieve(tokens, k=cfg.top_k, show_progress=False, n_threads=0)
        for idx in range(docs.shape[1]):
            doc = docs[0, idx]
            if doc is None:
                continue
            score = float(scores[0, idx])
            seq += 1
            entry = (score, seq, dict(doc))
            if len(heap) < cfg.top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    hits: list[dict] = []
    for rank, (score, _, doc) in enumerate(sorted(heap, key=lambda item: item[0], reverse=True), 1):
        hits.append(
            {
                "content_id": content_id,
                "content_title": title,
                "reference_target": text,
                "retrieval_method": "bm25",
                "corpus_name": f"dolma3_{cfg.variant}",
                "doc_id": str(doc.get("doc_id", "")),
                "doc_source": str(doc.get("doc_source", "")),
                "doc_category": str(doc.get("doc_category", "")),
                "unit_id": str(doc.get("unit_id", "")),
                "unit_char_start": int(doc.get("unit_char_start", 0)),
                "unit_char_end": int(doc.get("unit_char_end", 0)),
                "rank": rank,
                "score": score,
                "retrieved_text": str(doc.get("retrieved_text", "")),
                "metadata": json.dumps(
                    {
                        "tokenization": {
                            "lower": True,
                            "stopwords": "english",
                            "stemming": False,
                            "lemmatization": False,
                        }
                    }
                ),
            }
        )
    return hits

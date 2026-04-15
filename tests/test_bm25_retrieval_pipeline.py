"""Integration tests for the BM25 retrieval pipeline."""

import json
from pathlib import Path
from concurrent.futures import Future

import pyarrow.parquet as pq
import zstandard as zstd

from curation.retrieval.backend_bm25 import BM25_TOKENIZE_KWARGS, segment_shard
from curation.retrieval.schema import Cfg
from curation.retrieval.stages import export, run_all, search


class InlineExecutor:
    def __init__(self, max_workers: int | None = None, initializer=None, initargs=()) -> None:
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self) -> "InlineExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover
            future.set_exception(exc)
        return future

    def map(self, fn, iterable):
        return map(fn, iterable)


def inline_as_completed(futures):
    return futures


def patch_inline_executors(monkeypatch) -> None:
    monkeypatch.setattr("curation.retrieval.backend_bm25.ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr("curation.retrieval.backend_bm25.as_completed", inline_as_completed)
    monkeypatch.setattr("curation.retrieval.stages.ProcessPoolExecutor", InlineExecutor)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_jsonl_zst(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor()
    with path.open("wb") as handle:
        with compressor.stream_writer(handle) as writer:
            for row in rows:
                writer.write(json.dumps(row).encode("utf-8"))
                writer.write(b"\n")


def _make_cfg(tmp_path: Path) -> Cfg:
    dolma_dir = tmp_path / "dolma3"
    shard_path = dolma_dir / "raw" / "10B" / "data" / "common_crawl-literature-0000" / "00000.jsonl.zst"
    _write_jsonl_zst(
        shard_path,
        [
            {
                "id": "doc-1",
                "text": "the quick brown fox jumps over the lazy dog and keeps running",
                "source": "literature",
            },
            {
                "id": "doc-2",
                "text": "sing the chorus softly under moonlight and starlight for everyone",
                "source": "lyrics",
            },
        ],
    )

    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(
        queries_path,
        [
            {
                "content_id": "song-1",
                "content_title": "Moonlight Song",
                "reference_target": "chorus softly under moonlight",
            },
            {
                "content_id": "song-2",
                "content_title": "Fox Song",
                "reference_target": "quick brown fox",
            },
        ],
    )

    return Cfg(
        variant="10B",
        method="bm25",
        top_k=3,
        segment_chars=32,
        segment_overlap=8,
        min_doc_chars=4,
        min_segment_chars=4,
        workers=1,
        index_workers=1,
        num_shards=2,
        dolma_root=str(dolma_dir),
        curation_root=str(tmp_path / "curation"),
        dataset_path=str(queries_path),
    )


def test_bm25_run_all_end_to_end(tmp_path: Path, monkeypatch) -> None:
    patch_inline_executors(monkeypatch)
    cfg = _make_cfg(tmp_path)

    matches_path = run_all(cfg, push_to_hub=False)

    rows = pq.read_table(matches_path).to_pylist()
    assert any(row["content_id"] == "song-1" and row["doc_id"] == "doc-2" for row in rows)
    assert any(row["content_id"] == "song-2" and row["doc_id"] == "doc-1" for row in rows)
    assert rows[0]["retrieval_method"] == "bm25"
    assert rows[0]["corpus_name"] == "dolma3_10B"
    assert "reference_target" in rows[0]
    assert "unit_char_start" in rows[0]
    assert "retrieved_text" in rows[0]
    assert "metadata" in rows[0]

    summary_rows = pq.read_table(cfg.summary_path).to_pylist()
    assert {row["content_id"] for row in summary_rows} == {"song-1", "song-2"}
    assert all(row["num_hits"] >= 1 for row in summary_rows)


def test_query_resume_skips_completed_items(tmp_path: Path, monkeypatch) -> None:
    patch_inline_executors(monkeypatch)
    cfg = _make_cfg(tmp_path)

    run_all(cfg, push_to_hub=False)
    first_mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in cfg.items_dir.glob("*.parquet")
    }

    search(cfg)
    second_mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in cfg.items_dir.glob("*.parquet")
    }
    assert second_mtimes == first_mtimes

    matches_path = export(cfg, push_to_hub=False)
    rows = pq.read_table(matches_path).to_pylist()
    assert rows


def test_segment_shard_uses_word_boundaries(tmp_path: Path) -> None:
    doc_text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    raw_path = tmp_path / "dolma3" / "raw" / "10B" / "data" / "category-0000" / "00000.jsonl.zst"
    _write_jsonl_zst(
        raw_path,
        [
            {
                "id": "doc-1",
                "text": doc_text,
                "source": "synthetic",
            }
        ],
    )

    cfg = Cfg(
        variant="10B",
        method="bm25",
        segment_chars=18,
        segment_overlap=6,
        min_doc_chars=4,
        min_segment_chars=4,
        dolma_root=str(tmp_path / "dolma3"),
        curation_root=str(tmp_path / "curation"),
        dataset_path=str(tmp_path / "queries.jsonl"),
    )
    out_path = tmp_path / "segments.parquet"

    segment_shard(raw_path, out_path, cfg)

    rows = pq.read_table(out_path).to_pylist()
    assert rows
    for row in rows:
        start = row["char_start"]
        end = row["char_end"]
        segment_text = row["text"]
        assert doc_text[start:end] == segment_text
        assert not segment_text.startswith(" ")
        assert not segment_text.endswith(" ")
        if start > 0:
            assert doc_text[start - 1].isspace()
        if end < len(doc_text):
            assert doc_text[end].isspace()


def test_bm25_uses_no_stemming_or_lemmatization() -> None:
    assert BM25_TOKENIZE_KWARGS["lower"] is True
    assert BM25_TOKENIZE_KWARGS["stopwords"] == "english"
    assert BM25_TOKENIZE_KWARGS["stemmer"] is None

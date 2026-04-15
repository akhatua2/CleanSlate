"""CLI-level end-to-end test for the BM25 pipeline."""

import sys
from pathlib import Path

import pyarrow.parquet as pq

from curation.retrieval.cli import main
from tests.test_bm25_retrieval_pipeline import _make_cfg
from tests.test_bm25_retrieval_pipeline import patch_inline_executors


def test_bm25_cli_run_e2e(tmp_path: Path, monkeypatch) -> None:
    patch_inline_executors(monkeypatch)
    cfg = _make_cfg(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "curation.retrieval.cli",
            "run",
            "--method",
            "bm25",
            "--corpus",
            cfg.variant,
            "--dolma-dir",
            str(cfg.dolma_path),
            "--curation-root",
            str(cfg.curation_path),
            "--num-shards",
            str(cfg.normalized_num_shards),
            "--segment-chars",
            str(cfg.segment_chars),
            "--segment-overlap",
            str(cfg.segment_overlap),
            "--min-doc-chars",
            str(cfg.min_doc_chars),
            "--min-segment-chars",
            str(cfg.min_segment_chars),
            "--dataset-path",
            str(cfg.dataset_path),
            "--top-k",
            str(cfg.top_k),
            "--workers",
            "1",
            "--index-workers",
            "1",
        ],
    )

    main()

    rows = pq.read_table(cfg.matches_path).to_pylist()
    assert any(row["content_id"] == "song-1" and row["doc_id"] == "doc-2" for row in rows)
    assert any(row["content_id"] == "song-2" and row["doc_id"] == "doc-1" for row in rows)

"""I/O utilities for QA benchmark datasets."""

import json
from pathlib import Path

import polars as pl

from cleanslate.fsid.qa.base import QACandidate, ValidatedQA


def save_candidates(candidates: list[QACandidate], path: str | Path) -> Path:
    """Save QA candidates to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame([c.to_dict() for c in candidates])
    df.write_parquet(path)
    return path


def load_candidates(path: str | Path) -> list[QACandidate]:
    """Load QA candidates from parquet."""
    df = pl.read_parquet(path)
    return [
        QACandidate(
            content_id=row["content_id"],
            content_title=row["content_title"],
            question=row["question"],
            answer=row["answer"],
        )
        for row in df.to_dicts()
    ]


def save_validated(validated: list[ValidatedQA], path: str | Path) -> Path:
    """Save validated QA pairs to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame([v.to_dict() for v in validated])
    df.write_parquet(path)
    return path


def load_validated(path: str | Path) -> list[ValidatedQA]:
    """Load validated QA pairs from parquet."""
    df = pl.read_parquet(path)
    return [
        ValidatedQA(
            content_id=row["content_id"],
            content_title=row["content_title"],
            question=row["question"],
            answer=row["answer"],
            verified_models=row["verified_models"],
        )
        for row in df.to_dicts()
    ]


def export_for_dataset(
    validated: list[ValidatedQA],
    path: str | Path,
) -> Path:
    """Export validated QA as JSON grouped by content_id (for HF dataset)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group by content_id
    by_content: dict[str, list[dict[str, str]]] = {}
    for v in validated:
        if v.content_id not in by_content:
            by_content[v.content_id] = []
        by_content[v.content_id].append({"question": v.question, "answer": v.answer})

    # Format for HF dataset: list of {content_id, content_qa: [...]}
    records = [
        {"content_id": cid, "content_qa": qas} for cid, qas in by_content.items()
    ]

    with open(path, "w") as f:
        json.dump(records, f, indent=2)

    return path

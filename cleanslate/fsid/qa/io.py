"""JSON I/O for QA pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from cleanslate.fsid.qa.base import Proposition, QACandidate, ValidatedQA

T = TypeVar("T", Proposition, QACandidate, ValidatedQA)


def save_rows(rows: list[T], path: str | Path) -> Path:
    """Save a list of dataclass rows to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump([r.to_dict() for r in rows], f, indent=2, ensure_ascii=False)
    return out


def load_propositions(path: str | Path) -> list[Proposition]:
    with Path(path).open() as f:
        return [Proposition(**row) for row in json.load(f)]


def load_candidates(path: str | Path) -> list[QACandidate]:
    with Path(path).open() as f:
        return [QACandidate(**row) for row in json.load(f)]


def load_validated(path: str | Path) -> list[ValidatedQA]:
    with Path(path).open() as f:
        return [ValidatedQA(**row) for row in json.load(f)]

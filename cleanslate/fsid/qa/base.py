"""Dataclasses for the QA benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Proposition:
    """An interior fact extracted from a content item."""

    content_id: str
    content_title: str
    content_creators: str
    fact: str
    source_span: str

    def to_dict(self) -> dict[str, str]:
        return {
            "content_id": self.content_id,
            "content_title": self.content_title,
            "content_creators": self.content_creators,
            "fact": self.fact,
            "source_span": self.source_span,
        }


@dataclass
class QACandidate:
    """An atomic QA candidate generated from a proposition."""

    content_id: str
    content_title: str
    content_creators: str
    proposition: str
    question: str
    answer: str
    source_span: str

    def to_dict(self) -> dict[str, str]:
        return {
            "content_id": self.content_id,
            "content_title": self.content_title,
            "content_creators": self.content_creators,
            "proposition": self.proposition,
            "question": self.question,
            "answer": self.answer,
            "source_span": self.source_span,
        }


@dataclass
class ValidatedQA:
    """A QA pair that passed the filter stage."""

    content_id: str
    content_title: str
    question: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return {
            "content_id": self.content_id,
            "content_title": self.content_title,
            "question": self.question,
            "answer": self.answer,
        }

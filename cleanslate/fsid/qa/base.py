"""Dataclasses for QA benchmark generation and evaluation."""

from dataclasses import dataclass, field


@dataclass
class QAPair:
    """An atomic question-answer pair from song lyrics."""

    question: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return {"question": self.question, "answer": self.answer}


@dataclass
class QACandidate:
    """A candidate QA pair for a content item."""

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


@dataclass
class ValidatedQA:
    """A QA pair validated against all base models."""

    content_id: str
    content_title: str
    question: str
    answer: str
    verified_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "content_id": self.content_id,
            "content_title": self.content_title,
            "question": self.question,
            "answer": self.answer,
            "verified_models": self.verified_models,
        }


@dataclass
class ContentQABenchmark:
    """QA benchmark for a single content item."""

    content_id: str
    content_title: str
    qa_pairs: list[QAPair] = field(default_factory=list)

    @property
    def num_questions(self) -> int:
        return len(self.qa_pairs)

    def to_dict(self) -> dict[str, object]:
        return {
            "content_id": self.content_id,
            "content_title": self.content_title,
            "content_qa": [qa.to_dict() for qa in self.qa_pairs],
        }


@dataclass
class QAEvalResult:
    """Result of evaluating a model on QA benchmark."""

    content_id: str
    model_name: str
    num_questions: int
    num_correct: int
    accuracy: float
    results: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "content_id": self.content_id,
            "model_name": self.model_name,
            "num_questions": self.num_questions,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "results": self.results,
        }


@dataclass
class QABenchmarkSummary:
    """Aggregate QA benchmark evaluation results."""

    model_name: str
    total_questions: int
    total_correct: int
    overall_accuracy: float
    per_content: list[QAEvalResult] = field(default_factory=list)
    eval_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "overall_accuracy": self.overall_accuracy,
            "per_content": [r.to_dict() for r in self.per_content],
            "eval_time_seconds": self.eval_time_seconds,
        }

"""QA Benchmark Generation for CleanSlate.

Generate atomic QA pairs from song lyrics, validate against base models,
and create a benchmark for evaluating semantic retention after unlearning.
"""

from cleanslate.fsid.qa.base import (
    QAPair,
    QACandidate,
    ValidatedQA,
    ContentQABenchmark,
    QAEvalResult,
    QABenchmarkSummary,
)
from cleanslate.fsid.qa.generator import generate_qa_candidates, generate_qa_for_content
from cleanslate.fsid.qa.judge import is_correct
from cleanslate.fsid.qa.validator import validate_qa_candidates, validate_single_candidate
from cleanslate.fsid.qa.io import (
    save_candidates,
    load_candidates,
    save_validated,
    load_validated,
    export_for_dataset,
)

__all__ = [
    # Dataclasses
    "QAPair",
    "QACandidate",
    "ValidatedQA",
    "ContentQABenchmark",
    "QAEvalResult",
    "QABenchmarkSummary",
    # Generation
    "generate_qa_candidates",
    "generate_qa_for_content",
    # Judging
    "is_correct",
    # Validation
    "validate_qa_candidates",
    "validate_single_candidate",
    # I/O
    "save_candidates",
    "load_candidates",
    "save_validated",
    "load_validated",
    "export_for_dataset",
]

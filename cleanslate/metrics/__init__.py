"""Evaluation metrics for sequence comparison."""

from .base import Metric, MetricResult
from .bleu import BLEU, BLEU1, BLEU2
from .rouge import ROUGE1, ROUGE2, ROUGEL
from .probability import (
    ExtractionResult,
    compute_extraction_probability,
    prompts_for_probability,
    probability_after_n_prompts,
)

__all__ = [
    # Base
    "Metric",
    "MetricResult",
    # N-gram metrics
    "BLEU",
    "BLEU1",
    "BLEU2",
    "ROUGE1",
    "ROUGE2",
    "ROUGEL",
    # Probability metrics (Hayes et al.)
    "ExtractionResult",
    "compute_extraction_probability",
    "prompts_for_probability",
    "probability_after_n_prompts",
]

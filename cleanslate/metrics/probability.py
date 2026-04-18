"""Extraction probability (p_z) metric.

p_z = P(suffix | prefix) = product of per-token conditional probabilities.
High p_z means the model is likely to produce this exact text (memorised).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    log_prob: float
    prob: float
    num_tokens: int


def compute_extraction_probability(token_log_probs: list[float]) -> ExtractionResult:
    n = len(token_log_probs)
    if n == 0:
        return ExtractionResult(log_prob=0.0, prob=1.0, num_tokens=0)
    if any(not math.isfinite(lp) for lp in token_log_probs):
        return ExtractionResult(log_prob=float("-inf"), prob=0.0, num_tokens=n)

    log_prob = sum(token_log_probs)
    return ExtractionResult(log_prob=log_prob, prob=math.exp(log_prob), num_tokens=n)

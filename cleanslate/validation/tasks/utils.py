"""Custom lm-eval metrics for CleanSlate QA benchmark."""

from __future__ import annotations

from cleanslate.validation.judge import is_correct


def llm_judge_match(references: list[str], predictions: list[str]) -> float:
    """Score answer correctness via LLM judge."""
    if not references or not predictions:
        return 0.0

    ref = references[0].strip()
    pred = predictions[0].strip()
    return 1.0 if is_correct(ref, pred) else 0.0

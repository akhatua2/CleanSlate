"""Custom lm-eval metrics for CleanSlate QA benchmark."""

from __future__ import annotations


def llm_judge_match(references: list[str], predictions: list[str]) -> float:
    """Score answer correctness via Gemini LLM judge."""
    if not references or not predictions:
        return 0.0

    ref = references[0].strip()
    pred = predictions[0].strip()

    # Skip API call for obvious matches
    if ref.lower() == pred.lower():
        return 1.0
    if ref.lower() in pred.lower() or pred.lower() in ref.lower():
        return 1.0

    from cleanslate.validation.judge import is_correct

    return 1.0 if is_correct(ref, pred) else 0.0

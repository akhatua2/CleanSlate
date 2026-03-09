"""Custom metrics for CleanSlate QA benchmark.

For fast evaluation, use contains_match (heuristic).
For accurate evaluation, use llm_judge_match (calls Gemini, slower).

Note: lm-eval metrics only receive (references, predictions), not the question.
The llm_judge_match metric works without the question since our judge prompt
only compares reference vs model_answer semantically.
"""

import os
from functools import lru_cache


def contains_match(references: list[str], predictions: list[str]) -> float:
    """Fast heuristic: check if reference is contained in prediction or vice versa.

    Handles cases like:
    - Target: "The 1980s" vs Prediction: "1980s"
    - Target: "Baltimora" vs Prediction: "Baltimora (the band)"
    """
    if not references or not predictions:
        return 0.0

    ref = references[0].lower().strip()
    pred = predictions[0].lower().strip()

    if ref in pred or pred in ref:
        return 1.0
    return 0.0


def llm_judge_match(references: list[str], predictions: list[str]) -> float:
    """LLM-based semantic matching using Gemini judge.

    Slower but more accurate than contains_match.
    Requires GEMINI_API_KEY environment variable.
    """
    if not references or not predictions:
        return 0.0

    ref = references[0].strip()
    pred = predictions[0].strip()

    # Skip API call if obvious match
    if ref.lower() == pred.lower():
        return 1.0
    if ref.lower() in pred.lower() or pred.lower() in ref.lower():
        return 1.0

    # Call LLM judge
    try:
        from cleanslate.fsid.qa.judge import is_correct
        # Note: is_correct expects (question, reference, model_answer)
        # We pass empty question since judge prompt doesn't use it
        result = is_correct("", ref, pred)
        return 1.0 if result else 0.0
    except Exception:
        # Fallback to contains_match on error
        return contains_match(references, predictions)

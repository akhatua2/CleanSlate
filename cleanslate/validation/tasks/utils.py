"""Custom metrics for CleanSlate QA benchmark."""


def contains_match(references: list[str], predictions: list[str]) -> float:
    """Fast heuristic: check if reference is contained in prediction or vice versa."""
    if not references or not predictions:
        return 0.0

    ref = references[0].lower().strip()
    pred = predictions[0].lower().strip()

    if ref in pred or pred in ref:
        return 1.0
    return 0.0

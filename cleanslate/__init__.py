"""CleanSlate - Sequence likelihood estimation for LLMs."""

from cleanslate.evaluator import (
    MemorizationConfig,
    MemorizationEvaluator,
    MemorizationResult,
    SlidingWindowResult,
    ExperimentSummary,
    save_results,
)
from cleanslate.experiment import Experiment
from cleanslate.generation import GenerationExperiment

__all__ = [
    # New unified evaluator
    "MemorizationConfig",
    "MemorizationEvaluator",
    "MemorizationResult",
    "SlidingWindowResult",
    "ExperimentSummary",
    "save_results",
    # Legacy classes
    "Experiment",
    "GenerationExperiment",
]

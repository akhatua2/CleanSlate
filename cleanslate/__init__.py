"""Cleanslate - Sequence likelihood estimation for LLMs."""

from .experiment import Experiment
from .generation import GenerationExperiment, GenerationRow
from .jailbreak import JailbreakExperiment, JailbreakRow
from .results import ExperimentResults, SampleResult
from .backends.base import GenerationResult, ProbabilityResult
from .prompts import PromptTemplate, get_template, get_templates, list_templates, ALL_TEMPLATES
from .evaluation import (
    evaluate_lyrics,
    evaluate_jailbreak_results,
    summarize_evaluations,
    LyricsEvalResult,
)

__version__ = "0.1.0"
__all__ = [
    # Experiments
    "Experiment",
    "GenerationExperiment",
    "JailbreakExperiment",
    # Results
    "ExperimentResults",
    "SampleResult",
    "GenerationResult",
    "GenerationRow",
    "JailbreakRow",
    "ProbabilityResult",
    # Prompts
    "PromptTemplate",
    "get_template",
    "get_templates",
    "list_templates",
    "ALL_TEMPLATES",
    # Evaluation
    "evaluate_lyrics",
    "evaluate_jailbreak_results",
    "summarize_evaluations",
    "LyricsEvalResult",
]

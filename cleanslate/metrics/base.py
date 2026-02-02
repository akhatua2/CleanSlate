"""Base class for evaluation metrics.

Metrics compare predicted output against target/reference text to measure
how similar they are. Different metrics capture different aspects:

- N-gram metrics (BLEU, ROUGE): Count overlapping words/phrases
- Sequence metrics (ROUGE-L): Check structural similarity
- Probability metrics: How likely the model was to generate the target

All scores are typically 0-1, where higher = more similar.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result from computing a metric."""
    
    name: str  # Metric name (e.g., "BLEU-1", "ROUGE-L")
    score: float  # Score value (typically 0-1)
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.4f}"


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Metrics compare predicted tokens/text against target tokens/text.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass
    
    @abstractmethod
    def compute(
        self,
        predicted: list[str],
        target: list[str],
    ) -> MetricResult:
        """
        Compute the metric.
        
        Args:
            predicted: List of predicted tokens
            target: List of target/reference tokens
            
        Returns:
            MetricResult with name and score
        """
        pass
    
    def compute_text(
        self,
        predicted: str,
        target: str,
    ) -> MetricResult:
        """
        Compute metric from text strings.
        
        Default implementation splits on whitespace.
        Override for different tokenization.
        """
        pred_tokens = predicted.split()
        tgt_tokens = target.split()
        return self.compute(pred_tokens, tgt_tokens)

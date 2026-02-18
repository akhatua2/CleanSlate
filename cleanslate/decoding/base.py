"""Base classes for decoding strategies.

This module provides the abstract interface for pluggable decoding strategies
(greedy, beam search, sampling, etc.) used in text generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class GenerationSequence:
    """A single generated sequence with token-level data."""

    generated_text: str
    generated_token_ids: list[int]
    generated_tokens: list[str]

    # Token-level probability data
    token_log_probs: list[float]
    top_k_tokens: list[list[int]] = field(default_factory=list)
    top_k_probs: list[list[float]] = field(default_factory=list)

    # Sequence-level metrics
    total_log_prob: float = 0.0
    score: float = 0.0  # May include length penalty

    generation_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecodingOutput:
    """Result from a decoding strategy for a single seed.

    Contains all hypotheses (beams) for analysis, not just the best one.
    For greedy decoding, sequences will have length 1.
    For beam search, sequences will have length beam_width.
    """

    sequences: list[GenerationSequence]  # All hypotheses, ranked by score
    strategy_name: str
    strategy_params: dict = field(default_factory=dict)

    # Convenience accessor
    @property
    def best_sequence(self) -> GenerationSequence:
        """The top-scoring sequence."""
        return self.sequences[0] if self.sequences else None

    @property
    def best_text(self) -> str:
        """The generated text from the top sequence."""
        return self.best_sequence.generated_text if self.best_sequence else ""

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    def to_dict(self) -> dict:
        return {
            "sequences": [s.to_dict() for s in self.sequences],
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
        }


class DecodingStrategy(ABC):
    """Abstract base class for decoding strategies.

    Subclasses implement different decoding algorithms:
    - GreedyDecoding: Always pick the highest probability token
    - BeamSearchDecoding: Maintain top-k hypotheses at each step
    - SamplingDecoding: Sample from the probability distribution (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'greedy', 'beam')."""
        ...

    @abstractmethod
    def decode(
        self,
        engine: Any,
        tokenizer: Any,
        seeds: list[str],
        max_new_tokens: int,
        capture_top_k: int = 50,
    ) -> list[DecodingOutput]:
        """Decode seeds into generated sequences.

        Args:
            engine: SGLang Engine instance
            tokenizer: Tokenizer for encoding/decoding
            seeds: List of prompt strings
            max_new_tokens: Maximum tokens to generate per seed
            capture_top_k: Number of top tokens to capture per position

        Returns:
            List of DecodingOutput, one per seed
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize strategy configuration for reproducibility."""
        ...

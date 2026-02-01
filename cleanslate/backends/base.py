"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProbabilityResult:
    """Result of a probability computation."""

    log_prob: float  # Total log probability: sum of log P(token_i | context)
    perplexity: float  # exp(-log_prob / num_tokens)
    num_tokens: int  # Number of target tokens
    token_log_probs: list[float]  # Per-token log probabilities
    token_ranks: list[int] | None = None  # Rank of each target token (1 = top choice)

    @property
    def avg_log_prob(self) -> float:
        """Average log probability per token."""
        if self.num_tokens == 0:
            return 0.0
        return self.log_prob / self.num_tokens

    @property
    def rank_1_fraction(self) -> float:
        """Fraction of tokens where model's top choice matched target."""
        if self.token_ranks is None or len(self.token_ranks) == 0:
            return 0.0
        return sum(1 for r in self.token_ranks if r == 1) / len(self.token_ranks)

    @property
    def avg_rank(self) -> float:
        """Average rank of target tokens (lower = more memorized)."""
        if self.token_ranks is None or len(self.token_ranks) == 0:
            return float("inf")
        return sum(self.token_ranks) / len(self.token_ranks)


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.

    Backends compute P(target | seed) by:
    1. Tokenizing seed + target
    2. Running forward pass to get logits
    3. Extracting log probabilities for target tokens
    4. Summing log probs in log-space
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this backend is using (cpu, cuda, mps, etc)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/path."""
        pass

    @abstractmethod
    def compute(self, seed: str, target: str) -> ProbabilityResult:
        """
        Compute P(target | seed).

        Args:
            seed: The prefix/prompt text
            target: The continuation text whose probability we want

        Returns:
            ProbabilityResult with log probability, perplexity, etc.
        """
        pass

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[ProbabilityResult]:
        """
        Compute probabilities for multiple (seed, target) pairs.

        Default implementation just loops. Backends can override for efficiency.
        """
        return [self.compute(seed, target) for seed, target in pairs]

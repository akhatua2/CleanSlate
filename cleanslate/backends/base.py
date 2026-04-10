"""Inference backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProbabilityResult:
    """P(target | seed) computation result."""

    log_prob: float
    num_tokens: int
    token_log_probs: list[float]
    token_ranks: list[int] = field(default_factory=list)


class InferenceBackend(ABC):
    """Abstract inference backend. Only computes P(target | seed)."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def tokenizer(self) -> object: ...

    @abstractmethod
    def compute(self, seed: str, target: str) -> ProbabilityResult: ...

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[ProbabilityResult]:
        return [self.compute(s, t) for s, t in pairs]
